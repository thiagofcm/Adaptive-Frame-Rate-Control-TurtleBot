#!/usr/bin/env python3
"""AdaptiveFPSEnv - Gymnasium environment for learning the adaptive LiDAR
sensing-frequency policy on the TurtleBot3 Stage 9 ROS2/Gazebo stack, with
the frozen Stage 9 TD3 navigation actor unchanged underneath it.

Sibling to FixedFPS/ (the validated fixed-interval reference), not a
dependent of it. This is a behavior-preserving EXTRACTION of the already
validated sensing/synchronization mechanics from
AdaptiveFPS/scripts/eval_adaptive_fps.py -- that file is imported from,
never modified, never duplicated logic-for-logic. Naming and step()
structure deliberately mirror the F1TENTH adaptive_fps_env.py reference
wherever the underlying concepts are equivalent (see per-method comments
for the TurtleBot-specific correspondences, since ROS/Gazebo asynchrony
means the mechanism can't be identical).

Reused, unmodified, from eval_adaptive_fps.py:
    - FixedSensingEvaluator: the ROS2 Node class itself (scan gate,
      step_comm/goal_comm/pause/unpause clients, goal/odom/clock tracking,
      frozen TD3 model loading -- all of it, instantiated as-is).
    - wait_for_episode_ready() / _safe_stop(): the full validated
      episode-ready handshake (new /goal_pose, fresh /clock, fresh /odom
      with reset-pose verification when appropriate, forced-fresh
      /scan_gated), bounded-retry, "don't proceed on timeout" semantics.
    - MAX_INIT_ATTEMPTS, NATIVE_SCAN_HZ.
Reused, unmodified, from common/utilities.py:
    - step(), init_episode(), pause_simulation(), unpause_simulation().
Reused, unmodified:
    - AdaptiveFPS/rewards/adaptive_fps_reward.py's get_adaptive_reward().

What's genuinely new: reset()/step() drive the control loop directly
(instead of eval_adaptive_fps.py's own run_episode()), and the adaptive
action dynamically retargets the existing gate's `node.k` (exposed here
as `self.obs_interval`) after each consumed frame, rather than
eval_adaptive_fps.py's convention of a single `k` fixed for an entire
episode from the CLI --fps argument.

Not included in this first version (deliberately, per spec): PPO/LSTM,
augmented/temporal observation features (fps_ratio, obs_age_ratio,
frame_ratio), frame-cost reward shaping, success/failure/budget reward
terms, and any evaluation/export logic (episodes.csv/steps.csv/etc. stay
the evaluation script's responsibility, added in a later task).
"""
import copy
import os
import sys
import time

import numpy as np
import gymnasium
from gymnasium import spaces

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from turtlebot3_drl.common import utilities as util
from turtlebot3_drl.common.settings import SUCCESS, EPISODE_TIMEOUT_SECONDS
from turtlebot3_drl.drl_environment.drl_environment import NUM_SCAN_SAMPLES, MAX_GOAL_DISTANCE

BASE_PATH = os.environ["DRLNAV_BASE_PATH"]
sys.path.insert(0, os.path.join(BASE_PATH, "AdaptiveFPS", "scripts"))
from eval_adaptive_fps import (  # noqa: E402  (path must be set up first)
    FixedSensingEvaluator,
    wait_for_episode_ready,
    _safe_stop,
    MAX_INIT_ATTEMPTS,
    NATIVE_SCAN_HZ,
    stamp_to_sec,
)

sys.path.insert(0, BASE_PATH)
from AdaptiveFPS.rewards.adaptive_fps_reward import get_adaptive_reward  # noqa: E402

# From AdaptiveFPS/action_space.txt -- reused, not re-derived.

class AdaptiveFPSEnv(gymnasium.Env):
    """One Gym step() == one navigation/control transition (one step_comm
    RPC to DRLEnvironment) -- NOT one full sensing interval. Native /scan
    arrivals are counted independently of the control loop (see
    FixedSensingEvaluator._on_real_scan, reused unmodified), since the
    control loop runs at ~hundreds-to-thousands of Hz while /scan is fixed
    at NATIVE_SCAN_HZ (~50 Hz) -- counting control steps would not measure
    the sensing interval correctly. F1TENTH's `steps_since_last_obs`
    counts control steps because its control frequency IS its sensing
    reference rate; the TurtleBot equivalent is native /scan arrivals
    (`scans_since_last_obs`, i.e. `node.scans_since_forward`), never
    control transitions.

    Observation (frozen TD3 navigation input, section 1 of step()): the
    raw 44-D state vector DRLEnvironment's own get_state() produces --
    unaugmented, unchanged. Observation (PPO-facing, returned by
    reset()/step()): that same 44-D vector plus 3 sensing-state features
    appended at the end (fps_ratio, obs_age_ratio,
    episode_frame_count_ratio -- see _augmented_features()), 47-D total.
    The frozen TD3 actor never sees the augmented 3 dims.

    TurtleBot equivalent of F1TENTH's self.last_sampled_scan: there is no
    separate held-scan object here. The fresh/held LiDAR observation is
    already cached inside DRLEnvironment itself (self.scan_ranges),
    updated only when /scan_gated delivers a newly forwarded native scan
    -- self.current_observation's LiDAR portion already *is* that cache's
    current value, fresh or stale, exactly as drl_environment.py's own
    unmodified sample-and-hold get_state() produces it. No redundant
    variable is introduced just to mirror the name.

    Reward: get_adaptive_reward()'s normalized goal-progress signal alone
    (see adaptive_fps_reward.py). No frame-cost, success/failure, or
    budget terms are added here -- deferred to a later task, per spec.
    The raw navigation reward is still exposed in `info["nav_reward"]`
    for diagnostics.

    Threading/locking: none used, none needed. Every ROS callback here
    (the gate, goal/odom/clock tracking, this env's own native-scan
    counter) only ever runs synchronously inside `rclpy.spin_once()`
    calls made from this same Python thread (inside util.step() and the
    handshake's _spin_until()) -- exactly like the rest of this codebase.
    There is no background executor thread and therefore no concurrent
    access to any shared counter.
    """

    def __init__(self):
        super().__init__() 

        if not rclpy.ok():
            rclpy.init()

        self.node = FixedSensingEvaluator()
        self.native_scan_count = 0
        self.node.create_subscription(LaserScan, "scan", self._on_native_scan, qos_profile_sensor_data)

        # Action Space definition
        self.fps_choices = [0.2, 0.5, 1.0, 5.0, 10.0]
        self.action_space = spaces.Discrete(len(self.fps_choices))

        # Fixed normalization denominators for the augmented sensing-state
        # features (fps_ratio/obs_age_ratio/episode_frame_count_ratio,
        # see _augmented_features()) -- computed once from fps_choices/
        # NATIVE_SCAN_HZ/EPISODE_TIMEOUT_SECONDS, mirroring the F1TENTH
        # reference's own fixed-denominator convention
        # (self.max_obs_interval = int(self.control_frequency /
        # min(self.fps_choices)); episode_frame_count normalized by
        # self.budget), adapted to TurtleBot's native-scan-count gate
        # instead of control-step counting, and to the fact this env has
        # no "budget" constructor concept -- the frame-count bound is
        # instead derived from the episode timeout and the fastest
        # available sensing rate.
        self._min_obs_interval = max(1, round(NATIVE_SCAN_HZ / max(self.fps_choices)))  # fastest possible k (10 Hz)
        self._max_obs_interval = max(1, round(NATIVE_SCAN_HZ / min(self.fps_choices)))  # slowest possible k (0.2 Hz)
        self._max_episode_frame_count = max(
            1, round((EPISODE_TIMEOUT_SECONDS * NATIVE_SCAN_HZ) / self._min_obs_interval))

        # Observation Space definition: original 44-D navigation state +
        # 3 sensing-state features (fps_ratio, obs_age_ratio,
        # episode_frame_count_ratio), all three already clipped to [0, 1]
        # by _augmented_features() below.
        low = np.array([0.0] * NUM_SCAN_SAMPLES + [0.0, -1.0, -1.0, -1.0] + [0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0] * NUM_SCAN_SAMPLES + [1.0, 1.0, 1.0, 1.0] + [1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # flag for reseting GAZEBO env
        self._expect_reset = True

        # Adaptive FPS Variables:
        self.world_step_count = 0
        self.prev_navigation_action = [0.0, 0.0]
        self.last_gated_scan_count = 0
        self.episode_frame_count = 0
        self.current_fps = None
        self.obs_interval = None
        self.current_observation = None
        self._initial_goal_distance = None
        self._previous_goal_distance = None
        self.frame_cost = 0.0

    def _on_native_scan(self, msg):
        self.native_scan_count += 1

    def _augmented_features(self):
        """fps_ratio / obs_age_ratio / episode_frame_count_ratio -- the
        three sensing-state features appended to the PPO-facing
        observation (never to the frozen TD3 navigation observation).
        Reuses only existing, already-validated state -- no new counters,
        no changes to the gate or to frame_consumed semantics:

        - fps_ratio: the currently ACTIVE sensing rate (self.current_fps,
          not a just-requested action that hasn't taken effect yet)
          normalized by the fastest available choice.
        - obs_age_ratio: TurtleBot equivalent of F1TENTH's
          steps_since_last_obs / max_obs_interval, substituting native
          /scan arrivals for control steps (see class docstring) --
          node.scans_since_forward is the exact same counter the
          validated gate (_on_real_scan, unmodified) already maintains:
          incremented on every native scan, reset to 0 the instant a
          forward happens. Reading it here is purely observational.
          Normalized by the fixed _max_obs_interval (the slowest
          available rate's interval), matching F1TENTH's fixed-
          denominator convention, then clipped to [0, 1].
        - episode_frame_count_ratio: self.episode_frame_count (already
          includes the initial reset observation, semantics unchanged)
          normalized by the derived _max_episode_frame_count and clipped
          to [0, 1].
        """
        node = self.node
        fps_ratio = self.current_fps / max(self.fps_choices)
        obs_age_ratio = float(np.clip(node.scans_since_forward / self._max_obs_interval, 0.0, 1.0))
        episode_frame_count_ratio = float(np.clip(self.episode_frame_count / self._max_episode_frame_count, 0.0, 1.0))
        return fps_ratio, obs_age_ratio, episode_frame_count_ratio

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        node = self.node
        node.episode_index += 1

        # Reset Adaptive FPS variables:
        self.current_fps = max(self.fps_choices)
        self.obs_interval = max(1, round(NATIVE_SCAN_HZ / self.current_fps))
        node.k = self.obs_interval #interval for the gate mechanism
        node.scans_since_forward = 0 #?
        node.gated_scan_count = 0 # Current not-skipped scans (How many were actually seen)

        # Reset ROS settings:
        # /odom, /clock and /goal_pose only advance while Gazebo is running
        util.unpause_simulation(node, 0)
        goal_before = node.goal_baseline

        # Handshake: wait for /goal_pose, /clock, /odom to advance, then force a fresh /scan_gated forward.
        ready, status, attempt = False, "", 0
        for attempt in range(MAX_INIT_ATTEMPTS):
            ready, status = wait_for_episode_ready(node, self._expect_reset, goal_before)
            if ready:
                break
            node.get_logger().warning(
                f"[episode {node.episode_index}] handshake FAILED (attempt {attempt + 1}/{MAX_INIT_ATTEMPTS}): "
                f"{status} -- not starting TD3 control, episode NOT counted")
            _safe_stop(node)
            if attempt < MAX_INIT_ATTEMPTS - 1:
                util.unpause_simulation(node, 0)

        if not ready:
            # Fail clearly rather than silently returning a bogus
            # observation -- matches this project's established
            # "don't proceed on timeout" handshake semantics.
            raise RuntimeError(
                f"AdaptiveFPSEnv.reset(): episode-ready handshake failed after "
                f"{MAX_INIT_ATTEMPTS} attempts: {status}")

        # Save current goal
        node.goal_baseline = node.goal_msg_count
        node.get_logger().info(
            f"[episode {node.episode_index}] ready after {attempt} retries: "
            f"initial_fps={self.current_fps} obs_interval={self.obs_interval}")

        # Reset Observation and scan ages
        observation = util.init_episode(node) # initial observation (44D)
        self.current_observation = observation
        self.world_step_count = 0
        self.prev_navigation_action = [0.0, 0.0]
        self.native_scan_count = 1 #GT scan acquired
        
        #last_gated_scan_count is the last saved value of scans 
        # that have been forwarded to the navigation stack
        self.last_gated_scan_count = node.gated_scan_count
        self.episode_frame_count = 1 #Counted frame/scan acquired
        self._initial_goal_distance = observation[NUM_SCAN_SAMPLES] * MAX_GOAL_DISTANCE
        self._previous_goal_distance = self._initial_goal_distance

        # Diagnostics-only episode-time origin (x/y/wall_time/sim_time in
        # info, for steps.csv/trajectory.csv) -- reuses node.latest_odom,
        # already maintained by FixedSensingEvaluator (no new subscriber).
        # sim_time is derived from the odom message's own header.stamp,
        # not the most recently arrived /clock tick, so it stays exactly
        # synchronized with whichever odom reading also supplies x/y at
        # each step. Same pattern eval_adaptive_fps.py's run_episode()
        # already uses for episode_start_sim.
        self._episode_start_wall = time.perf_counter()
        self._episode_start_sim = stamp_to_sec(node.latest_odom.header.stamp) if node.latest_odom is not None else None

        # Create reset observation and Info
        fps_ratio, obs_age_ratio, episode_frame_count_ratio = self._augmented_features()
        ppo_observation = np.concatenate(
            [np.asarray(observation, dtype=np.float32), [fps_ratio, obs_age_ratio, episode_frame_count_ratio]]
        ).astype(np.float32)
        info = {
            "current_fps": self.current_fps,
            "obs_interval": self.obs_interval,
            "scan_interval_k": self.obs_interval,  # backward-compatible alias
            "fps_ratio": fps_ratio,
            "obs_age_ratio": obs_age_ratio,
            "episode_frame_count_ratio": episode_frame_count_ratio,
        }
        return ppo_observation, info

    def _navigation_step(self, navigation_action):
        """TurtleBot equivalent of F1TENTH's self._physics_step(): one
        control transition -- DRLEnvironment applies navigation_action via
        /cmd_vel and returns its next 44-D state. Thin wrapper purely for
        structural readability; changes no behavior."""
        return util.step(self.node, navigation_action, self.prev_navigation_action)

    def step(self, action):
        node = self.node
        self.world_step_count += 1
        requested_fps = self.fps_choices[int(action)]

        # ---------------------------------
        # 1. Frozen navigation uses currently held observation
        # ---------------------------------
        navigation_action = node.model.get_action(self.current_observation, False, self.world_step_count, False)

        # ---------------------------------
        # 2. One control transition
        # ---------------------------------
        observation, nav_reward, done, outcome, dist_trav = self._navigation_step(navigation_action)
        self.current_observation = observation

        # ---------------------------------
        # 3. Sampling Timer
        # ---------------------------------
        new_gated_scans = node.gated_scan_count - self.last_gated_scan_count
        frame_consumed = new_gated_scans > 0
        self.last_gated_scan_count = node.gated_scan_count

        if frame_consumed:
            self.episode_frame_count += new_gated_scans  # exact, even if >1 scan
                                                           # forwarded within one step's gap
            # The action selected at THIS sampling instant controls the
            # FUTURE sensing rate -- never retroactive to navigation_action
            # above, which already used the previously-held (or
            # just-forced) scan. _on_real_scan re-reads node.k fresh on
            # every real /scan arrival, so this takes effect starting
            # with the NEXT interval only.
            self.current_fps = requested_fps
            self.obs_interval = max(1, round(NATIVE_SCAN_HZ / self.current_fps))
            node.k = self.obs_interval


        # ---------------------------------
        # 4. Reward
        # ---------------------------------
        goal_distance_m = observation[NUM_SCAN_SAMPLES] * MAX_GOAL_DISTANCE
        adaptive_reward = get_adaptive_reward(
            self._previous_goal_distance, goal_distance_m, self._initial_goal_distance)
        self._previous_goal_distance = goal_distance_m
        frame_penalty = self.frame_cost if frame_consumed else 0.0
        reward = adaptive_reward - frame_penalty


        self.prev_navigation_action = copy.deepcopy(navigation_action)
        terminated = bool(done)
        truncated = False  # DRLEnvironment's own TIMEOUT outcome is already
                            # reported via `done`/`outcome`; no separate
                            # Gym-level truncation concept is introduced here.

        if terminated:
            if outcome == SUCCESS:
                reward += 0.5
            else:
                reward -= 0.5
            util.pause_simulation(node, 0)
            self._expect_reset = not (outcome == SUCCESS)

        # ---------------------------------
        # Sensing-state features -- appended to the PPO-facing observation
        # only, never fed to the frozen TD3 navigation policy (section 1
        # above already ran on the un-augmented self.current_observation).
        # TEMPORARY debug print for this validation stage.
        # ---------------------------------
        fps_ratio, obs_age_ratio, episode_frame_count_ratio = self._augmented_features()
        # if frame_consumed or self.world_step_count % 200 == 0:
        #     print(
        #         f"[AdaptiveFPS] "
        #         f"step={self.world_step_count:06d} "
        #         f"fresh={int(frame_consumed)} "
        #         f"fps={self.current_fps:4.1f} "
        #         f"fps_ratio={fps_ratio:.3f} "
        #         f"obs_age={node.scans_since_forward:3d} "
        #         f"obs_age_ratio={obs_age_ratio:.3f} "
        #         f"frame_count={self.episode_frame_count:4d} "
        #         f"frame_count_ratio={episode_frame_count_ratio:.3f}"
        #     )
        # ---------------------------------
        # 5. Adaptive-policy observation (PPO-facing)
        # ---------------------------------
        ppo_observation = np.concatenate(
            [np.asarray(observation, dtype=np.float32), [fps_ratio, obs_age_ratio, episode_frame_count_ratio]]
        ).astype(np.float32)

        # ---------------------------------
        # Diagnostics only: x/y/wall_time/sim_time, for steps.csv/
        # trajectory.csv logging by the evaluator. Reuses node.latest_odom
        # (already maintained by FixedSensingEvaluator) -- no new ROS
        # subscriber, no effect on control/reward/timing.
        # ---------------------------------
        x = y = float("nan")
        sim_time = None
        if node.latest_odom is not None:
            x = node.latest_odom.pose.pose.position.x
            y = node.latest_odom.pose.pose.position.y
            if self._episode_start_sim is not None:
                sim_time = stamp_to_sec(node.latest_odom.header.stamp) - self._episode_start_sim
        wall_time = time.perf_counter() - self._episode_start_wall

        # ---------------------------------
        # 6. Info
        # ---------------------------------
        info = {
            "frame_penalty": frame_penalty,
            "frame_cost": self.frame_cost,
            "navigation_action": navigation_action,
            "nav_reward": nav_reward,
            "current_fps": self.current_fps,
            "obs_interval": self.obs_interval,
            "scans_since_last_obs": node.scans_since_forward,
            "frame_consumed": frame_consumed,
            "lidar_fresh": frame_consumed,
            "episode_frame_count": self.episode_frame_count,
            "goal_distance_m": goal_distance_m,
            "outcome": outcome,
            "outcome_str": util.translate_outcome(outcome),
            "native_scan_count": self.native_scan_count,
            "x": x,
            "y": y,
            "wall_time": wall_time,
            "sim_time": sim_time,
            "fps_ratio": fps_ratio,
            "obs_age_ratio": obs_age_ratio,
            "episode_frame_count_ratio": episode_frame_count_ratio,
            # backward-compatible aliases
            "scan_interval_k": self.obs_interval,
            "scans_since_last_observation": node.scans_since_forward,
        }
        return ppo_observation, float(reward), terminated, truncated, info

    def close(self):
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


gymnasium.register(id="AdaptiveFPSTurtleBot-v0", entry_point="AdaptiveFPS.env.adaptive_fps_env:AdaptiveFPSEnv")
