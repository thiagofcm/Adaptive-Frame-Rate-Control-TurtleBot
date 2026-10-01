import copy
import math
import os
import sys
import time
import torch
import math 
import numpy as np
import gymnasium
from gymnasium import spaces

import rclpy
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty as EmptyMsg
from std_srvs.srv import Empty

from turtlebot3_drl.common import utilities as util
from turtlebot3_drl.common.storagemanager import StorageManager
from turtlebot3_drl.common.settings import (
    SUCCESS, COLLISION_WALL, COLLISION_OBSTACLE, TIMEOUT, TUMBLE, ENABLE_STACKING)
from turtlebot3_drl.drl_agent.td3 import TD3
from turtlebot3_drl.drl_environment.drl_environment import NUM_SCAN_SAMPLES, MAX_GOAL_DISTANCE

BASE_PATH = os.environ["DRLNAV_BASE_PATH"]
sys.path.insert(0, BASE_PATH)

from AdaptiveFPS.env.gazebo_bridge import (
    GazeboSensingBridge,
    EPISODE_READY_TIMEOUT,
    HEARTBEAT_PERIOD_S,
    STAGE9_RESET_X,
    STAGE9_RESET_Y,
    RESET_POSITION_TOLERANCE_M,
    RESET_VELOCITY_TOLERANCE_MPS,
    MAX_INIT_ATTEMPTS,
    stamp_to_sec,
    quaternion_to_yaw,
)
from AdaptiveFPS.env.adaptive_obs import ADAPTIVE_FRAME_BUDGET, ADAPTIVE_OBS_DIM  # noqa: E402

PPO_RATE_HZ = 10.0
PPO_DT = 1.0 / PPO_RATE_HZ

# Frozen TD3 navigation controller checkpoint (moved from gazebo_bridge.py).
MODEL_RUN_NAME    = "examples/td3_0_stage9"
LOAD_EPISODE      = 7400
ALGORITHM         = "td3"

# Reward Constants
R_SUCCESS = 15.0
R_COLLISION = -15.0
R_TIMEOUT = -15.0


class AdaptiveFPSEnv(gymnasium.Env):
    def __init__(self):
        super().__init__()

        if not rclpy.ok():
            rclpy.init()

        self.node = GazeboSensingBridge()
        self.navigation_model = self._load_navigation_model()
        self.native_scan_count = 0
        self.node.create_subscription(LaserScan, "scan", self._on_native_scan, qos_profile_sensor_data)
        self.node.create_subscription(Odometry, "obstacle/odom", self._on_obstacle_odom, QoSProfile(depth=10))

        # Per-episode recording buffers (lidar + obstacle positions over time)
        self._recording_active = False
        self._lidar_t_wall = []
        self._lidar_t_sim = []
        self._lidar_ranges = []
        self._obstacle_t_wall = []
        self._obstacle_t_sim = []
        self._obstacle_frame_id = []
        self._obstacle_x = []
        self._obstacle_y = []
        self._obstacle_yaw = []

        # Action Space definition
        self.fps_choices = [0.2, 0.5, 1.0, 5.0, 10.0]
        self.action_space = spaces.Discrete(len(self.fps_choices))

        # Observation Space definition: canonical 43-D sensing observation
        # (retained 40-beam scan + fps_ratio, obs_age_ratio, frame_ratio),
        # every entry in [0, 1].
        self.observation_space = spaces.Box(
            low=np.zeros(ADAPTIVE_OBS_DIM, dtype=np.float32),
            high=np.ones(ADAPTIVE_OBS_DIM, dtype=np.float32),
            dtype=np.float32)


        # Adaptive FPS Variables:
        self.world_step_count = 0
        self.prev_navigation_action = [0.0, 0.0]
        self.steps_since_last_obs = 0
        self.fps_ratio = None                  # last values inserted into the 43-D PPO observation
        self.obs_age_ratio = None              # (set by get_augmented_obs; None until the first reset)
        self.episode_scan_count_ratio = None
        self.prev_gated_scan_count = 0
        self.episode_scan_count = 0
        self.current_fps = None
        self.obs_interval = None
        self.current_observation = None
        self._initial_goal_distance = None
        self._previous_goal_distance = None
        self.frame_cost = 0.005
        self._max_obs_interval = PPO_RATE_HZ / (min(self.fps_choices))
        self.budget = 450

    def _load_navigation_model(self):
        """Load the frozen TD3 navigation controller;
        it consumes the unchanged 44-D navigation state. Returns the TD3 agent."""
        # --- load the frozen actor exactly as drl_agent.py does ---
        device = torch.device("cpu")
        sim_speed = util.get_simulation_speed(util.stage)
        sm = StorageManager(ALGORITHM, MODEL_RUN_NAME, LOAD_EPISODE, device, util.stage)
        model = sm.load_model()
        model.device = device
        sm.load_weights(model.networks)

        assert isinstance(model, TD3), f"expected a TD3 model, got {type(model)}"
        assert model.state_size == NUM_SCAN_SAMPLES + 4, \
            f"state_size mismatch: model={model.state_size} env={NUM_SCAN_SAMPLES + 4}"
        assert not ENABLE_STACKING, "AdaptiveFPSEnv does not support ENABLE_STACKING"

        self.node.get_logger().info(
            f"loaded {MODEL_RUN_NAME} episode {LOAD_EPISODE}: state_size={model.state_size} "
            f"step_time={model.step_time} sim_speed={sim_speed} device={device}")
        return model

    # Recording-related callbacks and utilities
    def _on_native_scan(self, msg):
        self.native_scan_count += 1
        if self._recording_active:
            t_wall, t_sim = self._recording_times(msg.header.stamp)
            self._lidar_t_wall.append(t_wall)
            self._lidar_t_sim.append(t_sim)
            self._lidar_ranges.append(list(msg.ranges))

    def _on_obstacle_odom(self, msg):
        if not self._recording_active:
            return
        t_wall, t_sim = self._recording_times(msg.header.stamp)
        self._obstacle_t_wall.append(t_wall)
        self._obstacle_t_sim.append(t_sim)
        self._obstacle_frame_id.append(msg.child_frame_id)
        self._obstacle_x.append(msg.pose.pose.position.x)
        self._obstacle_y.append(msg.pose.pose.position.y)
        self._obstacle_yaw.append(quaternion_to_yaw(msg.pose.pose.orientation))

    def _recording_times(self, stamp):
        """(t_wall, t_sim) relative to this episode's origin, same
        convention as the x/y/wall_time/sim_time diagnostics in step()
        (reset()'s _episode_start_wall/_episode_start_sim) -- t_sim is
        None if sim time wasn't available at episode start, matching
        record_episode.py's own NaN-on-missing convention (translated to
        None here, NaN only at the final np.array() call in eval.py)."""
        t_wall = time.perf_counter() - self._episode_start_wall
        t_sim = (stamp_to_sec(stamp) - self._episode_start_sim) if self._episode_start_sim is not None else None
        return t_wall, t_sim

    def get_recording_arrays(self):
        """Snapshot this episode's buffered lidar/obstacle recordings as
        numpy arrays, in exactly the schema
        tools/episode_recorder/record_episode.py's Episode.write() saves
        to lidar.npz/obstacles.npz -- so the same
        tools/episode_recorder/make_video_fps_v2.py reads either source
        unmodified. Intended to be called once, right after an episode
        ends (terminated/truncated=True), by whichever script owns
        writing eval output files (this env itself never writes files,
        matching its existing "no evaluation/export logic" boundary --
        see class docstring). Does not clear/reset anything itself; the
        next reset() call does that.

        Returns (lidar_dict, obstacle_dict), each a plain dict of
        already-`np.array`-wrapped fields ready for
        np.savez_compressed(path, **lidar_dict)."""
        if self._lidar_ranges:
            max_len = max(len(r) for r in self._lidar_ranges)
            ranges_arr = np.full((len(self._lidar_ranges), max_len), np.nan, dtype=np.float32)
            for i, r in enumerate(self._lidar_ranges):
                ranges_arr[i, :len(r)] = r
        else:
            ranges_arr = np.zeros((0, 0), dtype=np.float32)
        lidar = {
            "t_wall": np.array(self._lidar_t_wall, dtype=np.float64),
            "t_sim": np.array([t if t is not None else np.nan for t in self._lidar_t_sim], dtype=np.float64),
            "ranges": ranges_arr,
        }
        obstacles = {
            "t_wall": np.array(self._obstacle_t_wall, dtype=np.float64),
            "t_sim": np.array([t if t is not None else np.nan for t in self._obstacle_t_sim], dtype=np.float64),
            "frame_id": np.array(self._obstacle_frame_id, dtype="<U64"),
            "x": np.array(self._obstacle_x, dtype=np.float64),
            "y": np.array(self._obstacle_y, dtype=np.float64),
            "yaw": np.array(self._obstacle_yaw, dtype=np.float64),
        }
        return lidar, obstacles

    def get_augmented_obs(self, nav_observation):
        """Canonical 43-D PPO observation: the retained LiDAR scan (first NUM_SCAN_SAMPLES dims of the
        44-D TD3 state, i.e. the held /scan_gated scan) followed by fps_ratio, obs_age_ratio and
        episode_scan_count_ratio. The three features are stored on the env (self.fps_ratio, ...) so
        info reports exactly the values placed in the observation.
        - fps_ratio: active sensing rate / PPO_RATE_HZ (10)
        - obs_age_ratio: PPO steps since the last fresh observation / _max_obs_interval, clipped to [0, 1]
        - episode_scan_count_ratio: episode_scan_count / budget, clipped to [0, 1]
        """
        self.fps_ratio = self.current_fps / PPO_RATE_HZ
        self.obs_age_ratio = float(np.clip(self.steps_since_last_obs / self._max_obs_interval, 0.0, 1.0))
        self.episode_scan_count_ratio = float(np.clip(self.episode_scan_count / self.budget, 0.0, 1.0))
        return np.concatenate(
            [np.asarray(nav_observation[:NUM_SCAN_SAMPLES], dtype=np.float32),
             [self.fps_ratio, self.obs_age_ratio, self.episode_scan_count_ratio]]
        ).astype(np.float32)

    def _spin_until(self, condition, timeout, label=None):
        """Spin this node's own callbacks until condition() is True or timeout
        elapses. Returns condition()'s final value -- never claims success on a
        timeout that condition() itself would reject.

        If `label` is given, logs an INFO heartbeat roughly every
        HEARTBEAT_PERIOD_S seconds while waiting, purely so a real (possibly
        long) wait is visibly distinguishable from a genuine hang, and so a
        stuck run shows exactly which condition it's stuck on. Purely
        diagnostic -- does not affect timing or the return value."""
        node = self.node
        start = time.monotonic()
        deadline = start + timeout
        next_heartbeat = start + HEARTBEAT_PERIOD_S
        while time.monotonic() < deadline:
            if condition():
                return True
            now = time.monotonic()
            if label is not None and now >= next_heartbeat:
                node.get_logger().info(
                    f"[episode {node.episode_index}] still waiting for {label} "
                    f"({now - start:.0f}s/{timeout:.0f}s)")
                next_heartbeat = now + HEARTBEAT_PERIOD_S
            rclpy.spin_once(node, timeout_sec=0.05)
        return condition()

    def _at_reset_pose(self):
        node = self.node
        if node.latest_odom is None:
            return False
        p = node.latest_odom.pose.pose.position
        t = node.latest_odom.twist.twist
        pos_ok = math.hypot(p.x - STAGE9_RESET_X, p.y - STAGE9_RESET_Y) <= RESET_POSITION_TOLERANCE_M
        vel_ok = abs(t.linear.x) <= RESET_VELOCITY_TOLERANCE_MPS and abs(t.angular.z) <= RESET_VELOCITY_TOLERANCE_MPS
        return pos_ok and vel_ok

    def _safe_stop(self):
        """Best-effort: zero the robot's commanded velocity and pause physics
        before a retry, so a stale command doesn't keep driving the robot while
        we wait, and so the next attempt starts from a known (paused) state."""
        node = self.node
        node.cmd_vel_pub.publish(Twist())
        util.pause_simulation(node, 0)

    def _issue_reset_recovery(self):
        """Issue Gazebo's /reset_simulation. Used by reset() for the single
        episode-boundary reset of every episode (this env owns all resets;
        gazebo_goals runs with external_reset:=true), and again for recovery
        after a failed handshake attempt -- as opposed to _safe_stop() (which
        only pauses/zeros velocity and changes no physical state). Teleports
        every model -- including the robot -- back to its world-file insertion
        pose and resets sim time to 0.

        Deliberately does NOT touch goal state: the goal marker's own Gazebo
        insertion pose is wherever it was last spawned for the CURRENT
        episode's intended goal, so this call cannot move or regenerate it.
        Nothing here calls task_fail/task_succeed, and goal_baseline/
        goal_msg_count/episode_index are all untouched -- this is a pure
        physics-state reset.

        Blocks (call_async + spin-until-future.done(), same idiom as
        utilities.py's pause_simulation/unpause_simulation) until Gazebo has
        ACKNOWLEDGED the request -- this confirms the request was issued,
        NOT that the reset has already propagated to /odom. The caller must
        run _wait_for_episode_ready() afterwards and rely on ITS existing
        fresh-/odom-at-reset-pose check for that -- same
        as any other handshake attempt, never assumed complete from this
        call alone."""
        node = self.node
        req = Empty.Request()
        while not node.reset_simulation_client.wait_for_service(timeout_sec=1.0):
            node.get_logger().info("reset_simulation service not available, waiting again...")
        future = node.reset_simulation_client.call_async(req)
        while rclpy.ok():
            rclpy.spin_once(node)
            if future.done():
                return

    def _wait_for_episode_ready(self, goal_before):
        """One handshake ATTEMPT for the current episode slot. `goal_before` is
        fixed for the whole slot (captured once in run_episode) -- retrying this
        function does NOT wait for another new /goal_pose, since the goal does
        not change again until the next episode; only the clock/odom/scan
        freshness checks are re-run per attempt. Confirms, in order:
          1. a /goal_pose newer than goal_before has already been observed;
          2. a /clock tick newer than when this attempt started (physics is
             actually advancing -- Gazebo starts paused and stays paused across
             every boundary until unpause_simulation() runs);
          3. /odom newer than this attempt's start, at the confirmed reset
             pose with near-zero twist (every episode starts from a reset);
          4. one forced-fresh /scan_gated forward, so the episode never starts
             on a scan cached from the previous one.
        Returns (ready, status). A timeout at any stage returns (False, reason)
        and never treats stale/incomplete data as readiness."""
        node = self.node
        if not self._spin_until(lambda: node.goal_msg_count > goal_before, EPISODE_READY_TIMEOUT,
                            label="a new /goal_pose"):
            return False, "timeout waiting for a new /goal_pose"

        clock_before = node.clock_msg_count
        if not self._spin_until(lambda: node.clock_msg_count > clock_before, EPISODE_READY_TIMEOUT,
                            label="a fresh /clock tick"):
            return False, "timeout waiting for a fresh /clock tick (is Gazebo unpaused?)"

        odom_before = node.odom_msg_count
        if not self._spin_until(
            lambda: node.odom_msg_count > odom_before and self._at_reset_pose(),
            EPISODE_READY_TIMEOUT,
            label="/odom at the confirmed reset pose",
        ):
            odom = node.latest_odom
            detail = "no /odom received" if odom is None else (
                f"x={odom.pose.pose.position.x:.3f} y={odom.pose.pose.position.y:.3f} "
                f"vlin={odom.twist.twist.linear.x:.3f} vang={odom.twist.twist.angular.z:.3f}")
            return False, f"timeout waiting for /odom at confirmed reset pose ({detail})"

        node.request_fresh_scan()
        scan_before = node.gated_scan_count
        if not self._spin_until(lambda: node.gated_scan_count > scan_before, EPISODE_READY_TIMEOUT,
                            label="a forced-fresh /scan_gated forward"):
            return False, "timeout waiting for a forced-fresh /scan_gated"

        return True, "ready"

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        node = self.node
        node.episode_index += 1
        goal_before = node.goal_baseline

        # Reset Simulation owned by the env.
        # Gazebo is already paused at the episode boundary.
        node.stop_distance()   # nothing from the reset/handshake (teleports) is counted
        node.get_logger().info(f"[episode {node.episode_index}] episode reset: issuing /reset_simulation")
        self._issue_reset_recovery()
        util.unpause_simulation(node, 0)

        # Handshake
        ready, status, attempt = False, "", 0
        for attempt in range(MAX_INIT_ATTEMPTS):
            ready, status = self._wait_for_episode_ready(goal_before)
            if ready:
                break
            node.get_logger().warning(
                f"[episode {node.episode_index}] handshake FAILED (attempt {attempt + 1}/{MAX_INIT_ATTEMPTS}): "
                f"{status} -- not starting TD3 control, episode NOT counted")
            self._safe_stop()
            if attempt < MAX_INIT_ATTEMPTS - 1:
                # Recovery (exceptional, reported separately from the episode reset above):
                # _safe_stop paused Gazebo; reset again, unpause, and re-verify.
                node.get_logger().warning(
                    f"[episode {node.episode_index}] initiating reset recovery...")
                self._issue_reset_recovery()
                node.get_logger().warning(
                    f"[episode {node.episode_index}] reset recovery issued")
                util.unpause_simulation(node, 0)
                node.get_logger().warning(
                    f"[episode {node.episode_index}] retrying episode-ready verification...")

        if not ready:
            raise RuntimeError(
                f"AdaptiveFPSEnv.reset(): episode-ready handshake failed after "
                f"{MAX_INIT_ATTEMPTS} attempts: {status}")

        # Save current goal
        node.goal_baseline = node.goal_msg_count
        retry_word = "recovery" if attempt else "retries"
        node.get_logger().info(
            f"[episode {node.episode_index}] ready after {attempt} {retry_word}: "
            f"initial_fps={self.current_fps} obs_interval={self.obs_interval}")

        # Reset Adaptive FPS variables:
        self.current_fps = max(self.fps_choices)
        self.obs_interval = node.reset_gate(self.current_fps)        
        observation = util.init_episode(node) # initial observation (44D)
        self.current_observation = observation
        self.steps_since_last_obs = 0
        self.world_step_count = 0
        self.prev_navigation_action = [0.0, 0.0]
        self.native_scan_count = 1 #GT scan acquired
        self.prev_gated_scan_count = node.gated_scan_count
        self.episode_scan_count = 1  # fresh scan acquired during episode initialization
        self._initial_goal_distance = observation[NUM_SCAN_SAMPLES] * MAX_GOAL_DISTANCE
        self._previous_goal_distance = self._initial_goal_distance
        self._episode_start_wall = time.perf_counter()
        self._episode_start_sim = stamp_to_sec(node.latest_odom.header.stamp) if node.latest_odom is not None else None
        # Odometry distance starts here, from the handshake-confirmed reset pose
        start_pos = node.latest_odom.pose.pose.position
        node.start_distance(start_pos.x, start_pos.y)

        # Clear any previous episode's recording buffers 
        self._lidar_t_wall = []
        self._lidar_t_sim = []
        self._lidar_ranges = []
        self._obstacle_t_wall = []
        self._obstacle_t_sim = []
        self._obstacle_frame_id = []
        self._obstacle_x = []
        self._obstacle_y = []
        self._obstacle_yaw = []
        self._recording_active = True

        # Published exactly once per valid episode -- purely informational
        # (no subscriber affects control/reward/sensing), but required
        # for tools/episode_recorder/record_episode.py's EpisodeRecorder
        # (if run alongside this env) to anchor its own t_wall=0/t_sim=0
        # origin correctly, exactly as it already does for that script.
        node.ready_pub.publish(EmptyMsg())

        # Create reset observation and Info
        ppo_observation = self.get_augmented_obs(observation)
        info = {
            "current_fps": self.current_fps,
            "obs_interval": self.obs_interval,
            "scan_interval_k": self.obs_interval,
            "fps_ratio": self.fps_ratio,
            "obs_age_ratio": self.obs_age_ratio,
            "episode_scan_count_ratio": self.episode_scan_count_ratio,
            "budget": self.budget,
        }
        return ppo_observation, info

    def dist_reward(self, previous_distance, current_distance):
        """Normalized progress toward goal."""
        if self._initial_goal_distance <= 0.0:
            return 0.0
        reward = 10.0 * (previous_distance - current_distance) / self._initial_goal_distance
        return float(reward)

    def _navigation_step(self, navigation_action):
        """TurtleBot equivalent of F1TENTH's self._physics_step(): one
        control transition -- DRLEnvironment applies navigation_action via
        /cmd_vel and returns its next 44-D state. Thin wrapper purely for
        structural readability; changes no behavior."""
        return util.step(self.node, navigation_action, self.prev_navigation_action)

    def step(self, action):
        
        node = self.node
        self.steps_since_last_obs += 1
        step_start_sim_time = node.latest_sim_time

        frame_consumed = False  # sticky for the whole PPO step: set once any new gated scan arrives
        while True:
            self.world_step_count += 1

            # 1. Frozen navigation uses currently held observation
            navigation_action = self.navigation_model.get_action(self.current_observation, False, self.world_step_count, False)

            # 2. One control transition
            nav_observation, _, done, outcome, _ = self._navigation_step(navigation_action)
            self.current_observation = nav_observation
            self.prev_navigation_action = copy.deepcopy(navigation_action)

            # 3. Sampling Timer
            # Newly forwarded scans since the previous tick (exact, even if >1 arrive in one gap)
            new_gated_scans = node.gated_scan_count - self.prev_gated_scan_count
            self.prev_gated_scan_count = node.gated_scan_count

            if new_gated_scans > 0:
                frame_consumed = True
                self.steps_since_last_obs = 0
                self.episode_scan_count += new_gated_scans
                self.current_fps = self.fps_choices[int(action)]
                self.obs_interval = node.set_sensing_rate(self.current_fps)

            # Termination checks
            terminated = bool(done)
            if terminated:
                break

            # Enforce PPO simulation interval
            if step_start_sim_time is not None and node.latest_sim_time is not None:
                elapsed_sim_time = node.latest_sim_time - step_start_sim_time
                if (elapsed_sim_time > PPO_DT or math.isclose(elapsed_sim_time,PPO_DT,rel_tol=0.0,abs_tol=1e-6,)):
                    break   

        # 4. Get Nav Reward
        goal_distance_norm = float(nav_observation[NUM_SCAN_SAMPLES])
        current_goal_distance = goal_distance_norm * MAX_GOAL_DISTANCE
        nav_reward = self.dist_reward(self._previous_goal_distance, current_goal_distance)

        # 5. Get Frame Penalty and Adaptive Reward
        self._previous_goal_distance = current_goal_distance
        frame_penalty = self.frame_cost if frame_consumed else 0.0
        adaptive_reward = nav_reward - frame_penalty

        # Terminal reward/penalty
        terminal_reward = 0.0
        if terminated:
            if outcome == SUCCESS:
                terminal_reward = R_SUCCESS
            elif outcome in (COLLISION_WALL, COLLISION_OBSTACLE, TUMBLE):
                terminal_reward = R_COLLISION
            elif outcome == TIMEOUT:
                terminal_reward = R_TIMEOUT
            adaptive_reward += terminal_reward
            util.pause_simulation(node, 0)   # keep Gazebo paused until the next reset()
            node.stop_distance()

        truncated = False

        ppo_observation = self.get_augmented_obs(nav_observation)

        # Collect step diagnostics for logging purposes
        x = y = float("nan")
        yaw = float("nan")
        sim_time = None
        if node.latest_odom is not None:
            x = node.latest_odom.pose.pose.position.x
            y = node.latest_odom.pose.pose.position.y
            yaw = quaternion_to_yaw(node.latest_odom.pose.pose.orientation)
            if self._episode_start_sim is not None:
                sim_time = stamp_to_sec(node.latest_odom.header.stamp) - self._episode_start_sim
        wall_time = time.perf_counter() - self._episode_start_wall

        # 6. Info
        info = {
            # Step and timing
            "wall_time": wall_time,
            "sim_time": sim_time,
            "ppo_step_sim_dt": (
                node.latest_sim_time - step_start_sim_time
                if (step_start_sim_time is not None and node.latest_sim_time is not None)
                else None
            ),

            # Rewards
            "nav_reward": nav_reward,
            "frame_cost": self.frame_cost,
            "frame_penalty": frame_penalty,
            "terminal_reward": terminal_reward,

            # Budget
            "budget": self.budget,
            "episode_scan_count": self.episode_scan_count,

            # Sensing frequency
            "current_fps": self.current_fps,
            "obs_interval": self.obs_interval,

            # Scan consumption
            "frame_consumed": frame_consumed,
            "native_scan_count": self.native_scan_count,
            "scans_since_last_obs": node.scans_since_forward,

            # Augmented observation
            "fps_ratio": self.fps_ratio,
            "obs_age_ratio": self.obs_age_ratio,
            "episode_scan_count_ratio": self.episode_scan_count_ratio,

            # Navigation and goal
            "navigation_action": navigation_action,
            "x": x,
            "y": y,
            "yaw": yaw,
            "goal_x": node.latest_goal_xy[0] if node.latest_goal_xy is not None else None,
            "goal_y": node.latest_goal_xy[1] if node.latest_goal_xy is not None else None,
            "goal_distance_m": current_goal_distance,
            "goal_distance_norm": goal_distance_norm,
            "distance_traveled_m": node.distance_traveled_m,   # odometry path length this episode (m)

            # Episode outcome
            "outcome": outcome,
            "outcome_str": util.translate_outcome(outcome),
        }
        return ppo_observation, float(adaptive_reward), terminated, truncated, info

    def close(self):
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


gymnasium.register(id="AdaptiveFPSTurtleBot-v0", entry_point="AdaptiveFPS.env.adaptive_fps_env:AdaptiveFPSEnv")
