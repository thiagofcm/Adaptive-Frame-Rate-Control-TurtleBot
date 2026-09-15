#!/usr/bin/env python3
"""FixedFPS evaluator for the frozen Stage 9 TD3 navigation policy.

Characterizes how the policy behaves as LiDAR acquisition rate drops below
control rate: the control loop runs at its normal (stock) rate, but the
LiDAR observation it sees is refreshed only at the requested sensing rate,
holding the last acquired scan in between (sample-and-hold).

Architecture (see the approved plan for the full derivation):

    Gazebo LiDAR --/scan (real, 50Hz)--> this evaluator (scan gate)
        --/scan_gated (forwards every k-th real scan, nothing in between)-->
    DRLEnvironment (launched via environment_gated.py with its 'scan'
    subscription remapped to /scan_gated -- file itself byte-for-byte
    unmodified) --44-D state--> frozen TD3 actor --action--> /cmd_vel

Between forwards, DRLEnvironment's own scan_callback simply doesn't fire,
so its cached self.scan_ranges keeps holding the last forwarded value --
this is the environment's own existing behavior, verified directly against
get_state() (drl_environment.py:203-208), which unconditionally does
`state = copy.deepcopy(self.scan_ranges)` with no freshness check.

This script itself replaces test_agent for the duration of the study,
because reward/done/outcome only ever exist in the DrlStep service
RESPONSE, seen only by whoever calls step_comm directly (drl_gazebo.py and
drl_environment.py remain completely unmodified and untouched).

This is a controlled experiment around ONE frozen checkpoint, not a
generic framework -- only the sensing rate(s) and episode count are CLI
arguments; everything else below is a fixed experimental constant.

Usage:
    python3 evaluate_fixed_sensing.py --fps 10 --n-episodes 100
    python3 evaluate_fixed_sensing.py --fps 50 25 10 5 2 --n-episodes 100
"""
import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data

from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty as EmptyMsg
from std_srvs.srv import Empty
from turtlebot3_msgs.srv import DrlStep, Goal

from turtlebot3_drl.common import utilities as util
from turtlebot3_drl.common.storagemanager import StorageManager
from turtlebot3_drl.common.settings import (
    ENABLE_BACKWARD, ENABLE_STACKING, ENABLE_MOTOR_NOISE,
    ENABLE_DYNAMIC_GOALS, ENABLE_TRUE_RANDOM_GOALS,
    SPEED_LINEAR_MAX, SPEED_ANGULAR_MAX,
    THRESHOLD_COLLISION, THREHSOLD_GOAL, LIDAR_DISTANCE_CAP, EPISODE_TIMEOUT_SECONDS,
    UNKNOWN, SUCCESS, COLLISION_WALL, COLLISION_OBSTACLE, TIMEOUT, TUMBLE,
)
from turtlebot3_drl.drl_agent.td3 import TD3
from turtlebot3_drl.drl_environment.drl_environment import NUM_SCAN_SAMPLES, MAX_GOAL_DISTANCE

BASE_PATH = os.environ["DRLNAV_BASE_PATH"]
sys.path.insert(0, os.path.join(BASE_PATH, "tools", "episode_recorder"))
from record_episode import EpisodeRecorder  # noqa: E402  (path must be set up first)

sys.path.insert(0, BASE_PATH)
from AdaptiveFPS.rewards.adaptive_fps_reward import get_adaptive_reward  # noqa: E402


# ===================================================================== #
#   Fixed experimental constants -- NOT exposed as CLI arguments.
#   This is a controlled experiment around one frozen checkpoint; the
#   only things that actually vary between runs are --fps and
#   --n-episodes.
# ===================================================================== #
MODEL_RUN_NAME    = "examples/td3_0_stage9"
LOAD_EPISODE      = 7400
ALGORITHM         = "td3"
NATIVE_SCAN_HZ    = 50.0   # models/turtlebot3_burger/model.sdf's LiDAR <update_rate>
GATED_TOPIC       = "scan_gated"
EVAL_ROOT         = "AdaptiveFPS/eval"

# Episode-ready handshake: each individual wait (goal / clock / odom / forced
# scan) gets this long before that sub-check is declared timed out.
EPISODE_READY_TIMEOUT = 10.0
# Bounded retries per episode SLOT for the sync sub-checks (goal/clock/odom/
# scan) -- NOT for waiting for a brand-new /goal_pose, which only happens
# once per slot (see run_episode). If every attempt times out, this rate
# block is abandoned early rather than looping forever.
MAX_INIT_ATTEMPTS = 5

# Stage 9's robot reset pose, verified directly from
# worlds/turtlebot3_drl_stage9/burger.model: <pose>2.5 2.5 0 0 0 ...</pose>
# (yaw pointed toward (0.0, 2.0) -- position only matters here, since
# _at_reset_pose() below never checks orientation).
STAGE9_RESET_X = 2.5
STAGE9_RESET_Y = 2.5
RESET_POSITION_TOLERANCE_M = 1.0     # intentionally very permissive
RESET_VELOCITY_TOLERANCE_MPS = 0.03  # must stay a meaningful "robot settled" check

LINEAR, ANGULAR = 0, 1

EPISODE_CSV_FIELDS = [
    "episode_index", "success", "crashed", "collision_wall", "collision_obstacle", "tumble", "timeout",
    "outcome_code", "outcome_str", "mean_fps", "episode_time", "episode_time_sim_s", "episode_return",
    "adaptive_reward_total", "final_progress",
    "steps_length", "n_fresh_observations", "fresh_observation_ratio", "distance_traveled",
    "start_x", "start_y", "final_x", "final_y", "final_goal_distance", "goal_x", "goal_y",
    "fps_requested", "fps_effective", "scan_divisor_k", "min_lidar_m", "valid", "init_retries",
]

STEP_CSV_FIELDS = [
    "step", "t_wall_s", "t_sim_s", "x", "y", "yaw",
    "fresh_observation", "fresh_observation_count",
    "reward", "reward_cumulative",
    "adaptive_reward", "adaptive_reward_cumulative",
    "outcome", "done",
    "goal_distance_norm", "goal_distance_m",
    "action_linear", "action_angular",
    "fps_requested", "fps_effective", "scan_divisor_k",
]


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def fmt_hz(hz):
    if abs(hz - round(hz)) < 1e-9:
        return str(int(round(hz)))
    return f"{hz:.2f}".rstrip("0").rstrip(".")


def git_rev():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=BASE_PATH, text=True).strip()
    except Exception:
        return None


class FixedSensingEvaluator(Node):
    """Drives step_comm directly (replacing test_agent for this study) and
    owns the /scan -> /scan_gated sensing-rate gate in the same process, so
    per-step reward/outcome and per-step gate state can be logged together."""

    def __init__(self):
        super().__init__("fixed_sensing_evaluator")

        # --- required by common/utilities.py's step()/wait_new_goal()/
        #     pause_simulation()/unpause_simulation() -- duck-typed on these
        #     exact attribute names, reused verbatim, unmodified. ---
        self.step_comm_client = self.create_client(DrlStep, "step_comm")
        self.goal_comm_client = self.create_client(Goal, "goal_comm")
        self.gazebo_pause = self.create_client(Empty, "/pause_physics")
        self.gazebo_unpause = self.create_client(Empty, "/unpause_physics")
        # Used only by _issue_reset_recovery() (handshake retry recovery,
        # not part of the normal episode-boundary path) -- same service
        # drl_gazebo.py's own task_fail_callback() already calls on every
        # ordinary episode failure, called here directly since it's a
        # plain Gazebo service, not something drl_gazebo.py exclusively
        # owns. Deliberately does NOT touch goal state (see
        # _issue_reset_recovery's docstring).
        self.reset_simulation_client = self.create_client(Empty, "reset_simulation")

        # --- scan gate state ---
        # gated_scan_count is the single source of truth for "how many scans
        # have actually been forwarded to /scan_gated so far" -- run_episode()
        # diffs it against its own last-seen snapshot each control step, so
        # counting stays exact even if the control loop is briefly slower than
        # the incoming scan rate and several forwards land in one step's gap.
        self.k = 1
        self.scans_since_forward = 0
        self.force_next_fresh = True
        self.gated_scan_count = 0
        self.gated_pub = self.create_publisher(LaserScan, GATED_TOPIC, QoSProfile(depth=10))
        self.create_subscription(LaserScan, "scan", self._on_real_scan, qos_profile_sensor_data)

        # --- independent goal-freshness tracking (drl_environment.py's
        #     new_goal flag is a latch never reset to False after episode 1
        #     -- confirmed present, unmodified, see plan -- so this evaluator
        #     verifies episode boundaries itself rather than trusting it). ---
        self.goal_msg_count = 0
        self.latest_goal_xy = None
        self.create_subscription(Pose, "goal_pose", self._on_goal_pose, QoSProfile(depth=10))
        # Locked in right after each successful handshake (see run_episode),
        # NOT recomputed fresh per call: drl_gazebo publishes the next
        # episode's goal fire-and-forget, asynchronously, as soon as the
        # previous episode's terminal step is serviced -- it can arrive
        # during THIS episode's own pause_simulation() spin, before the next
        # run_episode() call even starts. A freshly-captured "goal_before"
        # would then already include it and could never see it as "new".
        self.goal_baseline = 0

        # --- odom, for x/y/yaw/velocity logging (not part of the 44-D state)
        #     and for the episode-ready handshake's reset-pose confirmation ---
        self.latest_odom = None
        self.odom_msg_count = 0
        self.create_subscription(Odometry, "odom", self._on_odom, QoSProfile(depth=10))

        # --- sim clock, so the handshake can confirm physics is actually
        #     advancing (Gazebo starts paused and stays paused across every
        #     episode boundary until unpause_simulation() is called) ---
        self.latest_sim_time = None
        self.clock_msg_count = 0
        self.create_subscription(Clock, "/clock", self._on_clock, QoSProfile(depth=10))

        # --- safe-stop publisher: used only when a handshake attempt times
        #     out, to make sure the robot isn't left driving on a stale
        #     command while we pause and retry ---
        self.cmd_vel_pub = self.create_publisher(Twist, "cmd_vel", QoSProfile(depth=10))

        # --- episode-ready signal for the passive recorder: published
        #     exactly once per valid episode, right before the first TD3
        #     control step, so the recorder can anchor its own t_wall=0/
        #     t_sim=0 at the same instant instead of at /goal_pose arrival
        #     (which precedes the handshake and is therefore too early). ---
        self.ready_pub = self.create_publisher(EmptyMsg, "episode_ready", QoSProfile(depth=10))

        # episode_index is reset per rate block (see evaluate_fixed_rate) so
        # it stays numerically aligned with EpisodeRecorder's own per-instance
        # episode_count, which also restarts at 0 for each new rate's recorder.
        self.episode_index = 0

        # --- load the frozen actor exactly as drl_agent.py does ---
        self.device = util.check_gpu()
        self.sim_speed = util.get_simulation_speed(util.stage)
        self.sm = StorageManager(ALGORITHM, MODEL_RUN_NAME, LOAD_EPISODE, self.device, util.stage)
        self.model = self.sm.load_model()
        self.model.device = self.device
        self.sm.load_weights(self.model.networks)

        assert isinstance(self.model, TD3), f"expected a TD3 model, got {type(self.model)}"
        assert self.model.state_size == NUM_SCAN_SAMPLES + 4, \
            f"state_size mismatch: model={self.model.state_size} env={NUM_SCAN_SAMPLES + 4}"
        assert not ENABLE_STACKING, "FixedFPS evaluator does not support ENABLE_STACKING"

        self.get_logger().info(
            f"loaded {MODEL_RUN_NAME} episode {LOAD_EPISODE}: state_size={self.model.state_size} "
            f"step_time={self.model.step_time} sim_speed={self.sim_speed} device={self.device}")

    # ------------------------------------------------------------------ #
    #   Scan gate
    # ------------------------------------------------------------------ #
    def _on_real_scan(self, msg):
        self.scans_since_forward += 1
        if self.force_next_fresh or self.scans_since_forward >= self.k:
            self.gated_pub.publish(msg)
            self.gated_scan_count += 1
            self.scans_since_forward = 0
            self.force_next_fresh = False

    def _on_goal_pose(self, msg):
        self.goal_msg_count += 1
        self.latest_goal_xy = (msg.position.x, msg.position.y)

    def _on_odom(self, msg):
        self.latest_odom = msg
        self.odom_msg_count += 1

    def _on_clock(self, msg):
        self.latest_sim_time = stamp_to_sec(msg.clock)
        self.clock_msg_count += 1


HEARTBEAT_PERIOD_S = 3.0       # handshake wait heartbeat
STEP_HEARTBEAT_PERIOD_S = 5.0  # in-episode control-loop heartbeat


def _spin_until(node, condition, timeout, label=None):
    """Spin this node's own callbacks until condition() is True or timeout
    elapses. Returns condition()'s final value -- never claims success on a
    timeout that condition() itself would reject.

    If `label` is given, logs an INFO heartbeat roughly every
    HEARTBEAT_PERIOD_S seconds while waiting, purely so a real (possibly
    long) wait is visibly distinguishable from a genuine hang, and so a
    stuck run shows exactly which condition it's stuck on. Purely
    diagnostic -- does not affect timing or the return value."""
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


def _at_reset_pose(node):
    if node.latest_odom is None:
        return False
    p = node.latest_odom.pose.pose.position
    t = node.latest_odom.twist.twist
    pos_ok = math.hypot(p.x - STAGE9_RESET_X, p.y - STAGE9_RESET_Y) <= RESET_POSITION_TOLERANCE_M
    vel_ok = abs(t.linear.x) <= RESET_VELOCITY_TOLERANCE_MPS and abs(t.angular.z) <= RESET_VELOCITY_TOLERANCE_MPS
    return pos_ok and vel_ok


def _safe_stop(node):
    """Best-effort: zero the robot's commanded velocity and pause physics
    before a retry, so a stale command doesn't keep driving the robot while
    we wait, and so the next attempt starts from a known (paused) state."""
    node.cmd_vel_pub.publish(Twist())
    util.pause_simulation(node, 0)


def _issue_reset_recovery(node):
    """Real recovery action for a failed handshake attempt -- as opposed
    to _safe_stop() (which only pauses/zeros velocity and changes no
    physical state), this issues Gazebo's own /reset_simulation service
    directly: the exact same recovery drl_gazebo.py's own
    task_fail_callback() already performs on every ordinary episode
    failure (see drl_gazebo.py's reset_simulation()). Teleports every
    model -- including the robot -- back to its world-file insertion
    pose and resets sim time to 0.

    Deliberately does NOT touch goal state: the goal marker's own Gazebo
    insertion pose is wherever it was last spawned for the CURRENT
    episode's intended goal, so this call cannot move or regenerate it.
    Nothing here calls task_fail/task_succeed, and goal_baseline/
    goal_msg_count/expect_reset/episode_index are all untouched -- this
    is a pure physics-state recovery, not a new episode-boundary event.

    Blocks (call_async + spin-until-future.done(), same idiom as
    utilities.py's pause_simulation/unpause_simulation) until Gazebo has
    ACKNOWLEDGED the request -- this confirms the request was issued,
    NOT that the reset has already propagated to /odom. The caller must
    re-run wait_for_episode_ready() afterwards and rely on ITS existing
    fresh-/odom (+ reset-pose, when expect_reset) check for that -- same
    as any other handshake attempt, never assumed complete from this
    call alone."""
    req = Empty.Request()
    while not node.reset_simulation_client.wait_for_service(timeout_sec=1.0):
        node.get_logger().info("reset_simulation service not available, waiting again...")
    future = node.reset_simulation_client.call_async(req)
    while rclpy.ok():
        rclpy.spin_once(node)
        if future.done():
            return


def wait_for_episode_ready(node, expect_reset, goal_before):
    """One handshake ATTEMPT for the current episode slot. `goal_before` is
    fixed for the whole slot (captured once in run_episode) -- retrying this
    function does NOT wait for another new /goal_pose, since the goal does
    not change again until the next episode; only the clock/odom/scan
    freshness checks are re-run per attempt. Confirms, in order:
      1. a /goal_pose newer than goal_before has already been observed;
      2. a /clock tick newer than when this attempt started (physics is
         actually advancing -- Gazebo starts paused and stays paused across
         every boundary until unpause_simulation() runs);
      3. /odom newer than this attempt's start, and (when expect_reset) at
         the confirmed Stage 9 reset pose with near-zero twist;
      4. one forced-fresh /scan_gated forward, so the episode never starts
         on a scan cached from the previous one.
    Returns (ready, status). A timeout at any stage returns (False, reason)
    and never treats stale/incomplete data as readiness."""
    if not _spin_until(node, lambda: node.goal_msg_count > goal_before, EPISODE_READY_TIMEOUT,
                        label="a new /goal_pose"):
        return False, "timeout waiting for a new /goal_pose"

    clock_before = node.clock_msg_count
    if not _spin_until(node, lambda: node.clock_msg_count > clock_before, EPISODE_READY_TIMEOUT,
                        label="a fresh /clock tick"):
        return False, "timeout waiting for a fresh /clock tick (is Gazebo unpaused?)"

    odom_before = node.odom_msg_count
    if expect_reset:
        if not _spin_until(
            node,
            lambda: node.odom_msg_count > odom_before and _at_reset_pose(node),
            EPISODE_READY_TIMEOUT,
            label="/odom at the confirmed reset pose",
        ):
            odom = node.latest_odom
            detail = "no /odom received" if odom is None else (
                f"x={odom.pose.pose.position.x:.3f} y={odom.pose.pose.position.y:.3f} "
                f"vlin={odom.twist.twist.linear.x:.3f} vang={odom.twist.twist.angular.z:.3f}")
            return False, f"timeout waiting for /odom at confirmed reset pose ({detail})"
    else:
        if not _spin_until(node, lambda: node.odom_msg_count > odom_before, EPISODE_READY_TIMEOUT,
                            label="fresh /odom"):
            return False, "timeout waiting for fresh /odom"

    node.force_next_fresh = True
    scan_before = node.gated_scan_count
    if not _spin_until(node, lambda: node.gated_scan_count > scan_before, EPISODE_READY_TIMEOUT,
                        label="a forced-fresh /scan_gated forward"):
        return False, "timeout waiting for a forced-fresh /scan_gated"

    return True, "ready"


def goal_consistent(node, state, tol=0.15):
    """Cross-check: does the just-returned state's encoded goal distance
    match this evaluator's own odom+goal_pose reading? If not, drl_environment
    likely hasn't ingested the new goal yet (the new_goal-latch symptom)."""
    if node.latest_odom is None or node.latest_goal_xy is None:
        return False
    rx = node.latest_odom.pose.pose.position.x
    ry = node.latest_odom.pose.pose.position.y
    gx, gy = node.latest_goal_xy
    true_dist = math.hypot(gx - rx, gy - ry)
    encoded_dist = state[NUM_SCAN_SAMPLES] * MAX_GOAL_DISTANCE
    return abs(encoded_dist - true_dist) <= tol


def unnormalize_action(action):
    if ENABLE_BACKWARD:
        cmd_linear = action[LINEAR] * SPEED_LINEAR_MAX
    else:
        cmd_linear = (action[LINEAR] + 1) / 2 * SPEED_LINEAR_MAX
    cmd_angular = action[ANGULAR] * SPEED_ANGULAR_MAX
    return cmd_linear, cmd_angular


def run_episode(node, fps_requested, fps_effective, k, rate_dir, expect_reset):
    """One episode SLOT at a fixed sensing rate. Mirrors drl_agent.py's
    process() loop structure, with the scan gate, per-step logging, and a
    bounded episode-ready handshake layered in.

    Returns (row, ready, init_retries). If the handshake never becomes ready
    within MAX_INIT_ATTEMPTS, returns (None, False, MAX_INIT_ATTEMPTS): no
    TD3 step is ever taken, nothing is written to steps.csv/episodes.csv, and
    the caller must not count this slot as an evaluated episode."""
    node.k = k
    node.scans_since_forward = 0
    node.gated_scan_count = 0

    node.episode_index += 1
    episode_index = node.episode_index

    # /odom, /clock and /goal_pose only advance while Gazebo is running --
    # the stage launch starts paused, and the previous episode (or the
    # previous failed attempt) left it paused too. Must happen BEFORE the
    # handshake, not after, or the freshness waits below spin forever.
    util.unpause_simulation(node, 0)
    goal_before = node.goal_baseline

    ready, status, attempt = False, "", 0
    for attempt in range(MAX_INIT_ATTEMPTS):
        ready, status = wait_for_episode_ready(node, expect_reset, goal_before)
        if ready:
            break
        node.get_logger().warning(
            f"[episode {episode_index}] handshake FAILED (attempt {attempt + 1}/{MAX_INIT_ATTEMPTS}): "
            f"{status} -- not starting TD3 control, episode NOT counted")
        _safe_stop(node)
        if attempt < MAX_INIT_ATTEMPTS - 1:
            util.unpause_simulation(node, 0)  # re-arm physics for the next attempt

    init_retries = attempt  # attempt is 0-indexed: ready on attempt N means N prior failures
    if not ready:
        node.get_logger().error(
            f"[episode {episode_index}] giving up after {MAX_INIT_ATTEMPTS} failed handshake "
            f"attempts ({status}) -- this episode slot produced no evaluated episode")
        return None, False, MAX_INIT_ATTEMPTS

    # Lock in the baseline for the NEXT slot now, at the moment THIS goal is
    # confirmed consumed -- not later, when the next run_episode() call
    # happens to start (see the comment on self.goal_baseline).
    node.goal_baseline = node.goal_msg_count

    goal_x, goal_y = node.latest_goal_xy if node.latest_goal_xy else (float("nan"), float("nan"))
    node.get_logger().info(
        f"[episode {episode_index}] ready after {init_retries} retries: "
        f"goal=({goal_x:.2f},{goal_y:.2f}) "
        f"odom=({node.latest_odom.pose.pose.position.x:.3f},"
        f"{node.latest_odom.pose.pose.position.y:.3f}) "
        f"gated_scans={node.gated_scan_count}")

    state = util.init_episode(node)
    if not goal_consistent(node, state):
        node.get_logger().warning(
            f"[episode {episode_index}] goal_consistent() secondary check failed after a "
            f"successful handshake -- proceeding, but flagging for later inspection")

    # AdaptiveFPS reward bookkeeping (logging-only, see get_adaptive_reward
    # in AdaptiveFPS/rewards/adaptive_fps_reward.py) -- uses the same
    # goal-distance convention already used elsewhere in this function
    # (state[NUM_SCAN_SAMPLES] * MAX_GOAL_DISTANCE), not a separate
    # Euclidean computation.
    initial_goal_distance = state[NUM_SCAN_SAMPLES] * MAX_GOAL_DISTANCE
    previous_goal_distance = initial_goal_distance
    adaptive_reward_cumulative = 0.0

    start_x = node.latest_odom.pose.pose.position.x
    start_y = node.latest_odom.pose.pose.position.y

    step_rows = []
    action_past = [0.0, 0.0]
    reward_cum = 0.0
    n_fresh = 0
    fps_trace = []
    episode_start_wall = time.perf_counter()
    episode_start_sim = stamp_to_sec(node.latest_odom.header.stamp) if node.latest_odom else None

    # Exactly one READY per valid episode (unreachable on a failed handshake,
    # since this function already returned above in that case), published at
    # the same instant steps.csv's own t_wall=0/t_sim=0 origin is captured --
    # the passive recorder uses this to anchor trajectory.csv identically,
    # instead of at /goal_pose arrival (before the handshake, too early).
    node.ready_pub.publish(EmptyMsg())

    step_idx = 0
    done = False
    outcome = UNKNOWN
    distance_traveled = 0.0
    final_goal_distance = float("nan")
    final_x, final_y = start_x, start_y
    episode_min_lidar_m = float("inf")

    # Handshake-phase scans (goal/clock/odom waits + the guaranteed forced-
    # fresh forward) must not inflate episode accounting -- the evaluated
    # episode conceptually begins with exactly one fresh observation: the
    # one already consumed, during the handshake, to obtain `state` above.
    # node.gated_scan_count is guaranteed unchanged since state was fetched
    # (nothing between init_episode() and here spins the executor).
    last_gated_count = node.gated_scan_count
    n_fresh = 1
    next_step_heartbeat_wall = STEP_HEARTBEAT_PERIOD_S
    while not done:
        action = node.model.get_action(state, False, step_idx, False)

        if step_idx == 0:
            # Step 0's fresh observation was already accounted for above
            # (the handshake's guaranteed forced-fresh scan) -- not a delta
            # observed here, so it must not be double-counted against n_fresh.
            fresh_now = True
        else:
            gated_delta = node.gated_scan_count - last_gated_count
            fresh_now = gated_delta > 0
            n_fresh += gated_delta  # accumulate newly forwarded scans during
                                     # evaluated stepping -- gated_delta > 1 means
                                     # multiple scans were forwarded between control
                                     # iterations, not multiple TD3 decisions
        last_gated_count = node.gated_scan_count

        next_state, reward, done, outcome, dist_trav = util.step(node, action, action_past)

        t_wall = time.perf_counter() - episode_start_wall
        t_sim = None
        if node.latest_odom is not None and episode_start_sim is not None:
            t_sim = stamp_to_sec(node.latest_odom.header.stamp) - episode_start_sim

        x = y = yaw = float("nan")
        vlin = vang = float("nan")
        if node.latest_odom is not None:
            x = node.latest_odom.pose.pose.position.x
            y = node.latest_odom.pose.pose.position.y
            yaw = quaternion_to_yaw(node.latest_odom.pose.pose.orientation)
            vlin = node.latest_odom.twist.twist.linear.x
            vang = node.latest_odom.twist.twist.angular.z
            final_x, final_y = x, y

        cmd_linear, cmd_angular = unnormalize_action(action)
        reward_cum += reward

        lidar_min_norm = min(next_state[0:NUM_SCAN_SAMPLES])
        lidar_min_m = lidar_min_norm * LIDAR_DISTANCE_CAP
        episode_min_lidar_m = min(episode_min_lidar_m, lidar_min_m)
        goal_distance_norm = next_state[NUM_SCAN_SAMPLES]
        goal_distance_m = goal_distance_norm * MAX_GOAL_DISTANCE

        # AdaptiveFPS reward: logging-only, computed alongside (never in
        # place of) the original navigation reward above.
        adaptive_reward = get_adaptive_reward(previous_goal_distance, goal_distance_m, initial_goal_distance)
        adaptive_reward_cumulative += adaptive_reward
        previous_goal_distance = goal_distance_m

        # Diagnostic only, logging-only: makes a long-running episode (tens
        # of thousands of control steps can take a real minute or more)
        # visibly distinguishable from a hang, since the step loop otherwise
        # prints nothing at all until the episode-end summary. Does not
        # touch control, timing, reward, or sensing.
        if t_wall >= next_step_heartbeat_wall:
            node.get_logger().info(
                f"[episode {episode_index}] running: steps={step_idx} "
                f"elapsed={t_wall:.1f}s goal_distance={goal_distance_m:.2f}m "
                f"fresh_obs={n_fresh}")
            next_step_heartbeat_wall += STEP_HEARTBEAT_PERIOD_S
        goal_angle_norm = next_state[NUM_SCAN_SAMPLES + 1]
        goal_angle_rad = goal_angle_norm * math.pi
        prev_action_linear = next_state[NUM_SCAN_SAMPLES + 2]
        prev_action_angular = next_state[NUM_SCAN_SAMPLES + 3]

        fps_trace.append(fps_effective)
        final_goal_distance = goal_distance_m

        step_rows.append({
            "step": step_idx,
            "t_wall_s": t_wall,
            "t_sim_s": t_sim,
            "x": x, "y": y, "yaw": yaw,
            "fresh_observation": fresh_now,
            "fresh_observation_count": n_fresh,
            "reward": reward,
            "reward_cumulative": reward_cum,
            "adaptive_reward": adaptive_reward,
            "adaptive_reward_cumulative": adaptive_reward_cumulative,
            "outcome": outcome,
            "done": done,
            "goal_distance_norm": goal_distance_norm,
            "goal_distance_m": goal_distance_m,
            "action_linear": action[LINEAR],
            "action_angular": action[ANGULAR],
            "fps_requested": fps_requested,
            "fps_effective": fps_effective,
            "scan_divisor_k": k,
        })

        action_past = copy.deepcopy(action)
        state = next_state
        step_idx += 1
        if done:
            distance_traveled = dist_trav

    util.pause_simulation(node, 0)
    episode_time = time.perf_counter() - episode_start_wall
    episode_time_sim_s = None
    if node.latest_odom is not None and episode_start_sim is not None:
        episode_time_sim_s = stamp_to_sec(node.latest_odom.header.stamp) - episode_start_sim

    episode_dir = os.path.join(rate_dir, f"episode_{episode_index:04d}")
    os.makedirs(episode_dir, exist_ok=True)
    with open(os.path.join(episode_dir, "steps.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STEP_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(step_rows)

    steps_length = len(step_rows)
    outcome_code = outcome
    success = outcome_code == SUCCESS
    collision_wall = outcome_code == COLLISION_WALL
    collision_obstacle = outcome_code == COLLISION_OBSTACLE
    tumble = outcome_code == TUMBLE
    timeout_flag = outcome_code == TIMEOUT
    crashed = collision_wall or collision_obstacle or tumble

    if initial_goal_distance > 0.0:
        final_progress = (initial_goal_distance - final_goal_distance) / initial_goal_distance
    else:
        final_progress = 0.0
    adaptive_reward_total = adaptive_reward_cumulative

    row = {
        "episode_index": episode_index,
        "success": success,
        "crashed": crashed,
        "collision_wall": collision_wall,
        "collision_obstacle": collision_obstacle,
        "tumble": tumble,
        "timeout": timeout_flag,
        "outcome_code": outcome_code,
        "outcome_str": util.translate_outcome(outcome_code),
        "mean_fps": (sum(fps_trace) / len(fps_trace)) if fps_trace else 0.0,
        "episode_time": episode_time,
        "episode_time_sim_s": episode_time_sim_s,
        "episode_return": reward_cum,
        "adaptive_reward_total": adaptive_reward_total,
        "final_progress": final_progress,
        "steps_length": steps_length,
        "n_fresh_observations": n_fresh,
        "fresh_observation_ratio": (n_fresh / steps_length) if steps_length else 0.0,
        "distance_traveled": distance_traveled,
        "start_x": start_x, "start_y": start_y,
        "final_x": final_x, "final_y": final_y,
        "final_goal_distance": final_goal_distance,
        "goal_x": goal_x, "goal_y": goal_y,
        "fps_requested": fps_requested,
        "fps_effective": fps_effective,
        "scan_divisor_k": k,
        "min_lidar_m": float("nan") if steps_length == 0 else episode_min_lidar_m,
        "valid": 0 if steps_length <= 30 else 1,
        "init_retries": init_retries,
    }
    return row, True, init_retries


def evaluate_fixed_rate(node, fps_requested, n_episodes, rate_dir):
    k = max(1, round(NATIVE_SCAN_HZ / fps_requested))
    fps_effective = NATIVE_SCAN_HZ / k
    os.makedirs(rate_dir, exist_ok=True)

    episodes_csv_path = os.path.join(rate_dir, "episodes.csv")
    write_header = not os.path.exists(episodes_csv_path)
    with open(episodes_csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EPISODE_CSV_FIELDS)
        if write_header:
            writer.writeheader()

        # episode 1 begins from init_callback's hard reset; after that,
        # expect_reset tracks whether the PREVIOUS episode ended in failure
        # (task_fail_callback resets the robot to origin) or success
        # (task_succeed_callback does not reset it -- the next episode
        # legitimately starts wherever the robot stopped).
        expect_reset = True
        evaluated = 0
        while evaluated < n_episodes:
            row, ok, init_retries = run_episode(node, fps_requested, fps_effective, k, rate_dir, expect_reset)
            if not ok:
                node.get_logger().error(
                    f"[{fps_requested}Hz] episode slot abandoned after {init_retries} failed "
                    f"handshake attempts -- stopping this rate block early "
                    f"({evaluated}/{n_episodes} episodes evaluated)")
                break
            writer.writerow(row)
            f.flush()
            evaluated += 1
            expect_reset = not row["success"]
            node.get_logger().info(
                f"[{fps_requested}Hz -> k={k}, eff={fps_effective:.2f}Hz] "
                f"episode {evaluated}/{n_episodes}: {row['outcome_str']:<10} "
                f"steps={row['steps_length']:<6} return={row['episode_return']:.1f} "
                f"fresh={row['n_fresh_observations']}/{row['steps_length']} "
                f"init_retries={init_retries}")


def start_recorder(rate_dir):
    """One EpisodeRecorder per rate block, spun on its own executor/thread
    since util.step()'s rclpy.spin_once(node) only services this evaluator's
    own node, not a second Node instance living in the same process."""
    recorder = EpisodeRecorder(rate_dir)
    executor = SingleThreadedExecutor()
    executor.add_node(recorder)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    return recorder, executor, thread


def stop_recorder(recorder, executor, thread):
    if recorder is None:
        return
    recorder.shutdown()  # flushes any in-progress episode
    executor.shutdown(timeout_sec=2.0)  # makes executor.spin() in `thread` return
    thread.join(timeout=2.0)
    executor.remove_node(recorder)
    recorder.destroy_node()


def write_run_config(eval_root, args, node):
    config = {
        "model_run_name": MODEL_RUN_NAME,
        "load_episode": LOAD_EPISODE,
        "algorithm": ALGORITHM,
        "native_scan_hz": NATIVE_SCAN_HZ,
        "gated_topic": GATED_TOPIC,
        "fps_requested": args.fps,
        "n_episodes": args.n_episodes,
        "stage": util.stage,
        "sim_speed": node.sim_speed,
        "model_step_time": node.model.step_time,
        "num_scan_samples": NUM_SCAN_SAMPLES,
        "max_goal_distance": MAX_GOAL_DISTANCE,
        "settings": {
            "ENABLE_BACKWARD": ENABLE_BACKWARD,
            "ENABLE_STACKING": ENABLE_STACKING,
            "ENABLE_MOTOR_NOISE": ENABLE_MOTOR_NOISE,
            "ENABLE_DYNAMIC_GOALS": ENABLE_DYNAMIC_GOALS,
            "ENABLE_TRUE_RANDOM_GOALS": ENABLE_TRUE_RANDOM_GOALS,
            "SPEED_LINEAR_MAX": SPEED_LINEAR_MAX,
            "SPEED_ANGULAR_MAX": SPEED_ANGULAR_MAX,
            "THRESHOLD_COLLISION": THRESHOLD_COLLISION,
            "THREHSOLD_GOAL": THREHSOLD_GOAL,
            "EPISODE_TIMEOUT_SECONDS": EPISODE_TIMEOUT_SECONDS,
            "LIDAR_DISTANCE_CAP": LIDAR_DISTANCE_CAP,
        },
        "state_vector_layout": {
            "0:40": "normalized LiDAR ranges",
            "40": "goal_distance_norm",
            "41": "goal_angle_norm",
            "42": "prev_action_linear",
            "43": "prev_action_angular",
        },
        "note": (
            "Gating /scan also gates the obstacle-distance term used for collision "
            "detection (drl_environment.py's obstacle_distance) and the reward's obstacle "
            "penalty (reward.py). This is intentional -- it's what the study is about. "
            "SUCCESS/TIMEOUT/TUMBLE remain odom/clock-driven and are not gated."
        ),
        "git_rev": git_rev(),
    }
    with open(os.path.join(eval_root, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fps", type=float, nargs="+", required=True,
                         help="one or more fixed LiDAR sensing rates in Hz, e.g. --fps 50 25 10 5 2")
    parser.add_argument("--n-episodes", type=int, default=20, help="episodes evaluated per rate")
    args = parser.parse_args()

    rclpy.init()
    node = FixedSensingEvaluator()

    eval_root = os.path.join(BASE_PATH, EVAL_ROOT)
    os.makedirs(eval_root, exist_ok=True)
    write_run_config(eval_root, args, node)

    recorder = executor = thread = None

    try:
        for fps in args.fps:
            rate_dir = os.path.join(eval_root, f"fixed_{fmt_hz(fps)}Hz")
            os.makedirs(rate_dir, exist_ok=True)

            stop_recorder(recorder, executor, thread)
            node.episode_index = 0  # realign with the new EpisodeRecorder's own fresh episode_count
            recorder, executor, thread = start_recorder(rate_dir)

            evaluate_fixed_rate(node, fps, args.n_episodes, rate_dir)
    finally:
        stop_recorder(recorder, executor, thread)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
