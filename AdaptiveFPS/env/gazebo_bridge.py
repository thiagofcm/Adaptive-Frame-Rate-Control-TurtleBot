"""ROS/Gazebo bridge used by AdaptiveFPSEnv: the single rclpy node that owns the /scan -> /scan_gated
LiDAR gate, the step_comm/goal_comm/pause/unpause/reset_simulation clients and goal/odom/clock tracking.
The episode-ready handshake and the frozen TD3 navigation controller live in AdaptiveFPSEnv; the handshake
constants below are still defined here and imported by it.

Originally part of AdaptiveFPS/scripts/eval_adaptive_fps.py (class formerly FixedSensingEvaluator; that
legacy script has been removed). The ROS node name is still "fixed_sensing_evaluator".
"""
import math

from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data

from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty as EmptyMsg
from std_srvs.srv import Empty
from turtlebot3_msgs.srv import DrlStep, Goal


# Native LiDAR rate, gated topic, episode-ready handshake settings.

NATIVE_SCAN_HZ    = 50.0   # models/turtlebot3_burger/model.sdf's LiDAR <update_rate>
GATED_TOPIC       = "scan_gated"

# Episode-ready handshake: each individual wait (goal / clock / odom / forced
# scan) gets this long before that sub-check is declared timed out.
EPISODE_READY_TIMEOUT = 10.0
# Bounded retries per episode SLOT for the sync sub-checks (goal/clock/odom/
# scan) -- NOT for waiting for a brand-new /goal_pose, which only happens
# once per slot (see run_episode). If every attempt times out, this rate
# block is abandoned early rather than looping forever.
MAX_INIT_ATTEMPTS = 5

# Stage 9's robot reset pose, verified directly from
# worlds/turtlebot3_drl_stage9/burger.model: <pose>-0.9 2.0 0 0 0 ...</pose>
# (position only matters here, since AdaptiveFPSEnv._at_reset_pose() never checks
# orientation).
STAGE9_RESET_X = 2.5
STAGE9_RESET_Y = 2.5
RESET_POSITION_TOLERANCE_M = 1.0     # intentionally very permissive
RESET_VELOCITY_TOLERANCE_MPS = 0.03  # must stay a meaningful "robot settled" check
HEARTBEAT_PERIOD_S = 3.0       # handshake wait heartbeat
# Odometry distance accumulation: a single /odom displacement above this is not motion
# (at most ~0.0044 m per 50 Hz message at 0.22 m/s) -- it is ignored and logged.
ODOM_JUMP_GUARD_M = 0.5

def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9

class GazeboSensingBridge(Node):
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
        # Used only by AdaptiveFPSEnv._issue_reset_recovery() (handshake retry recovery,
        # not part of the normal episode-boundary path) -- same service
        # drl_gazebo.py's own task_fail_callback() already calls on every
        # ordinary episode failure, called here directly since it's a
        # plain Gazebo service, not something drl_gazebo.py exclusively
        # owns. Deliberately does NOT touch goal state (see
        # AdaptiveFPSEnv._issue_reset_recovery's docstring).
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

        # --- odometry path length (m), summed per /odom message while an episode is running.
        #     AdaptiveFPSEnv starts it after the episode-ready handshake and stops it at reset and
        #     termination, so reset/teleport motion is never counted. ---
        self.distance_traveled_m = 0.0
        self.distance_odom_msgs = 0
        self._distance_prev_xy = None
        self._distance_active = False

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

    # Gate control used by AdaptiveFPSEnv: it requests a sensing frequency, this node realizes it as
    # "forward every k-th native /scan". gated_scan_count and scans_since_forward stay readable counters.
    def scan_interval_for(self, fps):
        """Native-scan interval k that realizes `fps` (NATIVE_SCAN_HZ / fps, rounded, at least 1)."""
        return max(1, round(NATIVE_SCAN_HZ / fps))

    def set_sensing_rate(self, fps):
        """Forward every k-th native scan from now on (takes effect for the next interval). Returns k."""
        self.k = self.scan_interval_for(fps)
        return self.k

    def reset_gate(self, fps):
        """Episode start: set the rate and zero the gate counters. Leaves force_next_fresh unchanged
        (the episode-ready handshake requests the forced scan). Returns k."""
        k = self.set_sensing_rate(fps)
        self.scans_since_forward = 0
        self.gated_scan_count = 0
        return k

    def request_fresh_scan(self):
        """Forward the next native scan regardless of k."""
        self.force_next_fresh = True

    def _on_goal_pose(self, msg):
        self.goal_msg_count += 1
        self.latest_goal_xy = (msg.position.x, msg.position.y)

    def _on_odom(self, msg):
        self.latest_odom = msg
        self.odom_msg_count += 1
        if self._distance_active:
            x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
            step = math.hypot(x - self._distance_prev_xy[0], y - self._distance_prev_xy[1])
            if step > ODOM_JUMP_GUARD_M:
                self.get_logger().warning(
                    f"[episode {self.episode_index}] ignoring /odom jump of {step:.3f} m "
                    f"(> {ODOM_JUMP_GUARD_M} m) in distance_traveled_m")
            else:
                self.distance_traveled_m += step
            self._distance_prev_xy = (x, y)
            self.distance_odom_msgs += 1

    def start_distance(self, x, y):
        """Zero distance_traveled_m and accumulate from (x, y) on every following /odom message."""
        self.distance_traveled_m = 0.0
        self.distance_odom_msgs = 0
        self._distance_prev_xy = (x, y)
        self._distance_active = True

    def stop_distance(self):
        """Stop accumulating; distance_traveled_m keeps its last value."""
        self._distance_active = False

    def _on_clock(self, msg):
        self.latest_sim_time = stamp_to_sec(msg.clock)
        self.clock_msg_count += 1

