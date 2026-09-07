#!/usr/bin/env python3
"""Deterministic drop-in replacement for `gazebo_goals` (drl_gazebo.py),
used during reproducible Stage 9 evaluation runs (FixedFPS/AdaptiveFPS).

Run this INSTEAD OF `ros2 run turtlebot3_drl gazebo_goals` -- everything
else (`environment`, `test_agent`, TD3, the Gazebo world/SDF files) stays
exactly as-is. It works because drl_environment.py only cares that *some*
node publishes `goal_pose` and serves `task_succeed`/`task_fail`; it has no
idea which node that is. drl_gazebo.py itself is never imported or modified
by this file.

Why: in the stock pipeline, `task_succeed_callback` never resets the
simulation (only `task_fail_callback` does -- see drl_gazebo.py:99-124), so
a successful episode leaves the robot wherever it stopped and the next
episode's starting condition silently depends on the previous episode's
outcome. This node removes that dependency: at *every* episode boundary,
success or failure, it unconditionally resets Gazebo, teleports the robot
to a prescribed (x, y, yaw), and publishes a prescribed goal -- taken from a
deterministic chain built out of the curated Stage 9 goal points.

The core logic lives in `ScenarioGazebo.reset_scenario()`, deliberately
factored out of the service callbacks so it can be called directly (e.g. by
the --validate mode below, and later reused as `env.reset_scenario(...)`
from a FixedFPS/AdaptiveFPS driver).

Usage:
    # normal operation (run in place of `gazebo_goals`)
    python3 scenario_gazebo.py

    # reproducibility check: repeat one scenario N times, report drift
    python3 scenario_gazebo.py --validate --scenario-index 0 --repeats 5
"""
import argparse
import math
import os
import sys
import threading
import time
from dataclasses import dataclass

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile

from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import DeleteEntity, SpawnEntity, SetEntityState
from turtlebot3_msgs.srv import RingGoal


# ===================================================================== #
#   Curated Stage 9 goal points -- copied verbatim (data only, no logic)
#   from drl_gazebo.py's generate_goal_pose(), the
#   `stage == 8 or stage == 9 or stage == 12` branch:
#   src/turtlebot3_drl/turtlebot3_drl/drl_gazebo/drl_gazebo.py:186-193
#   Duplicated here rather than imported because that list is a local
#   literal inside a method, not a reusable module-level constant, and
#   this file must not modify drl_gazebo.py.
# ===================================================================== #
STAGE9_GOAL_POINTS = [
    (2.0, 2.0), (2.0, 1.5), (2.0, -0.5), (2.0, -1.0), (2.0, -2.0), (1.3, 1.0),
    (1.0, 0.3), (1.0, -2.0), (0.3, -1.0), (0.0, 2.0), (0.0, -1.0), (-1.0, 1.0),
    (-1.0, -1.2), (-2.0, 1.0), (-2.2, 0.0), (-2.0, -2.2), (-2.4, 2.4),
]

MIN_SEPARATION_M = 2.0    # same numeric threshold generate_goal_pose() uses (there: L1, here: Euclidean)
ORIGIN = (0.0, 0.0)
ROBOT_ENTITY_NAME = "turtlebot3_burger"   # <model name=...> in models/turtlebot3_burger/model.sdf
GOAL_ENTITY_NAME = "goal"                 # matches drl_gazebo.py's self.entity_name


@dataclass(frozen=True)
class Scenario:
    index: int
    start_x: float
    start_y: float
    start_yaw: float
    goal_x: float
    goal_y: float


def build_scenarios(points=STAGE9_GOAL_POINTS, origin=ORIGIN, min_separation=MIN_SEPARATION_M):
    """Deterministically chain the curated points into a walk starting at
    `origin`, so scenario i's goal becomes scenario i+1's start (per the
    approved design) -- ScenarioGazebo makes that explicit/robust via an
    actual Gazebo reset rather than leaving it as an implicit side effect.

    Order is deterministic (no RNG): at each step, walk the still-unused
    points in their original list order and take the first one at least
    `min_separation` away from the current point, to avoid the
    trivially-short pairs that raw list order alone can produce. If every
    remaining point is closer than that (shouldn't happen with this point
    set), fall back to the farthest remaining point so the chain still
    terminates deterministically.
    """
    unused = list(points)
    chain = [origin]
    current = origin
    while unused:
        candidates = [p for p in unused if math.dist(current, p) >= min_separation]
        nxt = candidates[0] if candidates else max(unused, key=lambda p: math.dist(current, p))
        chain.append(nxt)
        unused.remove(nxt)
        current = nxt

    scenarios = []
    for i in range(len(chain) - 1):
        sx, sy = chain[i]
        gx, gy = chain[i + 1]
        yaw = math.atan2(gy - sy, gx - sx)   # start yaw points at the goal, per current requirement
        scenarios.append(Scenario(i, sx, sy, yaw, gx, gy))
    return scenarios


def yaw_to_quaternion_zw(yaw):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def load_sdf(path):
    with open(path) as f:
        return f.read()


class ScenarioGazebo(Node):
    def __init__(self, scenarios, loop=True):
        super().__init__("scenario_gazebo")
        self.scenarios = scenarios
        self.loop = loop
        self.scenario_ptr = 0

        self.cb_group = ReentrantCallbackGroup()
        # Serializes _on_episode_boundary(): without this, two overlapping
        # task_succeed/task_fail calls (ReentrantCallbackGroup permits them
        # to run concurrently on a MultiThreadedExecutor) could interleave
        # their Gazebo reset sequences and race on scenario_ptr.
        self.reset_lock = threading.Lock()

        base_path = os.environ.get("DRLNAV_BASE_PATH")
        if not base_path:
            self.get_logger().error("DRLNAV_BASE_PATH is not set -- cannot locate model SDF files.")
            raise RuntimeError("DRLNAV_BASE_PATH not set")
        gazebo_pkg = os.path.join(base_path, "src", "turtlebot3_simulations", "turtlebot3_gazebo")
        self.goal_sdf_path = os.path.join(gazebo_pkg, "models", "turtlebot3_drl_world", "goal_box", "model.sdf")
        self.robot_sdf_path = os.path.join(gazebo_pkg, "models", "turtlebot3_burger", "model.sdf")
        self.goal_xml = load_sdf(self.goal_sdf_path)
        self._robot_xml = None   # lazily loaded only if the respawn fallback is actually needed

        qos = QoSProfile(depth=10)
        self.goal_pose_pub = self.create_publisher(Pose, "goal_pose", qos)

        self.latest_odom = None
        self.latest_obstacles = {}
        self.create_subscription(Odometry, "odom", self._on_odom, qos)
        self.create_subscription(Odometry, "obstacle/odom", self._on_obstacle_odom, qos)

        self.reset_client = self.create_client(Empty, "reset_simulation", callback_group=self.cb_group)
        self.delete_client = self.create_client(DeleteEntity, "delete_entity", callback_group=self.cb_group)
        self.spawn_client = self.create_client(SpawnEntity, "spawn_entity", callback_group=self.cb_group)

        self.set_entity_state_client = self._probe_set_entity_state()

        self.task_succeed_srv = self.create_service(
            RingGoal, "task_succeed", self._on_episode_boundary, callback_group=self.cb_group)
        self.task_fail_srv = self.create_service(
            RingGoal, "task_fail", self._on_episode_boundary, callback_group=self.cb_group)

        self.get_logger().info(
            f"scenario_gazebo: {len(scenarios)} scenarios loaded, robot placement via "
            f"{'SetEntityState' if self.set_entity_state_client else 'delete+respawn fallback'}")

        # Place scenario 0 immediately, mirroring drl_gazebo.py's own init_callback() behavior,
        # so the first episode also starts from a prescribed (not baked-in-SDF-default) pose.
        self.reset_scenario(self.scenarios[0])

    # ------------------------------------------------------------------ #
    #   Startup: is a SetEntityState-style teleport service available?
    # ------------------------------------------------------------------ #
    def _probe_set_entity_state(self):
        for name in ("set_entity_state", "/set_entity_state", "gazebo/set_entity_state"):
            client = self.create_client(SetEntityState, name, callback_group=self.cb_group)
            if client.wait_for_service(timeout_sec=2.0):
                self.get_logger().info(f"found SetEntityState service at '{name}' -- using it for robot placement")
                return client
            client.destroy()
        self.get_logger().warning(
            "No SetEntityState-style service found (tried 'set_entity_state', '/set_entity_state', "
            "'gazebo/set_entity_state', 2s each). Falling back to delete+respawn for robot placement "
            "-- this briefly tears down and recreates the robot's diff-drive/LiDAR/IMU plugins each "
            "episode boundary, so /odom and /scan will have a short gap after every reset.\n"
            "Smallest fix to enable direct teleport instead: add ONE line inside the <world> element "
            "of worlds/turtlebot3_drl_stage9/burger.model (no other file needs to change):\n"
            '    <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so"/>\n'
            "then relaunch Gazebo. No Python/RL code is affected either way.")
        return None

    # ------------------------------------------------------------------ #
    #   Subscriptions (used by --validate; harmless overhead otherwise)
    # ------------------------------------------------------------------ #
    def _on_odom(self, msg):
        self.latest_odom = msg

    def _on_obstacle_odom(self, msg):
        if "obstacle" in msg.child_frame_id:
            self.latest_obstacles[msg.child_frame_id] = (
                msg.pose.pose.position.x, msg.pose.pose.position.y)

    # ------------------------------------------------------------------ #
    #   Episode boundary: identical handling for success AND failure --
    #   this is the deliberate deviation from drl_gazebo.py that makes
    #   scenario N+1 independent of scenario N's outcome.
    # ------------------------------------------------------------------ #
    def _on_episode_boundary(self, request, response):
        # --- FREEZE DIAGNOSTIC LOGGING ---
        self.get_logger().info(f"[FREEZE-DIAG] task_succeed/task_fail received (thread={threading.get_ident()})")

        # `with self.reset_lock:` replaced by manual acquire/release *only* to
        # log immediately before/after the (unchanged, still plain-blocking)
        # acquire call -- semantics are identical to the previous `with` block.
        self.get_logger().info("[FREEZE-DIAG] reset_lock ACQUIRE")
        self.reset_lock.acquire()
        self.get_logger().info("[FREEZE-DIAG] reset_lock ACQUIRED")
        try:
            prev_ptr = self.scenario_ptr
            self.scenario_ptr += 1
            if self.scenario_ptr >= len(self.scenarios):
                if self.loop:
                    self.scenario_ptr = 0
                else:
                    self.get_logger().info("scenario_gazebo: scenario list exhausted, holding on the last scenario")
                    self.scenario_ptr = len(self.scenarios) - 1
            # thread id is logged deliberately: if two boundary calls ever overlap
            # (see diagnosis notes), this is what reveals it in the log stream.
            self.get_logger().info(
                f"episode boundary: scenario_ptr {prev_ptr} -> {self.scenario_ptr} "
                f"(callback thread={threading.get_ident()})")
            self.reset_scenario(self.scenarios[self.scenario_ptr])
        finally:
            self.reset_lock.release()
            self.get_logger().info("[FREEZE-DIAG] reset_lock RELEASED")

        self.get_logger().info("[FREEZE-DIAG] episode boundary COMPLETE")
        return response

    # ------------------------------------------------------------------ #
    #   The reusable core -- intended to later back env.reset_scenario()
    # ------------------------------------------------------------------ #
    def reset_scenario(self, scenario: Scenario, settle_sec: float = 0.0):
        """Reset Gazebo, place the robot at scenario.start_{x,y,yaw}, and
        publish/spawn scenario.goal_{x,y}. Safe to call directly (not just
        from the task_succeed/task_fail callbacks) -- this is exactly the
        method meant to be reused later as env.reset_scenario(scenario).

        NOTE (see investigation notes / commit message): nothing here blocks
        drl_agent.py from starting the next episode before this method
        returns -- see the diagnosis of the false-early-SUCCESS bug. The
        BEGIN/END log lines with elapsed time exist specifically to make
        that race visible against drl_agent.py's own timing.
        """
        t0 = time.monotonic()
        distance = math.dist((scenario.start_x, scenario.start_y), (scenario.goal_x, scenario.goal_y))
        self.get_logger().info(
            f"[reset_scenario BEGIN] scenario_index={scenario.index} "
            f"start=(x={scenario.start_x:.3f}, y={scenario.start_y:.3f}, "
            f"yaw={math.degrees(scenario.start_yaw):.1f}deg) "
            f"goal=(x={scenario.goal_x:.3f}, y={scenario.goal_y:.3f}) "
            f"start_to_goal_distance={distance:.3f}m")

        self.get_logger().info("[FREEZE-DIAG] reset_simulation CALL")
        self._call(self.reset_client, Empty.Request(), label="reset_simulation")
        self.get_logger().info("[FREEZE-DIAG] reset_simulation DONE")

        self.get_logger().info(
            f"[FREEZE-DIAG] robot placement CALL "
            f"({'SetEntityState' if self.set_entity_state_client is not None else 'delete+respawn fallback'})")
        self._place_robot(scenario.start_x, scenario.start_y, scenario.start_yaw)
        self.get_logger().info("[FREEZE-DIAG] robot placement DONE")

        self.get_logger().info("[FREEZE-DIAG] delete goal CALL")
        self._call(self.delete_client, self._delete_request(GOAL_ENTITY_NAME), label="delete_entity(goal)")
        self.get_logger().info("[FREEZE-DIAG] delete goal DONE")

        self.get_logger().info("[FREEZE-DIAG] spawn goal CALL")
        self._call(self.spawn_client, self._goal_spawn_request(scenario.goal_x, scenario.goal_y),
                   label="spawn_entity(goal)")
        self.get_logger().info("[FREEZE-DIAG] spawn goal DONE")

        self.get_logger().info("[FREEZE-DIAG] publish goal")
        goal_msg = Pose()
        goal_msg.position.x = scenario.goal_x
        goal_msg.position.y = scenario.goal_y
        self.goal_pose_pub.publish(goal_msg)
        self.get_logger().info("[FREEZE-DIAG] publish goal DONE")

        if settle_sec > 0:
            time.sleep(settle_sec)

        elapsed = time.monotonic() - t0
        self.get_logger().info(
            f"[reset_scenario END]   scenario_index={scenario.index} elapsed={elapsed:.3f}s")

    def _place_robot(self, x, y, yaw):
        if self.set_entity_state_client is not None:
            req = SetEntityState.Request()
            req.state.name = ROBOT_ENTITY_NAME
            req.state.pose.position.x = float(x)
            req.state.pose.position.y = float(y)
            req.state.pose.position.z = 0.0
            qz, qw = yaw_to_quaternion_zw(yaw)
            req.state.pose.orientation.z = qz
            req.state.pose.orientation.w = qw
            req.state.twist.linear.x = 0.0
            req.state.twist.linear.y = 0.0
            req.state.twist.linear.z = 0.0
            req.state.twist.angular.x = 0.0
            req.state.twist.angular.y = 0.0
            req.state.twist.angular.z = 0.0
            req.state.reference_frame = "world"
            self._call(self.set_entity_state_client, req, label="set_entity_state(robot)")
        else:
            self._respawn_robot(x, y, yaw)

    def _respawn_robot(self, x, y, yaw):
        if self._robot_xml is None:
            self._robot_xml = load_sdf(self.robot_sdf_path)
        self._call(self.delete_client, self._delete_request(ROBOT_ENTITY_NAME), label="delete_entity(robot)")
        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        qz, qw = yaw_to_quaternion_zw(yaw)
        pose.orientation.z = qz
        pose.orientation.w = qw
        req = SpawnEntity.Request()
        req.name = ROBOT_ENTITY_NAME
        req.xml = self._robot_xml
        req.initial_pose = pose
        self._call(self.spawn_client, req, label="spawn_entity(robot)")

    def _delete_request(self, name):
        req = DeleteEntity.Request()
        req.name = name
        return req

    def _goal_spawn_request(self, x, y):
        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        req = SpawnEntity.Request()
        req.name = GOAL_ENTITY_NAME
        req.xml = self.goal_xml
        req.initial_pose = pose
        return req

    def _call(self, client, request, timeout_sec=5.0, label=None):
        """Synchronous service call from within a node that may itself be
        mid-callback. Requires the node to be spun by a MultiThreadedExecutor
        with clients/servers on a ReentrantCallbackGroup (both true here) --
        otherwise this would deadlock waiting on its own executor thread.

        The three [FREEZE-DIAG] sub-steps below are the actual candidate
        blocking waits inside this path: wait_for_service() (service never
        advertised) and spin_until_future_complete() (request sent but no
        response ever arrives/gets processed) are two functionally different
        ways to hang, and this pinpoints which one it is. Note timeout_sec
        is passed to spin_until_future_complete() but whether it is honored
        when called reentrantly (from inside another callback, via the
        MultiThreadedExecutor + ReentrantCallbackGroup pattern used here) is
        exactly one of the things this logging is meant to reveal -- if it
        hangs, look for whether "waiting on future" is the last line ever
        printed for that label (timeout not honored) vs. a "TIMED OUT" line
        eventually appearing (timeout honored, something else is frozen).
        """
        label = label or client.srv_name
        self.get_logger().info(f"[FREEZE-DIAG]   _call({label}): wait_for_service...")
        if not client.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error(f"service '{client.srv_name}' not available")
            self.get_logger().info(f"[FREEZE-DIAG]   _call({label}): wait_for_service TIMED OUT (service unavailable)")
            return None
        self.get_logger().info(f"[FREEZE-DIAG]   _call({label}): service available, call_async...")
        future = client.call_async(request)
        self.get_logger().info(f"[FREEZE-DIAG]   _call({label}): waiting on future (spin_until_future_complete)...")
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if not future.done():
            self.get_logger().error(f"service '{client.srv_name}' call timed out")
            self.get_logger().info(f"[FREEZE-DIAG]   _call({label}): spin_until_future_complete TIMED OUT (future not done)")
            return None
        self.get_logger().info(f"[FREEZE-DIAG]   _call({label}): future completed")
        return future.result()

    # ------------------------------------------------------------------ #
    #   Reproducibility validation
    # ------------------------------------------------------------------ #
    def run_validation(self, scenario: Scenario, repeats: int, settle_sec: float):
        self.get_logger().info(f"=== reproducibility validation: scenario {scenario.index}, {repeats} repeats ===")
        rows = []
        for i in range(repeats):
            self.latest_odom = None
            self.latest_obstacles = {}
            self.reset_scenario(scenario, settle_sec=settle_sec)
            odom = self.latest_odom
            rows.append({
                "repeat": i,
                "observed_odom": (
                    odom.pose.pose.position.x, odom.pose.pose.position.y,
                    quaternion_to_yaw(odom.pose.pose.orientation),
                ) if odom is not None else None,
                "obstacles": dict(self.latest_obstacles),
            })
        self._print_validation_report(scenario, rows)

    def _print_validation_report(self, scenario, rows):
        print()
        print(f"Scenario {scenario.index}")
        print(f"  commanded start pose : x={scenario.start_x:.3f} y={scenario.start_y:.3f} "
              f"yaw={math.degrees(scenario.start_yaw):.1f} deg")
        print(f"  goal pose            : x={scenario.goal_x:.3f} y={scenario.goal_y:.3f}")
        print()
        print(f"  {'repeat':>6} | {'odom x':>8} {'odom y':>8} {'odom yaw':>9} | dx from commanded | dy | dyaw")
        for row in rows:
            if row["observed_odom"] is None:
                print(f"  {row['repeat']:>6} | NO /odom RECEIVED -- increase --settle-sec")
                continue
            ox, oy, oyaw = row["observed_odom"]
            dx, dy = ox - scenario.start_x, oy - scenario.start_y
            dyaw = math.degrees(((oyaw - scenario.start_yaw + math.pi) % (2 * math.pi)) - math.pi)
            print(f"  {row['repeat']:>6} | {ox:8.3f} {oy:8.3f} {math.degrees(oyaw):9.1f} | "
                  f"{dx:+.3f}             | {dy:+.3f} | {dyaw:+.1f} deg")

        obstacle_ids = sorted({oid for row in rows for oid in row["obstacles"]})
        if not obstacle_ids:
            print("\n  no obstacle/odom messages observed (no dynamic obstacles on this stage, or none arrived in time)")
            return
        print("\n  initial moving-obstacle positions per repeat:")
        for oid in obstacle_ids:
            positions = [row["obstacles"].get(oid) for row in rows]
            print(f"    {oid}:")
            for i, p in enumerate(positions):
                print(f"      repeat {i}: {p if p is not None else 'not observed'}")
            seen = [p for p in positions if p is not None]
            if len(seen) > 1:
                spread_x = max(p[0] for p in seen) - min(p[0] for p in seen)
                spread_y = max(p[1] for p in seen) - min(p[1] for p in seen)
                verdict = "IDENTICAL across repeats" if (spread_x < 1e-3 and spread_y < 1e-3) else \
                    f"DRIFTS across repeats (spread x={spread_x:.3f} y={spread_y:.3f}) -- animation phase is not resetting"
                print(f"      -> {verdict}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-separation", type=float, default=MIN_SEPARATION_M,
                         help="Minimum distance (m) enforced between consecutive scenario waypoints")
    parser.add_argument("--no-loop", action="store_true",
                         help="Hold on the last scenario instead of looping back to scenario 0")
    parser.add_argument("--validate", action="store_true",
                         help="Run reproducibility validation instead of serving task_succeed/task_fail")
    parser.add_argument("--scenario-index", type=int, default=0, help="[--validate] which scenario to repeat")
    parser.add_argument("--repeats", type=int, default=5, help="[--validate] how many times to repeat it")
    parser.add_argument("--settle-sec", type=float, default=0.5,
                         help="Seconds to wait after each reset before reading back /odom and obstacle/odom")
    args = parser.parse_args()

    scenarios = build_scenarios(min_separation=args.min_separation)

    rclpy.init()
    try:
        node = ScenarioGazebo(scenarios, loop=not args.no_loop)
    except RuntimeError as exc:
        print(f"scenario_gazebo: {exc}", file=sys.stderr)
        rclpy.shutdown()
        sys.exit(1)

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    if args.validate:
        if not (0 <= args.scenario_index < len(scenarios)):
            print(f"--scenario-index must be in [0, {len(scenarios) - 1}]", file=sys.stderr)
            sys.exit(1)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()
        try:
            node.run_validation(scenarios[args.scenario_index], args.repeats, args.settle_sec)
        finally:
            node.destroy_node()
            rclpy.shutdown()
        return

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
