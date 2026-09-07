#!/usr/bin/env python3
"""Passive, read-only recorder for turtlebot3_drlnav TD3 evaluation episodes.

Subscribes to /goal_pose, /odom, /scan, /cmd_vel, /clock and obstacle/odom
(if present). Never publishes anything and never calls a service, so it
cannot affect TD3 actions, reward, or Gazebo physics.

Episode boundaries are inferred purely from /goal_pose arrivals: gazebo_goals
(drl_gazebo.py) publishes exactly one Pose on that topic at startup and again
after every task_succeed/task_fail call, so each message marks the start of a
new episode (and the end of the previous one, if any).

This script does NOT classify SUCCESS/COLLISION/TIMEOUT/TUMBLE. That decision
is made inside drl_environment.py's get_state() and is only ever exposed as
the `success` field of the DrlStep *service* response -- a passive subscriber
cannot observe service traffic between other nodes the way it can a topic.
Ground truth for outcomes is the evaluation log the unmodified Logger class
already writes (common/logger.py -> _test_stage<N>_eps<E>_<timestamp>.txt);
join recorded episodes to that log's rows by episode order after a run.

Run (after sourcing the workspace, so `turtlebot3_drl` is importable):
    python3 record_episode.py --out-dir ~/drlnav_recordings/td3_stage9_eval
"""
import argparse
import csv
import json
import math
import os
import signal
import time
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data

from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Empty

try:
    from turtlebot3_drl.common.settings import ARENA_LENGTH, ARENA_WIDTH
except ImportError:
    ARENA_LENGTH, ARENA_WIDTH = None, None


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def mean_dt(times):
    if len(times) < 2:
        return None
    ordered = sorted(times)
    diffs = [b - a for a, b in zip(ordered, ordered[1:])]
    return sum(diffs) / len(diffs)


# --- Static world geometry (walls) -----------------------------------------
#
# The maze/boundary walls for every stage are static Gazebo models loaded
# once, directly in the stage's world SDF (see worlds/turtlebot3_drl_stage<N>
# /<robot>.model), not spawned/deleted through a service the way the goal
# marker is. So they never change during or between episodes of a run, and
# are captured once per recording run (world_geometry.json) instead of once
# per episode. Geometry is parsed straight from the actual SDF files Gazebo
# itself loads for the current stage -- nothing here is stage-specific or
# hardcoded; it works the same way for any stage number.

def parse_pose(elem):
    """Parse an SDF <pose>x y z roll pitch yaw</pose> element; identity if absent."""
    if elem is None or not elem.text:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    vals = [float(v) for v in elem.text.split()]
    vals += [0.0] * (6 - len(vals))
    return tuple(vals[:6])


def compose_pose_2d(parent, child):
    """Compose two SDF poses in the (x, y, yaw) plane. Ignores z/roll/pitch,
    which is exact for this repo's wall models (axis-aligned yaw-only boxes)."""
    px, py, _, _, _, pyaw = parent
    cx, cy, _, _, _, cyaw = child
    wx = px + cx * math.cos(pyaw) - cy * math.sin(pyaw)
    wy = py + cx * math.sin(pyaw) + cy * math.cos(pyaw)
    return (wx, wy, 0.0, 0.0, 0.0, pyaw + cyaw)


def model_uri_to_name(uri):
    prefix = "model://"
    return uri[len(prefix):] if uri.startswith(prefix) else uri


def parse_model_sdf_boxes(model_sdf_path, model_world_pose):
    """Return one {cx, cy, size_x, size_y, yaw} rectangle (world frame) per
    box-collision <link> in an SDF model file."""
    root = ET.parse(model_sdf_path).getroot()
    model_elem = root.find("model")
    if model_elem is None:
        return []
    combined_pose = compose_pose_2d(model_world_pose, parse_pose(model_elem.find("pose")))
    rects = []
    for link in model_elem.findall("link"):
        box_size = link.find("collision/geometry/box/size")
        if box_size is None or not box_size.text:
            continue
        size_x, size_y = (float(v) for v in box_size.text.split()[:2])
        link_pose = parse_pose(link.find("pose"))
        if abs(link_pose[3]) > 1e-6 or abs(link_pose[4]) > 1e-6:
            print(f"episode_recorder: warning: non-planar link pose in {model_sdf_path}, "
                  f"2D wall rendering will be approximate for this link")
        wx, wy, _, _, _, wyaw = compose_pose_2d(combined_pose, link_pose)
        rects.append({"cx": wx, "cy": wy, "size_x": size_x, "size_y": size_y, "yaw": wyaw})
    return rects


def load_world_geometry(base_path, stage):
    """Parse the current stage's world SDF for its static scene geometry
    (boundary/maze walls, and any static-obstacle models), by walking only
    the world's direct <include> children -- this naturally excludes the
    moving-obstacle models (each wrapped in a <model> with a <plugin>, not a
    bare <include>) and the ground plane / sun / robot, without needing to
    know anything stage-specific in advance."""
    robot_model = os.environ.get("TURTLEBOT3_MODEL", "burger")
    gazebo_pkg = os.path.join(base_path, "src", "turtlebot3_simulations", "turtlebot3_gazebo")
    world_path = os.path.join(gazebo_pkg, "worlds", f"turtlebot3_drl_stage{stage}", f"{robot_model}.model")
    models_dir = os.path.join(gazebo_pkg, "models")
    exclude_names = {"ground_plane", "sun", f"turtlebot3_{robot_model}"}

    world_elem = ET.parse(world_path).getroot().find("world")
    walls = []
    source_models = []
    for include in world_elem.findall("include"):
        uri = include.find("uri")
        if uri is None or not uri.text:
            continue
        name = model_uri_to_name(uri.text.strip())
        if name in exclude_names:
            continue
        model_sdf_path = os.path.join(models_dir, name, "model.sdf")
        if not os.path.exists(model_sdf_path):
            continue
        include_pose = parse_pose(include.find("pose"))
        rects = parse_model_sdf_boxes(model_sdf_path, include_pose)
        if rects:
            walls.extend(rects)
            source_models.append(name)

    return {"stage": stage, "world_file": world_path, "source_models": source_models, "walls": walls}


class Episode:
    def __init__(self, index, goal_x, goal_y):
        self.index = index
        self.goal_x = goal_x
        self.goal_y = goal_y
        # Timing origin is latched by on_episode_ready(), not at creation
        # (creation happens at /goal_pose, which precedes the evaluator's
        # handshake -- too early to be t=0). Both pairs below stay absolute
        # epoch/sim-clock values; duration is tracked separately as a
        # relative offset so the two are never subtracted from each other
        # (see add_odom / write()).
        self.ready = False
        self.start_wall = None
        self.start_sim = None
        self.end_wall = None
        self.end_sim = None
        self.end_wall_rel = 0.0
        self.end_sim_rel = None
        self.latest_odom_msg = None  # cached continuously, pre- and post-READY
        self.trajectory_rows = []
        self.lidar_t_wall = []
        self.lidar_t_sim = []
        self.lidar_ranges = []
        self.last_cmd_linear = 0.0
        self.last_cmd_angular = 0.0
        self.cmd_vel_wall_times = []
        self.obstacle_t_wall = []
        self.obstacle_t_sim = []
        self.obstacle_frame_id = []
        self.obstacle_x = []
        self.obstacle_y = []
        self.obstacle_yaw = []

    def add_odom(self, t_wall, t_sim, x, y, yaw, tilt):
        self.trajectory_rows.append((
            t_wall, t_sim, x, y, yaw, tilt,
            self.last_cmd_linear, self.last_cmd_angular,
            self.goal_x, self.goal_y,
        ))
        self.end_wall_rel = t_wall
        self.end_wall = (self.start_wall + t_wall) if self.start_wall is not None else None
        if t_sim is not None:
            self.end_sim_rel = t_sim
            self.end_sim = (self.start_sim + t_sim) if self.start_sim is not None else None

    def add_scan(self, t_wall, t_sim, ranges):
        self.lidar_t_wall.append(t_wall)
        self.lidar_t_sim.append(t_sim)
        self.lidar_ranges.append(ranges)

    def set_cmd(self, linear, angular, t_wall):
        self.last_cmd_linear = linear
        self.last_cmd_angular = angular
        self.cmd_vel_wall_times.append(t_wall)

    def add_obstacle(self, t_wall, t_sim, frame_id, x, y, yaw):
        self.obstacle_t_wall.append(t_wall)
        self.obstacle_t_sim.append(t_sim)
        self.obstacle_frame_id.append(frame_id)
        self.obstacle_x.append(x)
        self.obstacle_y.append(y)
        self.obstacle_yaw.append(yaw)

    def write(self, out_dir):
        ep_dir = os.path.join(out_dir, f"episode_{self.index:04d}")
        os.makedirs(ep_dir, exist_ok=True)

        with open(os.path.join(ep_dir, "trajectory.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t_wall", "t_sim", "x", "y", "yaw", "robot_tilt_raw",
                              "cmd_linear", "cmd_angular", "goal_x", "goal_y"])
            writer.writerows(self.trajectory_rows)

        if self.lidar_ranges:
            max_len = max(len(r) for r in self.lidar_ranges)
            ranges_arr = np.full((len(self.lidar_ranges), max_len), np.nan, dtype=np.float32)
            for i, r in enumerate(self.lidar_ranges):
                ranges_arr[i, :len(r)] = r
        else:
            ranges_arr = np.zeros((0, 0), dtype=np.float32)
        np.savez_compressed(
            os.path.join(ep_dir, "lidar.npz"),
            t_wall=np.array(self.lidar_t_wall, dtype=np.float64),
            t_sim=np.array([t if t is not None else np.nan for t in self.lidar_t_sim], dtype=np.float64),
            ranges=ranges_arr,
        )

        np.savez_compressed(
            os.path.join(ep_dir, "obstacles.npz"),
            t_wall=np.array(self.obstacle_t_wall, dtype=np.float64),
            t_sim=np.array([t if t is not None else np.nan for t in self.obstacle_t_sim], dtype=np.float64),
            frame_id=np.array(self.obstacle_frame_id, dtype="<U64"),
            x=np.array(self.obstacle_x, dtype=np.float64),
            y=np.array(self.obstacle_y, dtype=np.float64),
            yaw=np.array(self.obstacle_yaw, dtype=np.float64),
        )

        # Both durations are already-relative offsets tracked directly by
        # add_odom (end_wall_rel/end_sim_rel) -- never an absolute-minus-
        # absolute subtraction, since start_wall/start_sim are absolute
        # epoch/sim-clock values and mixing the two previously produced a
        # nonsensical duration once any odom had been recorded.
        duration_wall = self.end_wall_rel
        duration_sim = self.end_sim_rel

        odom_wall_times = [row[0] for row in self.trajectory_rows]
        meta = {
            "episode_index": self.index,
            "goal_x": self.goal_x,
            "goal_y": self.goal_y,
            "start_wall": self.start_wall,
            "end_wall": self.end_wall,
            "start_sim": self.start_sim,
            "end_sim": self.end_sim,
            "duration_wall_s": duration_wall,
            "duration_sim_s": duration_sim,
            "odom_rows": len(self.trajectory_rows),
            "scan_rows": len(self.lidar_ranges),
            "obstacle_rows": len(self.obstacle_frame_id),
            "obstacle_ids": sorted(set(self.obstacle_frame_id)),
            "mean_dt_odom_s": mean_dt(odom_wall_times),
            "mean_dt_scan_s": mean_dt(self.lidar_t_wall),
            "mean_dt_cmd_vel_s": mean_dt(self.cmd_vel_wall_times),
            "arena_length": ARENA_LENGTH,
            "arena_width": ARENA_WIDTH,
            "outcome": None,
            "outcome_note": (
                "Ground-truth outcome is not recorded here. Join this episode (by order) "
                "with the matching row of the test_agent evaluation log "
                "'_test_stage<N>_eps<E>_<timestamp>.txt' written by common/logger.py."
            ),
        }
        with open(os.path.join(ep_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        return ep_dir, duration_wall


class EpisodeRecorder(Node):
    def __init__(self, out_dir):
        super().__init__("episode_recorder")
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.manifest_path = os.path.join(out_dir, "manifest.csv")
        if not os.path.exists(self.manifest_path):
            with open(self.manifest_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "episode_index", "dir", "start_wall", "end_wall",
                    "duration_s", "odom_rows", "scan_rows", "obstacle_rows",
                ])

        self.episode_count = 0
        self.current = None
        self.latest_sim_time = None

        self._write_world_geometry(out_dir)

        qos = QoSProfile(depth=10)
        self.create_subscription(Pose, "goal_pose", self.on_goal_pose, qos)
        self.create_subscription(Odometry, "odom", self.on_odom, qos)
        self.create_subscription(LaserScan, "scan", self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Twist, "cmd_vel", self.on_cmd_vel, qos)
        self.create_subscription(Clock, "/clock", self.on_clock, QoSProfile(depth=10))
        self.create_subscription(Odometry, "obstacle/odom", self.on_obstacle_odom, qos)
        self.create_subscription(Empty, "episode_ready", self.on_episode_ready, QoSProfile(depth=10))

        self.get_logger().info(f"episode_recorder: observational only, writing to {out_dir}")

    def _write_world_geometry(self, out_dir):
        """Best-effort, once per run: snapshot the current stage's static wall
        geometry from its actual world SDF files. Never raises -- if it fails
        (e.g. DRLNAV_BASE_PATH unset, stage file not written yet), recording
        motion data still proceeds without wall geometry."""
        geometry_path = os.path.join(out_dir, "world_geometry.json")
        if os.path.exists(geometry_path):
            return
        try:
            base_path = os.environ["DRLNAV_BASE_PATH"]
            with open("/tmp/drlnav_current_stage.txt") as f:
                stage = int(f.read())
            geometry = load_world_geometry(base_path, stage)
            with open(geometry_path, "w") as f:
                json.dump(geometry, f, indent=2)
            self.get_logger().info(
                f"world_geometry.json written for stage {stage} "
                f"({len(geometry['walls'])} wall segments from {geometry['source_models']})")
        except Exception as exc:
            self.get_logger().warning(f"could not record world geometry, continuing without it: {exc}")

    def on_clock(self, msg):
        # Pure tracking -- the sim-time origin is latched explicitly in
        # on_episode_ready(), not opportunistically here (that would set it
        # before READY and reintroduce the same "too early" problem this
        # fix removes for wall time).
        self.latest_sim_time = stamp_to_sec(msg.clock)

    def on_goal_pose(self, msg):
        if self.current is not None and self.current.trajectory_rows:
            self._finalize_current()
        self.episode_count += 1
        self.current = Episode(self.episode_count, msg.position.x, msg.position.y)
        self.get_logger().info(
            f"episode {self.episode_count} goal received "
            f"({msg.position.x:.2f}, {msg.position.y:.2f}); waiting for READY")

    def on_episode_ready(self, msg):
        """The evaluator publishes this exactly once per valid episode,
        right before its first TD3 control step -- this is the canonical
        t_wall=0/t_sim=0 origin, not /goal_pose arrival (which precedes the
        evaluator's own reset/readiness handshake)."""
        if self.current is None:
            self.get_logger().warning("received /episode_ready with no pending episode; ignoring")
            return
        if self.current.ready:
            self.get_logger().warning(
                f"episode {self.current.index}: duplicate /episode_ready received; ignoring")
            return
        self.current.ready = True
        self.current.start_wall = time.time()
        self.current.start_sim = self.latest_sim_time
        self.get_logger().info(f"episode {self.current.index} READY; origin latched")

        odom = self.current.latest_odom_msg
        if odom is not None:
            yaw = quaternion_to_yaw(odom.pose.pose.orientation)
            tilt = odom.pose.pose.orientation.y
            self.current.add_odom(0.0, 0.0, odom.pose.pose.position.x,
                                   odom.pose.pose.position.y, yaw, tilt)
            self.get_logger().info(
                f"initial trajectory pose: t_wall=0.000 x={odom.pose.pose.position.x:.3f} "
                f"y={odom.pose.pose.position.y:.3f}")
        else:
            self.get_logger().warning(
                f"episode {self.current.index}: no /odom cached at READY; "
                f"first trajectory row will come from the next /odom message")

    def _sim_relative(self, stamp_sim):
        if self.current.start_sim is None:
            return None
        if stamp_sim:
            return stamp_sim - self.current.start_sim
        if self.latest_sim_time is not None:
            return self.latest_sim_time - self.current.start_sim
        return None

    def on_odom(self, msg):
        if self.current is None:
            return
        # Cached unconditionally (pre- and post-READY) so on_episode_ready
        # always has the latest synchronized pose available to write as the
        # t_wall=0 trajectory row -- never a fabricated (0, 0).
        self.current.latest_odom_msg = msg
        if not self.current.ready:
            return
        t_wall = time.time() - self.current.start_wall
        t_sim = self._sim_relative(stamp_to_sec(msg.header.stamp))
        yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        tilt = msg.pose.pose.orientation.y
        self.current.add_odom(t_wall, t_sim, msg.pose.pose.position.x,
                               msg.pose.pose.position.y, yaw, tilt)

    def on_scan(self, msg):
        if self.current is None or not self.current.ready:
            return
        t_wall = time.time() - self.current.start_wall
        t_sim = self._sim_relative(stamp_to_sec(msg.header.stamp))
        self.current.add_scan(t_wall, t_sim, list(msg.ranges))

    def on_cmd_vel(self, msg):
        if self.current is None or not self.current.ready:
            return
        t_wall = time.time() - self.current.start_wall
        self.current.set_cmd(msg.linear.x, msg.angular.z, t_wall)

    def on_obstacle_odom(self, msg):
        if self.current is None or not self.current.ready:
            return
        t_wall = time.time() - self.current.start_wall
        t_sim = self._sim_relative(stamp_to_sec(msg.header.stamp))
        yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        self.current.add_obstacle(t_wall, t_sim, msg.child_frame_id,
                                   msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)

    def _finalize_current(self):
        ep_dir, duration = self.current.write(self.out_dir)
        with open(self.manifest_path, "a", newline="") as f:
            csv.writer(f).writerow([
                self.current.index, ep_dir, self.current.start_wall, self.current.end_wall,
                duration, len(self.current.trajectory_rows), len(self.current.lidar_ranges),
                len(self.current.obstacle_frame_id),
            ])
        self.get_logger().info(
            f"episode {self.current.index} saved to {ep_dir} "
            f"({len(self.current.trajectory_rows)} odom rows, {len(self.current.lidar_ranges)} scans)")

    def shutdown(self):
        if self.current is not None and self.current.trajectory_rows:
            self.get_logger().info("shutdown: flushing in-progress episode before exit")
            self._finalize_current()
            self.current = None


def main():
    parser = argparse.ArgumentParser(
        description="Passive recorder for turtlebot3_drlnav TD3 evaluation episodes.")
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory (default: ~/drlnav_recordings/<timestamp>)")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(
        "/home/turtlebot3_drlnav",
        "evaluation_records",
        time.strftime("%Y%m%d-%H%M%S"))

    # Make SIGTERM behave like SIGINT so `finally` below always runs and flushes
    # the in-progress episode, whether the process is Ctrl+C'd or killed.
    signal.signal(signal.SIGTERM, signal.default_int_handler)

    rclpy.init()
    node = EpisodeRecorder(out_dir)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
