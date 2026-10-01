#!/usr/bin/env python3
"""Standalone check of the Stage 13 OB3 triggered_chase obstacle (no policy, no evaluator).

Moves the TurtleBot with `gz model` teleports and watches OB3 on /obstacle/odom (index 2) and the
robot on /odom, checking against CORRIDOR_DYNAMIC_OB3_CHASE in
adaptive_sensor_policy/envs/turtlebot_var_scan_rate.py:
  A  /reset_simulation -> OB3 at start (-1, -1), robot at (2.5, 2.5)
  B  robot parked at start                         -> OB3 stationary
  C  robot teleported to x = 0.50 (no crossing)    -> OB3 stationary
  D  robot teleported to x = 0.21 (not <= 0.2)     -> OB3 stationary
  E  robot teleported to x = 0.19 (crosses 0.2)    -> OB3 starts moving at ~0.1 m/s toward the robot
  F  robot teleported to (-1.0, 0.8) mid-chase     -> OB3 heading turns toward the new robot position
  G  10 s after the trigger                        -> OB3 stops where it is (~1.0 m travelled)
  then /reset_simulation again; with --cycles 2 the whole sequence repeats to show it re-triggers.
The plugin also prints TRIGGERED / chase ended / reset lines in the Gazebo terminal.

Run with Stage 13 launched and unpaused, and nothing else driving the robot:
    ros2 launch turtlebot3_gazebo turtlebot3_drl_stage13.launch.py pause:=false
    python3 tools/evaluation/validate_stage13_ob3.py
"""
import argparse
import math
import subprocess
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rosgraph_msgs.msg import Clock
from std_srvs.srv import Empty

# CORRIDOR_DYNAMIC_OB3_CHASE in adaptive_sensor_policy/envs/turtlebot_var_scan_rate.py
START = np.array([-1.0, -1.0])
SPEED = 0.1
CHASE_DURATION = 10.0
TRIGGER_X = 0.2
ROBOT_START = np.array([2.5, 2.5])
OB3_INDEX = 2  # DRLEnvironment: int(child_frame_id[-1]) - 1

STILL_TOL_M = 0.002
POS_TOL_M = 0.005


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


class Validator(Node):
    def __init__(self, robot_model):
        super().__init__("validate_stage13_ob3")
        self.robot_model = robot_model
        self.ob3 = []            # (t, x, y)
        self.robot = []          # (t, x, y)
        self.frames = set()
        self.sim_t = None
        q = QoSProfile(depth=100)
        self.create_subscription(Odometry, "obstacle/odom", self.on_obstacle, q)
        self.create_subscription(Odometry, "odom", self.on_odom, q)
        self.create_subscription(Clock, "/clock", self.on_clock, QoSProfile(depth=10))
        self.reset_client = self.create_client(Empty, "reset_simulation")
        self.ok = True

    def stale(self, t):
        # messages stamped before a /reset_simulation that arrive after it
        return self.sim_t is not None and t > self.sim_t + 0.5

    def on_obstacle(self, msg):
        self.frames.add(msg.child_frame_id)
        if int(msg.child_frame_id[-1]) - 1 == OB3_INDEX and not self.stale(stamp(msg)):
            p = msg.pose.pose.position
            self.ob3.append((stamp(msg), p.x, p.y))

    def on_odom(self, msg):
        if not self.stale(stamp(msg)):
            p = msg.pose.pose.position
            self.robot.append((stamp(msg), p.x, p.y))

    def on_clock(self, msg):
        self.sim_t = msg.clock.sec + msg.clock.nanosec * 1e-9

    # ---------------------------------------------------------------- helpers
    def spin_until(self, cond, timeout_wall):
        deadline = time.monotonic() + timeout_wall
        while rclpy.ok() and not cond() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        return cond()

    def wait_sim(self, seconds):
        t0 = self.sim_t
        self.spin_until(lambda: self.sim_t is not None and self.sim_t >= t0 + seconds, seconds * 10 + 20)

    def window(self, samples, t_from, t_to=math.inf):
        a = np.array([s for s in samples if t_from <= s[0] <= t_to])
        return a.reshape(-1, 3)

    def check(self, cond, text):
        print(f"  [{'OK' if cond else 'FAIL'}] {text}")
        self.ok &= bool(cond)

    def teleport(self, x, y, yaw=math.pi):
        cmd = ["gz", "model", "-m", self.robot_model, "-x", str(x), "-y", str(y), "-z", "0",
               "-R", "0", "-P", "0", "-Y", str(yaw)]
        t_cmd = self.sim_t
        n_before = len(self.robot)
        subprocess.run(cmd, check=True)
        # first /odom sample after the command that is at the target (the previous pose may be only 2 cm away)
        at_target = lambda: next((s for s in self.robot[n_before:] if math.hypot(s[1] - x, s[2] - y) < 0.003), None)
        arrived = self.spin_until(lambda: at_target() is not None, 10.0)
        t_arrived = at_target()[0] if arrived else float("nan")
        print(f"  robot teleported to ({x:.2f}, {y:.2f}) at t_sim~{t_cmd:.2f}s (seen on /odom at {t_arrived:.2f}s)")
        if not arrived:
            self.check(False, "robot reached the teleport target on /odom")
        return t_arrived

    def reset(self):
        self.reset_client.wait_for_service(timeout_sec=10.0)
        before = self.sim_t
        fut = self.reset_client.call_async(Empty.Request())
        self.spin_until(fut.done, 10.0)
        self.spin_until(lambda: self.sim_t is not None and self.sim_t < before, 10.0)
        # sim time restarts at 0: drop the previous cycle's samples so time windows never mix cycles
        self.ob3, self.robot = [], []
        self.wait_sim(1.5)

    def ob3_still(self, t_from, label):
        w = self.window(self.ob3, t_from)
        if len(w) < 2:
            self.check(False, f"{label}: OB3 samples received")
            return
        span = np.hypot(w[:, 1] - w[0, 1], w[:, 2] - w[0, 2]).max()
        self.check(span < STILL_TOL_M, f"{label}: OB3 stationary at ({w[-1, 1]:.3f}, {w[-1, 2]:.3f}), "
                                        f"moved {span * 1000:.2f} mm over {w[-1, 0] - w[0, 0]:.1f}s")

    def robot_at(self, t):
        r = np.array(self.robot)
        i = max(0, np.searchsorted(r[:, 0], t, side="right") - 1)
        return r[i, 1:3]

    def headings(self, t_from, t_to, stride_s=0.2):
        """Median angle between OB3's motion and the bearing to the robot, plus mean motion heading."""
        w = self.window(self.ob3, t_from, t_to)
        errs, heads, bearings = [], [], []
        j = 0
        for i in range(len(w)):
            if w[i, 0] - w[j, 0] < stride_s:
                continue
            d = w[i, 1:3] - w[j, 1:3]
            if np.hypot(*d) > 1e-4:
                b = self.robot_at(w[j, 0]) - w[j, 1:3]
                h, bh = math.atan2(d[1], d[0]), math.atan2(b[1], b[0])
                errs.append(abs(math.remainder(h - bh, 2 * math.pi)))
                heads.append(h)
                bearings.append(bh)
            j = i
        if not errs:
            return None
        return (math.degrees(float(np.median(errs))), math.degrees(float(np.median(heads))),
                math.degrees(float(np.median(bearings))))

    # ---------------------------------------------------------------- one cycle
    def cycle(self, n):
        print(f"\n=== cycle {n} ===")
        print("A: /reset_simulation")
        self.reset()
        t = self.sim_t
        ob = self.window(self.ob3, t - 0.5)
        rb = self.robot[-1]
        self.check(len(ob) and np.hypot(*(ob[-1, 1:3] - START)) < POS_TOL_M,
                   f"OB3 at start {tuple(START)}: ({ob[-1, 1]:.3f}, {ob[-1, 2]:.3f})" if len(ob) else "OB3 at start")
        self.check(np.hypot(rb[1] - ROBOT_START[0], rb[2] - ROBOT_START[1]) < 0.05,
                   f"robot at reset pose {tuple(ROBOT_START)}: ({rb[1]:.3f}, {rb[2]:.3f})")

        print("B: robot parked at its start pose for 3 s")
        t0 = self.sim_t
        self.wait_sim(3.0)
        self.ob3_still(t0, "B")

        print("C: robot x = 0.50 (no crossing)")
        t0 = self.teleport(0.50, 0.5)
        self.wait_sim(2.0)
        self.ob3_still(t0, "C")

        print(f"D: robot x = 0.21 (still > trigger_x = {TRIGGER_X})")
        t0 = self.teleport(0.21, 0.5)
        self.wait_sim(2.0)
        self.ob3_still(t0, "D")

        print("E: robot x = 0.19 (crosses trigger_x, decreasing)")
        t_cross = self.teleport(0.19, 0.5)
        self.wait_sim(4.0)
        w = self.window(self.ob3, t_cross - 1.0)
        moved = np.hypot(w[:, 1] - START[0], w[:, 2] - START[1]) > 0.001
        if not moved.any():
            self.check(False, "OB3 started moving after the crossing")
            return
        t_move = w[np.argmax(moved), 0]
        # robot x just before OB3 started moving
        r = np.array(self.robot)
        rx_before = r[np.searchsorted(r[:, 0], t_move) - 1, 1]
        # /odom (~30 Hz) and /obstacle/odom (50 Hz) are sampled, so allow a few tens of ms either way
        self.check(-0.05 <= t_move - t_cross < 0.1,
                   f"OB3 started moving at t={t_move:.3f}s, robot first seen at x<=0.2 at t={t_cross:.3f}s "
                   f"(robot x just before: {rx_before:.3f})")
        seg = self.window(self.ob3, t_move + 0.2, self.sim_t)
        path = np.hypot(np.diff(seg[:, 1]), np.diff(seg[:, 2])).sum()
        speed = path / (seg[-1, 0] - seg[0, 0])
        self.check(abs(speed - SPEED) < 0.005, f"chase speed {speed:.4f} m/s (expected {SPEED})")
        h = self.headings(t_move + 0.2, self.sim_t)
        self.check(h and h[0] < 3.0, f"OB3 heading {h[1]:.1f} deg vs bearing to robot {h[2]:.1f} deg "
                                     f"(median error {h[0]:.2f} deg)" if h else "OB3 heading measurable")

        print("F: robot moved to (-1.0, 0.8) during the chase")
        t_f = self.teleport(-1.0, 0.8)
        self.wait_sim(max(0.5, t_move + CHASE_DURATION + 3.0 - self.sim_t))
        h2 = self.headings(t_f + 0.5, min(t_f + 3.0, t_move + CHASE_DURATION - 0.2))
        self.check(h2 and h2[0] < 3.0 and h and abs(math.remainder(math.radians(h2[1] - h[1]), 2 * math.pi)) > 0.3,
                   f"OB3 heading changed {h[1]:.1f} -> {h2[1]:.1f} deg, bearing to new robot pose "
                   f"{h2[2]:.1f} deg (median error {h2[0]:.2f} deg)" if (h and h2) else "OB3 heading after F")

        print(f"G: chase stops {CHASE_DURATION} s after the trigger")
        w = self.window(self.ob3, t_move - 0.1)
        step = np.hypot(np.diff(w[:, 1]), np.diff(w[:, 2]))
        moving_idx = np.nonzero(step > 1e-5)[0]
        t_stop = w[moving_idx[-1] + 1, 0]
        total = step.sum()
        self.check(abs((t_stop - t_move) - CHASE_DURATION) < 0.1,
                   f"moved for {t_stop - t_move:.2f}s (t={t_move:.2f}..{t_stop:.2f}), expected {CHASE_DURATION}s")
        self.check(abs(total - SPEED * CHASE_DURATION) < 0.02,
                   f"total path {total:.3f} m (expected {SPEED * CHASE_DURATION:.3f} m)")
        self.ob3_still(t_stop + 0.1, "after the chase")

    def run(self, cycles):
        print("waiting for /clock, /odom and /obstacle/odom (is Gazebo unpaused?) ...")
        if not self.spin_until(lambda: self.sim_t is not None and self.robot and self.ob3, 30.0):
            print("no data -- is Stage 13 running and unpaused?")
            return False
        self.wait_sim(1.0)
        print(f"/obstacle/odom child_frame_ids: {sorted(self.frames)}")
        self.check(any(f.endswith("link_obstacle3") for f in self.frames), "link_obstacle3 on /obstacle/odom")
        for n in range(1, cycles + 1):
            self.cycle(n)
        print("\nfinal /reset_simulation")
        self.reset()
        ob = self.ob3[-1]
        self.check(np.hypot(ob[1] - START[0], ob[2] - START[1]) < POS_TOL_M,
                   f"OB3 back at start after the final reset: ({ob[1]:.3f}, {ob[2]:.3f})")
        return self.ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cycles", type=int, default=2, help="trigger/chase/reset cycles (2 shows re-triggering)")
    ap.add_argument("--robot-model", default="turtlebot3_burger")
    args = ap.parse_args()
    rclpy.init()
    node = Validator(args.robot_model)
    ok = node.run(args.cycles)
    print("\nRESULT:", "PASS" if ok else "CHECK OUTPUT ABOVE")
    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
