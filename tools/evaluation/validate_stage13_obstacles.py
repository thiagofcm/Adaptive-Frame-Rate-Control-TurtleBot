#!/usr/bin/env python3
"""Check the Stage 13 ping-pong obstacles (OB1, OB2) against the simplified env's definition.

Listens to /obstacle/odom for --duration seconds of sim time and, per obstacle, compares every
received position with the ping_pong formula of TurtleBot_VarScanRate._obstacle_positions()
evaluated at the message's sim-time stamp. Reports the position error, the measured speed on
each leg, and the observed reversal times. With --reset it then calls /reset_simulation and
checks that both obstacles are back at their t = 0 state.

Run with Stage 13 launched and unpaused, and nothing else driving the episode:
    ros2 launch turtlebot3_gazebo turtlebot3_drl_stage13.launch.py pause:=false
    python3 tools/evaluation/validate_stage13_obstacles.py --duration 40 --reset
"""
import argparse
import math

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_srvs.srv import Empty

# CORRIDOR_DYNAMIC_CHASE_OBSTACLES[:2] in adaptive_sensor_policy/envs/turtlebot_var_scan_rate.py,
# keyed by the index DRLEnvironment.obstacle_odom_callback derives from the child frame id.
EXPECTED = {
    0: dict(name="OB1", start=(1.4, 2.0), end=(2.4, 2.0), speed=0.06),
    1: dict(name="OB2", start=(2.4, 1.4), end=(1.4, 1.4), speed=0.06),
}
POS_TOL_M = 0.01
SPEED_TOL = 0.005


def expected_pos(spec, t):
    a, b = np.asarray(spec["start"]), np.asarray(spec["end"])
    leg = float(np.hypot(*(b - a))) / spec["speed"]
    ph = t % (2.0 * leg)
    frac = (ph if ph <= leg else 2.0 * leg - ph) / leg
    return a + (b - a) * frac, leg


class Checker(Node):
    def __init__(self):
        super().__init__("validate_stage13_obstacles")
        self.samples = {}      # index -> list of (t_sim, x, y)
        self.frames = {}       # index -> child_frame_id
        self.create_subscription(Odometry, "obstacle/odom", self.on_odom, QoSProfile(depth=50))
        self.reset_client = self.create_client(Empty, "reset_simulation")

    def on_odom(self, msg):
        idx = int(msg.child_frame_id[-1]) - 1   # same mapping as DRLEnvironment
        self.frames[idx] = msg.child_frame_id
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        self.samples.setdefault(idx, []).append((t, p.x, p.y))

    def latest_t(self):
        return max((s[-1][0] for s in self.samples.values() if s), default=None)

    def spin_until(self, cond, timeout_wall=120.0):
        import time
        deadline = time.monotonic() + timeout_wall
        while rclpy.ok() and not cond() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return cond()


def report_motion(node, t_from):
    ok = True
    for idx, spec in EXPECTED.items():
        s = np.array([r for r in node.samples.get(idx, []) if r[0] >= t_from])
        if len(s) < 2:
            print(f"{spec['name']}: NO DATA on /obstacle/odom (index {idx})")
            ok = False
            continue
        t, x, y = s[:, 0], s[:, 1], s[:, 2]
        exp = np.array([expected_pos(spec, ti)[0] for ti in t])
        err = np.hypot(x - exp[:, 0], y - exp[:, 1])
        _, leg = expected_pos(spec, 0.0)
        axis = np.asarray(spec["end"]) - np.asarray(spec["start"])
        axis /= np.linalg.norm(axis)
        along = (s[:, 1:3] - np.asarray(spec["start"])) @ axis
        v = np.diff(along) / np.maximum(np.diff(t), 1e-9)
        moving = np.abs(v) > 1e-4
        sign = np.sign(v[moving])
        t_mid = ((t[1:] + t[:-1]) / 2)[moving]
        reversals = t_mid[1:][np.diff(sign) != 0]
        out_speed = v[moving & (v > 0)] if np.any(moving & (v > 0)) else np.array([np.nan])
        back_speed = -v[moving & (v < 0)] if np.any(moving & (v < 0)) else np.array([np.nan])
        speeds_ok = all(np.isnan(sp) or abs(sp - spec["speed"]) < SPEED_TOL
                        for sp in (np.median(out_speed), np.median(back_speed)))
        this_ok = err.max() < POS_TOL_M and speeds_ok
        ok &= this_ok
        print(f"{spec['name']} (index {idx}, child_frame_id '{node.frames.get(idx)}'): {len(s)} samples, "
              f"t_sim {t[0]:.2f}..{t[-1]:.2f} s")
        print(f"  first sample ({x[0]:.3f}, {y[0]:.3f}) at t={t[0]:.2f}; expected "
              f"({exp[0, 0]:.3f}, {exp[0, 1]:.3f}); start={spec['start']} end={spec['end']}")
        print(f"  position error vs formula: max {err.max() * 1000:.2f} mm, mean {err.mean() * 1000:.2f} mm")
        print(f"  median speed start->end {np.median(out_speed):.4f} m/s, end->start {np.median(back_speed):.4f} m/s "
              f"(expected {spec['speed']})")
        exp_rev = [k * leg for k in range(1, int(t[-1] / leg) + 1) if k * leg > t[0]]
        print(f"  reversals observed at t={np.round(reversals, 2).tolist()} s, expected {np.round(exp_rev, 2).tolist()} s")
        print(f"  x range {x.min():.3f}..{x.max():.3f}, y range {y.min():.3f}..{y.max():.3f}  -> "
              f"{'OK' if this_ok else 'CHECK'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration", type=float, default=40.0, help="sim seconds to record (one full cycle is ~33.3 s)")
    ap.add_argument("--reset", action="store_true", help="afterwards call /reset_simulation and check the t=0 state")
    args = ap.parse_args()

    rclpy.init()
    node = Checker()
    print("waiting for /obstacle/odom from both obstacles (is Gazebo unpaused?) ...")
    if not node.spin_until(lambda: all(node.samples.get(i) for i in EXPECTED), timeout_wall=30.0):
        print(f"only received indices {sorted(node.samples)} -- expected {sorted(EXPECTED)}")
        rclpy.shutdown()
        raise SystemExit(1)
    t0 = node.latest_t()
    print(f"recording from t_sim={t0:.2f} s for {args.duration:.0f} s of sim time ...")
    node.spin_until(lambda: node.latest_t() >= t0 + args.duration, timeout_wall=args.duration * 5 + 30)
    ok = report_motion(node, t0)

    if args.reset:
        print("\ncalling /reset_simulation ...")
        node.reset_client.wait_for_service(timeout_sec=10.0)
        before = {i: len(s) for i, s in node.samples.items()}
        fut = node.reset_client.call_async(Empty.Request())
        node.spin_until(fut.done, timeout_wall=10.0)
        # first messages stamped after the sim-time jump back towards 0
        got = lambda: all(any(r[0] < 1.0 for r in node.samples[i][before[i]:]) for i in EXPECTED)
        node.spin_until(got, timeout_wall=10.0)
        for idx, spec in EXPECTED.items():
            post = [r for r in node.samples[idx][before[idx]:] if r[0] < 1.0]
            if not post:
                print(f"{spec['name']}: no post-reset sample with t_sim < 1 s")
                ok = False
                continue
            t, x, y = post[0]
            exp, _ = expected_pos(spec, t)
            d_start = math.hypot(x - spec["start"][0], y - spec["start"][1])
            err = math.hypot(x - exp[0], y - exp[1])
            this_ok = err < POS_TOL_M
            ok &= this_ok
            print(f"{spec['name']}: after reset t_sim={t:.3f} s at ({x:.3f}, {y:.3f}), {d_start * 1000:.1f} mm from "
                  f"start, {err * 1000:.2f} mm from formula -> {'OK' if this_ok else 'CHECK'}")

    print("\nRESULT:", "PASS" if ok else "CHECK OUTPUT ABOVE")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
