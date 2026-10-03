#!/usr/bin/env python3
"""Measure the Stage 14 pre-roll: the sim time between the obstacle reset/randomization (/reset_simulation,
sim time 0) and the start of the PPO episode (AdaptiveFPSEnv.reset() returning), and how far OB1/OB2 move in it.

Run inside the container through tools/stage14/run_preroll_measurement.sh, which starts Gazebo (Stage 14, with
the obstacle event log), the environment node and gazebo_goals exactly as the training script does.
Each episode is driven to its end with a constant 10 Hz sensing action and the frozen navigation policy; no
PPO training, nothing is saved besides the CSV.
"""
import argparse
import csv
import os
import statistics
import sys
import time

sys.path.insert(0, os.environ["DRLNAV_BASE_PATH"])

from nav_msgs.msg import Odometry                              # noqa: E402
from rclpy.qos import QoSProfile                               # noqa: E402

from AdaptiveFPS.env.adaptive_fps_env import AdaptiveFPSEnv    # noqa: E402
from AdaptiveFPS.env.gazebo_bridge import stamp_to_sec         # noqa: E402

SPEED = 0.06
START_X = {0: 1.4, 1: 2.4}        # world x of s = 0 for obstacle_index 0 / 1
SIGN_X = {0: 1.0, 1: -1.0}        # world x direction of increasing s


def read_log(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="plugin event log (CORRIDOR_OBSTACLE_LOG)")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    env = AdaptiveFPSEnv()
    node = env.node
    latest = {}

    def on_obstacle(msg):
        latest[msg.child_frame_id[-1]] = (stamp_to_sec(msg.header.stamp), msg.pose.pose.position.x)

    node.create_subscription(Odometry, "obstacle/odom", on_obstacle, QoSProfile(depth=10))

    fields = ["episode", "plugin_episode", "resets_this_episode", "reset_wall_s", "preroll_clock_s",
              "preroll_robot_odom_s", "first_step_end_s"]
    for i in (1, 2):
        fields += [f"ob{i}_s0", f"ob{i}_dir0", f"ob{i}_path_m", f"ob{i}_net_displacement_m",
                   f"ob{i}_reversed_in_preroll", f"ob{i}_odom_displacement_m"]
    fields += ["steps", "outcome"]
    rows = []
    prev_plugin_episode = None
    for ep in range(1, args.episodes + 1):
        wall = time.perf_counter()
        env.reset()
        reset_wall = time.perf_counter() - wall
        t_clock = node.latest_sim_time                       # sim time when the PPO episode starts
        t_odom = env._episode_start_sim                      # robot /odom stamp AdaptiveFPSEnv anchors on
        snapshot = dict(latest)

        log = read_log(args.log)
        plugin_episode = max(int(r["episode"]) for r in log)
        row = {"episode": ep, "plugin_episode": plugin_episode,
               "resets_this_episode": (plugin_episode - prev_plugin_episode) if prev_plugin_episode is not None else "",
               "reset_wall_s": reset_wall, "preroll_clock_s": t_clock, "preroll_robot_odom_s": t_odom}
        prev_plugin_episode = plugin_episode
        for index in (0, 1):
            mine = [r for r in log if int(r["index"]) == index and int(r["episode"]) == plugin_episode]
            reset_row = next(r for r in mine if r["event"] == "reset")
            s0, s, d, t0 = float(reset_row["s"]), float(reset_row["s"]), int(reset_row["dir"]), 0.0
            reversed_ = False
            for r in mine:
                if r["event"] == "reversal" and float(r["sim_time"]) <= t_clock:
                    s, d, t0, reversed_ = float(r["s"]), int(r["dir"]), float(r["sim_time"]), True
            s_now = s + d * SPEED * (t_clock - t0)
            key = f"ob{index + 1}"
            row[f"{key}_s0"] = s0
            row[f"{key}_dir0"] = int(reset_row["dir"])
            row[f"{key}_path_m"] = SPEED * t_clock
            row[f"{key}_net_displacement_m"] = abs(s_now - s0)
            row[f"{key}_reversed_in_preroll"] = int(reversed_)
            odom = snapshot.get(str(index + 1))
            row[f"{key}_odom_displacement_m"] = (
                abs(odom[1] - (START_X[index] + SIGN_X[index] * s0)) if odom is not None else "")

        steps, info, terminated = 0, {}, False
        while not terminated and steps < 1000:
            _, _, terminated, _, info = env.step(4)          # 10 Hz sensing
            steps += 1
            if steps == 1:
                row["first_step_end_s"] = node.latest_sim_time
        row["steps"], row["outcome"] = steps, info.get("outcome_str", "")
        rows.append(row)
        print(f"episode {ep}: pre-roll {t_clock:.3f} s sim (robot odom stamp {t_odom:.3f} s), "
              f"reset() took {reset_wall:.2f} s wall, OB1/OB2 path {SPEED * t_clock * 1e3:.1f} mm, "
              f"plugin episode {plugin_episode}, outcome {row['outcome']} after {steps} steps", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    def summary(name, values, scale=1.0, unit="s"):
        print(f"{name}: mean {statistics.mean(values) * scale:.3f}, min {min(values) * scale:.3f}, "
              f"max {max(values) * scale:.3f} {unit} (n={len(values)})")

    pre = [r["preroll_clock_s"] for r in rows]
    print("\n=== pre-roll summary ===")
    summary("pre-roll sim time, all episodes", pre)
    if len(pre) > 1:
        summary("pre-roll sim time, episodes 2..N", pre[1:])
    summary("obstacle path during pre-roll (0.06 m/s x pre-roll)", [SPEED * p for p in pre], 1e3, "mm")
    summary("obstacle net displacement during pre-roll",
            [r[f"ob{i}_net_displacement_m"] for r in rows for i in (1, 2)], 1e3, "mm")
    print("episodes with a reversal during pre-roll:",
          sum(r["ob1_reversed_in_preroll"] or r["ob2_reversed_in_preroll"] for r in rows), "of", len(rows))
    print("extra (recovery) resets:", sum((r["resets_this_episode"] or 1) - 1 for r in rows))
    print("saved ->", args.out)
    env.close()


if __name__ == "__main__":
    main()
