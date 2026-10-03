#!/usr/bin/env python3
"""Validation of the Stage 14 randomized OB1/OB2 in Gazebo (run inside the container, see
tools/stage14/run_gazebo_validation.sh which launches Gazebo and calls this script).

    python3 tools/stage14/validate_gazebo.py run --log <event log csv> [--resets 150] [--long-episodes 4]
    python3 tools/stage14/validate_gazebo.py compare <same-seed A.csv> <same-seed B.csv> <other-seed C.csv>

`run` needs only gzserver with the Stage 14 world (no environment / goals / trainer node; the robot stays at
its start). It drives /reset_simulation, /pause_physics and /unpause_physics itself and compares every
/obstacle/odom sample (50 Hz, sim-time stamped) with the trajectory rebuilt from the plugin's event log
(reset: s0, direction; reversal: time, position), so it checks start positions, direction, speed, reversal
locations and reset behaviour against what the plugin says it sampled.
"""
import argparse
import csv
import math
import sys
import time

L, SPEED = 1.0, 0.06
SEGMENTS = {0: ((1.4, 2.0), (2.4, 2.0)), 1: ((2.4, 1.4), (1.4, 1.4))}   # obstacle_index -> (start, end)
OB3_START = (-1.0, -1.0)
POS_TOL = 5e-4      # m; p3d samples the pose within a physics tick or two (0.06 mm each) of its stamp
VEL_TOL = 1e-3      # m/s
FAILED = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""), flush=True)
    if not ok:
        FAILED.append(name)


def ks_uniform(x, lo, hi):
    u = sorted((v - lo) / (hi - lo) for v in x)
    n = len(u)
    d = max(max((k + 1) / n - v for k, v in enumerate(u)), max(v - k / n for k, v in enumerate(u)))
    return d, 1.63 / math.sqrt(n)


def read_log(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("index", "seed", "episode", "dir"):
            r[k] = int(r[k])
        for k in ("sim_time", "s", "x", "y", "target"):
            r[k] = float(r[k])
    return rows


def episodes_of(rows):
    """{(index, episode): {"reset": row, "reversals": [rows]}}"""
    eps = {}
    for r in rows:
        e = eps.setdefault((r["index"], r["episode"]), {"reversals": []})
        if r["event"] == "reset":
            e["reset"] = r
        else:
            e["reversals"].append(r)
    return eps


def model(ep, t):
    """(s, dir) at sim time t from the logged reset and reversals."""
    t0, s, d = 0.0, ep["reset"]["s"], ep["reset"]["dir"]
    for r in ep["reversals"]:
        if r["sim_time"] > t:
            break
        t0, s, d = r["sim_time"], r["s"], r["dir"]
    return s + d * SPEED * (t - t0), d


def xy(index, s):
    a, b = SEGMENTS[index]
    return a[0] + (b[0] - a[0]) * s / L, a[1] + (b[1] - a[1]) * s / L


def run(args):
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.qos import QoSProfile
    from rosgraph_msgs.msg import Clock
    from std_srvs.srv import Empty

    rclpy.init()
    node = rclpy.create_node("stage14_obstacle_validation")
    reset_cli = node.create_client(Empty, "/reset_simulation")
    pause_cli = node.create_client(Empty, "/pause_physics")
    unpause_cli = node.create_client(Empty, "/unpause_physics")
    state = {"clock": None, "odom": []}

    def on_clock(msg):
        state["clock"] = msg.clock.sec + msg.clock.nanosec * 1e-9

    def on_odom(msg):
        state["odom"].append((msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, msg.child_frame_id,
                              msg.pose.pose.position.x, msg.pose.pose.position.y,
                              msg.twist.twist.linear.x, msg.twist.twist.linear.y))

    node.create_subscription(Clock, "/clock", on_clock, QoSProfile(depth=10))
    node.create_subscription(Odometry, "obstacle/odom", on_odom, QoSProfile(depth=1000))

    def call(client):
        while not client.wait_for_service(timeout_sec=1.0):
            node.get_logger().info(f"waiting for {client.srv_name}")
        future = client.call_async(Empty.Request())
        while not future.done():
            rclpy.spin_once(node, timeout_sec=0.05)

    def spin(seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.02)

    def collect(dwell):
        """Unpause, spin until sim time (restarted from 0 by the reset) reaches dwell, pause.
        Returns this episode's odom samples: everything from the first sample stamped < 0.1 s on."""
        state["odom"].clear()
        state["clock"] = None
        call(unpause_cli)
        armed, deadline = False, time.monotonic() + dwell * 5 + 30
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
            t = state["clock"]
            if t is None:
                continue
            armed = armed or t < min(dwell, 0.2)
            if armed and t >= dwell:
                break
        call(pause_cli)
        spin(0.3)
        samples = list(state["odom"])
        first = next((k for k, s in enumerate(samples) if s[0] < 0.1), len(samples))
        return samples[first:]

    n_log = [len(read_log(args.log))]
    stats = {"pos": 0.0, "vel": 0.0, "n": 0, "ob3": 0.0, "bad_episode": 0, "first": 0.0, "y": 0.0}
    s0 = {0: [], 1: []}
    d0 = {0: [], 1: []}
    reversals = []
    last_episode = [max(r["episode"] for r in read_log(args.log))]

    def evaluate(samples, expected_step=1, label=""):
        rows = read_log(args.log)
        new, n_log[0] = rows[n_log[0]:], len(rows)
        eps = episodes_of(new)
        episode = last_episode[0] + expected_step
        last_episode[0] = episode
        seen_first = set()
        for i in (0, 1):
            ep = eps.get((i, episode))
            if ep is None or "reset" not in ep:
                stats["bad_episode"] += 1
                print(f"  {label}: no reset row for obstacle {i} episode {episode}; rows: "
                      f"{sorted(eps)}", flush=True)
                continue
            s0[i].append(ep["reset"]["s"])
            d0[i].append(ep["reset"]["dir"])
            reversals.extend(ep["reversals"])
            for t, frame, x, y, vx, vy in samples:
                if not frame.endswith(str(i + 1)):
                    continue
                s, d = model(ep, t)
                ex, ey = xy(i, s)
                err = math.hypot(x - ex, y - ey)
                stats["pos"] = max(stats["pos"], err)
                stats["y"] = max(stats["y"], abs(y - SEGMENTS[i][0][1]))
                if i not in seen_first:
                    seen_first.add(i)
                    stats["first"] = max(stats["first"], err)
                if all(abs(t - r["sim_time"]) > 0.005 for r in ep["reversals"]):
                    sign = 1.0 if SEGMENTS[i][1][0] > SEGMENTS[i][0][0] else -1.0
                    stats["vel"] = max(stats["vel"], abs(vx - sign * d * SPEED), abs(vy))
                stats["n"] += 1
        for t, frame, x, y, vx, vy in samples:
            if frame.endswith("3"):
                stats["ob3"] = max(stats["ob3"], math.hypot(x - OB3_START[0], y - OB3_START[1]))

    # Phase 1: many short episodes -> start positions / directions.
    call(pause_cli)
    spin(0.5)
    for k in range(args.resets):
        call(reset_cli)
        evaluate(collect(args.dwell), label=f"short {k}")
    # Phase 2: long episodes -> reversals.
    for k in range(args.long_episodes):
        call(reset_cli)
        evaluate(collect(args.long_dwell), label=f"long {k}")
    # Phase 3: reset edge cases. (a) two resets in a row while paused; (b) reset while running, mid-leg.
    call(reset_cli)
    call(reset_cli)
    evaluate(collect(args.dwell), expected_step=2, label="double reset")
    call(unpause_cli)
    spin(1.0)
    call(reset_cli)                      # physics keeps running through this reset
    evaluate(collect(args.dwell), label="reset while running")

    n = len(s0[0])
    check("every reset produced exactly one new episode per obstacle (episode counter == reset count)",
          stats["bad_episode"] == 0, f"{n} episodes, last episode index {last_episode[0]}")
    check("every /obstacle/odom sample of OB1/OB2 matches the logged trajectory (start, direction, speed, "
          "reversal points)", stats["n"] > 0 and stats["pos"] < POS_TOL,
          f"{stats['n']} samples, max position error {stats['pos'] * 1e3:.3f} mm (tol {POS_TOL * 1e3} mm)")
    check("first sample after each reset is at the newly sampled start (no jump, no stale state)",
          stats["first"] < POS_TOL, f"max {stats['first'] * 1e3:.3f} mm")
    check("odometry twist == +-0.06 m/s along the segment with the logged direction",
          stats["vel"] < VEL_TOL, f"max velocity error {stats['vel']:.2e} m/s")
    check("OB1/OB2 stay on their segment line (y constant)", stats["y"] < POS_TOL, f"max {stats['y'] * 1e3:.3f} mm")
    check("OB3 stays at its start, untriggered, through every reset", stats["ob3"] < 2e-3,
          f"max distance from (-1, -1): {stats['ob3'] * 1e3:.3f} mm")
    for i in (0, 1):
        D, crit = ks_uniform(s0[i], 0.0, L)
        frac = sum(d == 1 for d in d0[i]) / len(d0[i])
        lim = 3.0 * 0.5 / math.sqrt(len(d0[i]))
        check(f"OB{i + 1} start positions ~ U(0, L) over the full segment", D < crit
              and min(s0[i]) < 0.05 and max(s0[i]) > 0.95,
              f"n={len(s0[i])}, KS {D:.4f} < {crit:.4f}, range [{min(s0[i]):.3f}, {max(s0[i]):.3f}]")
        check(f"OB{i + 1} start direction 50/50", abs(frac - 0.5) < lim, f"P(+1) = {frac:.3f} (+-{lim:.3f})")
        check(f"OB{i + 1} new realization at every reset", len(set(s0[i])) == len(s0[i]))
    if args.long_episodes:
        inside = all((0.75 <= r["s"] <= 1.0) if r["dir"] == -1 else (0.0 <= r["s"] <= 0.25) for r in reversals)
        check("reversals happen inside the outer-25% regions and flip toward the other region",
              len(reversals) > 0 and inside
              and all((r["target"] <= 0.25) if r["dir"] == -1 else (r["target"] >= 0.75) for r in reversals),
              f"{len(reversals)} reversals observed in Gazebo, "
              f"s in [{min(r['s'] for r in reversals):.3f}, {max(r['s'] for r in reversals):.3f}]")
    node.destroy_node()
    rclpy.shutdown()


def compare(args):
    def resets(path):
        return {(r["index"], r["episode"]): (r["s"], r["dir"], r["target"]) for r in read_log(path)
                if r["event"] == "reset"}

    def reversal_lists(path):
        return {k: [(r["sim_time"], r["s"], r["dir"], r["target"]) for r in e["reversals"]]
                for k, e in episodes_of(read_log(path)).items()}

    a, b, c = resets(args.a), resets(args.b), resets(args.c)
    common = sorted(set(a) & set(b))
    check("same seed, two separate Gazebo launches: identical start position / direction / first target",
          len(common) >= 20 and all(a[k] == b[k] for k in common), f"{len(common)} (obstacle, episode) pairs")
    ra, rb = reversal_lists(args.a), reversal_lists(args.b)
    n = [min(len(ra[k]), len(rb[k])) for k in common]
    check("same seed: identical reversal times / points / targets (common prefix of each episode)",
          sum(n) > 0 and all(ra[k][:m] == rb[k][:m] for k, m in zip(common, n)), f"{sum(n)} reversals")
    common = sorted(set(a) & set(c))
    check("different seed: different realizations", len(common) >= 20 and all(a[k][0] != c[k][0] for k in common),
          f"{len(common)} pairs")
    check("OB1 and OB2 use different streams", all(a[(0, e)] != a[(1, e)] for (i, e) in a if i == 0 and (1, e) in a))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--log", required=True)
    r.add_argument("--resets", type=int, default=150)
    r.add_argument("--dwell", type=float, default=0.5)
    r.add_argument("--long-episodes", type=int, default=4)
    r.add_argument("--long-dwell", type=float, default=40.0)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("c")
    args = ap.parse_args()
    (run if args.cmd == "run" else compare)(args)
    print(f"\n{'ALL CHECKS PASSED' if not FAILED else f'{len(FAILED)} CHECK(S) FAILED: {FAILED}'}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
