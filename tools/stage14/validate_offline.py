#!/usr/bin/env python3
"""Offline validation of the Stage 14 OB1/OB2 motion logic (random_ping_pong.hh), no ROS/Gazebo.

Run on the host with the simplified environment's interpreter:

    <adaptive_sensor_policy>/.venv/bin/python tools/stage14/validate_offline.py [--simple-repo PATH]

Compiles tools/stage14/random_ping_pong_trace.cc with g++ and checks
  A. the statistics of the C++ realizations against the specification of corridor_dynamic_chase_random
     (same checks as adaptive_sensor_policy/tools/validate_random_obstacles.py, items 1-14);
  B. that the C++ kinematics equal TurtleBot_VarScanRate's when both are given the same random draws.
Exit status 1 if any check fails.
"""
import argparse
import csv
import io
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PLUGIN_DIR = (HERE.parents[1] / "src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_drl_world"
              / "obstacle_plugin")
L, SPEED, FRAC = 1.0, 0.06, 0.25
EPISODES, DURATION = 2000, 70.0
FAILED = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def ks_uniform(x, lo, hi):
    """Kolmogorov-Smirnov statistic of x against U(lo, hi) and the 1% critical value."""
    u = np.sort((np.asarray(x) - lo) / (hi - lo))
    n = len(u)
    d = max(np.max(np.arange(1, n + 1) / n - u), np.max(u - np.arange(n) / n))
    return d, 1.63 / math.sqrt(n)


def trace(binary, seed, episodes, duration, dt):
    """{(index, episode): {"reset": row, "reversals": [rows], "end": row}}; a row is (t, s, dir, target)."""
    out = subprocess.run([binary, "trace", str(seed), str(episodes), str(duration), str(dt)],
                         check=True, capture_output=True, text=True).stdout
    eps = {}
    for r in csv.DictReader(io.StringIO(out)):
        key = (int(r["index"]), int(r["episode"]))
        row = (float(r["t"]), float(r["s"]), int(r["dir"]), float(r["target"]))
        e = eps.setdefault(key, {"reversals": []})
        if r["event"] == "reversal":
            e["reversals"].append(row)
        else:
            e[r["event"]] = row
    return eps


def statistics(binary):
    fine = trace(binary, 1, EPISODES, DURATION, 0.001)     # Gazebo physics step
    coarse = trace(binary, 1, EPISODES, DURATION, 0.1)     # simplified-environment step
    for i in (0, 1):
        name = f"OB{i + 1}"
        eps = [fine[(i, e)] for e in range(EPISODES)]
        s0 = np.array([e["reset"][1] for e in eps])
        d0 = np.array([e["reset"][2] for e in eps])
        D, crit = ks_uniform(s0, 0.0, L)
        check(f"{name} initial s0 ~ U(0, L) and covers the path", D < crit and s0.min() < 0.02 and s0.max() > 0.98,
              f"KS {D:.4f} < {crit:.4f}, range [{s0.min():.4f}, {s0.max():.4f}]")
        frac = float(np.mean(d0 == 1))
        check(f"{name} initial direction ~ {{-1, +1}}", 0.46 < frac < 0.54, f"P(+1) = {frac:.3f}")
        t0 = np.array([e["reset"][3] for e in eps])
        ahead = np.where(d0 == 1, (t0 >= np.maximum(0.75, s0)) & (t0 <= L), (t0 <= np.minimum(0.25, s0)) & (t0 >= 0))
        check(f"{name} first target in the reversal region ahead of s0", bool(ahead.all()))

        ok_region = ok_target = ok_flip = ok_path = ok_time = True
        legs, rev_start, rev_end, n_rev = [], [], [], 0
        for e in eps:
            t_prev, s_prev, d_prev, target = e["reset"]
            path = 0.0
            for k, (t, s, d, new_target) in enumerate(e["reversals"]):
                n_rev += 1
                ok_target &= s == target                       # reverses exactly at the drawn target
                ok_flip &= d == -d_prev
                ok_region &= (s >= 0.75 and s <= L) if d_prev == 1 else (s >= 0.0 and s <= 0.25)
                ok_region &= (new_target <= 0.25) if d == -1 else (new_target >= 0.75)
                ok_time &= abs((t - t_prev) * SPEED - abs(s - s_prev)) < 1e-9   # constant speed on every leg
                (rev_end if d_prev == 1 else rev_start).append(s)
                if k > 0:
                    legs.append(abs(s - s_prev))
                path += abs(s - s_prev)
                t_prev, s_prev, d_prev, target = t, s, d, new_target
            t_end, s_end = e["end"][0], e["end"][1]
            path += abs(s_end - s_prev)
            ok_path &= abs(path - SPEED * t_end) < 1e-9        # no pause, no lost distance at reversals
        legs = np.array(legs)
        check(f"{name} every reversal at its target, direction flips, inside the outer-25% regions",
              ok_target and ok_flip and ok_region and n_rev > 0, f"{n_rev} reversals")
        De, ce = ks_uniform(rev_end, 0.75, 1.0)
        Ds, cs = ks_uniform(rev_start, 0.0, 0.25)
        # The first reversal of an episode is not uniform over the region (target drawn ahead of s0), hence
        # the test on later reversals only would be cleaner; the pooled sample is dominated by later ones.
        check(f"{name} reversal points spread over both regions",
              min(rev_end) < 0.76 and max(rev_end) > 0.99 and min(rev_start) < 0.01 and max(rev_start) > 0.24,
              f"end KS {De:.4f} (crit {ce:.4f}), start KS {Ds:.4f} (crit {cs:.4f})")
        check(f"{name} leg lengths between reversals within [0.5 L, L] and varying",
              legs.min() >= 0.5 - 1e-12 and legs.max() <= L + 1e-12 and legs.std() > 0.05,
              f"min {legs.min():.4f}, max {legs.max():.4f}, std {legs.std():.4f}")
        check(f"{name} constant 0.06 m/s on every leg (reversal times)", ok_time)
        check(f"{name} path length == v * T in every episode (no pause, leftover distance kept)", ok_path)

        same = all(fine[(i, e)]["reset"] == coarse[(i, e)]["reset"]
                   and len(fine[(i, e)]["reversals"]) == len(coarse[(i, e)]["reversals"])
                   and all(abs(a[0] - b[0]) < 1e-6 and abs(a[1] - b[1]) < 1e-12 and a[2:] == b[2:]
                           for a, b in zip(fine[(i, e)]["reversals"], coarse[(i, e)]["reversals"]))
                   for e in range(EPISODES))
        check(f"{name} realization independent of the step size (1 ms vs 0.1 s)", same)
        distinct = len({fine[(i, e)]["reset"][1] for e in range(EPISODES)})
        check(f"{name} new realization at every reset", distinct == EPISODES, f"{distinct} distinct s0")

    a = np.array([fine[(0, e)]["reset"][1] for e in range(EPISODES)])
    b = np.array([fine[(1, e)]["reset"][1] for e in range(EPISODES)])
    da = np.array([fine[(0, e)]["reset"][2] for e in range(EPISODES)])
    db = np.array([fine[(1, e)]["reset"][2] for e in range(EPISODES)])
    r_s, r_d = np.corrcoef(a, b)[0, 1], np.corrcoef(da, db)[0, 1]
    lim = 3.0 / math.sqrt(EPISODES)
    check("OB1 / OB2 independent (s0 and direction uncorrelated, not anti-phase)",
          abs(r_s) < lim and abs(r_d) < lim and np.abs(a + b - L).max() > 0.5,
          f"corr s0 {r_s:+.3f}, dir {r_d:+.3f} (limit {lim:.3f})")

    again = trace(binary, 1, 50, DURATION, 0.001)
    check("same seed -> identical realizations", all(again[k] == fine[k] for k in again))
    other = trace(binary, 2, 50, DURATION, 0.001)
    check("different seed -> different realizations",
          all(other[k]["reset"][1] != fine[k]["reset"][1] for k in other))


def cross_check(binary, simple_repo, seeds=50, steps=700):
    """Feed the Python environment's own draws to the C++ logic and compare s after every 0.1 s step."""
    sys.path.insert(0, str(simple_repo))
    import envs.turtlebot_var_scan_rate as tb
    env = tb.TurtleBot_VarScanRate(navigation_model_path=None, scene="corridor_dynamic_chase_random")
    worst, n_rev = 0.0, 0
    for seed in range(seeds):
        env.reset(seed=seed)
        s_py = {i: [env.rpp_s[i]] for i in env.rpp_ids}
        for k in range(steps):
            env.world_step_count = k + 1
            env.sim_time += env.dt
            env._advance_random_ping_pong()
            for i in env.rpp_ids:
                s_py[i].append(env.rpp_s[i])
        for i, log in zip(env.rpp_ids, env.obstacle_event_log["obstacles"]):
            o = env.obstacle_defs[i]
            length = log["length"]
            assert (length, o["speed"]) == (L, SPEED)
            # Recover the U[0, 1) draw behind every target from the bounds _rpp_sample_target used.
            draws = [log["s0"] / length, 0.25 if log["dir0"] > 0 else 0.75]
            s, d = log["s0"], log["dir0"]
            for target in [log["first_target"]] + [r["new_target"] for r in log["reversals"]]:
                lo, hi = (max(0.75 * length, s), length) if d > 0 else (0.0, min(0.25 * length, s))
                draws.append((target - lo) / (hi - lo) if hi > lo else 0.0)
                s, d = target, -d
            n_rev += len(log["reversals"])
            out = subprocess.run([binary, "replay", str(length), str(o["speed"]), str(env.dt), str(steps)],
                                 input="\n".join(repr(u) for u in draws) + "\n",
                                 check=True, capture_output=True, text=True).stdout
            s_cpp = np.array([float(v) for v in out.split()])
            worst = max(worst, float(np.abs(s_cpp - np.array(s_py[i])).max()))
    check("C++ kinematics == TurtleBot_VarScanRate for identical draws (s after every 0.1 s step)",
          worst < 1e-9, f"{seeds} seeds x 2 obstacles x {steps} steps, {n_rev} reversals, max |ds| = {worst:.2e} m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simple-repo", type=Path, default=HERE.parents[2] / "adaptive_sensor_policy")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        binary = str(Path(tmp) / "random_ping_pong_trace")
        subprocess.run(["g++", "-std=c++14", "-O2", "-Wall", "-Wextra", f"-I{PLUGIN_DIR}",
                        str(HERE / "random_ping_pong_trace.cc"), "-o", binary], check=True)
        statistics(binary)
        if (args.simple_repo / "envs/turtlebot_var_scan_rate.py").exists():
            cross_check(binary, args.simple_repo)
        else:
            check("simplified environment found for the cross-check", False, str(args.simple_repo))
    print(f"\n{'ALL CHECKS PASSED' if not FAILED else f'{len(FAILED)} CHECK(S) FAILED: {FAILED}'}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
