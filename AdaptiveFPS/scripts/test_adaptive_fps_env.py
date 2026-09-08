#!/usr/bin/env python3
"""Smoke test for AdaptiveFPSEnv: forces a constant FPS action every step
for one episode and reports the native-scan delta between consecutive
complete fresh (/scan_gated) events, so each individual interval can be
compared against the expected k = max(1, round(50/fps)) -- a stronger
check than an aggregate ratio, since partial first/last intervals could
otherwise mask a real per-interval error.

This validates SENSING SEMANTICS only (gate ratio, sample-and-hold,
non-retroactive action timing) -- it is not a policy-performance
evaluation, and does not write any of episodes.csv/steps.csv/trajectory.csv
(that stays evaluate_fixed_sensing.py's / eval_adaptive_fps.py's job).

Usage:
    python3 AdaptiveFPS/scripts/test_adaptive_fps_env.py --fps 5
    python3 AdaptiveFPS/scripts/test_adaptive_fps_env.py --fps 1 --max-steps 5000
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.environ["DRLNAV_BASE_PATH"]))
from AdaptiveFPS.env.adaptive_fps_env import AdaptiveFPSEnv  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fps", type=float, required=True, help="constant sensing rate to force every step")
    parser.add_argument("--max-steps", type=int, default=3000, help="stop after this many control steps if the episode hasn't ended")
    args = parser.parse_args()

    env = AdaptiveFPSEnv()
    if args.fps not in env.fps_choices:
        env.close()
        raise SystemExit(f"--fps must be one of {env.fps_choices}, got {args.fps}")
    action = env.fps_choices.index(args.fps)
    expected_k = max(1, round(50.0 / args.fps))

    obs, info = env.reset()
    print(
        f"reset: current_fps={info['current_fps']} obs_interval={info['obs_interval']} "
        f"(expected k={expected_k}) episode_frame_count={env.episode_frame_count} (expected 1)")

    fresh_count = 0
    last_native_at_fresh = env.native_scan_count  # 0, baselined post-handshake in reset()
    deltas = []
    info = {}
    try:
        for step_idx in range(args.max_steps):
            obs, reward, terminated, truncated, info = env.step(action)
            if info["frame_consumed"]:
                fresh_count += 1
                delta = info["native_scan_count"] - last_native_at_fresh
                deltas.append(delta)
                last_native_at_fresh = info["native_scan_count"]
                print(f"fresh event {fresh_count} -> native delta = {delta} (expected k={info['obs_interval']})")
            if step_idx % 500 == 0 or terminated or truncated:
                print(
                    f"step={step_idx:<6} fresh_count={fresh_count:<5} "
                    f"native_scan_count={info['native_scan_count']:<6} "
                    f"obs_interval={info['obs_interval']} episode_frame_count={info['episode_frame_count']:<4} "
                    f"outcome={info['outcome_str']}")
            if terminated or truncated:
                break
        else:
            print(f"reached --max-steps={args.max_steps} without episode termination")

        if deltas:
            ratio = info["native_scan_count"] / fresh_count
            all_match = all(d == expected_k for d in deltas)
            print(
                f"\n{len(deltas)} complete fresh interval(s) observed, deltas={deltas}\n"
                f"all deltas == k={expected_k}? {all_match}\n"
                f"aggregate native/fresh ratio: {ratio:.2f} (for reference only -- per-interval deltas above are the stronger check)")
        else:
            print("\nno complete fresh interval observed -- episode ended too early to validate")
    finally:
        env.close()


if __name__ == "__main__":
    main()
