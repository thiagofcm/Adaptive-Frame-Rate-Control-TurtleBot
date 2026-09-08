#!/usr/bin/env python3
"""Fixed-policy evaluator for AdaptiveFPSEnv.

Forces a single constant sensing-rate action on every env.step(), for
--episodes episodes, and logs episodes.csv / steps.csv / metadata.json per
rate -- purely a thin driver around AdaptiveFPSEnv. All ROS/sensing logic
(scan gate, episode-ready handshake, frozen TD3, reward) lives in the env
itself; this script never touches rclpy, /scan, /scan_gated, node.k, or
the TD3 model directly.

Gazebo, environment_gated.py, and gazebo_goals must already be running
(launched separately by the bash/tmux script) before this is started.

Usage:
    python3 AdaptiveFPS/scripts/evaluate_fixed_policy_env.py --fps 5 --episodes 20
"""
import argparse
import atexit
import csv
import glob
import json
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.environ["DRLNAV_BASE_PATH"])
from AdaptiveFPS.env.adaptive_fps_env import AdaptiveFPSEnv  # noqa: E402

sys.path.insert(0, os.path.join(os.environ["DRLNAV_BASE_PATH"], "AdaptiveFPS", "scripts"))
from eval_adaptive_fps import fmt_hz  # noqa: E402  (reused, not re-derived)

EPISODE_CSV_FIELDS = [
    "episode", "fixed_fps", "success", "outcome", "outcome_str",
    "n_steps", "episode_return", "fresh_observations", "native_scans",
    "final_goal_distance_m", "final_obs_interval",
    "fresh_observation_ratio", "native_scans_per_fresh_observation",
]

STEP_CSV_FIELDS = [
    "step", "wall_time", "sim_time", "x", "y", "goal_distance_m",
    "current_fps", "obs_interval", "frame_consumed", "frame_penalty", "frame_cost",
    "episode_frame_count", "scans_since_last_obs", "instant_reward", "cumulative_reward",
    "outcome", "outcome_str",
]

TRAJECTORY_CSV_FIELDS = ["step", "x", "y", "wall_time", "sim_time"]

SUMMARY_CSV_NAME = "summary.csv"
SUMMARY_CSV_FIELDS = [
    "run_name", "mean_fps", "n_episodes", "success_rate",
    "mean_episode_return", "std_episode_return",
    "mean_n_steps", "std_n_steps",
    "mean_final_goal_distance_m",
    "mean_fresh_observations", "std_fresh_observations",
    "mean_native_scans_per_fresh_observation",
]


def run_episode(env, action_index, requested_fps, episode_num):
    """Drives one episode with a fixed action every step. Returns
    (episode_data, step_rows); episode_data is a superset of
    EPISODE_CSV_FIELDS (extra keys feed metadata.json only)."""
    observation, reset_info = env.reset()
    initial_current_fps = reset_info["current_fps"]
    initial_obs_interval = reset_info["obs_interval"]

    step_rows = []
    cumulative_reward = 0.0
    step = 0
    terminated = truncated = False
    info = reset_info
    while not (terminated or truncated):
        observation, reward, terminated, truncated, info = env.step(action_index)
        cumulative_reward += reward
        step += 1
        step_rows.append({
            "step": step,
            "wall_time": info["wall_time"],
            "sim_time": info["sim_time"],
            "x": info["x"],
            "y": info["y"],
            "goal_distance_m": info["goal_distance_m"],
            "current_fps": info["current_fps"],
            "obs_interval": info["obs_interval"],
            "frame_consumed": info["frame_consumed"],
            "frame_penalty": info["frame_penalty"],
            "frame_cost": info["frame_cost"],
            "episode_frame_count": info["episode_frame_count"],
            "scans_since_last_obs": info["scans_since_last_obs"],
            "instant_reward": reward,
            "cumulative_reward": cumulative_reward,
            "outcome": info["outcome"],
            "outcome_str": info["outcome_str"],
        })

    success = info["outcome_str"] == "SUCCESS"
    native_scans = info["native_scan_count"]
    fresh_observations = info["episode_frame_count"]
    fresh_observation_ratio = (fresh_observations / native_scans) if native_scans else 0.0
    native_scans_per_fresh_observation = (native_scans / fresh_observations) if fresh_observations else 0.0

    episode_data = {
        "episode": episode_num,
        "fixed_fps": requested_fps,
        "success": success,
        "outcome": info["outcome"],
        "outcome_str": info["outcome_str"],
        "n_steps": step,
        "episode_return": cumulative_reward,
        "fresh_observations": fresh_observations,
        "native_scans": native_scans,
        "final_goal_distance_m": info["goal_distance_m"],
        "final_obs_interval": info["obs_interval"],
        "fresh_observation_ratio": fresh_observation_ratio,
        "native_scans_per_fresh_observation": native_scans_per_fresh_observation,
        # metadata.json-only fields, not written to episodes.csv
        "initial_current_fps": initial_current_fps,
        "initial_obs_interval": initial_obs_interval,
        "final_current_fps": info["current_fps"],
    }
    return episode_data, step_rows


def build_metadata(env, args, action_index, episode_data):
    return {
        "episode": episode_data["episode"],
        "policy_type": "fixed",
        "requested_fixed_fps": args.fps,
        "fixed_action_index": action_index,
        "environment_fps_choices": env.fps_choices,
        "initial_current_fps": episode_data["initial_current_fps"],
        "initial_obs_interval": episode_data["initial_obs_interval"],
        "final_current_fps": episode_data["final_current_fps"],
        "final_obs_interval": episode_data["final_obs_interval"],
        "n_steps": episode_data["n_steps"],
        "episode_return": episode_data["episode_return"],
        "fresh_observations": episode_data["fresh_observations"],
        "native_scans": episode_data["native_scans"],
        "outcome": episode_data["outcome"],
        "outcome_str": episode_data["outcome_str"],
        "success": episode_data["success"],
    }


def summarize_results(eval_root):
    """Read every */episodes.csv under AdaptiveFPS/eval/ -- fixed_*Hz/ today,
    any other evaluated-config subdirectory later -- and combine them into
    one comparison table: one row per run (run_name = the directory
    containing that run's episodes.csv), with mean/std stats computed
    across every evaluated episode for that run.

    Registered to run on interpreter exit (atexit) so it always reflects
    the latest accumulated results, whether the run finished normally or
    was interrupted -- same pattern as evaluate_adaptive_fps.py's own
    summarize_results()/summary.csv for the F1TENTH/Lunar Lander studies.
    """
    episode_csv_paths = sorted(glob.glob(os.path.join(eval_root, "*", "episodes.csv")))
    if not episode_csv_paths:
        return

    summary_rows = []
    for path in episode_csv_paths:
        with open(path, "r", newline="") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            continue

        run_name = os.path.basename(os.path.dirname(path))

        fixed_fps = np.array([float(row["fixed_fps"]) for row in rows])
        successes = np.array([row["success"] == "True" for row in rows])
        returns = np.array([float(row["episode_return"]) for row in rows])
        n_steps = np.array([float(row["n_steps"]) for row in rows])
        final_goal_distance_m = np.array([float(row["final_goal_distance_m"]) for row in rows])
        fresh_observations = np.array([float(row["fresh_observations"]) for row in rows])
        native_per_fresh = np.array([float(row["native_scans_per_fresh_observation"]) for row in rows])

        summary_rows.append({
            "run_name": run_name,
            "mean_fps": float(fixed_fps.mean()),
            "n_episodes": len(rows),
            "success_rate": float(successes.mean() * 100),
            "mean_episode_return": float(returns.mean()),
            "std_episode_return": float(returns.std()),
            "mean_n_steps": float(n_steps.mean()),
            "std_n_steps": float(n_steps.std()),
            "mean_final_goal_distance_m": float(final_goal_distance_m.mean()),
            "mean_fresh_observations": float(fresh_observations.mean()),
            "std_fresh_observations": float(fresh_observations.std()),
            "mean_native_scans_per_fresh_observation": float(native_per_fresh.mean()),
        })

    summary_rows.sort(key=lambda row: row["mean_fps"])

    summary_path = os.path.join(eval_root, SUMMARY_CSV_NAME)
    with open(summary_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("_________________________________________________________")
    print(f"Run summary (from all */episodes.csv found under {eval_root}):")
    for row in summary_rows:
        print(
            f"  {row['run_name']:<20} mean_fps={row['mean_fps']:>6.2f}  n_episodes={row['n_episodes']:>3}  "
            f"success_rate={row['success_rate']:6.2f}%  "
            f"return mean={row['mean_episode_return']:.4f} std={row['std_episode_return']:.4f}  "
            f"fresh_obs mean={row['mean_fresh_observations']:.2f} std={row['std_fresh_observations']:.2f}")
    print(f"Summary written to: {summary_path}")


def mean_std(values):
    if not values:
        return 0.0, 0.0
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return m, s


def print_summary(episode_rows):
    n = len(episode_rows)
    successes = sum(1 for r in episode_rows if r["success"])
    success_rate = (successes / n) if n else 0.0
    ret_m, ret_s = mean_std([r["episode_return"] for r in episode_rows])
    fresh_m, fresh_s = mean_std([r["fresh_observations"] for r in episode_rows])
    npf_m, npf_s = mean_std([r["native_scans_per_fresh_observation"] for r in episode_rows])

    print()
    print(f"Success rate: {success_rate:.2%} ({successes}/{n})")
    print(f"Episode return: {ret_m:.2f} +/- {ret_s:.2f}")
    print(f"Fresh observations: {fresh_m:.2f} +/- {fresh_s:.2f}")
    print(f"Native scans per fresh observation: {npf_m:.2f} +/- {npf_s:.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fps", type=float, required=True, help="constant sensing rate to force every step")
    parser.add_argument("--episodes", type=int, default=20, help="number of episodes to evaluate")
    args = parser.parse_args()

    env = AdaptiveFPSEnv()
    try:
        if args.fps not in env.fps_choices:
            raise SystemExit(f"--fps must be one of {env.fps_choices}, got {args.fps}")
        action_index = env.fps_choices.index(args.fps)

        eval_root = os.path.join(os.environ["DRLNAV_BASE_PATH"], "AdaptiveFPS", "eval")
        rate_dir = os.path.join(eval_root, f"fixed_{fmt_hz(args.fps)}Hz")
        os.makedirs(rate_dir, exist_ok=True)

        atexit.register(summarize_results, eval_root)

        print("AdaptiveFPSEnv fixed-policy evaluation")
        print(f"  requested fps: {args.fps}")
        print(f"  action index:  {action_index} (fps_choices={env.fps_choices})")
        print(f"  episodes:      {args.episodes}")
        print(f"  output dir:    {rate_dir}")

        episodes_csv_path = os.path.join(rate_dir, "episodes.csv")
        write_header = not os.path.exists(episodes_csv_path)
        episode_rows = []

        with open(episodes_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EPISODE_CSV_FIELDS, extrasaction="ignore")
            if write_header:
                writer.writeheader()

            for ep in range(1, args.episodes + 1):
                episode_data, step_rows = run_episode(env, action_index, args.fps, ep)

                episode_dir = os.path.join(rate_dir, f"episode_{ep:04d}")
                os.makedirs(episode_dir, exist_ok=True)

                with open(os.path.join(episode_dir, "steps.csv"), "w", newline="") as sf:
                    swriter = csv.DictWriter(sf, fieldnames=STEP_CSV_FIELDS)
                    swriter.writeheader()
                    swriter.writerows(step_rows)

                # Same per-step data already collected above -- no separate
                # ROS recorder/subscriber, just a projection to its own file.
                with open(os.path.join(episode_dir, "trajectory.csv"), "w", newline="") as tf:
                    twriter = csv.DictWriter(tf, fieldnames=TRAJECTORY_CSV_FIELDS)
                    twriter.writeheader()
                    twriter.writerows({k: row[k] for k in TRAJECTORY_CSV_FIELDS} for row in step_rows)

                metadata = build_metadata(env, args, action_index, episode_data)
                with open(os.path.join(episode_dir, "metadata.json"), "w") as mf:
                    json.dump(metadata, mf, indent=2)

                writer.writerow(episode_data)
                f.flush()
                episode_rows.append(episode_data)

                print(
                    f"Episode {ep}/{args.episodes}: {episode_data['outcome_str']} | "
                    f"steps={episode_data['n_steps']} | return={episode_data['episode_return']:.2f} | "
                    f"fresh={episode_data['fresh_observations']} | native={episode_data['native_scans']}")

        print_summary(episode_rows)
    finally:
        env.close()


if __name__ == "__main__":
    main()
