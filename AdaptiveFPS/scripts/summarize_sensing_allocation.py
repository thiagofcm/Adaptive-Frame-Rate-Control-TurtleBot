"""
summarize_sensing_allocation.py
───────────────────────────────
Per-episode summary of where the fresh LiDAR observations of an evaluation
run were acquired, and where the episode time was spent, by ground-truth
angular obstruction level.

For every PPO step of every episode the script takes the ground-truth
Angular Obstruction (%) of plot_fps_vs_free_space.py (same LiDAR geometry,
raycasting and FREE_SPACE_RADIUS, imported, never redefined), classifies the
step as low / moderate / high obstruction, and adds up the fresh observations
actually acquired in that step and the simulation time the step lasted.

Fresh observations are counted from steps.csv's episode_frame_count, the
env's running total of scans forwarded to the policy -- not from current_fps
and not from the per-step fresh_observation flag, which stays a single True
when two scans arrive within one step. The counter starts at 1: the scan
taken at reset, before the first step. That scan is attributed to the first
step's obstruction level, so the three counts of an episode add up to its
recorded fresh_observations total.

Time is measured on steps.csv's t_sim_s, the simulation time at the END of
each step, counted from the episode origin (t = 0 at reset). Step k lasts
from the previous row's t_sim_s (0 for the first step) to its own, and that
whole interval goes to step k's obstruction level -- the same attribution as
the fresh observations acquired during the step. Steps are not all 0.1 s
long (the first and last ones in particular), so time is summed, not steps
counted; the intervals add up to the last t_sim_s, the episode duration.

Two CSVs are written into the run directory:
    per_episode_sensing_alloc.csv  one row per episode (all episodes)
    mean_sensing_alloc.csv         one row: the model (the run directory's
                                   name), its type (zero_shot / transfer, from
                                   the checkpoint provenance recorded in
                                   metadata.json), the success rate over all episodes,
                                   and the mean and sample standard deviation
                                   (<metric>_std, ddof=1) of every per-episode
                                   metric over the successful episodes only
Every successful episode weighs the same in those statistics -- the per-episode
values are averaged, counts and durations are never pooled across episodes.

Offline and read-only with respect to the recorded evaluation data. The only
files written are the two CSVs; existing ones are replaced, but only once
every episode has been summarized cleanly.

Usage:
    python AdaptiveFPS/scripts/summarize_sensing_allocation.py \
        --eval-dir AdaptiveFPS/eval_stage14_50ep_seed_1000/adaptive_zeroshot/<run>
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

# Ground-truth obstruction metric, reused as is.
from plot_fps_vs_free_space import FREE_SPACE_RADIUS, LIDAR_NUM_RAYS, load_episode

# ══════════════════════════════════════════════════════
# Defines
# ══════════════════════════════════════════════════════
OUTPUT_NAME = "per_episode_sensing_alloc.csv"
MEAN_OUTPUT_NAME = "mean_sensing_alloc.csv"

# Obstruction levels: (name, lower bound in % -- inclusive). Each level runs up
# to the next one's lower bound (exclusive); the last one includes 100 %.
OBSTRUCTION_LEVELS = [("low", 0.0), ("moderate", 25.0), ("high", 50.0)]

# Scans already counted when the first step starts: the one acquired at reset.
RESET_FRAMES = 1

METRIC_FIELDS = ([f"pct_frames_{name}" for name, _ in OBSTRUCTION_LEVELS]
                 + [f"n_frames_{name}" for name, _ in OBSTRUCTION_LEVELS]
                 + [f"pct_time_{name}" for name, _ in OBSTRUCTION_LEVELS])
CSV_FIELDS = ["episode"] + METRIC_FIELDS + ["outcome"]
MEAN_CSV_FIELDS = (["model", "model_type"]
                   + [column for field in METRIC_FIELDS for column in (field, f"{field}_std")]
                   + ["success_rate"])


# Env the simplified-environment policies are trained in, and the Gazebo one.
SIMPLE_ENV_ID = "TurtleBot_VarScanRate"
GAZEBO_ENV_ID = "AdaptiveFPSTurtleBot-v0"


# ══════════════════════════════════════════════════════
# Per-episode summary
# ══════════════════════════════════════════════════════

def model_type(meta):
    """How the evaluated checkpoint relates to Gazebo, from the provenance eval.py
    records in metadata.json:
        zero_shot  trained in the simplified env only, evaluated in Gazebo as is
        transfer   trained in Gazebo starting from a simplified-env policy
        scratch    trained in Gazebo without a pretrained start
        unknown    provenance not recorded (older checkpoints) / fixed-rate run"""
    env_id = meta.get("checkpoint_env_id")
    if env_id == SIMPLE_ENV_ID:
        return "zero_shot"
    if env_id == GAZEBO_ENV_ID:
        init = meta.get("checkpoint_init")
        if isinstance(init, dict):
            pretrained = init.get("mode") == "pretrained" and init.get("source_env_id") == SIMPLE_ENV_ID
            return "transfer" if pretrained else "scratch"
    return "unknown"


def obstruction_levels(free_fraction):
    """Index into OBSTRUCTION_LEVELS for every step. Classified on the number of
    obstructed rays (obstruction % = 100 * blocked / LIDAR_NUM_RAYS), so the
    25 % and 50 % boundaries are exact."""
    blocked_rays = LIDAR_NUM_RAYS - np.rint(free_fraction * LIDAR_NUM_RAYS).astype(int)
    lower_bounds = [lower * LIDAR_NUM_RAYS / 100.0 for _, lower in OBSTRUCTION_LEVELS]
    return np.searchsorted(lower_bounds, blocked_rays, side="right") - 1


def read_step_log(episode_dir):
    """Per PPO step: fresh observations acquired (the reset scan is added to the
    first step) and simulation time the step lasted. Also returns the outcome
    recorded on the last step."""
    with open(os.path.join(episode_dir, "steps.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    t_sim = np.array([float(r["t_sim_s"]) for r in rows])
    frame_count = np.array([int(r["episode_frame_count"]) for r in rows])
    fresh_flag = np.array([r["fresh_observation"] == "True" for r in rows])

    acquired = np.diff(frame_count, prepend=RESET_FRAMES)
    if (acquired < 0).any():
        raise ValueError("episode_frame_count decreases")
    if not np.array_equal(acquired > 0, fresh_flag):
        raise ValueError("episode_frame_count and fresh_observation disagree on which steps acquired a scan")
    acquired[0] += RESET_FRAMES

    # t_sim_s is the end of each step; the episode origin (t = 0) starts the first one.
    durations = np.diff(t_sim, prepend=0.0)
    if not np.isfinite(t_sim).all() or (durations < 0).any():
        raise ValueError("t_sim_s is missing or not increasing")
    if not t_sim[-1] > 0:
        raise ValueError("episode has no duration")
    return acquired, durations, rows[-1]["outcome_str"]


def summarize_episode(eval_dir, episode_num, episode_record):
    """One per-episode row, with unrounded values (formatted only when written)."""
    episode_dir = os.path.join(eval_dir, f"episode_{episode_num:04d}")
    with open(os.path.join(episode_dir, "metadata.json")) as f:
        meta = json.load(f)

    levels = obstruction_levels(load_episode(eval_dir, episode_num)["free_fraction"])
    acquired, durations, last_step_outcome = read_step_log(episode_dir)

    if not (len(levels) == len(acquired) == meta["n_steps"] == int(episode_record["n_steps"])):
        raise ValueError(f"step counts differ: metric {len(levels)}, steps.csv {len(acquired)}, "
                         f"metadata {meta['n_steps']}, episodes.csv {episode_record['n_steps']}")

    counts = [int(acquired[levels == i].sum()) for i in range(len(OBSTRUCTION_LEVELS))]
    total = sum(counts)
    if not (total == meta["fresh_observations"] == int(episode_record["fresh_observations"])):
        raise ValueError(f"fresh observations do not reconcile: counted {total}, "
                         f"metadata {meta['fresh_observations']}, "
                         f"episodes.csv {episode_record['fresh_observations']}")

    outcome = episode_record["outcome_str"]
    if not (outcome == meta["outcome_str"] == last_step_outcome):
        raise ValueError(f"outcomes differ: episodes.csv {outcome}, metadata {meta['outcome_str']}, "
                         f"steps.csv {last_step_outcome}")

    row = {"episode": episode_num, "outcome": outcome, "model_type": model_type(meta)}
    for (name, _), count in zip(OBSTRUCTION_LEVELS, counts):
        row[f"n_frames_{name}"] = count
        # No fresh observation at all: the shares are undefined, not 0 %.
        row[f"pct_frames_{name}"] = 100.0 * count / total if total else None
    for i, (name, _) in enumerate(OBSTRUCTION_LEVELS):
        row[f"pct_time_{name}"] = 100.0 * durations[levels == i].sum() / durations.sum()
    return row


def format_value(value):
    """CSV text of one value: integers as is, other numbers with 4 decimals, undefined as empty."""
    if value is None:
        return ""
    return str(value) if isinstance(value, (int, str)) else f"{value:.4f}"


# ══════════════════════════════════════════════════════
# Mean over the successful episodes
# ══════════════════════════════════════════════════════

def summarize_successful(rows):
    """The single aggregate row: success rate over all episodes, and the plain
    mean and sample standard deviation (ddof=1) of every per-episode metric over
    the successful ones (each episode weighs the same). The means are undefined
    with no successful episode, the standard deviations with fewer than two."""
    successful = [r for r in rows if r["outcome"] == "SUCCESS"]
    mean_row = {"success_rate": 100.0 * len(successful) / len(rows)}
    for field in METRIC_FIELDS:
        undefined = [r["episode"] for r in successful if r[field] is None]
        if undefined:
            raise ValueError(f"{field} is undefined for successful episodes {undefined}")
        values = [r[field] for r in successful]
        mean_row[field] = float(np.mean(values)) if values else None
        mean_row[f"{field}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
    return mean_row, len(successful)


# ══════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=str, required=True,
                        help="Run directory holding episode_NNNN/, episodes.csv and world_geometry.json")
    parser.add_argument("--output",   type=str, default=None,
                        help=f"Per-episode CSV path (default: <eval-dir>/{OUTPUT_NAME})")
    parser.add_argument("--mean-output", type=str, default=None,
                        help=f"Successful-episode mean CSV path (default: <eval-dir>/{MEAN_OUTPUT_NAME})")
    args = parser.parse_args()

    eval_dir = args.eval_dir.rstrip("/")
    output = args.output or os.path.join(eval_dir, OUTPUT_NAME)
    mean_output = args.mean_output or os.path.join(eval_dir, MEAN_OUTPUT_NAME)

    # episodes.csv is the evaluator's per-episode record: it defines which
    # episodes exist and holds their authoritative outcome and totals.
    with open(os.path.join(eval_dir, "episodes.csv"), newline="") as f:
        records = list(csv.DictReader(f))
    episode_records = {int(r["episode"]): r for r in records}
    episode_dirs = {int(os.path.basename(d).split("_")[1])
                    for d in glob.glob(os.path.join(eval_dir, "episode_[0-9][0-9][0-9][0-9]"))}

    problems = []
    if len(episode_records) != len(records):
        problems.append("episodes.csv lists an episode more than once")
    if set(episode_records) != episode_dirs:
        problems.append(f"episodes.csv and the episode folders differ: "
                        f"only in episodes.csv {sorted(set(episode_records) - episode_dirs)}, "
                        f"only on disk {sorted(episode_dirs - set(episode_records))}")

    rows = []
    for episode_num in sorted(set(episode_records) & episode_dirs):
        try:
            rows.append(summarize_episode(eval_dir, episode_num, episode_records[episode_num]))
        except (OSError, KeyError, ValueError) as error:
            problems.append(f"episode {episode_num}: {error}")

    model_types = sorted({row["model_type"] for row in rows})
    if len(model_types) > 1:
        problems.append(f"episodes disagree on the model type: {model_types}")

    if not problems:
        try:
            mean_row, n_successful = summarize_successful(rows)
        except ValueError as error:
            problems.append(str(error))

    # Nothing is written unless every episode was summarized cleanly.
    if problems:
        sys.exit("Summary not written:\n  " + "\n  ".join(problems))

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({field: format_value(row[field]) for field in CSV_FIELDS} for row in rows)

    with open(mean_output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MEAN_CSV_FIELDS)
        writer.writeheader()
        # Means are not whole numbers, so every value gets 4 decimals here.
        writer.writerow({"model": os.path.basename(eval_dir), "model_type": model_types[0],
                         **{field: format_value(None if mean_row[field] is None else float(mean_row[field]))
                            for field in MEAN_CSV_FIELDS if field not in ("model", "model_type")}})

    totals = [sum(r[f"n_frames_{name}"] for r in rows) for name, _ in OBSTRUCTION_LEVELS]
    print(f"{len(rows)} episodes, obstruction within {FREE_SPACE_RADIUS:g} m, {sum(totals)} fresh observations: "
          + ", ".join(f"{name} {count}" for (name, _), count in zip(OBSTRUCTION_LEVELS, totals)))
    print(f"Saved: {output}")
    print(f"{n_successful} of {len(rows)} episodes successful ({mean_row['success_rate']:g} %)"
          + ("" if n_successful else " -- the means are undefined and left empty"))
    print(f"Saved: {mean_output}")
