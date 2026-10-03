#!/usr/bin/env python3
"""Fixed-policy AND adaptive (trained recurrent-PPO) evaluator for
AdaptiveFPSEnv.

Fixed mode (--fps) forces a single constant sensing-rate action on every
env.step(), for --episodes episodes. Adaptive mode (--model) loads a
checkpoint saved by train_adaptive_fps_ppo.py and drives the env with
that policy's deterministic (argmax) action every step, maintaining/
resetting its LSTM hidden state per episode -- architecture and
checkpoint format ported to match train_adaptive_fps_ppo.py's Agent
exactly (see AgentEval below), following the same fixed-vs-adaptive
evaluator pattern as the F1TENTH/LunarLander evaluate_adaptive_fps.py
reference. Both modes log episodes.csv / steps.csv / metadata.json /
summary.csv -- purely a thin driver around AdaptiveFPSEnv. All ROS/
sensing logic (scan gate, episode-ready handshake, frozen TD3, reward)
lives in the env itself; this script never touches rclpy, /scan,
/scan_gated, node.k, or the TD3 model directly, and never modifies
AdaptiveFPSEnv or train_adaptive_fps_ppo.py.

Gazebo, environment_gated.py, and gazebo_goals must already be running
(launched separately by the bash/tmux script) before this is started.

--frame-cost and --budget configure the env (Python defaults 0.0 / 450 from adaptive_obs.py; the launch
scripts pass FRAME_COST, default 0.005). The budget scales
the frame_ratio policy input, so a --model checkpoint must have been trained with the same budget. The
frame cost only affects the reported reward; the checkpoint's training frame cost is recorded alongside.

Output layout (AdaptiveFPS/env/experiment_layout.py): <eval-root>/<group>/<run>/ with group fixed,
adaptive_scratch, adaptive_transfer (Gazebo-trained, from the checkpoint's init provenance), adaptive_zeroshot
(SimpleEnv policy) or adaptive_unrecorded (Gazebo checkpoint without provenance); default eval-root
AdaptiveFPS/eval/stage<N>. summary.csv at <eval-root> covers grouped runs and flat (historical) run folders.

Usage:
    python3 AdaptiveFPS/scripts/eval.py --fps 5 --episodes 20 --frame-cost 0.005 --budget 450
    python3 AdaptiveFPS/scripts/eval.py --model AdaptiveFPS/runs/<run>/model.pt --episodes 20
"""
import argparse
import atexit
import csv
import glob
import json
import os
import statistics
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

sys.path.insert(0, os.environ["DRLNAV_BASE_PATH"])
from AdaptiveFPS.env.adaptive_fps_env import AdaptiveFPSEnv  # noqa: E402
from AdaptiveFPS.env.adaptive_obs import (  # noqa: E402
    ADAPTIVE_FRAME_BUDGET, ADAPTIVE_FRAME_COST, ADAPTIVE_OBS_DIM, ADAPTIVE_OBS_LAYOUT, check_adaptive_checkpoint)
from AdaptiveFPS.env.checkpoint_init import checkpoint_source_env  # noqa: E402
from AdaptiveFPS.env.experiment_layout import default_eval_root, eval_group  # noqa: E402

sys.path.insert(0, os.path.join(os.environ["DRLNAV_BASE_PATH"], "tools", "episode_recorder"))
from record_episode import load_world_geometry  # noqa: E402  (reused, not re-derived --
                                                  # same SDF-parsing used by FixedFPS's
                                                  # EpisodeRecorder, for wall geometry only;
                                                  # AdaptiveFPSEnv's own obstacle/lidar
                                                  # recording is separate, see
                                                  # get_recording_arrays())

# Rate label formatting for run folders and logs.
def fmt_hz(hz):
    if abs(hz - round(hz)) < 1e-9:
        return str(int(round(hz)))
    return f"{hz:.2f}".rstrip("0").rstrip(".")


EPISODE_CSV_FIELDS = [
    "episode", "policy_type", "fixed_fps", "mean_fps", "success", "outcome", "outcome_str",
    "n_steps", "episode_return", "fresh_observations", "native_scans",
    "final_goal_distance_m", "final_obs_interval", "distance_traveled_m",
    "fresh_observation_ratio", "native_scans_per_fresh_observation",
]

# Schema matches tools/episode_recorder/record_episode.py's
# EpisodeRecorder trajectory.csv exactly (t_wall/t_sim naming, plus
# yaw/goal_x/goal_y), so make_video_fps_v2.py can consume either
# source unmodified.
TRAJECTORY_CSV_FIELDS = ["t_wall", "t_sim", "x", "y", "yaw", "goal_x", "goal_y"]

# steps.csv: one row per completed AdaptiveFPSEnv.step() call (one PPO
# step; the inner TD3/control ticks are only summarized via
# ppo_step_sim_dt). The per-run constant frame_cost is in metadata.json.
STEP_CSV_FIELDS = [
    "ppo_step", "t_wall_s", "t_sim_s", "ppo_step_sim_dt",
    "x", "y", "yaw", "goal_distance_m", "goal_distance_norm", "distance_traveled_m",
    "instant_adaptive_reward", "cumulative_adaptive_reward",
    "instant_nav_reward", "cumulative_nav_reward", "terminal_reward", "frame_penalty",
    "fresh_observation", "frame_consumed", "episode_frame_count",
    "current_fps", "obs_interval",
    "obs_age_ratio", "fps_ratio", "frame_count_ratio",
    "action_linear", "action_angular",
    "outcome", "outcome_str", "done",
]

# Adaptive mode + --diagnose-probs only: one row per causal sensing
# decision (info["frame_consumed"]=True), holding the policy's full
# action-probability distribution at that decision. prob_<fps>hz columns
# are added at write time from env.fps_choices, in that exact order.
CAUSAL_DECISIONS_CSV_BASE_FIELDS = ["step", "chosen_fps"]

SUMMARY_CSV_NAME = "summary.csv"
SUMMARY_CSV_FIELDS = [
    "run_name", "mean_fps", "n_episodes", "success_rate",
    "mean_episode_return", "std_episode_return",
    "mean_n_steps", "std_n_steps",
    "mean_final_goal_distance_m",
    "mean_distance_traveled_m", "std_distance_traveled_m",
    "mean_fresh_observations", "std_fresh_observations",
    "mean_native_scans_per_fresh_observation",
]


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class AgentEval(nn.Module):
    """Inference-only mirror of train_adaptive_fps_ppo.py's Agent --
    architecture and submodule names (network/lstm/critic/actor) copied
    verbatim so a training checkpoint's model_state_dict loads directly
    via load_state_dict(). Not imported from the trainer (same reason
    the F1TENTH/LunarLander reference keeps its own AgentEval separate
    from Agent): this evaluator must keep working even if the trainer's
    Agent class changes shape for an unrelated reason, and must never
    accidentally pull in any training-only code (optimizer, rollout
    buffers, etc.)."""

    def __init__(self, obs_dim, n_actions, lstm_hidden_size=64):
        super().__init__()

        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
        )

        self.lstm = nn.LSTM(64, lstm_hidden_size)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(lstm_hidden_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(lstm_hidden_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01),
        )

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x)

        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))

        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden.append(h)

        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def predict(self, obs, lstm_state, done, deterministic=True, return_probs=False):
        """Single-observation convenience wrapper for evaluation: adds
        the batch-of-1 dimension, runs get_states + actor head, returns
        (action:int, new_lstm_state) -- or (action, new_lstm_state, probs)
        when return_probs=True, where probs is a length-n_actions
        np.ndarray from softmax(logits), ordered exactly like the actor
        head's output (i.e. same index ordering as env.fps_choices).
        Mirrors the F1TENTH reference's AgentEval.predict() exactly, with
        return_probs added purely as an evaluation-time diagnostic --
        deterministic action selection (argmax) is unaffected either way."""
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        done_tensor = torch.tensor([float(done)], dtype=torch.float32)

        with torch.no_grad():
            hidden, lstm_state = self.get_states(obs_tensor, lstm_state, done_tensor)
            logits = self.actor(hidden)
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = Categorical(logits=logits).sample()

            if return_probs:
                probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
                return int(action.item()), lstm_state, probs

        return int(action.item()), lstm_state


def run_episode(env, episode_num, action_index=None, requested_fps=None, model=None, diagnose_probs=False,
                 episode_dir=None):
    """Drives one episode with either a fixed action every step
    (action_index/requested_fps set, model=None) or a trained recurrent
    policy's deterministic action every step (model set). Returns
    (episode_data, step_rows, causal_decision_rows); episode_data is a
    superset of EPISODE_CSV_FIELDS (extra keys feed metadata.json only).
    causal_decision_rows is only ever non-empty in adaptive mode with
    diagnose_probs=True (see CAUSAL_DECISIONS_CSV_FIELDS).

    Causal-only FPS accounting: info["frame_consumed"] (same key/
    semantics used for causal-tick masking in train_adaptive_fps_ppo.py)
    gates which ticks' current_fps gets counted into causal_fps_counts /
    mean_fps -- PPO's action on the (much more numerous) non-causal
    ticks in between is deliberately never counted, matching training's
    own mask semantics exactly. diagnose_probs reuses this exact same
    gate (info["frame_consumed"]) to decide when to print/log the
    policy's action probabilities -- deterministic action selection
    itself (argmax) is unchanged, this is a read-only diagnostic.

    If episode_dir is given, steps.csv is written there incrementally
    (one row per completed env.step() call, flushed immediately) so the
    file stays valid even if the process is interrupted mid-episode --
    unlike trajectory.csv, which is only written once the whole
    episode's rows are already collected in memory."""
    observation, reset_info = env.reset()
    initial_current_fps = reset_info["current_fps"]
    initial_obs_interval = reset_info["obs_interval"]

    lstm_state = None
    if model is not None:
        lstm_state = (
            torch.zeros(1, 1, model.lstm.hidden_size),
            torch.zeros(1, 1, model.lstm.hidden_size),
        )
    done = False

    step_rows = []
    causal_decision_rows = []
    cumulative_reward = 0.0
    step = 0
    # ppo_step / cumulative_adaptive_reward / cumulative_nav_reward: reset here, i.e. once per episode
    # (run_episode() is called once per episode), matching the requested
    # "reset at the beginning of every episode" semantics.
    ppo_step = 0
    cumulative_adaptive_reward = 0.0
    cumulative_nav_reward = 0.0
    causal_fps_counts = Counter()
    causal_fps_sum = 0.0
    causal_ticks = 0
    terminated = truncated = False
    info = reset_info

    steps_file = None
    steps_writer = None
    if episode_dir is not None:
        steps_file = open(os.path.join(episode_dir, "steps.csv"), "w", newline="")
        steps_writer = csv.DictWriter(steps_file, fieldnames=STEP_CSV_FIELDS)
        steps_writer.writeheader()
        steps_file.flush()

    try:
        while not (terminated or truncated):
            probs = None
            if model is not None:
                if diagnose_probs:
                    action, lstm_state, probs = model.predict(
                        observation, lstm_state, done, deterministic=True, return_probs=True)
                else:
                    action, lstm_state = model.predict(observation, lstm_state, done, deterministic=True)
            else:
                action = action_index

            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            cumulative_reward += reward
            step += 1

            if info["frame_consumed"]:
                causal_fps_counts[info["current_fps"]] += 1
                causal_fps_sum += info["current_fps"]
                causal_ticks += 1

                if diagnose_probs and probs is not None:
                    # env.fps_choices is the authoritative action-index -> FPS
                    # mapping (env.step() does fps_choices[int(action)]
                    # directly) -- the actor head's logits/probs share that
                    # exact same index ordering, so probs[i] <-> fps_choices[i]
                    # always, never assumed/hardcoded here.
                    prob_str = "  ".join(
                        f"{fmt_hz(fps)}={probs[i]:.2f}" for i, fps in enumerate(env.fps_choices))
                    print(f"[causal decision] step={step}")
                    print(f"  chosen={fmt_hz(info['current_fps'])} Hz")
                    print(f"  probs: {prob_str}")

                    row = {"step": step, "chosen_fps": info["current_fps"]}
                    for i, fps in enumerate(env.fps_choices):
                        row[f"prob_{fmt_hz(fps)}hz"] = float(probs[i])
                    causal_decision_rows.append(row)

            step_rows.append({
                "step": step,
                "wall_time": info["wall_time"],
                "sim_time": info["sim_time"],
                "x": info["x"],
                "y": info["y"],
                "yaw": info["yaw"],
                "goal_x": info["goal_x"],
                "goal_y": info["goal_y"],
                "goal_distance_m": info["goal_distance_m"],
                "current_fps": info["current_fps"],
                "obs_interval": info["obs_interval"],
                "frame_consumed": info["frame_consumed"],
                "frame_penalty": info["frame_penalty"],
                "frame_cost": info["frame_cost"],
                "episode_frame_count": info["episode_scan_count"],
                "scans_since_last_obs": info["scans_since_last_obs"],
                "instant_reward": reward,
                "cumulative_reward": cumulative_reward,
                "outcome": info["outcome"],
                "outcome_str": info["outcome_str"],
            })

            if steps_writer is not None:
                # One row per completed AdaptiveFPSEnv.step() call -- i.e.
                # one row per outer PPO transition, never per inner TD3/
                # control tick (that granularity is only ever visible
                # inside AdaptiveFPSEnv.step()'s own inner loop, and is
                # summarized here only via ppo_step_sim_dt).
                # Every field below is either an existing info[...] value
                # used directly, or one of the counters (ppo_step and the two
                # cumulative rewards) this function already owns --
                # no value is recomputed from anything else.
                ppo_step += 1
                cumulative_adaptive_reward += reward
                cumulative_nav_reward += info["nav_reward"]
                navigation_action = info["navigation_action"]
                steps_writer.writerow({
                    "ppo_step": ppo_step,
                    "t_wall_s": info["wall_time"],
                    "t_sim_s": info["sim_time"],
                    "ppo_step_sim_dt": info["ppo_step_sim_dt"],
                    "x": info["x"],
                    "y": info["y"],
                    "yaw": info["yaw"],
                    "goal_distance_m": info["goal_distance_m"],
                    "goal_distance_norm": info["goal_distance_norm"],
                    "distance_traveled_m": info["distance_traveled_m"],
                    "instant_adaptive_reward": reward,
                    "cumulative_adaptive_reward": cumulative_adaptive_reward,
                    "instant_nav_reward": info["nav_reward"],
                    "cumulative_nav_reward": cumulative_nav_reward,
                    "terminal_reward": info["terminal_reward"],
                    "frame_penalty": info["frame_penalty"],
                    "fresh_observation": info["frame_consumed"],
                    "frame_consumed": info["frame_consumed"],
                    "episode_frame_count": info["episode_scan_count"],
                    "current_fps": info["current_fps"],
                    "obs_interval": info["obs_interval"],
                    "obs_age_ratio": info["obs_age_ratio"],
                    "fps_ratio": info["fps_ratio"],
                    "frame_count_ratio": info["episode_scan_count_ratio"],
                    "action_linear": navigation_action[0],
                    "action_angular": navigation_action[1],
                    "outcome": info["outcome"],
                    "outcome_str": info["outcome_str"],
                    "done": terminated,
                })
                steps_file.flush()
    finally:
        if steps_file is not None:
            steps_file.close()

    success = info["outcome_str"] == "SUCCESS"
    native_scans = info["native_scan_count"]
    fresh_observations = info["episode_scan_count"]
    fresh_observation_ratio = (fresh_observations / native_scans) if native_scans else 0.0
    native_scans_per_fresh_observation = (native_scans / fresh_observations) if fresh_observations else 0.0
    mean_fps = (causal_fps_sum / causal_ticks) if causal_ticks else 0.0

    episode_data = {
        "episode": episode_num,
        "policy_type": "fixed" if model is None else "adaptive",
        "fixed_fps": requested_fps if requested_fps is not None else "",
        "mean_fps": mean_fps,
        "success": success,
        "outcome": info["outcome"],
        "outcome_str": info["outcome_str"],
        "n_steps": step,
        "episode_return": cumulative_reward,
        "fresh_observations": fresh_observations,
        "native_scans": native_scans,
        "final_goal_distance_m": info["goal_distance_m"],
        "final_obs_interval": info["obs_interval"],
        "distance_traveled_m": info["distance_traveled_m"],
        "fresh_observation_ratio": fresh_observation_ratio,
        "native_scans_per_fresh_observation": native_scans_per_fresh_observation,
        # metadata.json-only fields, not written to episodes.csv
        "initial_current_fps": initial_current_fps,
        "initial_obs_interval": initial_obs_interval,
        "final_current_fps": info["current_fps"],
        "causal_fps_action_counts": dict(causal_fps_counts),
        "causal_ticks": causal_ticks,
    }
    return episode_data, step_rows, causal_decision_rows


def build_metadata(env, args, action_index, episode_data, model_path=None, lstm_hidden_size=None,
                   checkpoint_summary=None):
    metadata = {
        "episode": episode_data["episode"],
        "policy_type": episode_data["policy_type"],
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
        "mean_fps": episode_data["mean_fps"],
        "causal_ticks": episode_data["causal_ticks"],
        "causal_fps_action_counts": episode_data["causal_fps_action_counts"],
        "policy_obs_dim": int(np.prod(env.observation_space.shape)),
        "policy_obs_layout": ADAPTIVE_OBS_LAYOUT,
        "frame_budget": env.budget,
        "frame_cost": env.frame_cost,
    }
    if model_path is None:
        metadata["requested_fixed_fps"] = args.fps
        metadata["fixed_action_index"] = action_index
    else:
        metadata["model_path"] = model_path
        metadata["lstm_hidden_size"] = lstm_hidden_size
        if checkpoint_summary is not None:
            metadata["checkpoint_env_id"] = checkpoint_summary["env_id"]
            metadata["checkpoint_scene"] = checkpoint_summary["scene"]
            metadata["checkpoint_budget"] = checkpoint_summary["budget"]
            metadata["checkpoint_budget_stored"] = checkpoint_summary["budget_stored"]
            metadata["checkpoint_frame_cost"] = checkpoint_summary["frame_cost"]
            metadata["checkpoint_gazebo_stage"] = checkpoint_summary["gazebo_stage"]
            metadata["checkpoint_init"] = checkpoint_summary["init"]
            metadata["checkpoint_resumed_from"] = checkpoint_summary["resumed_from"]
    return metadata


def summarize_results(eval_root):
    """Read every episodes.csv one level (historical flat layout: <eval_root>/<run>/) or two levels
    (grouped layout: <eval_root>/<group>/<run>/) below eval_root and combine them into
    one comparison table: one row per run (run_name = that run's directory relative to eval_root, i.e.
    "<run>" or "<group>/<run>"), with mean/std stats computed
    across every evaluated episode for that run.

    Registered to run on interpreter exit (atexit) so it always reflects
    the latest accumulated results, whether the run finished normally or
    was interrupted -- same pattern as evaluate_adaptive_fps.py's own
    summarize_results()/summary.csv for the F1TENTH/Lunar Lander studies.
    """
    episode_csv_paths = sorted(glob.glob(os.path.join(eval_root, "*", "episodes.csv")) +
                               glob.glob(os.path.join(eval_root, "*", "*", "episodes.csv")))
    if not episode_csv_paths:
        return

    summary_rows = []
    for path in episode_csv_paths:
        with open(path, "r", newline="") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            continue

        run_name = os.path.relpath(os.path.dirname(path), eval_root)

        # mean_fps (not fixed_fps) is used here so this works for both
        # policy types: fixed_fps is blank for adaptive-mode rows, while
        # mean_fps (causal-tick-averaged, see run_episode()) is always
        # populated and equals the constant fixed rate in fixed mode.
        # Falls back to fixed_fps for episodes.csv files written before
        # this column existed, so old runs on disk don't break this scan.
        mean_fps_col = np.array([
            float(row["mean_fps"]) if row.get("mean_fps") not in (None, "") else float(row["fixed_fps"])
            for row in rows])
        successes = np.array([row["success"] == "True" for row in rows])
        returns = np.array([float(row["episode_return"]) for row in rows])
        n_steps = np.array([float(row["n_steps"]) for row in rows])
        final_goal_distance_m = np.array([float(row["final_goal_distance_m"]) for row in rows])
        fresh_observations = np.array([float(row["fresh_observations"]) for row in rows])
        native_per_fresh = np.array([float(row["native_scans_per_fresh_observation"]) for row in rows])
        # episodes.csv files written before distance_traveled_m existed: leave the columns blank
        distance = [float(row["distance_traveled_m"]) for row in rows if row.get("distance_traveled_m") not in (None, "")]

        summary_rows.append({
            "run_name": run_name,
            "mean_fps": float(mean_fps_col.mean()),
            "n_episodes": len(rows),
            "success_rate": float(successes.mean() * 100),
            "mean_episode_return": float(returns.mean()),
            "std_episode_return": float(returns.std()),
            "mean_n_steps": float(n_steps.mean()),
            "std_n_steps": float(n_steps.std()),
            "mean_final_goal_distance_m": float(final_goal_distance_m.mean()),
            "mean_distance_traveled_m": float(np.mean(distance)) if distance else "",
            "std_distance_traveled_m": float(np.std(distance)) if distance else "",
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


def print_causal_fps_distribution(fps_choices, episode_rows):
    """Adaptive mode only: aggregate every episode's causal_fps_action_counts
    (populated in run_episode(), gated on info["frame_consumed"] -- never
    counts PPO's output on non-causal/stale ticks) into one run-level
    distribution and print it."""
    total_counts = Counter()
    total_ticks = 0
    for row in episode_rows:
        for fps, count in row.get("causal_fps_action_counts", {}).items():
            total_counts[fps] += count
            total_ticks += count

    if total_ticks == 0:
        return

    print()
    print(f"Causal sensing-decision FPS distribution ({total_ticks} causal ticks total, "
          f"i.e. frame_consumed=True -- PPO's output on non-causal ticks is not counted):")
    for fps in fps_choices:
        count = total_counts.get(fps, 0)
        pct = 100.0 * count / total_ticks
        print(f"  {fps:>5.1f} Hz: {count:>6} ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--fps", type=float, default=None,
                             help="constant sensing rate to force every step (fixed-policy mode)")
    mode_group.add_argument("--model", type=str, default=None,
                             help="path to a train_adaptive_fps_ppo.py checkpoint .pt file (adaptive mode)")
    parser.add_argument("--episodes", type=int, default=20, help="number of episodes to evaluate")
    parser.add_argument("--diagnose-probs", action="store_true",
                         help="adaptive mode only: print the policy's action-probability "
                              "distribution at each causal sensing decision (frame_consumed=True), "
                              "and save it to causal_decisions.csv per episode. Deterministic "
                              "(argmax) action selection is unaffected -- diagnostic only, off by default.")
    parser.add_argument("--eval-root", type=str, default=None,
                         help="directory that receives the fixed_*Hz/ or adaptive_*/ run folders and "
                              "summary.csv (default: $DRLNAV_BASE_PATH/AdaptiveFPS/eval/stage<N>). Use a separate "
                              "root per stage so results from different worlds never mix.")
    parser.add_argument("--frame-cost", type=float, default=ADAPTIVE_FRAME_COST,
                         help="reward penalty per PPO step that consumes a fresh scan (default: %(default)s)")
    parser.add_argument("--budget", type=int, default=ADAPTIVE_FRAME_BUDGET,
                         help="frame budget, frame_ratio = episode_scan_count / budget; must match a --model "
                              "checkpoint's training budget (default: %(default)s)")
    args = parser.parse_args()

    # Every episode starts from the same reset pose regardless of the previous outcome
    # (AdaptiveFPSEnv owns one /reset_simulation per episode; gazebo_goals runs with external_reset:=true).
    env = AdaptiveFPSEnv(frame_cost=args.frame_cost, budget=args.budget)
    try:
        # The sensing policy's observation is the canonical 43-D one (AdaptiveFPS/env/adaptive_obs.py);
        # refuse to run if the environment was edited to produce anything else.
        obs_dim = int(np.prod(env.observation_space.shape))
        if obs_dim != ADAPTIVE_OBS_DIM:
            raise SystemExit(f"AdaptiveFPSEnv provides a {obs_dim}-D observation; expected {ADAPTIVE_OBS_DIM}-D")
        print(f"Experiment settings: frame_cost={env.frame_cost} budget={env.budget}")
        eval_root = (os.path.abspath(args.eval_root) if args.eval_root is not None
                     else default_eval_root(os.environ["DRLNAV_BASE_PATH"], env.stage))

        action_index = None
        model = None
        lstm_hidden_size = None
        checkpoint_summary = None

        if args.fps is not None:
            if args.fps not in env.fps_choices:
                raise SystemExit(f"--fps must be one of {env.fps_choices}, got {args.fps}")
            action_index = env.fps_choices.index(args.fps)

            out_dir = os.path.join(eval_root, "fixed", f"fixed_{fmt_hz(args.fps)}Hz")
            print("AdaptiveFPSEnv fixed-policy evaluation")
            print(f"  requested fps: {args.fps}")
            print(f"  action index:  {action_index} (fps_choices={env.fps_choices})")
            print(f"  policy obs:    {obs_dim}-D ({ADAPTIVE_OBS_LAYOUT}), frame budget {env.budget} "
                  f"(fixed mode: no policy is loaded)")
        else:
            checkpoint = torch.load(args.model, map_location="cpu")
            try:
                checkpoint_summary = check_adaptive_checkpoint(checkpoint, env.fps_choices, env.budget)
            except ValueError as exc:
                raise SystemExit(f"cannot evaluate {args.model}:\n  {exc}")
            # Initialization provenance written by train_adaptive_fps_ppo.py (absent from SimpleEnv/older checkpoints).
            # A scene name only exists for SimpleEnv checkpoints; Gazebo checkpoints are identified by gazebo_stage.
            source_env = checkpoint_source_env(checkpoint)
            if source_env == "gazebo":
                checkpoint_summary["scene"] = "not applicable"
                checkpoint_summary["gazebo_stage"] = checkpoint.get("gazebo_stage", "not recorded")
            else:
                checkpoint_summary["gazebo_stage"] = "not applicable"
            checkpoint_summary["init"] = checkpoint.get("init", "not recorded")
            checkpoint_summary["resumed_from"] = checkpoint.get("resumed_from", "not recorded")
            n_actions = env.action_space.n
            lstm_hidden_size = checkpoint.get("args", {}).get("lstm_hidden_size", 64)

            model = AgentEval(obs_dim=obs_dim, n_actions=n_actions, lstm_hidden_size=lstm_hidden_size)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            # Both the run directory name and the checkpoint's own
            # basename are folded in -- final checkpoints are always
            # literally named "model.pt" (see train_adaptive_fps_ppo.py),
            # so the checkpoint basename alone would collide across every
            # training run and silently mix their evaluation results
            # into the same output directory.
            run_dir_name = os.path.basename(os.path.dirname(os.path.abspath(args.model)))
            checkpoint_basename = os.path.splitext(os.path.basename(args.model))[0]
            group = eval_group(source_env, checkpoint_summary["init"] if isinstance(checkpoint_summary["init"], dict) else None)
            out_dir = os.path.join(eval_root, group, f"{run_dir_name}_{checkpoint_basename}")
            print("AdaptiveFPSEnv adaptive (recurrent PPO) evaluation")
            print(f"  loaded model:     {args.model}")
            print(f"  obs_dim:          {obs_dim} ({ADAPTIVE_OBS_LAYOUT})")
            print(f"  frame budget:     {env.budget} (checkpoint: {checkpoint_summary['budget']}"
                  f"{'' if checkpoint_summary['budget_stored'] else ', not stored -> legacy default'})")
            print(f"  frame cost:       {env.frame_cost} (checkpoint trained with: {checkpoint_summary['frame_cost']})")
            trained_on = (f"gazebo_stage={checkpoint_summary['gazebo_stage']}" if source_env == "gazebo"
                          else f"scene={checkpoint_summary['scene']}")
            print(f"  trained on:       env_id={checkpoint_summary['env_id']} {trained_on}")
            print(f"  output group:     {group}")
            init = checkpoint_summary["init"]
            print(f"  initialization:   {init if not isinstance(init, dict) else init.get('mode')}"
                  f"{' from ' + str(init.get('source_path')) if isinstance(init, dict) and 'source_path' in init else ''}")
            print(f"  n_actions:        {n_actions} (fps_choices={env.fps_choices})")
            print(f"  lstm_hidden_size: {lstm_hidden_size}")
            print(f"  diagnose_probs:   {args.diagnose_probs}")

        if args.diagnose_probs and model is None:
            print("--diagnose-probs has no effect in fixed-policy mode (no policy to diagnose); ignoring.")

        os.makedirs(out_dir, exist_ok=True)
        atexit.register(summarize_results, eval_root)

        # Best-effort, once per run: same static-wall-geometry snapshot
        # FixedFPS's EpisodeRecorder writes (record_episode.py's
        # Episode._write_world_geometry) -- never raises, since
        # trajectory/lidar/obstacle recording must proceed without it.
        # Also rewritten if the cached file was recorded for a different
        # stage, so a run folder never renders another world's walls.
        world_geometry_path = os.path.join(out_dir, "world_geometry.json")
        try:
            with open("/tmp/drlnav_current_stage.txt") as f:
                current_stage = int(f.read())
        except Exception:
            current_stage = None
        cached_stage = None
        if os.path.exists(world_geometry_path):
            try:
                with open(world_geometry_path) as f:
                    cached_stage = json.load(f).get("stage")
            except Exception:
                pass
            if cached_stage != current_stage:
                print(f"  world_geometry.json is for stage {cached_stage}, current stage is "
                      f"{current_stage} -- regenerating")
        if not os.path.exists(world_geometry_path) or cached_stage != current_stage:
            try:
                stage = current_stage
                if stage is None:
                    raise RuntimeError("/tmp/drlnav_current_stage.txt missing or unreadable")
                geometry = load_world_geometry(os.environ["DRLNAV_BASE_PATH"], stage)
                with open(world_geometry_path, "w") as f:
                    json.dump(geometry, f, indent=2)
                print(
                    f"  world_geometry.json written for stage {stage} "
                    f"({len(geometry['walls'])} wall segments from {geometry['source_models']})")
            except Exception as exc:
                print(f"  could not record world geometry, continuing without it: {exc}")

        print(f"  episodes:      {args.episodes}")
        print(f"  output dir:    {out_dir}")

        episodes_csv_path = os.path.join(out_dir, "episodes.csv")
        write_header = not os.path.exists(episodes_csv_path)
        episode_rows = []

        with open(episodes_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EPISODE_CSV_FIELDS, extrasaction="ignore")
            if write_header:
                writer.writeheader()

            for ep in range(1, args.episodes + 1):
                # Created before run_episode() so steps.csv can be opened
                # and written incrementally *during* the episode, not just
                # assembled in memory and written at the end like
                # trajectory.csv is.
                episode_dir = os.path.join(out_dir, f"episode_{ep:04d}")
                os.makedirs(episode_dir, exist_ok=True)

                episode_data, step_rows, causal_decision_rows = run_episode(
                    env, ep, action_index=action_index, requested_fps=args.fps, model=model,
                    diagnose_probs=args.diagnose_probs, episode_dir=episode_dir)

                # Same per-step data already collected above -- no separate
                # ROS recorder/subscriber, just a projection to its own file.
                # Explicit key rename (wall_time/sim_time -> t_wall/t_sim)
                # to match record_episode.py's EpisodeRecorder schema, which
                # make_video_fps_v2.py expects.
                with open(os.path.join(episode_dir, "trajectory.csv"), "w", newline="") as tf:
                    twriter = csv.DictWriter(tf, fieldnames=TRAJECTORY_CSV_FIELDS)
                    twriter.writeheader()
                    twriter.writerows({
                        "t_wall": row["wall_time"],
                        "t_sim": row["sim_time"],
                        "x": row["x"],
                        "y": row["y"],
                        "yaw": row["yaw"],
                        "goal_x": row["goal_x"],
                        "goal_y": row["goal_y"],
                    } for row in step_rows)

                # env-side lidar/obstacle recording, matching
                # record_episode.py's EpisodeRecorder npz schema exactly
                # (see AdaptiveFPSEnv.get_recording_arrays()) so this file
                # can be consumed the same way FixedFPS's lidar.npz/
                # obstacles.npz already are.
                lidar_arrays, obstacle_arrays = env.get_recording_arrays()
                np.savez_compressed(os.path.join(episode_dir, "lidar.npz"), **lidar_arrays)
                np.savez_compressed(os.path.join(episode_dir, "obstacles.npz"), **obstacle_arrays)

                if causal_decision_rows:
                    causal_fields = CAUSAL_DECISIONS_CSV_BASE_FIELDS + [
                        f"prob_{fmt_hz(fps)}hz" for fps in env.fps_choices]
                    with open(os.path.join(episode_dir, "causal_decisions.csv"), "w", newline="") as cf:
                        cwriter = csv.DictWriter(cf, fieldnames=causal_fields)
                        cwriter.writeheader()
                        cwriter.writerows(causal_decision_rows)

                metadata = build_metadata(
                    env, args, action_index, episode_data,
                    model_path=args.model, lstm_hidden_size=lstm_hidden_size,
                    checkpoint_summary=checkpoint_summary)
                with open(os.path.join(episode_dir, "metadata.json"), "w") as mf:
                    json.dump(metadata, mf, indent=2)

                writer.writerow(episode_data)
                f.flush()
                episode_rows.append(episode_data)

                print(
                    f"Episode {ep}/{args.episodes}: {episode_data['outcome_str']} | "
                    f"steps={episode_data['n_steps']} | return={episode_data['episode_return']:.2f} | "
                    f"mean_fps={episode_data['mean_fps']:.2f} | "
                    f"fresh={episode_data['fresh_observations']} | native={episode_data['native_scans']}")

        print_summary(episode_rows)
        if model is not None:
            print_causal_fps_distribution(env.fps_choices, episode_rows)
    finally:
        env.close()


if __name__ == "__main__":
    main()
