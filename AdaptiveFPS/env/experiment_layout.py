"""Experiment naming and directory layout for new AdaptiveFPS training and evaluation runs.

Training runs (train_adaptive_fps_ppo.py):

    AdaptiveFPS/runs/stage<N>/<condition>/adaptive_<condition>[_resume]_stage<N>_fc<fc>_bud<budget>_<YYYY-MM-DD_HH-MM-SS>/

    condition  "scratch" (random initialization) or "transfer" (--pretrained-path, full SimpleEnv model).
    A --resume-path run keeps the condition of the run it continues (from its stored "init" provenance) and
    adds "_resume" to its name, so it is never mistaken for a new initialization; "resumed_from" in its
    checkpoints records the lineage.

Evaluation runs (eval.py):

    AdaptiveFPS/eval/stage<N>/<group>/<run>/      plus one summary.csv at AdaptiveFPS/eval/stage<N>/

    group  "fixed" (--fps), "adaptive_scratch" / "adaptive_transfer" (Gazebo-trained, by init provenance),
           "adaptive_zeroshot" (a SimpleEnv policy evaluated directly in Gazebo) or "adaptive_unrecorded"
           (a Gazebo checkpoint written before initialization provenance was recorded).

Historical results (flat AdaptiveFPS/runs/<run>/ and eval_*/<run>/ folders) are left where they are; eval.py's
summary reads both the flat and the grouped layout.

No ROS imports, so this module can be used and tested offline.
"""
import os

import numpy as np

CONDITIONS = {"scratch": "scratch", "pretrained": "transfer"}   # checkpoint init["mode"] -> condition


def format_frame_cost(frame_cost):
    """Shortest round-trip decimal, never exponent notation: 0.035 -> "0.035", 0.0 -> "0", 1e-05 -> "0.00001"."""
    return np.format_float_positional(float(frame_cost), trim="-")


def stage_label(stage):
    return f"stage{stage}" if stage is not None else "stageunknown"


def experiment_condition(init):
    """"scratch", "transfer" or "unrecorded" from a run's init provenance."""
    return CONDITIONS.get((init or {}).get("mode"), "unrecorded")


def training_run_name(condition, resumed, stage, frame_cost, budget, now):
    return (f"adaptive_{condition}{'_resume' if resumed else ''}_{stage_label(stage)}"
            f"_fc{format_frame_cost(frame_cost)}_bud{budget}_{now.strftime('%Y-%m-%d_%H-%M-%S')}")


def training_run_dir(runs_root, init, resumed, stage, frame_cost, budget, now):
    """Unique new run directory path (not created): <runs_root>/stage<N>/<condition>/<run name>[_<k>]."""
    condition = experiment_condition(init)
    base = os.path.join(runs_root, stage_label(stage), condition,
                        training_run_name(condition, resumed, stage, frame_cost, budget, now))
    path, k = base, 2
    while os.path.exists(path):
        path, k = f"{base}_{k}", k + 1
    return path


def eval_group(source_env, init):
    """Evaluation group of an adaptive checkpoint (source_env from checkpoint_init.checkpoint_source_env)."""
    if source_env == "simpleenv":
        return "adaptive_zeroshot"
    return "adaptive_" + experiment_condition(init)


def default_eval_root(base_path, stage):
    return os.path.join(base_path, "AdaptiveFPS", "eval", stage_label(stage))
