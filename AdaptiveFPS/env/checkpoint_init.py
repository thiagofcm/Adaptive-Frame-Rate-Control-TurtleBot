"""Training initialization from checkpoints (used by train_adaptive_fps_ppo.py).

Two mutually exclusive ways to start a Gazebo training run from a checkpoint:

    --resume-path      continue an interrupted Gazebo run: model, optimizer, global_step, iteration and LSTM
                       state are all restored (validate_resume_checkpoint).
    --pretrained-path  start a NEW Gazebo run from a SimpleEnv policy: only the complete model_state_dict
                       (network + LSTM + actor + critic) is loaded, strictly; optimizer, progress, LSTM state
                       and everything else start fresh (validate_pretrained_checkpoint, load_pretrained_weights).

The source environment is identified from the checkpoint (checkpoint_source_env) so a SimpleEnv checkpoint
can't be resumed and a Gazebo checkpoint can't be used as a pretrained source. Provenance (how a run's weights
were initialized, and every resume since) is recorded in each Gazebo checkpoint's "init" / "resumed_from".

No ROS imports, so this module can be used and tested offline.
"""
import hashlib

import torch

from AdaptiveFPS.env.adaptive_obs import check_adaptive_checkpoint

GAZEBO_ENV_ID = "AdaptiveFPSTurtleBot-v0"     # gymnasium id registered by adaptive_fps_env.py
SIMPLE_ENV_ID = "TurtleBot_VarScanRate"       # args.env_id of SimpleEnv training checkpoints
PRETRAINED_COMPONENTS = ("network", "lstm", "actor", "critic")
RESUME_KEYS = ("model_state_dict", "optimizer_state_dict", "global_step", "iteration",
               "next_lstm_state_h", "next_lstm_state_c")


def checkpoint_env_id(checkpoint):
    """The checkpoint's env id: top-level (Gazebo checkpoints) or args.env_id (SimpleEnv), else None."""
    if checkpoint.get("env_id") is not None:
        return checkpoint["env_id"]
    args = checkpoint.get("args") or {}
    return args.get("env_id") if isinstance(args, dict) else None


def checkpoint_source_env(checkpoint):
    """"gazebo", "simpleenv" or "unknown".

    Gazebo checkpoints written before env_id was stored are recognized by the top-level obs_layout that
    only this trainer writes (SimpleEnv checkpoints store no obs_layout)."""
    env_id = checkpoint_env_id(checkpoint)
    if env_id == GAZEBO_ENV_ID:
        return "gazebo"
    if env_id == SIMPLE_ENV_ID:
        return "simpleenv"
    if env_id is None and "obs_layout" in checkpoint:
        return "gazebo"
    return "unknown"


def check_architecture(state_dict, agent):
    """Require the checkpoint's model_state_dict to match `agent` exactly: key set, shapes, dtypes, all finite."""
    expected = agent.state_dict()
    missing = [k for k in expected if k not in state_dict]
    unexpected = [k for k in state_dict if k not in expected]
    if missing or unexpected:
        raise ValueError(f"model_state_dict does not match the Agent architecture "
                         f"(missing {missing}, unexpected {unexpected})")
    for key, ref in expected.items():
        tensor = state_dict[key]
        if tuple(tensor.shape) != tuple(ref.shape) or tensor.dtype != ref.dtype:
            raise ValueError(f"model_state_dict['{key}'] is {tuple(tensor.shape)} {tensor.dtype}, "
                             f"the Agent expects {tuple(ref.shape)} {ref.dtype}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"model_state_dict['{key}'] contains non-finite values")


def _check_lstm_hidden_size(checkpoint, lstm_hidden_size):
    args = checkpoint.get("args") or {}
    stored = args.get("lstm_hidden_size") if isinstance(args, dict) else None
    if stored is not None and int(stored) != int(lstm_hidden_size):
        raise ValueError(f"checkpoint was trained with lstm_hidden_size={stored}, "
                         f"but --lstm-hidden-size is {lstm_hidden_size}")


def validate_pretrained_checkpoint(checkpoint, agent, env_fps_choices, env_budget, lstm_hidden_size):
    """Validate a --pretrained-path source. Raises ValueError; returns check_adaptive_checkpoint's summary."""
    source = checkpoint_source_env(checkpoint)
    if source == "gazebo":
        raise ValueError("this is a Gazebo training checkpoint, not a SimpleEnv policy -- "
                         "to continue a Gazebo run use --resume-path")
    if source != "simpleenv":
        raise ValueError(f"unknown source environment (env_id={checkpoint_env_id(checkpoint)!r}); "
                         f"--pretrained-path expects a SimpleEnv checkpoint (env_id {SIMPLE_ENV_ID!r})")
    summary = check_adaptive_checkpoint(checkpoint, env_fps_choices, env_budget)
    _check_lstm_hidden_size(checkpoint, lstm_hidden_size)
    check_architecture(checkpoint["model_state_dict"], agent)
    return summary


def validate_resume_checkpoint(checkpoint, agent, env_fps_choices, env_budget, frame_cost, num_envs):
    """Validate a --resume-path checkpoint. Raises ValueError; returns check_adaptive_checkpoint's summary."""
    source = checkpoint_source_env(checkpoint)
    if source == "simpleenv":
        raise ValueError("this is a SimpleEnv checkpoint -- it cannot be resumed in Gazebo; "
                         "to start a new Gazebo run from its weights use --pretrained-path")
    if source != "gazebo":
        raise ValueError(f"unknown source environment (env_id={checkpoint_env_id(checkpoint)!r}); "
                         f"--resume-path expects a Gazebo training checkpoint")
    summary = check_adaptive_checkpoint(checkpoint, env_fps_choices, env_budget)
    if summary["frame_cost"] != "not stored" and float(summary["frame_cost"]) != float(frame_cost):
        raise ValueError(f"checkpoint was trained with frame cost {summary['frame_cost']}, "
                         f"but --frame-cost is {frame_cost}")
    missing = [k for k in RESUME_KEYS if k not in checkpoint]
    if missing:
        raise ValueError(f"checkpoint lacks {missing} needed to resume (final model.pt files written before "
                         f"'iteration' was saved can't be resumed; use a ckpts/ checkpoint)")
    lstm_shape = (agent.lstm.num_layers, num_envs, agent.lstm.hidden_size)
    for key in ("next_lstm_state_h", "next_lstm_state_c"):
        if tuple(checkpoint[key].shape) != lstm_shape:
            raise ValueError(f"checkpoint {key} is {tuple(checkpoint[key].shape)}, expected {lstm_shape}")
    check_architecture(checkpoint["model_state_dict"], agent)
    return summary


def load_pretrained_weights(agent, checkpoint):
    """Strictly load the complete model_state_dict (network + LSTM + actor + critic) into `agent`, in place,
    so an optimizer already built over agent.parameters() stays valid and stateless. Nothing else from the
    checkpoint is read. Returns the loaded components."""
    state_dict = checkpoint["model_state_dict"]
    agent.load_state_dict(state_dict, strict=True)
    loaded = agent.state_dict()
    for key, tensor in state_dict.items():
        if not torch.equal(loaded[key].cpu(), tensor.cpu()):
            raise RuntimeError(f"pretrained load verification failed for '{key}'")
    return list(PRETRAINED_COMPONENTS)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pretrained_init_record(path, checkpoint, summary, loaded_components):
    """Provenance of a pretrained-initialized run, stored as checkpoint["init"]."""
    args = checkpoint.get("args") or {}
    return {
        "mode": "pretrained",
        "source_path": path,
        "source_sha256": file_sha256(path),
        "source_env_id": summary["env_id"],
        "source_scene": summary["scene"],
        "source_frame_cost": summary["frame_cost"],
        "source_budget": summary["budget"],
        "source_budget_stored": summary["budget_stored"],
        "source_global_step": checkpoint.get("global_step", "not stored"),
        "source_iteration": checkpoint.get("iteration", "not stored"),
        "source_lstm_hidden_size": args.get("lstm_hidden_size", "not stored"),
        "loaded_components": list(loaded_components),
    }


def resume_records(path, checkpoint):
    """(init, resumed_from) for a resumed run: the checkpoint's own initialization provenance is kept as is,
    and this resume is appended to its resume history."""
    init = dict(checkpoint.get("init") or {"mode": "not recorded"})
    resumed_from = list(checkpoint.get("resumed_from") or [])
    resumed_from.append({
        "path": path,
        "sha256": file_sha256(path),
        "global_step": checkpoint["global_step"],
        "iteration": checkpoint["iteration"],
    })
    return init, resumed_from
