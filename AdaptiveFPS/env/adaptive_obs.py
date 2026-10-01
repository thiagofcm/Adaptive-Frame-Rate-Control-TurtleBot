"""Canonical adaptive sensing-policy observation (shared by AdaptiveFPSEnv, eval.py and the trainer).

The sensing policy sees 43 values, the same definition as the simplified env
(adaptive_sensor_policy/envs/turtlebot_var_scan_rate.py, _get_augmented_obs):

    [0:40]  retained LiDAR scan (the held /scan_gated scan), normalised to [0, 1]
    [40]    fps_ratio     = current_fps / 10 (control/PPO rate)
    [41]    obs_age_ratio = age of the retained scan / 5 s (slowest rate's interval), clipped to [0, 1]
    [42]    frame_ratio   = episode_scan_count / ADAPTIVE_FRAME_BUDGET, clipped to [0, 1]

The frozen TD3 navigator keeps its own 44-D state; nothing here touches it. Checkpoints trained on the
former 47-D Gazebo observation (44-D TD3 state + 3 features) are legacy and are rejected, never converted.

No ROS imports, so this module can be used and tested offline.
"""

ADAPTIVE_FRAME_BUDGET = 450   # frame_ratio denominator; part of the observation definition (simplified env default)
ADAPTIVE_OBS_DIM = 43
ADAPTIVE_OBS_LAYOUT = "retained_scan[0:40], fps_ratio, obs_age_ratio, frame_ratio"
LEGACY_GAZEBO_OBS_DIM = 47


def _stored(checkpoint, key):
    """A value saved at the checkpoint's top level or in its saved training args, else None."""
    if key in checkpoint:
        return checkpoint[key]
    args = checkpoint.get("args") or {}
    return args.get(key) if isinstance(args, dict) else None


def check_adaptive_checkpoint(checkpoint, env_fps_choices, env_budget):
    """Check that an adaptive-policy checkpoint matches the canonical 43-D observation and this environment.

    Raises ValueError with an explanatory message on any mismatch. Returns a summary dict (input dim,
    env_id, scene, stored budget or "not stored") for logging and metadata.
    """
    if env_budget != ADAPTIVE_FRAME_BUDGET:
        raise ValueError(f"environment frame budget is {env_budget}, but the canonical adaptive observation "
                         f"uses ADAPTIVE_FRAME_BUDGET = {ADAPTIVE_FRAME_BUDGET}")

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None or "network.0.weight" not in state_dict:
        raise ValueError("checkpoint has no model_state_dict['network.0.weight'] -- not an adaptive-policy checkpoint")
    input_dim = int(state_dict["network.0.weight"].shape[1])
    args = checkpoint.get("args") or {}
    env_id = args.get("env_id", "unknown") if isinstance(args, dict) else "unknown"
    scene = args.get("scene", "unknown") if isinstance(args, dict) else "unknown"

    if input_dim == LEGACY_GAZEBO_OBS_DIM:
        raise ValueError(f"checkpoint expects {input_dim} inputs: a legacy 47-D Gazebo checkpoint (44-D TD3 state "
                         f"incl. goal/previous-action inputs + 3 sensing features). It is not supported by the "
                         f"canonical {ADAPTIVE_OBS_DIM}-D observation ({ADAPTIVE_OBS_LAYOUT}).")
    if input_dim != ADAPTIVE_OBS_DIM:
        raise ValueError(f"checkpoint expects {input_dim} inputs (env_id {env_id}), but the canonical adaptive "
                         f"observation has {ADAPTIVE_OBS_DIM} ({ADAPTIVE_OBS_LAYOUT})")

    stored_obs_dim = _stored(checkpoint, "obs_dim")
    if stored_obs_dim is not None and int(stored_obs_dim) != ADAPTIVE_OBS_DIM:
        raise ValueError(f"checkpoint records obs_dim={stored_obs_dim}, expected {ADAPTIVE_OBS_DIM}")

    stored_fps = _stored(checkpoint, "fps_choices")
    if stored_fps is not None and [float(f) for f in stored_fps] != [float(f) for f in env_fps_choices]:
        raise ValueError(f"checkpoint fps_choices {list(stored_fps)} differ from the environment's "
                         f"{list(env_fps_choices)} (the action index -> rate mapping would be wrong)")

    stored_budget = _stored(checkpoint, "budget")
    if stored_budget is not None and float(stored_budget) != float(env_budget):
        raise ValueError(f"checkpoint was trained with frame budget {stored_budget}, but this environment "
                         f"normalises frame_ratio by {env_budget}")

    return {
        "input_dim": input_dim,
        "env_id": env_id,
        "scene": scene,
        "budget": stored_budget if stored_budget is not None else "not stored",
        "fps_choices": list(stored_fps) if stored_fps is not None else "not stored",
    }
