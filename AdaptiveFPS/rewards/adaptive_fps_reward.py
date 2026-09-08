# AdaptiveFPS/rewards/adaptive_fps_reward.py

def get_adaptive_reward(
    previous_distance,
    current_distance,
    initial_distance,
):
    """
    Normalized progress toward the goal.

    r_t = (d_{t-1} - d_t) / d_0

    Positive: moved toward the goal
    Zero:     no progress
    Negative: moved away from the goal
    """

    if initial_distance <= 0.0:
        return 0.0

    reward = (
        previous_distance - current_distance
    ) / initial_distance

    return float(reward)