#!/usr/bin/env bash

set -e

SESSION="adaptivefps_stage14"
WORKSPACE="/home/turtlebot3_drlnav"

TOTAL_TIMESTEPS="${1:-1000000}"
NUM_STEPS="${2:-2048}"
# One seed for the PPO trainer and for the randomized OB1/OB2 of Stage 14 (corridor_random_obstacle plugin).
# OBSTACLE_LOG, if set, is the CSV the plugin appends one row per obstacle reset/reversal to.
SEED="${3:-1}"
FRAME_COST="${4:-0.03}"   # reward penalty per frame-consuming PPO step
BUDGET="${5:-450}"         # frame budget: frame_ratio = episode_scan_count / BUDGET (a policy input)
OBSTACLE_LOG="${OBSTACLE_LOG:-}"
# RESUME_PATH, if set, is a checkpoint (ckpts/ checkpoint or final model.pt, absolute or relative to
# $WORKSPACE) to continue training from. TOTAL_TIMESTEPS stays the overall target, not additional steps:
#   RESUME_PATH=AdaptiveFPS/runs/<run>/model.pt ./run_adaptive_fps_train_stage14.sh 2000000 2048 1 0.03 450
# NUM_STEPS, FRAME_COST and BUDGET must match the checkpoint (the trainer rejects a mismatch).
RESUME_PATH="${RESUME_PATH:-}"
# ros2 launch rejects an empty "obstacle_log:=", so the argument is only passed when a log is requested.
LAUNCH_ARGS="obstacle_seed:=$SEED"
if [ -n "$OBSTACLE_LOG" ]; then
    LAUNCH_ARGS="$LAUNCH_ARGS obstacle_log:=$OBSTACLE_LOG"
fi

TRAINER_ARGS="--total-timesteps $TOTAL_TIMESTEPS --num-steps $NUM_STEPS --seed $SEED --frame-cost $FRAME_COST --budget $BUDGET"
if [ -n "$RESUME_PATH" ]; then
    case "$RESUME_PATH" in
        /*) RESUME_FILE="$RESUME_PATH" ;;
        *)  RESUME_FILE="$WORKSPACE/$RESUME_PATH" ;;
    esac
    if [ ! -f "$RESUME_FILE" ]; then
        echo "RESUME_PATH not found: $RESUME_FILE" >&2
        exit 1
    fi
    TRAINER_ARGS="$TRAINER_ARGS --resume-path $RESUME_FILE"
fi

setup_cmd="cd $WORKSPACE && source /opt/ros/foxy/setup.bash && source install/setup.bash && export TURTLEBOT3_MODEL=burger"

tmux kill-session -t "$SESSION" 2>/dev/null || true

# Gazebo
tmux new-session -d -s "$SESSION" -n gazebo
tmux send-keys -t "$SESSION:gazebo" \
"$setup_cmd && ros2 launch turtlebot3_gazebo turtlebot3_drl_stage14.launch.py $LAUNCH_ARGS" C-m

sleep 5

# Environment
tmux new-window -t "$SESSION" -n environment
tmux send-keys -t "$SESSION:environment" \
"$setup_cmd && python3 AdaptiveFPS/wrapper/environment_gated.py --ros-args -r scan:=scan_gated" C-m

sleep 3

# PPO trainer
tmux new-window -t "$SESSION" -n trainer
tmux send-keys -t "$SESSION:trainer" \
"$setup_cmd && python3 AdaptiveFPS/scripts/train_adaptive_fps_ppo.py $TRAINER_ARGS" C-m

# gazebo_goals waits internally for the required /goal_pose subscribers
# before publishing the first goal. This short delay is only for cleaner
# process startup/log ordering.
sleep 2

# Goals
tmux new-window -t "$SESSION" -n goals
tmux send-keys -t "$SESSION:goals" \
"$setup_cmd && ros2 run turtlebot3_drl gazebo_goals --ros-args -p external_reset:=true" C-m

echo
echo "AdaptiveFPS PPO training started on Stage 14 (corridor + randomized OB1/OB2 + OB3 chase)."
echo "Total timesteps: $TOTAL_TIMESTEPS"
echo "Rollout steps:   $NUM_STEPS"
echo "Seed:            $SEED"
echo "Frame cost:      $FRAME_COST"
echo "Budget:          $BUDGET"
echo "Resume from:     ${RESUME_PATH:-none (new run)}"
echo
echo "Attach with:"
echo "  tmux attach -t $SESSION"