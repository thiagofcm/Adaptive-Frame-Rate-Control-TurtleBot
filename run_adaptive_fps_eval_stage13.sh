#!/usr/bin/env bash
# AdaptiveFPS evaluation on Stage 13 (corridor_dynamic_chase: corridor + OB1/OB2 ping-pong + OB3 chase).
#
# Usage:
#   ./run_adaptive_fps_eval_stage13.sh <model.pt> [episodes]
#   ./run_adaptive_fps_eval_stage13.sh AdaptiveFPS/runs/adaptive_fps_fc_0.005/model.pt 3
#
# <model.pt> is an adaptive sensing-policy checkpoint on the canonical 43-D observation
# (retained scan + fps_ratio, obs_age_ratio, frame_ratio with budget 450; AdaptiveFPS/env/adaptive_obs.py),
# e.g. Adaptive_Policies_SimpleEnv/models/corridor_dynamic_chase/<run>/model.pt or a new Gazebo run.
# Legacy 47-D checkpoints (AdaptiveFPS/runs/* trained before this change) are rejected by eval.py.
# Absolute or relative to $WORKSPACE.
# Results go to $EVAL_ROOT/<group>/<run dir>_<checkpoint name>/ with group adaptive_scratch, adaptive_transfer,
# adaptive_zeroshot (SimpleEnv policy) or adaptive_unrecorded, next to the Stage 13 fixed-rate baselines in
# $EVAL_ROOT/fixed/ so $EVAL_ROOT/summary.csv compares them (eval.py prints the group). Historical results remain
# in AdaptiveFPS/eval_stage13*/ (flat layout). Override per run, e.g.
#   EVAL_ROOT=/home/turtlebot3_drlnav/AdaptiveFPS/eval_stage13_other ./run_adaptive_fps_eval_stage13.sh <model.pt> 3
# Frame cost / budget (defaults 0.005 / 450; BUDGET must match the checkpoint's training budget), e.g.
#   FRAME_COST=0.035 BUDGET=450 ./run_adaptive_fps_eval_stage13.sh <model.pt> 3

set -e

SESSION="eval_adaptive_fps_stage13"
WORKSPACE="/home/turtlebot3_drlnav"

if [ -z "$1" ]; then
    echo "usage: $0 <model.pt> [episodes]" >&2
    exit 1
fi
MODEL="$1"
EPISODES="${2:-3}"
EVAL_ROOT="${EVAL_ROOT:-$WORKSPACE/AdaptiveFPS/eval/stage13}"
FRAME_COST="${FRAME_COST:-0.005}"
BUDGET="${BUDGET:-450}"

case "$MODEL" in
    /*) MODEL_PATH="$MODEL" ;;
    *)  MODEL_PATH="$WORKSPACE/$MODEL" ;;
esac
if [ ! -f "$MODEL_PATH" ]; then
    echo "model not found: $MODEL_PATH" >&2
    exit 1
fi

setup_cmd="cd $WORKSPACE && source /opt/ros/foxy/setup.bash && source install/setup.bash && export TURTLEBOT3_MODEL=burger"

tmux kill-session -t "$SESSION" 2>/dev/null || true

# Gazebo
tmux new-session -d -s "$SESSION" -n gazebo
tmux send-keys -t "$SESSION:gazebo" \
"$setup_cmd && ros2 launch turtlebot3_gazebo turtlebot3_drl_stage13.launch.py" C-m

sleep 5

# Environment
tmux new-window -t "$SESSION" -n environment
tmux send-keys -t "$SESSION:environment" \
"$setup_cmd && python3 AdaptiveFPS/wrapper/environment_gated.py --ros-args -r scan:=scan_gated" C-m

sleep 3

# Evaluator
tmux new-window -t "$SESSION" -n evaluator

tmux send-keys -t "$SESSION:evaluator" \
  "$setup_cmd && python3 AdaptiveFPS/scripts/eval.py \
  --model $MODEL_PATH \
  --episodes $EPISODES \
  --eval-root $EVAL_ROOT \
  --frame-cost $FRAME_COST \
  --budget $BUDGET \
  --diagnose-probs" C-m


# gazebo_goals waits internally (drl_gazebo.py's
# _wait_for_initial_goal_pose_subscribers()) for the required /goal_pose
# subscribers before publishing episode 1's one-shot goal, so this is just a
# short pause for tidy process startup/log ordering.
sleep 2

# Goals
tmux new-window -t "$SESSION" -n goals
tmux send-keys -t "$SESSION:goals" \
"$setup_cmd && ros2 run turtlebot3_drl gazebo_goals --ros-args -p external_reset:=true" C-m

echo
echo "AdaptiveFPS Stage 13 experiment started."
echo "MODEL:    $MODEL_PATH"
echo "Episodes: $EPISODES"
echo "Output:   $EVAL_ROOT/<group>/$(basename "$(dirname "$MODEL_PATH")")_$(basename "$MODEL_PATH" .pt)  (group printed by eval.py)"
echo
echo "Attach with:"
echo "  tmux attach -t $SESSION"
