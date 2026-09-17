#!/usr/bin/env bash

set -e

SESSION="adaptivefps"
WORKSPACE="/home/turtlebot3_drlnav"
EPISODES="${EPISODES:-50}"

# ============================================================
# CONFIGURABLE EVALUATION LIST
#
# Format:
#   "fps:<value>"
#   "model:<path>"
# ============================================================

EVALS=(
    #"model:AdaptiveFPS/runs/adaptive_fps_turtlebot_15-09-09-09-34/model.pt"
    "fps:10"
    "fps:5"
    "fps:1"
    "fps:0.5"
    "fps:0.2"
    # "model:AdaptiveFPS/runs/adaptive_fps_fc_0.0/model.pt",
    # "model:AdaptiveFPS/runs/adaptive_fps_fc_0.001/model.pt"
    # "model:AdaptiveFPS/runs/adaptive_fps_turtlebot_fc_0.003_ckpts/ckpts/timestep_20480_iterations_10/ckpt_20480_iterations_10.pt"
    # "model:AdaptiveFPS/runs/adaptive_fps_turtlebot_fc_0.003_ckpts/ckpts/timestep_675840_iterations_330/ckpt_675840_iterations_330.pt"
    # "model:AdaptiveFPS/runs/adaptive_fps_turtlebot_fc_0.003_ckpts/ckpts/timestep_512000_iterations_250/ckpt_512000_iterations_250.pt"
    # "model:AdaptiveFPS/runs/adaptive_fps_fc_0.0025/model.pt"
)

setup_cmd="cd $WORKSPACE && \
source /opt/ros/foxy/setup.bash && \
source install/setup.bash && \
export TURTLEBOT3_MODEL=burger"


run_eval() {
    ENTRY="$1"

    TYPE="${ENTRY%%:*}"
    VALUE="${ENTRY#*:}"

    echo
    echo "============================================================"
    echo "Starting evaluation"
    echo "Type:     $TYPE"
    echo "Value:    $VALUE"
    echo "Episodes: $EPISODES"
    echo "============================================================"
    echo

    # Kill any previous experiment session
    tmux kill-session -t "$SESSION" 2>/dev/null || true

    # --------------------------------------------------------
    # Gazebo
    # --------------------------------------------------------
    tmux new-session -d -s "$SESSION" -n gazebo

    tmux send-keys -t "$SESSION:gazebo" \
        "$setup_cmd && ros2 launch turtlebot3_gazebo turtlebot3_drl_stage9.launch.py" C-m

    sleep 5

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------
    tmux new-window -t "$SESSION" -n environment

    tmux send-keys -t "$SESSION:environment" \
        "$setup_cmd && python3 AdaptiveFPS/wrapper/environment_gated.py --ros-args -r scan:=scan_gated" C-m

    sleep 3

    # --------------------------------------------------------
    # Evaluator
    # --------------------------------------------------------
    tmux new-window -t "$SESSION" -n evaluator

    if [[ "$TYPE" == "fps" ]]; then

        EVAL_CMD="python3 AdaptiveFPS/scripts/eval.py \
            --fps $VALUE \
            --episodes $EPISODES"

    elif [[ "$TYPE" == "model" ]]; then

        EVAL_CMD="python3 AdaptiveFPS/scripts/eval.py \
            --model $VALUE \
            --episodes $EPISODES"

    else
        echo "ERROR: Unknown evaluation type '$TYPE'"
        exit 1
    fi

    tmux send-keys -t "$SESSION:evaluator" \
        "$setup_cmd && $EVAL_CMD" C-m

    sleep 2

    # --------------------------------------------------------
    # Goals
    # --------------------------------------------------------
    tmux new-window -t "$SESSION" -n goals

    tmux send-keys -t "$SESSION:goals" \
        "$setup_cmd && ros2 run turtlebot3_drl gazebo_goals" C-m

    echo "Evaluation launched."
    echo "Waiting for evaluator to finish..."

    # --------------------------------------------------------
    # Wait until evaluator process finishes
    # --------------------------------------------------------
    while tmux list-panes -t "$SESSION:evaluator" \
        -F '#{pane_current_command}' 2>/dev/null | grep -q "python3"
    do
        sleep 10
    done

    echo
    echo "Evaluation finished:"
    echo "  $ENTRY"
    echo

    # Give ROS/Gazebo a moment to finish writing files
    sleep 3

    # Kill experiment before launching the next one
    tmux kill-session -t "$SESSION" 2>/dev/null || true

    sleep 3
}


# ============================================================
# RUN ALL CONFIGURED EXPERIMENTS
# ============================================================

echo
echo "============================================================"
echo "AdaptiveFPS batch evaluation"
echo "Episodes per configuration: $EPISODES"
echo "Configurations: ${#EVALS[@]}"
echo "============================================================"
echo

for ENTRY in "${EVALS[@]}"; do
    run_eval "$ENTRY"
done


echo
echo "============================================================"
echo "ALL EVALUATIONS COMPLETED"
echo "============================================================"