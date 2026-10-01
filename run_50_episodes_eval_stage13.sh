#!/usr/bin/env bash
# Batch evaluation on Stage 13 (corridor_dynamic_chase: corridor + OB1/OB2 ping-pong + OB3 chase).
# Runs every entry of EVALS one after another, each in a fresh Gazebo instance.
#
# Usage:
#   ./run_50_episodes_eval_stage13.sh
#   EPISODES=10 ./run_50_episodes_eval_stage13.sh
#   EVAL_ROOT=/home/turtlebot3_drlnav/AdaptiveFPS/eval_stage13_other ./run_50_episodes_eval_stage13.sh
#
# Results go to $EVAL_ROOT/fixed_<fps>Hz/ and $EVAL_ROOT/adaptive_<run>_<checkpoint>/, with one
# $EVAL_ROOT/summary.csv comparing them all. Adaptive checkpoints must use the canonical 43-D
# observation (AdaptiveFPS/env/adaptive_obs.py); legacy 47-D checkpoints are rejected by eval.py.

set -e

SESSION="eval_stage13_batch"
WORKSPACE="/home/turtlebot3_drlnav"
EPISODES="${EPISODES:-2}"
EVAL_ROOT="${EVAL_ROOT:-$WORKSPACE/AdaptiveFPS/eval_stage13_50ep}"

# Every tmux session the Stage 13 scripts use. Any of them still running would attach a second
# environment/goals node to the same Gazebo world and corrupt the results.
STAGE13_SESSIONS=("$SESSION" "adaptivefps_stage13" "eval_adaptive_fps_stage13")

# ============================================================
# CONFIGURABLE EVALUATION LIST
#
# Format:
#   "fps:<value>"      fixed rate, one of 0.2 0.5 1 5 10
#   "model:<path>"     adaptive checkpoint, absolute or relative to $WORKSPACE
# ============================================================

EVALS=(
    "fps:10"
    #"fps:5"
    #"fps:1"
    #"fps:0.5"
    #"fps:0.2"
    #"model:Adaptive_Policies_SimpleEnv/models/corridor_dynamic_chase/adaptive_fps_26-09-16-13-56_fc_0.0/model.pt"
    #"model:Adaptive_Policies_SimpleEnv/models/corridor_dynamic_chase/adaptive_fps_26-09-16-13-57_fc_0.01/model.pt"
    #"model:Adaptive_Policies_SimpleEnv/models/corridor_dynamic_chase/adaptive_fps_26-09-16-13-58_fc_0.015/model.pt"
    #"model:Adaptive_Policies_SimpleEnv/models/corridor_dynamic_chase/adaptive_fps_26-09-16-14-00_fc_0.02/model.pt"
    # "model:Adaptive_Policies_SimpleEnv/models/corridor_dynamic_chase/adaptive_fps_26-09-16-14-01_fc_0.05/model.pt"
    # "model:Adaptive_Policies_SimpleEnv/models/corridor_dynamic_chase/adaptive_fps_27-09-14-05-58_fc_0.0_resume/model.pt"
    # "model:Adaptive_Policies_SimpleEnv/models/corridor_dynamic_chase/adaptive_fps_27-09-14-05-59_fc_0.01_resume/model.pt"
    #"model:Adaptive_Policies_SimpleEnv/models/corridor_dynamic_chase/adaptive_fps_27-09-14-06-00_fc_0.015_resume/model.pt"
    # "model:Adaptive_Policies_SimpleEnv/models/corridor_dynamic_chase/adaptive_fps_27-09-14-06-02_fc_0.02_resume/model.pt"
    "model:Adaptive_Policies_SimpleEnv/models/corridor_dynamic_chase/adaptive_fps_27-09-14-06-03_fc_0.05_resume/model.pt"
)

setup_cmd="cd $WORKSPACE && \
source /opt/ros/foxy/setup.bash && \
source install/setup.bash && \
export TURTLEBOT3_MODEL=burger"

STACK_PATTERN="gzserver|environment_gated.py|gazebo_goals|AdaptiveFPS/scripts/eval.py"


# Kill the Stage 13 tmux sessions and wait until no Gazebo/ROS process of the stack is left.
stop_stack() {
    for s in "${STAGE13_SESSIONS[@]}"; do
        tmux kill-session -t "$s" 2>/dev/null || true
    done
    for _ in $(seq 1 15); do
        pgrep -f "$STACK_PATTERN" >/dev/null || return 0
        sleep 1
    done
    echo "Leftover processes after killing the sessions, sending SIGTERM:"
    pgrep -af "$STACK_PATTERN" || true
    pkill -f "$STACK_PATTERN" 2>/dev/null || true
    sleep 3
    if pgrep -f "$STACK_PATTERN" >/dev/null; then
        echo "ERROR: could not stop the previous Gazebo/ROS stack:"
        pgrep -af "$STACK_PATTERN"
        exit 1
    fi
}


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
    echo "Output:   $EVAL_ROOT"
    echo "============================================================"
    echo

    if [[ "$TYPE" == "fps" ]]; then
        EVAL_CMD="python3 AdaptiveFPS/scripts/eval.py \
            --fps $VALUE \
            --episodes $EPISODES \
            --eval-root $EVAL_ROOT"
    elif [[ "$TYPE" == "model" ]]; then
        case "$VALUE" in
            /*) MODEL_PATH="$VALUE" ;;
            *)  MODEL_PATH="$WORKSPACE/$VALUE" ;;
        esac
        if [ ! -f "$MODEL_PATH" ]; then
            echo "ERROR: model not found: $MODEL_PATH -- skipping"
            return 0
        fi
        EVAL_CMD="python3 AdaptiveFPS/scripts/eval.py \
            --model $MODEL_PATH \
            --episodes $EPISODES \
            --eval-root $EVAL_ROOT \
            --diagnose-probs"
    else
        echo "ERROR: Unknown evaluation type '$TYPE'"
        exit 1
    fi

    # Fresh stack for every configuration
    stop_stack

    # --------------------------------------------------------
    # Gazebo
    # --------------------------------------------------------
    tmux new-session -d -s "$SESSION" -n gazebo
    tmux send-keys -t "$SESSION:gazebo" \
        "$setup_cmd && ros2 launch turtlebot3_gazebo turtlebot3_drl_stage13.launch.py" C-m

    sleep 5
    if ! pgrep -f gzserver >/dev/null; then
        echo "ERROR: gzserver did not start -- see 'tmux attach -t $SESSION' (gazebo window)"
        exit 1
    fi

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
    tmux send-keys -t "$SESSION:evaluator" \
        "$setup_cmd && $EVAL_CMD" C-m

    sleep 2

    # --------------------------------------------------------
    # Goals
    # --------------------------------------------------------
    tmux new-window -t "$SESSION" -n goals
    tmux send-keys -t "$SESSION:goals" \
        "$setup_cmd && ros2 run turtlebot3_drl gazebo_goals --ros-args -p external_reset:=true" C-m

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

    stop_stack
    sleep 3
}


# ============================================================
# RUN ALL CONFIGURED EXPERIMENTS
# ============================================================

echo
echo "============================================================"
echo "AdaptiveFPS Stage 13 batch evaluation"
echo "Episodes per configuration: $EPISODES"
echo "Configurations: ${#EVALS[@]}"
echo "Output root: $EVAL_ROOT"
echo "============================================================"
echo

for ENTRY in "${EVALS[@]}"; do
    run_eval "$ENTRY"
done


echo
echo "============================================================"
echo "ALL EVALUATIONS COMPLETED"
echo "Summary: $EVAL_ROOT/summary.csv"
echo "============================================================"
