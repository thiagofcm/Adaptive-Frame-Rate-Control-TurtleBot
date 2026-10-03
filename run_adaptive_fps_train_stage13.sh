#!/usr/bin/env bash
# AdaptiveFPS PPO training on Gazebo Stage 13: scratch, SimpleEnv -> Gazebo transfer, or resume.
#
# Scratch and transfer use the same launcher and settings; only CONDITION (+ PRETRAINED_PATH) differs:
#
#   CONDITION=scratch  FRAME_COST=0.035 BUDGET=450 SEED=1 ./run_adaptive_fps_train_stage13.sh
#   CONDITION=transfer FRAME_COST=0.035 BUDGET=450 SEED=1 \
#       PRETRAINED_PATH=Adaptive_Policies_SimpleEnv/models/corridor_dynamic_chase/<run>/model.pt \
#       ./run_adaptive_fps_train_stage13.sh
#   CONDITION=resume   FRAME_COST=0.035 BUDGET=450 SEED=1 \
#       RESUME_PATH=AdaptiveFPS/runs/stage13/<scratch|transfer>/<run>/ckpts/<...>/ckpt_<...>.pt \
#       ./run_adaptive_fps_train_stage13.sh
#
# Required:  CONDITION (scratch | transfer | resume), FRAME_COST, BUDGET
#            (always explicit; for transfer use the source policy's training frame cost and budget)
#            PRETRAINED_PATH for transfer only, RESUME_PATH for resume only (absolute or relative to $WORKSPACE)
# Optional:  TOTAL_TIMESTEPS (20000000), NUM_STEPS (2048), SEED (1), CHECKPOINT_INTERVAL (10),
#            CONFIG (YAML of further PPO hyperparameters, see train_adaptive_fps_ppo.py --config),
#            ROS_DOMAIN_ID / GAZEBO_MASTER_URI (isolate from other ROS/Gazebo stacks on the same host network),
#            DRY_RUN=1 (print the pre-flight block and exit without starting anything)
#
# Runs are written to AdaptiveFPS/runs/stage13/<scratch|transfer>/<run name>/ (see the trainer's startup block);
# the full trainer output is also saved to AdaptiveFPS/runs/stage13/logs/.

set -e

SESSION="adaptivefps_train_stage13"
WORKSPACE="/home/turtlebot3_drlnav"
STAGE=13

: "${CONDITION:?set CONDITION to scratch, transfer or resume}"
: "${FRAME_COST:?set FRAME_COST explicitly (e.g. FRAME_COST=0.035)}"
: "${BUDGET:?set BUDGET explicitly (e.g. BUDGET=450)}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-500000}"
NUM_STEPS="${NUM_STEPS:-2048}"
SEED="${SEED:-1}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10}"

resolve() { case "$1" in /*) echo "$1" ;; *) echo "$WORKSPACE/$1" ;; esac; }

case "$CONDITION" in
    scratch)
        [ -z "$PRETRAINED_PATH$RESUME_PATH" ] || { echo "CONDITION=scratch takes no PRETRAINED_PATH/RESUME_PATH" >&2; exit 1; }
        CKPT_LABEL="pretrained checkpoint"; CKPT="" ;;
    transfer)
        [ -n "$PRETRAINED_PATH" ] || { echo "CONDITION=transfer requires PRETRAINED_PATH (SimpleEnv model.pt)" >&2; exit 1; }
        [ -z "$RESUME_PATH" ]     || { echo "CONDITION=transfer takes no RESUME_PATH" >&2; exit 1; }
        CKPT_LABEL="pretrained checkpoint"; CKPT="$(resolve "$PRETRAINED_PATH")" ;;
    resume)
        [ -n "$RESUME_PATH" ]     || { echo "CONDITION=resume requires RESUME_PATH (Gazebo checkpoint)" >&2; exit 1; }
        [ -z "$PRETRAINED_PATH" ] || { echo "CONDITION=resume takes no PRETRAINED_PATH" >&2; exit 1; }
        CKPT_LABEL="resume checkpoint"; CKPT="$(resolve "$RESUME_PATH")" ;;
    *)
        echo "CONDITION must be scratch, transfer or resume (got '$CONDITION')" >&2; exit 1 ;;
esac
if [ -n "$CKPT" ] && [ ! -f "$CKPT" ]; then
    echo "$CKPT_LABEL not found: $CKPT" >&2
    exit 1
fi

TRAIN_ARGS="--total-timesteps $TOTAL_TIMESTEPS --num-steps $NUM_STEPS --seed $SEED \
--checkpoint-interval $CHECKPOINT_INTERVAL --frame-cost $FRAME_COST --budget $BUDGET"
[ -n "$CONFIG" ] && TRAIN_ARGS="--config $(resolve "$CONFIG") $TRAIN_ARGS"
[ "$CONDITION" = transfer ] && TRAIN_ARGS="$TRAIN_ARGS --pretrained-path $CKPT"
[ "$CONDITION" = resume ]   && TRAIN_ARGS="$TRAIN_ARGS --resume-path $CKPT"

LOG_DIR="$WORKSPACE/AdaptiveFPS/runs/stage$STAGE/logs"
LOG="$LOG_DIR/train_${CONDITION}_stage${STAGE}_fc${FRAME_COST}_bud${BUDGET}_$(date +%Y-%m-%d_%H-%M-%S).log"

rule="========================================================================"
echo "$rule"
echo "AdaptiveFPS Stage $STAGE training -- pre-flight"
echo "$rule"
echo "initialization mode   : $CONDITION"
if [ -n "$CKPT" ]; then
    echo "$CKPT_LABEL : $CKPT"
    echo "  sha256              : $(sha256sum "$CKPT" | cut -d' ' -f1)"
else
    echo "pretrained checkpoint : none"
fi
echo "gazebo stage          : $STAGE"
echo "gazebo frame cost     : $FRAME_COST"
echo "gazebo budget         : $BUDGET"
echo "total timesteps       : $TOTAL_TIMESTEPS"
echo "rollout steps         : $NUM_STEPS"
echo "seed                  : $SEED"
echo "checkpoint interval   : $CHECKPOINT_INTERVAL iterations"
echo "config                : ${CONFIG:-none}"
echo "ROS_DOMAIN_ID         : ${ROS_DOMAIN_ID:-<from ~/.bashrc>}   GAZEBO_MASTER_URI: ${GAZEBO_MASTER_URI:-<default>}"
echo "trainer args          : $TRAIN_ARGS"
echo "trainer log           : $LOG"
echo "$rule"
echo "Source details (environment, scene, frame cost, budget) are validated and printed by the trainer's"
echo "startup block and saved in info_settings.txt and every checkpoint."
[ "$DRY_RUN" = 1 ] && { echo "DRY_RUN=1: nothing started."; exit 0; }

setup_cmd="cd $WORKSPACE && source /opt/ros/foxy/setup.bash && source install/setup.bash && export TURTLEBOT3_MODEL=burger"
[ -n "$ROS_DOMAIN_ID" ]     && setup_cmd="$setup_cmd && export ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
[ -n "$GAZEBO_MASTER_URI" ] && setup_cmd="$setup_cmd && export GAZEBO_MASTER_URI=$GAZEBO_MASTER_URI"

tmux kill-session -t "$SESSION" 2>/dev/null || true
if pgrep -f "gzserver|environment_gated.py|gazebo_goals" >/dev/null; then
    echo "Another Gazebo/ROS stack is running in this container -- stop it first:" >&2
    pgrep -af "gzserver|environment_gated.py|gazebo_goals" >&2
    exit 1
fi
mkdir -p "$LOG_DIR"

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

# PPO trainer (output also saved to $LOG)
tmux new-window -t "$SESSION" -n trainer
tmux send-keys -t "$SESSION:trainer" \
"$setup_cmd && PYTHONUNBUFFERED=1 python3 AdaptiveFPS/scripts/train_adaptive_fps_ppo.py $TRAIN_ARGS 2>&1 | tee $LOG" C-m

# gazebo_goals waits internally for the required /goal_pose subscribers before publishing the first goal.
sleep 2

# Goals
tmux new-window -t "$SESSION" -n goals
tmux send-keys -t "$SESSION:goals" \
"$setup_cmd && ros2 run turtlebot3_drl gazebo_goals --ros-args -p external_reset:=true" C-m

echo
echo "AdaptiveFPS Stage $STAGE PPO training started ($CONDITION)."
echo "Attach with:  tmux attach -t $SESSION      Log:  $LOG"
