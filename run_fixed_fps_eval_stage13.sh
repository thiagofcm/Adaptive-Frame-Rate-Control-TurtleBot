#!/usr/bin/env bash

set -e

SESSION="adaptivefps_stage13"
WORKSPACE="/home/turtlebot3_drlnav"

FPS="${1:-10}"   # must be one of AdaptiveFPSEnv.fps_choices: 0.2 0.5 1 5 10
EPISODES="${2:-3}"
# Stage 13 results live apart from the Stage 9 ones in AdaptiveFPS/eval
# Results go to $EVAL_ROOT/fixed/fixed_<fps>Hz/ (override per run, e.g.
# EVAL_ROOT=$WORKSPACE/AdaptiveFPS/eval_stage13_ob12 ./run_fixed_fps_eval_stage13.sh 10 3).
# Historical results remain in AdaptiveFPS/eval_stage13*/ (flat layout).
EVAL_ROOT="${EVAL_ROOT:-$WORKSPACE/AdaptiveFPS/eval/stage13}"
FRAME_COST="${FRAME_COST:-0.005}"   # reward penalty per frame-consuming PPO step
BUDGET="${BUDGET:-450}"             # frame budget (frame_ratio denominator)

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
"$setup_cmd && python3 AdaptiveFPS/scripts/eval.py --fps $FPS --episodes $EPISODES --eval-root $EVAL_ROOT --frame-cost $FRAME_COST --budget $BUDGET" C-m

# gazebo_goals now waits internally (drl_gazebo.py's
# _wait_for_initial_goal_pose_subscribers()) for all 3 required /goal_pose
# subscribers (DRLEnvironment, evaluator, EpisodeRecorder) to be discovered
# before publishing episode 1's one-shot goal, so correctness no longer
# depends on this delay -- it's just a short pause for tidy process
# startup/log ordering.
sleep 2

# Goals
tmux new-window -t "$SESSION" -n goals
tmux send-keys -t "$SESSION:goals" \
"$setup_cmd && ros2 run turtlebot3_drl gazebo_goals --ros-args -p external_reset:=true" C-m

echo
echo "AdaptiveFPS Stage 13 (static corridor) experiment started."
echo "FPS:      $FPS"
echo "Episodes: $EPISODES"
echo "Output:   $EVAL_ROOT/fixed/fixed_${FPS}Hz"
echo
echo "Attach with:"
echo "  tmux attach -t $SESSION"