#!/usr/bin/env bash

set -e

SESSION="adaptivefps"
WORKSPACE="/home/turtlebot3_drlnav"

#FPS="${1:-10}"
EPISODES="${1:-10}"
MODEL="AdaptiveFPS/runs/adaptive_fps_turtlebot_08-09-16-18-28/model.pt"

setup_cmd="cd $WORKSPACE && source /opt/ros/foxy/setup.bash && source install/setup.bash && export TURTLEBOT3_MODEL=burger"

tmux kill-session -t "$SESSION" 2>/dev/null || true

# Gazebo
tmux new-session -d -s "$SESSION" -n gazebo
tmux send-keys -t "$SESSION:gazebo" \
"$setup_cmd && ros2 launch turtlebot3_gazebo turtlebot3_drl_stage9.launch.py" C-m

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
  --model $MODEL\
  --episodes $EPISODES \
  --diagnose-probs" C-m


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
"$setup_cmd && ros2 run turtlebot3_drl gazebo_goals" C-m

echo
echo "AdaptiveFPS experiment started."
echo "MODEL:      $MODEL"
echo "Episodes: $EPISODES"
echo
echo "Attach with:"
echo "  tmux attach -t $SESSION"