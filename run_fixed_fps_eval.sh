#!/usr/bin/env bash

set -e

SESSION="fixedfps"
WORKSPACE="/home/turtlebot3_drlnav"

FPS="${1:-50}"
EPISODES="${2:-3}"

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
"$setup_cmd && python3 FixedFPS/wrapper/environment_gated.py --ros-args -r scan:=scan_gated" C-m

sleep 3

# Evaluator
tmux new-window -t "$SESSION" -n evaluator
tmux send-keys -t "$SESSION:evaluator" \
"$setup_cmd && python3 FixedFPS/scripts/evaluate_fixed_sensing.py --fps $FPS --n-episodes $EPISODES" C-m

# gazebo_goals publishes episode 1's goal exactly once, with volatile QoS, as
# soon as it starts -- there is no resend. The evaluator's own goal_pose
# subscription must finish DDS discovery (and the evaluator must finish
# loading the TD3 checkpoint, which runs synchronously in its constructor)
# before that single message is published, or it is lost forever and no
# retry in the evaluator's episode-ready handshake can recover it. 3s proved
# too tight in practice; 10s gives comfortable margin.
sleep 10

# Goals
tmux new-window -t "$SESSION" -n goals
tmux send-keys -t "$SESSION:goals" \
"$setup_cmd && ros2 run turtlebot3_drl gazebo_goals" C-m

echo
echo "FixedFPS experiment started."
echo "FPS:      $FPS"
echo "Episodes: $EPISODES"
echo
echo "Attach with:"
echo "  tmux attach -t $SESSION"