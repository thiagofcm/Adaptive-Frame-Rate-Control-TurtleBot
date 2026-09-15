#!/usr/bin/env bash

set -e

SESSION="adaptivefps"
WORKSPACE="/home/turtlebot3_drlnav"

TOTAL_TIMESTEPS="${1:-20000000}"
NUM_STEPS="${2:-2048}"

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

# PPO trainer
tmux new-window -t "$SESSION" -n trainer
tmux send-keys -t "$SESSION:trainer" \
"$setup_cmd && python3 AdaptiveFPS/scripts/train_adaptive_fps_ppo.py --total-timesteps $TOTAL_TIMESTEPS --num-steps $NUM_STEPS " C-m

# --resume-path AdaptiveFPS/runs/adaptive_fps_turtlebot_09-09-14-19-39/ckpts/timestep_245760_iterations_120/ckpt_245760_iterations_120.pt 

# gazebo_goals waits internally for the required /goal_pose subscribers
# before publishing the first goal. This short delay is only for cleaner
# process startup/log ordering.
sleep 2

# Goals
tmux new-window -t "$SESSION" -n goals
tmux send-keys -t "$SESSION:goals" \
"$setup_cmd && ros2 run turtlebot3_drl gazebo_goals" C-m

echo
echo "AdaptiveFPS PPO training started."
echo "Total timesteps: $TOTAL_TIMESTEPS"
echo "Rollout steps:   $NUM_STEPS"
echo
echo "Attach with:"
echo "  tmux attach -t $SESSION"