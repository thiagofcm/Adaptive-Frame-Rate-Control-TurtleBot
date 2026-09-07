#!/bin/bash

SESSION="td3_deterministic_eval"
PROJECT="/home/turtlebot3_drlnav"

# Window 0 — Gazebo Stage 9
tmux new-session -d -s "$SESSION" -n gazebo
tmux send-keys -t "$SESSION:gazebo" \
"cd $PROJECT && source /opt/ros/foxy/setup.bash && source install/setup.bash && export TURTLEBOT3_MODEL=burger && ros2 launch turtlebot3_gazebo turtlebot3_drl_stage9.launch.py" C-m

# Window 1 — Environment
tmux new-window -t "$SESSION" -n environment
tmux send-keys -t "$SESSION:environment" \
"cd $PROJECT && source /opt/ros/foxy/setup.bash && source install/setup.bash && ros2 run turtlebot3_drl environment" C-m

# Window 2 — Recorder
# Start BEFORE scenario_gazebo.py so it catches the first /goal_pose
tmux new-window -t "$SESSION" -n recorder
tmux send-keys -t "$SESSION:recorder" \
"cd $PROJECT && source /opt/ros/foxy/setup.bash && source install/setup.bash && python3 tools/episode_recorder/record_episode.py" C-m

# Window 3 — Deterministic scenario manager
# Replaces the original: ros2 run turtlebot3_drl gazebo_goals
tmux new-window -t "$SESSION" -n scenarios
tmux send-keys -t "$SESSION:scenarios" \
"cd $PROJECT && source /opt/ros/foxy/setup.bash && source install/setup.bash && python3 tools/evaluation/scenario_gazebo.py" C-m

# Window 4 — TD3 test agent
tmux new-window -t "$SESSION" -n agent
tmux send-keys -t "$SESSION:agent" \
"cd $PROJECT && source /opt/ros/foxy/setup.bash && source install/setup.bash && ros2 run turtlebot3_drl test_agent td3 'examples/td3_0_stage9' 7400" C-m

# Return to Gazebo window and attach
tmux select-window -t "$SESSION:gazebo"
tmux attach-session -t "$SESSION"