#!/usr/bin/env bash
# Pre-roll measurement on Stage 14 (run inside the container, nothing else running):
#   ./tools/stage14/run_preroll_measurement.sh [episodes] [out_dir]      (out_dir must be fresh)
# Same process layout as run_adaptive_fps_train_stage14.sh, with measure_preroll.py in place of the trainer.

WORKSPACE="/home/turtlebot3_drlnav"
EPISODES="${1:-10}"
OUT="${2:-/tmp/stage14_preroll_$(date +%Y%m%d_%H%M%S)}"

cd "$WORKSPACE"
source /opt/ros/foxy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=1 DRLNAV_BASE_PATH="$WORKSPACE" TURTLEBOT3_MODEL=burger
export GAZEBO_MODEL_PATH="$GAZEBO_MODEL_PATH:$WORKSPACE/src/turtlebot3_simulations/turtlebot3_gazebo/models"
export GAZEBO_PLUGIN_PATH="$GAZEBO_PLUGIN_PATH:$WORKSPACE/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_drl_world/obstacle_plugin/lib"

if pgrep -x gzserver >/dev/null; then
    echo "gzserver is already running; stop it first" >&2
    exit 1
fi
mkdir -p "$OUT"
if [ -e "$OUT/obstacles.csv" ]; then
    echo "$OUT/obstacles.csv already exists; use a fresh out_dir" >&2
    exit 1
fi

setsid ros2 launch turtlebot3_gazebo turtlebot3_drl_stage14.launch.py \
    obstacle_seed:=1 obstacle_log:="$OUT/obstacles.csv" > "$OUT/gazebo.log" 2>&1 &
gazebo_pid=$!
sleep 5
setsid python3 AdaptiveFPS/wrapper/environment_gated.py --ros-args -r scan:=scan_gated > "$OUT/environment.log" 2>&1 &
env_pid=$!
sleep 3
python3 tools/stage14/measure_preroll.py --log "$OUT/obstacles.csv" --episodes "$EPISODES" \
    --out "$OUT/preroll.csv" > "$OUT/measure.log" 2>&1 &
measure_pid=$!
sleep 2
setsid ros2 run turtlebot3_drl gazebo_goals --ros-args -p external_reset:=true > "$OUT/goals.log" 2>&1 &
goals_pid=$!

wait "$measure_pid"
status=$?
grep -E "^(episode|===|pre-roll|obstacle|episodes|extra|saved)" "$OUT/measure.log"
[ "$status" -eq 0 ] || tail -30 "$OUT/measure.log"

kill -INT -- "-$goals_pid" "-$env_pid" "-$gazebo_pid" 2>/dev/null
for _ in $(seq 30); do pgrep -x gzserver >/dev/null || break; sleep 1; done
pkill -9 -x gzserver 2>/dev/null
exit $status
