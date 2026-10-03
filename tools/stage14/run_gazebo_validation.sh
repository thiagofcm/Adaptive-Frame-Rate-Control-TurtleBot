#!/usr/bin/env bash
# Gazebo validation of the Stage 14 randomized OB1/OB2 (run inside the container, nothing else running):
#   ./tools/stage14/run_gazebo_validation.sh [out_dir]      (out_dir must not contain earlier logs)
# Launches gzserver three times (seed 1, seed 1 again, seed 2), runs validate_gazebo.py against each and
# compares the event logs. Needs no environment / goals / trainer node.

WORKSPACE="/home/turtlebot3_drlnav"
OUT="${1:-/tmp/stage14_validation_$(date +%Y%m%d_%H%M%S)}"

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
status=0

run_one() {   # name seed validate-args...
    local name="$1" seed="$2"; shift 2
    if [ -e "$OUT/$name.csv" ]; then
        echo "$OUT/$name.csv already exists (the plugin appends); use a fresh out_dir" >&2
        exit 1
    fi
    setsid ros2 launch turtlebot3_gazebo turtlebot3_drl_stage14.launch.py \
        obstacle_seed:="$seed" obstacle_log:="$OUT/$name.csv" > "$OUT/$name.gazebo.log" 2>&1 &
    local launch_pid=$!
    for _ in $(seq 60); do [ -s "$OUT/$name.csv" ] && break; sleep 1; done
    sleep 5
    echo "=== run $name (seed $seed) ==="
    python3 tools/stage14/validate_gazebo.py run --log "$OUT/$name.csv" "$@" 2>&1 | tee "$OUT/$name.validation.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || status=1
    kill -INT -- "-$launch_pid" 2>/dev/null
    for _ in $(seq 30); do pgrep -x gzserver >/dev/null || break; sleep 1; done
    pkill -9 -x gzserver 2>/dev/null
    sleep 2
}

run_one A 1 --resets 150 --long-episodes 4
run_one B 1 --resets 40 --long-episodes 2
run_one C 2 --resets 40 --long-episodes 0

echo "=== compare ==="
python3 tools/stage14/validate_gazebo.py compare "$OUT/A.csv" "$OUT/B.csv" "$OUT/C.csv" | tee "$OUT/compare.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || status=1
echo "Stage 13 files changed vs git HEAD:"
git -C "$WORKSPACE" status --short -- src/turtlebot3_simulations/turtlebot3_gazebo/worlds/turtlebot3_drl_stage13 \
    src/turtlebot3_simulations/turtlebot3_gazebo/launch/turtlebot3_drl_stage13.launch.py \
    src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_drl_world/obstacle_plugin/corridor_obstacle.cc
echo "(end of list)"
exit $status
