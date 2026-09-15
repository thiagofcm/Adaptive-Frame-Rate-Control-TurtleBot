#!/usr/bin/env python3
"""Plot drl_gazebo.py's Stage 9 (8/9/12) goal_pose_list directly on the
map, instead of a recorded trajectory -- for visualizing where each
candidate goal actually sits before/without running an evaluation.

Reuses the exact same wall-geometry loading/drawing
(load_world_geometry(), rectangle_corners(), wall_bounds()) as
tools/episode_recorder/make_video_fps_v2.py, so the map matches what
episode videos already show. GOAL_POSE_LIST below is a deliberate,
read-only duplicate of drl_gazebo.py's own list (same convention
tools/evaluation/scenario_gazebo.py already uses) -- this script must
not import from or modify drl_gazebo.py.

Run (after sourcing the workspace, so rclpy/turtlebot3_drl are importable):
    python3 tools/episode_recorder/plot_goal_list.py
    python3 tools/episode_recorder/plot_goal_list.py --stage 9 -o goals.png
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from record_episode import load_world_geometry  # noqa: E402  (reused, not re-derived)
from make_video_fps_v2 import rectangle_corners, wall_bounds  # noqa: E402  (same wall-drawing helpers)

# drl_gazebo.py's generate_goal_pose() Stage 9 branch -- index i here IS
# the episode order itself (goal_pose_list is cycled in list order).
# GOAL_POSE_LIST = [[2.0, 2.0], [2.0, 1.5], [2.0, -0.5], [2.0, -1.0], [2.0, -2.0], [1.3, 1.0],
#                     [1.0, 0.3], [1.0, -2.0], [0.3, -1.0],  [0.0, 2.0], [0.0, -1.0], [-1.0, 1.0],
#                         [-1.0, -1.2], [-2.0, 1.0], [-2.2, 0.0], [-2.0, -2.2], [-2.4, 2.4]]

GOAL_POSE_LIST = [[0.0, 2.0], [0.2, 2.0], [0.3, 2.0], [0.4, 2.0], [0.5, 2.0], [0.6, 2.0], [0.8, 2.0],
                  [0.0, 2.1], [0.2, 2.2], [0.3, 2.3], [0.4, 2.4], [0.5, 2.5]]

# eval_adaptive_fps.py's STAGE9_RESET_X/STAGE9_RESET_Y -- the fixed pose
# reset_simulation() (and reset_on_success) put the robot back at.
ROBOT_START_X, ROBOT_START_Y = 0.0, 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", type=int, default=9,
                         help="stage number, for wall-geometry lookup only (default: 9)")
    parser.add_argument("-o", "--output", default=None,
                         help="output image path (default: goal_list_stage<N>.png in cwd)")
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args()

    geometry = load_world_geometry(os.environ["DRLNAV_BASE_PATH"], args.stage)
    walls = geometry["walls"]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")

    margin = 0.5
    bounds = wall_bounds(walls, margin)
    if bounds is None:
        xs = [p[0] for p in GOAL_POSE_LIST] + [ROBOT_START_X]
        ys = [p[1] for p in GOAL_POSE_LIST] + [ROBOT_START_Y]
        bounds = (min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Stage {args.stage} goal_pose_list ({len(GOAL_POSE_LIST)} candidates)")
    ax.grid(alpha=0.15, zorder=0)

    for w in walls:
        corners = rectangle_corners(w["cx"], w["cy"], w["yaw"], w["size_x"], w["size_y"])
        ax.add_patch(Polygon(corners, closed=True, facecolor="0.55", edgecolor="0.3", zorder=2))

    gx = [p[0] for p in GOAL_POSE_LIST]
    gy = [p[1] for p in GOAL_POSE_LIST]
    ax.plot(gx, gy, marker="*", markersize=16, color="tab:green", linestyle="none",
            label="candidate goal", zorder=5)
    for i, (x, y) in enumerate(GOAL_POSE_LIST):
        ax.annotate(f"{i}: ({x:g}, {y:g})", (x, y), textcoords="offset points",
                    xytext=(6, 6), fontsize=7, color="tab:green", zorder=6)

    ax.plot(ROBOT_START_X, ROBOT_START_Y, marker="^", markersize=14, color="tab:red",
            linestyle="none", label="robot start", zorder=6)
    ax.annotate(f"start ({ROBOT_START_X:g}, {ROBOT_START_Y:g})", (ROBOT_START_X, ROBOT_START_Y),
                textcoords="offset points", xytext=(8, -12), fontsize=8, color="tab:red", zorder=6)

    ax.legend(loc="upper right", fontsize=8)

    output = args.output or f"goal_list_stage{args.stage}.png"
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
