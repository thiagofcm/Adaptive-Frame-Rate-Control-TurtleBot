#!/usr/bin/env python3
"""Render a headless MP4 from one episode recorded by record_episode.py.

Reads <episode_dir>/trajectory.csv, lidar.npz, obstacles.npz (if present)
and metadata.json, plus <episode_dir>/../world_geometry.json (one file per
recording run, written once by record_episode.py -- see that script for why
walls aren't duplicated per episode). Pure visualization -- no outcome is
read or shown, since record_episode.py does not record one; see that
script's docstring for why.

By default this draws a clean navigation view: robot trajectory, orientation,
goal, and recorded moving-obstacle positions. Pass --lidar to also overlay
the LiDAR point cloud.

Usage:
    python3 make_video.py --eval-dir FixedFPS/eval/fixed_0.2Hz --n-ep 3
    python3 make_video.py --eval-dir FixedFPS/eval/fixed_0.2Hz --n-ep 3 --lidar --fps 20

--eval-dir is the rate/run folder (whatever record_episode.py's --out-dir
was), and --n-ep selects episode_%04d underneath it -- the example above
reads FixedFPS/eval/fixed_0.2Hz/episode_0003. By default the rendered MP4
is written inside that same episode folder; pass -o/--output to override.
"""
import argparse
import json
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.colors import BoundaryNorm
import numpy as np
import pandas as pd

try:
    from turtlebot3_drl.common.settings import OBSTACLE_RADIUS
except ImportError:
    OBSTACLE_RADIUS = 0.16


# Keep the same discrete viridis sensing-rate convention used by the
# F1TENTH/Lunar-Lander AdaptiveFPS plots: low rates are dark/purple and
# higher rates progress through viridis toward yellow.
FPS_CHOICES = [0.2, 0.5, 1.0, 5.0, 10.0]


def load_episode(episode_dir):
    traj = pd.read_csv(os.path.join(episode_dir, "trajectory.csv"))
    lidar_path = os.path.join(episode_dir, "lidar.npz")
    if os.path.exists(lidar_path):
        lidar = np.load(lidar_path)
    else:
        # No lidar.npz recorded for this episode -- keep the same
        # t_wall/t_sim/ranges keys, empty, so pick_timebase()/the
        # optional --lidar overlay degrade gracefully instead of crashing.
        lidar = {"t_wall": np.zeros(0), "t_sim": np.zeros(0), "ranges": np.zeros((0, 0))}
    obstacles_path = os.path.join(episode_dir, "obstacles.npz")
    obstacles = np.load(obstacles_path) if os.path.exists(obstacles_path) else None
    with open(os.path.join(episode_dir, "metadata.json")) as f:
        meta = json.load(f)
    return traj, lidar, obstacles, meta


def load_step_log(episode_dir):
    """Load the per-PPO-step log used to color the trajectory by sensing rate.

    The current evaluator writes step.csv; steps.csv is also accepted for
    compatibility with the naming used by the other AdaptiveFPS projects.
    """
    for filename in ("step.csv", "steps.csv"):
        path = os.path.join(episode_dir, filename)
        if os.path.exists(path):
            steps = pd.read_csv(path)
            required = {"x", "y", "current_fps"}
            missing = required - set(steps.columns)
            if missing:
                raise ValueError(f"{path}: missing required columns {sorted(missing)}")
            return steps
    return None


def pick_step_time_column(steps, timebase):
    """Return the step-log time column matching trajectory/LiDAR timebase."""
    if steps is None:
        return None
    preferred = "sim_time" if timebase == "t_sim" else "wall_time"
    if preferred in steps.columns and steps[preferred].notna().all():
        return preferred
    fallback = "wall_time" if preferred == "sim_time" else "sim_time"
    if fallback in steps.columns and steps[fallback].notna().all():
        return fallback
    return None


def validate_fps_values(values, source):
    present = np.asarray(values, dtype=float)
    unexpected = sorted({float(v) for v in np.unique(present)
                         if not any(np.isclose(v, choice) for choice in FPS_CHOICES)})
    if unexpected:
        raise ValueError(
            f"{source}: current_fps contains values outside the valid discrete "
            f"choices {FPS_CHOICES}: {unexpected}"
        )


def load_walls(episode_dir):
    """world_geometry.json is written once per recording run (a sibling of
    every episode_NNNN dir), not per episode -- see record_episode.py."""
    geometry_path = os.path.join(os.path.dirname(episode_dir.rstrip("/")), "world_geometry.json")
    if not os.path.exists(geometry_path):
        return []
    with open(geometry_path) as f:
        return json.load(f).get("walls", [])


def rectangle_corners(cx, cy, yaw, size_x, size_y):
    hx, hy = size_x / 2, size_y / 2
    local = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]])
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([cx, cy])


def wall_bounds(walls, margin):
    if not walls:
        return None
    xs, ys = [], []
    for w in walls:
        corners = rectangle_corners(w["cx"], w["cy"], w["yaw"], w["size_x"], w["size_y"])
        xs.extend(corners[:, 0])
        ys.extend(corners[:, 1])
    return min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin


def build_obstacle_streams(obstacles, timebase):
    """Group the flat obstacles.npz event log into one sorted time series per
    obstacle frame_id, so each can be looked up independently per frame."""
    if obstacles is None or len(obstacles["frame_id"]) == 0:
        return {}
    times = obstacles["t_sim"] if timebase == "t_sim" else obstacles["t_wall"]
    valid = np.isfinite(times)
    streams = {}
    for frame_id in sorted(set(obstacles["frame_id"][valid])):
        mask = valid & (obstacles["frame_id"] == frame_id)
        order = np.argsort(times[mask])
        streams[frame_id] = {
            "t": times[mask][order],
            "x": obstacles["x"][mask][order],
            "y": obstacles["y"][mask][order],
        }
    return streams


def pick_timebase(traj, lidar):
    """Prefer simulation time when it's available for the whole episode,
    since that's the clock the robot/environment actually run on; fall back
    to wall-clock receipt time otherwise."""
    traj_sim_ok = traj["t_sim"].notna().all() if len(traj) else False
    lidar_sim_ok = (not len(lidar["t_sim"])) or (not np.isnan(lidar["t_sim"]).any())
    if traj_sim_ok and lidar_sim_ok:
        return "t_sim"
    return "t_wall"


def robot_triangle(x, y, yaw, size):
    local = np.array([
        [size, 0.0],
        [-size * 0.6, size * 0.5],
        [-size * 0.6, -size * 0.5],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    world = local @ rot.T + np.array([x, y])
    return world


def lidar_points(x, y, yaw, ranges, max_range):
    n = len(ranges)
    if n == 0:
        return np.zeros((0, 2))
    angles = yaw + np.linspace(0, 2 * np.pi, n, endpoint=False)
    valid = np.isfinite(ranges) & (ranges > 0) & (ranges <= max_range)
    xs = x + ranges[valid] * np.cos(angles[valid])
    ys = y + ranges[valid] * np.sin(angles[valid])
    return np.column_stack([xs, ys])


def main():
    parser = argparse.ArgumentParser(description="Render an MP4 from a recorded episode.")
    parser.add_argument("--eval-dir", required=True,
                         help="Rate/run folder containing episode_NNNN dirs, e.g. FixedFPS/eval/fixed_0.2Hz")
    parser.add_argument("--n-ep", type=int, required=True,
                         help="Episode number, e.g. 3 -> <eval-dir>/episode_0003")
    parser.add_argument("-o", "--output", default=None,
                         help="Output MP4 path (default: inside the episode folder itself)")
    parser.add_argument("--fps", type=int, default=30, help="Output video frame rate")
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--lidar", dest="lidar", action="store_true", default=False,
                         help="Overlay LiDAR points (default: off, for a clean navigation view)")
    parser.add_argument("--lidar-max-range", type=float, default=3.5,
                         help="Discard LiDAR points beyond this range (meters)")
    parser.add_argument("--no-obstacles", dest="obstacles", action="store_false", default=True,
                         help="Hide recorded moving-obstacle positions (default: shown)")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH -- install it (e.g. apt-get install ffmpeg) and retry.")

    eval_dir = args.eval_dir.rstrip("/")
    episode_dir = os.path.join(eval_dir, f"episode_{args.n_ep:04d}")
    if not os.path.isdir(episode_dir):
        sys.exit(f"{episode_dir}: no such episode directory")
    traj, lidar, obstacles, meta = load_episode(episode_dir)
    steps = load_step_log(episode_dir)
    if len(traj) == 0:
        sys.exit(f"{episode_dir}: trajectory.csv has no rows, nothing to render.")
    walls = load_walls(episode_dir)

    # By default, write the video into the episode folder itself (the same
    # folder the CSV/npz/json inputs came from), named after --eval-dir so
    # it stays identifiable if later collected elsewhere alongside videos
    # from other rates/runs.
    output = args.output or os.path.join(episode_dir, "episode_" + f"{args.n_ep:04d}" + ".mp4")

    timebase = pick_timebase(traj, lidar)
    traj_t = traj[timebase].to_numpy()
    lidar_t = lidar["t_sim"] if timebase == "t_sim" else lidar["t_wall"]
    lidar_ranges = lidar["ranges"]
    obstacle_streams = build_obstacle_streams(obstacles, timebase) if args.obstacles else {}

    # Sensing-rate visualization comes from the per-step evaluator log.
    # Match it to the same simulation/wall clock already selected for the video.
    step_time_col = pick_step_time_column(steps, timebase)
    if steps is not None and step_time_col is None:
        sys.exit("step.csv/steps.csv has no usable sim_time or wall_time column.")
    if steps is not None:
        step_t = steps[step_time_col].to_numpy(dtype=float)
        step_x = steps["x"].to_numpy(dtype=float)
        step_y = steps["y"].to_numpy(dtype=float)
        step_fps = steps["current_fps"].to_numpy(dtype=float)
        validate_fps_values(step_fps, os.path.join(episode_dir, "step.csv/steps.csv"))
        fps_cmap = matplotlib.colormaps["viridis"].resampled(len(FPS_CHOICES))
        fps_norm = BoundaryNorm(np.arange(len(FPS_CHOICES) + 1) - 0.5, fps_cmap.N)
        fps_index = np.array([
            next(i for i, choice in enumerate(FPS_CHOICES) if np.isclose(v, choice))
            for v in step_fps
        ])
    else:
        step_t = step_x = step_y = step_fps = fps_index = None
        fps_cmap = fps_norm = None

    duration = float(traj_t[-1])
    if len(lidar_t):
        duration = max(duration, float(lidar_t[-1]))
    num_frames = max(1, int(duration * args.fps) + 1)
    frame_times = np.linspace(0, duration, num_frames)

    traj_idx = np.searchsorted(traj_t, frame_times, side="right") - 1
    traj_idx = np.clip(traj_idx, 0, len(traj_t) - 1)
    if len(lidar_t):
        lidar_idx = np.searchsorted(lidar_t, frame_times, side="right") - 1
        lidar_idx = np.clip(lidar_idx, 0, len(lidar_t) - 1)
    else:
        lidar_idx = None

    xs = traj["x"].to_numpy()
    ys = traj["y"].to_numpy()
    yaws = traj["yaw"].to_numpy()
    goal_x, goal_y = float(traj["goal_x"].iloc[0]), float(traj["goal_y"].iloc[0])

    arena_l = meta.get("arena_length")
    arena_w = meta.get("arena_width")
    margin = 0.5

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")

    bounds = wall_bounds(walls, margin)
    if bounds is None and arena_l and arena_w:
        bounds = (-arena_l / 2 - margin, arena_l / 2 + margin,
                  -arena_w / 2 - margin, arena_w / 2 + margin)
    if bounds is None:
        bounds = (min(xs.min(), goal_x) - margin, max(xs.max(), goal_x) + margin,
                  min(ys.min(), goal_y) - margin, max(ys.max(), goal_y) + margin)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    for w in walls:
        corners = rectangle_corners(w["cx"], w["cy"], w["yaw"], w["size_x"], w["size_y"])
        ax.add_patch(Polygon(corners, closed=True, facecolor="0.55", edgecolor="0.3", zorder=2))

    ax.plot(goal_x, goal_y, marker="*", markersize=18, color="tab:green", linestyle="none",
            label="goal", zorder=5)

    trail_line, = ax.plot([], [], color="tab:blue", linewidth=1.2, alpha=0.7, zorder=3)

    # Overlay small discrete-color markers showing the sensing rate applied along
    # the already-traversed path.  The original blue trajectory line is retained.
    if steps is not None:
        fps_scatter = ax.scatter([], [], c=[], cmap=fps_cmap, norm=fps_norm,
                                 s=16, alpha=0.95, edgecolors="none", zorder=4)
        cbar = fig.colorbar(fps_scatter, ax=ax, ticks=range(len(FPS_CHOICES)),
                            fraction=0.046, pad=0.04)
        cbar.ax.set_yticklabels([f"{fps:g} Hz" for fps in FPS_CHOICES])
        cbar.set_label("Sensing Rate (Hz)")
    else:
        fps_scatter = None

    robot_patch = Polygon(robot_triangle(xs[0], ys[0], yaws[0], size=0.15),
                           closed=True, color="tab:red", zorder=6)
    ax.add_patch(robot_patch)
    lidar_scatter = ax.scatter([], [], s=4, color="tab:orange", alpha=0.6, zorder=4)

    obstacle_patches = {}
    for frame_id, stream in obstacle_streams.items():
        patch = Circle((stream["x"][0], stream["y"][0]), radius=OBSTACLE_RADIUS,
                        color="tab:purple", alpha=0.6, zorder=4, visible=False)
        ax.add_patch(patch)
        obstacle_patches[frame_id] = patch
    if obstacle_patches:
        ax.add_patch(Circle((0, 0), radius=OBSTACLE_RADIUS, color="tab:purple", alpha=0.6,
                             visible=False, label="obstacle"))

    title = ax.set_title("")
    ax.legend(loc="upper right")

    episode_index = meta.get("episode_index", "?")

    def update(frame_i):
        ti = traj_idx[frame_i]
        x, y, yaw = xs[ti], ys[ti], yaws[ti]
        t_now = frame_times[frame_i]
        trail_line.set_data(xs[:ti + 1], ys[:ti + 1])
        if fps_scatter is not None:
            si = np.searchsorted(step_t, t_now, side="right")
            if si > 0:
                fps_scatter.set_offsets(np.column_stack([step_x[:si], step_y[:si]]))
                fps_scatter.set_array(fps_index[:si])
            else:
                fps_scatter.set_offsets(np.empty((0, 2)))
                fps_scatter.set_array(np.array([], dtype=float))
        robot_patch.set_xy(robot_triangle(x, y, yaw, size=0.15))
        if args.lidar and lidar_idx is not None and len(lidar_ranges):
            li = lidar_idx[frame_i]
            pts = lidar_points(x, y, yaw, lidar_ranges[li], args.lidar_max_range)
            lidar_scatter.set_offsets(pts)
        for frame_id, stream in obstacle_streams.items():
            patch = obstacle_patches[frame_id]
            if t_now < stream["t"][0]:
                patch.set_visible(False)
                continue
            oi = np.searchsorted(stream["t"], t_now, side="right") - 1
            patch.center = (stream["x"][oi], stream["y"][oi])
            patch.set_visible(True)
        if steps is not None:
            current_si = np.searchsorted(step_t, t_now, side="right") - 1
            current_fps_text = (f"   sensing={step_fps[current_si]:g} Hz"
                                if current_si >= 0 else "")
        else:
            current_fps_text = ""
        title.set_text(f"episode {episode_index}   t={t_now:.1f}s "
                       f"({'sim' if timebase == 't_sim' else 'wall'})"
                       f"{current_fps_text}")
        artists = [trail_line, robot_patch, lidar_scatter, title]
        if fps_scatter is not None:
            artists.append(fps_scatter)
        artists.extend(obstacle_patches.values())
        return tuple(artists)

    anim = animation.FuncAnimation(fig, update, frames=num_frames, blit=False)
    anim.save(output, fps=args.fps, dpi=args.dpi, writer="ffmpeg")
    plt.close(fig)
    print(f"wrote {output} ({num_frames} frames, {duration:.1f}s @ {args.fps} fps)")


if __name__ == "__main__":
    main()
