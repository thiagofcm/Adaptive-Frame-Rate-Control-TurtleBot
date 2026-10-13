#!/usr/bin/env python3
"""Render a headless side-by-side MP4 of one recorded episode.

Left panel:  the episode video of tools/episode_recorder/make_video_fps_v2.py
             (walls, goal, trail, sensing-rate markers, robot, moving obstacles).
Right panel: the two stacked plots of plot_fps_vs_free_space.py -- sensing
             rate (Hz) and ground-truth Angular Obstruction (%) -- animated:
             both traces are revealed progressively, up to the current PPO
             step, on axes that stay fixed for the whole episode.

Offline and read-only with respect to the recorded evaluation data. The
metric is imported from plot_fps_vs_free_space.py (same raycasting, same
FREE_SPACE_RADIUS), never recomputed here.

Synchronization: both panels run on the episode's simulation clock and on
make_video_fps_v2.py's frame schedule. Every video frame shows the latest
PPO step at or before the frame time -- the robot pose on the left and the
sensing rate / obstruction on the right come from that same step; nothing is
interpolated between steps.
Known small difference: as in make_video_fps_v2.py, the moving obstacles on
the left are drawn at their latest recorded position at the *frame* time
(50 Hz log), while the metric uses the obstacles at the *step* time. The two
can differ by up to one PPO step (~0.1 s, i.e. < 1 cm of obstacle travel).

Usage:
    python3 AdaptiveFPS/scripts/make_video_free_space.py --eval-dir <run dir> --n-ep 1
    python3 AdaptiveFPS/scripts/make_video_free_space.py --eval-dir <run dir> --n-ep 1 --speed 1   # real time (default is 4x)

--eval-dir is the run folder and --n-ep selects episode_%04d underneath it.
By default the MP4 is written inside that episode folder as
episode_NNNN_free_space.mp4; pass -o/--output to override. An existing
output file is overwritten.
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "tools", "episode_recorder"))

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np

# Episode-video helpers, reused as is (the drawing itself lives inside that
# script's main(), so it is replicated in draw_episode_panel() below).
import make_video_fps_v2 as episode_video
from make_video_fps_v2 import FPS_CHOICES, OBSTACLE_RADIUS

# Free-space metric + plot style, reused as is. Importing it switches the
# global matplotlib style to the SciencePlots one used by the static plots.
import plot_fps_vs_free_space as free_space

# The static plots are styled for a 15 x 18 in figure; the right panel is
# 7 x 7 in, so its fonts and line widths are scaled down by this factor.
PLOT_SCALE = 0.42

# Keep the SciencePlots style for the right panel only: the left panel is
# built and animated under matplotlib's own defaults, exactly like
# make_video_fps_v2.py renders it.
PLOT_STYLE = dict(plt.rcParams)
matplotlib.rc_file_defaults()


# ══════════════════════════════════════════════════════
# Frame schedule (same as make_video_fps_v2.py)
# ══════════════════════════════════════════════════════

def frame_schedule(traj_t, lidar_t, fps, speed):
    """Episode time shown by every output frame. Each frame advances
    speed / fps seconds of episode time."""
    duration = float(traj_t[-1])
    if len(lidar_t):
        duration = max(duration, float(lidar_t[-1]))
    num_frames = max(1, int(duration * fps / speed) + 1)
    return np.linspace(0, duration, num_frames), duration


def latest_index(times, frame_times):
    """Index of the latest sample at or before each frame time (first sample before any exists)."""
    return np.clip(np.searchsorted(times, frame_times, side="right") - 1, 0, len(times) - 1)


# ══════════════════════════════════════════════════════
# Left panel: episode video (drawing replicated from make_video_fps_v2.py)
# ══════════════════════════════════════════════════════

def draw_episode_panel(fig, traj, steps, step_t, walls, obstacle_streams, meta):
    xs = traj["x"].to_numpy()
    ys = traj["y"].to_numpy()
    yaws = traj["yaw"].to_numpy()
    goal_x, goal_y = float(traj["goal_x"].iloc[0]), float(traj["goal_y"].iloc[0])

    arena_l = meta.get("arena_length")
    arena_w = meta.get("arena_width")
    margin = 0.5

    # Left half of the figure, at make_video_fps_v2.py's default subplot position.
    ax = fig.add_axes([0.125 / 2, 0.11, 0.775 / 2, 0.77])
    ax.set_aspect("equal")

    bounds = episode_video.wall_bounds(walls, margin)
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
        corners = episode_video.rectangle_corners(w["cx"], w["cy"], w["yaw"], w["size_x"], w["size_y"])
        ax.add_patch(Polygon(corners, closed=True, facecolor="0.55", edgecolor="0.3", zorder=2))

    ax.plot(goal_x, goal_y, marker="*", markersize=18, color="tab:green", linestyle="none",
            label="goal", zorder=5)

    trail_line, = ax.plot([], [], color="tab:blue", linewidth=1.2, alpha=0.7, zorder=3)

    # Small discrete-color markers showing the sensing rate applied along the
    # already-traversed path. The blue trajectory line is retained.
    step_x = steps["x"].to_numpy(dtype=float)
    step_y = steps["y"].to_numpy(dtype=float)
    step_fps = steps["current_fps"].to_numpy(dtype=float)
    fps_cmap = matplotlib.colormaps["viridis"].resampled(len(FPS_CHOICES))
    fps_norm = BoundaryNorm(np.arange(len(FPS_CHOICES) + 1) - 0.5, fps_cmap.N)
    fps_index = np.array([
        next(i for i, choice in enumerate(FPS_CHOICES) if np.isclose(v, choice))
        for v in step_fps
    ])
    fps_scatter = ax.scatter([], [], c=[], cmap=fps_cmap, norm=fps_norm,
                             s=16, alpha=0.95, edgecolors="none", zorder=4)
    cbar = fig.colorbar(fps_scatter, ax=ax, ticks=range(len(FPS_CHOICES)),
                        fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels([f"{fps:g} Hz" for fps in FPS_CHOICES])
    cbar.set_label("Sensing Rate (Hz)")

    robot_patch = Polygon(episode_video.robot_triangle(xs[0], ys[0], yaws[0], size=0.15),
                          closed=True, color="tab:red", zorder=6)
    ax.add_patch(robot_patch)

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

    def update(t_now, ti):
        x, y, yaw = xs[ti], ys[ti], yaws[ti]
        trail_line.set_data(xs[:ti + 1], ys[:ti + 1])
        si = np.searchsorted(step_t, t_now, side="right")
        if si > 0:
            fps_scatter.set_offsets(np.column_stack([step_x[:si], step_y[:si]]))
            fps_scatter.set_array(fps_index[:si])
        else:
            fps_scatter.set_offsets(np.empty((0, 2)))
            fps_scatter.set_array(np.array([], dtype=float))
        robot_patch.set_xy(episode_video.robot_triangle(x, y, yaw, size=0.15))
        for frame_id, stream in obstacle_streams.items():
            patch = obstacle_patches[frame_id]
            if t_now < stream["t"][0]:
                patch.set_visible(False)
                continue
            oi = np.searchsorted(stream["t"], t_now, side="right") - 1
            patch.center = (stream["x"][oi], stream["y"][oi])
            patch.set_visible(True)

    return title, update


# ══════════════════════════════════════════════════════
# Right panel: animated sensing rate + angular obstruction
# ══════════════════════════════════════════════════════

def draw_metric_panel(fig, episode):
    """Same two stacked plots as plot_fps_vs_free_space.plot_combined, on fixed
    axes; update(si) reveals both traces up to PPO step index si."""
    timesteps = episode["timesteps"]
    fps_choices = episode["fps_choices"]
    current_fps = episode["current_fps"]
    # Same definition as the static plot: 1 - free-space fraction, in percent.
    obstruction = 100.0 * (1.0 - episode["free_fraction"])

    linewidth = 7.0 * PLOT_SCALE
    label_size = 56 * PLOT_SCALE
    tick_size = 50 * PLOT_SCALE

    ax_fps = fig.add_axes([0.625, 0.535, 0.355, 0.365])
    ax_free = fig.add_axes([0.625, 0.11, 0.355, 0.365], sharex=ax_fps)

    fps_line, = ax_fps.plot([], [], linewidth=linewidth, drawstyle="steps-post", color="tab:blue")
    ax_fps.set_ylabel("Sens. Rate (Hz)", fontsize=label_size, labelpad=16 * PLOT_SCALE)
    ax_fps.grid(True, axis="y", alpha=0.9)
    for fps in fps_choices:
        ax_fps.axhline(y=fps, linestyle=":", linewidth=3.5 * PLOT_SCALE, alpha=0.9, color="gray")
    ax_fps.set_yscale("log")
    ax_fps.set_yticks(fps_choices)
    ax_fps.set_yticklabels([f"{fps:g}" for fps in fps_choices], fontsize=40 * PLOT_SCALE)
    ax_fps.minorticks_off()
    ax_fps.set_ylim(min(fps_choices) / 1.4, max(fps_choices) * 1.4)
    ax_fps.tick_params(axis="x", labelbottom=False)

    free_line, = ax_free.plot([], [], linewidth=linewidth, color="tab:orange")
    ax_free.set_xlabel("Timesteps", fontsize=label_size, labelpad=10 * PLOT_SCALE)
    ax_free.set_ylabel("Angular\nObstruction (%)", fontsize=label_size, labelpad=30 * PLOT_SCALE)
    ax_free.grid(True, axis="y", alpha=0.9)
    ax_free.set_yticks([0, 25, 50, 75, 100])
    ax_free.set_ylim(-2, 102)
    ax_free.tick_params(axis="both", labelsize=tick_size)
    # The traces grow during the video, so the x range is fixed to the whole episode up front.
    x_margin = 0.05 * (timesteps[-1] - timesteps[0])
    ax_free.set_xlim(timesteps[0] - x_margin, timesteps[-1] + x_margin)
    ax_free.xaxis.set_major_locator(MaxNLocator(nbins=4))

    def update(si):
        fps_line.set_data(timesteps[:si + 1], current_fps[:si + 1])
        free_line.set_data(timesteps[:si + 1], obstruction[:si + 1])

    return update


# ══════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Render a side-by-side MP4: episode video + animated free-space plots.")
    parser.add_argument("--eval-dir", required=True,
                        help="Run folder containing episode_NNNN dirs and world_geometry.json")
    parser.add_argument("--n-ep", type=int, required=True,
                        help="Episode number, e.g. 3 -> <eval-dir>/episode_0003")
    parser.add_argument("-o", "--output", default=None,
                        help="Output MP4 path (default: inside the episode folder itself)")
    parser.add_argument("--fps", type=int, default=30, help="Output video frame rate")
    parser.add_argument("--speed", type=float, default=4.0,
                        help="Playback speed-up relative to episode time (default: 4 -> 4x; 1 = real time)")
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--no-obstacles", dest="obstacles", action="store_false", default=True,
                        help="Hide recorded moving-obstacle positions (default: shown)")
    args = parser.parse_args()
    if args.speed <= 0:
        sys.exit("--speed must be > 0")

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH -- install it (e.g. apt-get install ffmpeg) and retry.")

    eval_dir = args.eval_dir.rstrip("/")
    episode_dir = os.path.join(eval_dir, f"episode_{args.n_ep:04d}")
    if not os.path.isdir(episode_dir):
        sys.exit(f"{episode_dir}: no such episode directory")

    output = args.output or os.path.join(episode_dir, f"episode_{args.n_ep:04d}_free_space.mp4")

    traj, lidar, obstacles, meta = episode_video.load_episode(episode_dir)
    steps = episode_video.load_step_log(episode_dir)
    if len(traj) == 0:
        sys.exit(f"{episode_dir}: trajectory.csv has no rows, nothing to render.")
    if steps is None:
        sys.exit(f"{episode_dir}: no steps.csv, cannot plot the sensing rate / free-space metric.")
    walls = episode_video.load_walls(episode_dir)

    timebase = episode_video.pick_timebase(traj, lidar)
    if timebase != "t_sim":
        sys.exit(f"{episode_dir}: simulation timestamps are incomplete; the metric is defined on sim time.")
    traj_t = traj[timebase].to_numpy()
    lidar_t = lidar["t_sim"]
    obstacle_streams = episode_video.build_obstacle_streams(obstacles, timebase) if args.obstacles else {}

    step_time_col = episode_video.pick_step_time_column(steps, timebase)
    if step_time_col is None:
        sys.exit("steps.csv has no usable sim_time column.")
    step_t = steps[step_time_col].to_numpy(dtype=float)
    step_fps = steps["current_fps"].to_numpy(dtype=float)
    episode_video.validate_fps_values(step_fps, os.path.join(episode_dir, "steps.csv"))

    # Ground-truth metric per PPO step, from the static plot's own implementation.
    episode = free_space.load_episode(eval_dir, args.n_ep)
    if len(episode["timesteps"]) != len(step_t):
        sys.exit(f"{episode_dir}: metric has {len(episode['timesteps'])} steps, steps.csv has {len(step_t)}.")

    frame_times, duration = frame_schedule(traj_t, lidar_t, args.fps, args.speed)
    num_frames = len(frame_times)
    traj_idx = latest_index(traj_t, frame_times)
    step_idx = latest_index(step_t, frame_times)

    # Twice make_video_fps_v2.py's 7 x 7 in figure: one half per panel.
    fig = plt.figure(figsize=(14, 7))
    panel_title, update_episode_panel = draw_episode_panel(
        fig, traj, steps, step_t, walls, obstacle_streams, meta)
    fig.canvas.draw()  # instantiate the left panel's ticks under the default style
    with plt.rc_context(PLOT_STYLE):
        update_metric_panel = draw_metric_panel(fig, episode)
        fig.canvas.draw()  # ... and the right panel's under the static plots' style

    def update(frame_i):
        t_now = frame_times[frame_i]
        update_episode_panel(t_now, traj_idx[frame_i])
        current_si = np.searchsorted(step_t, t_now, side="right") - 1
        current_fps_text = f"   sensing={step_fps[current_si]:g} Hz" if current_si >= 0 else ""
        panel_title.set_text(f"episode {args.n_ep}   t={t_now:.1f}s (sim)"
                             f"{current_fps_text}   [{args.speed:g}x]")
        update_metric_panel(step_idx[frame_i])
        return ()

    anim = animation.FuncAnimation(fig, update, frames=num_frames, blit=False)
    anim.save(output, fps=args.fps, dpi=args.dpi, writer="ffmpeg")
    plt.close(fig)
    print(f"wrote {output} ({num_frames} frames, {duration:.1f}s episode at {args.speed:g}x "
          f"-> {num_frames / args.fps:.1f}s video @ {args.fps} fps)")


if __name__ == "__main__":
    main()
