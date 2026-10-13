#!/usr/bin/env python3
"""Render the side-by-side MP4 of make_video_free_space.py with the
ground-truth LiDAR rays drawn on the episode panel.

Left panel:  the episode video of make_video_free_space.py, plus the 40
             ground-truth LiDAR rays and a dashed circle of radius
             FREE_SPACE_RADIUS around the LiDAR origin. Green rays are free
             up to that radius and reach the circle; orange rays are
             obstructed within it and stop at their first intersection.
Right panel: unchanged -- sensing rate (Hz) and Angular Obstruction (%).

Offline and read-only with respect to the recorded evaluation data. Ray
geometry, raycasting and FREE_SPACE_RADIUS are imported from
plot_fps_vs_free_space.py, never redefined here.

Synchronization: every video frame shows one recorded PPO step -- the latest
one at or before the frame time (make_video_free_space.py's schedule). The
robot pose, the moving obstacles, the rays and the plotted values all belong
to that same step; nothing is interpolated between steps. The rays are the
very ones the metric counts, so
    orange rays = Angular Obstruction (%) * 40 / 100
holds on every frame (checked for the whole episode before rendering).
Unlike make_video_free_space.py, the moving obstacles are therefore drawn at
the PPO-step time (the positions the raycast uses), not at the frame time.

Usage:
    python3 AdaptiveFPS/scripts/make_video_free_space_rays.py --eval-dir <run dir> --n-ep 1
    python3 AdaptiveFPS/scripts/make_video_free_space_rays.py --eval-dir <run dir> --n-ep 1 --speed 1   # real time (default is 4x)

--eval-dir is the run folder and --n-ep selects episode_%04d underneath it.
By default the MP4 is written inside that episode folder as
episode_NNNN_free_space_rays.mp4; pass -o/--output to override. An existing
output file is overwritten.
"""
import argparse
import os
import shutil
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np

# Two-panel video (episode panel, metric panel, frame schedule, styles), reused as is.
import make_video_free_space as base
from make_video_free_space import episode_video, free_space
from make_video_fps_v2 import OBSTACLE_RADIUS
from plot_fps_vs_free_space import FREE_SPACE_RADIUS, LIDAR_NUM_RAYS

FREE_RAY_COLOR = "#59a14f"     # muted green
BLOCKED_RAY_COLOR = "#f28e2b"  # muted orange
RAY_LINEWIDTH = 0.9
RAY_ALPHA = 0.85


# ══════════════════════════════════════════════════════
# Ground-truth rays per PPO step
# ══════════════════════════════════════════════════════

def compute_rays(steps, walls, obstacle_tracks):
    """Ray geometry of every PPO step, from the metric's own lidar_rays()/raycast().

    Returns the LiDAR origins (n_steps, 2), the ray end points (n_steps, n_rays, 2)
    -- on the FREE_SPACE_RADIUS circle for free rays, at the first intersection
    for obstructed ones -- and the obstructed mask (n_steps, n_rays)."""
    n_steps = len(steps["ppo_step"])
    origins = np.empty((n_steps, 2))
    ends = np.empty((n_steps, LIDAR_NUM_RAYS, 2))
    blocked = np.empty((n_steps, LIDAR_NUM_RAYS), dtype=bool)
    for k in range(n_steps):
        origin, directions = free_space.lidar_rays(steps["x"][k], steps["y"][k], steps["yaw"][k])
        distances = free_space.raycast(origin, directions, walls, obstacle_tracks[:, k, :])
        # Same criterion as free_space_fraction(): free means distance >= FREE_SPACE_RADIUS.
        blocked[k] = ~(distances >= FREE_SPACE_RADIUS)
        origins[k] = origin
        ends[k] = origin + np.minimum(distances, FREE_SPACE_RADIUS)[:, None] * directions
    return origins, ends, blocked


# ══════════════════════════════════════════════════════
# Episode-panel overlay: obstacles at step time, rays, radius circle
# ══════════════════════════════════════════════════════

def draw_ray_overlay(ax, origins, ends, blocked, obstacle_tracks, show_obstacles):
    obstacle_patches = []
    if show_obstacles:
        for track in obstacle_tracks:
            patch = Circle(track[0], radius=OBSTACLE_RADIUS, color="tab:purple", alpha=0.6, zorder=4)
            ax.add_patch(patch)
            obstacle_patches.append(patch)

    rays = LineCollection([], linewidths=RAY_LINEWIDTH, alpha=RAY_ALPHA, zorder=3.5)
    ax.add_collection(rays)
    radius_circle = Circle(origins[0], radius=FREE_SPACE_RADIUS, fill=False, edgecolor="0.25",
                           linewidth=0.9, linestyle=(0, (4, 3)), alpha=0.8, zorder=3.6)
    ax.add_patch(radius_circle)

    handles, _ = ax.get_legend_handles_labels()
    if obstacle_patches:
        handles.append(Line2D([], [], marker="o", markersize=9, color="tab:purple", alpha=0.6,
                              linestyle="none", label="obstacle"))
    handles += [
        Line2D([], [], color=FREE_RAY_COLOR, linewidth=1.5, label="free ray"),
        Line2D([], [], color=BLOCKED_RAY_COLOR, linewidth=1.5, label="obstructed ray"),
        Line2D([], [], color="0.25", linewidth=0.9, linestyle=(0, (4, 3)),
               label=f"{FREE_SPACE_RADIUS:g} m radius"),
    ]
    # One compact row under the axes, so the legend never hides the scene.
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(handles),
              fontsize=8, handlelength=1.6, columnspacing=1.2, borderpad=0.4, frameon=False)

    def update(si):
        origin = origins[si]
        rays.set_segments([np.array([origin, end]) for end in ends[si]])
        rays.set_colors([BLOCKED_RAY_COLOR if b else FREE_RAY_COLOR for b in blocked[si]])
        radius_circle.center = tuple(origin)
        for patch, track in zip(obstacle_patches, obstacle_tracks):
            patch.center = tuple(track[si])

    return rays, radius_circle, update


# ══════════════════════════════════════════════════════
# Scene assembly
# ══════════════════════════════════════════════════════

def build_video(eval_dir, n_ep, fps, speed, show_obstacles=True):
    """Build the figure and its per-frame update. Returns a dict with the
    figure, the update(frame_i) callback, the frame schedule and the artists."""
    episode_dir = os.path.join(eval_dir, f"episode_{n_ep:04d}")
    if not os.path.isdir(episode_dir):
        sys.exit(f"{episode_dir}: no such episode directory")

    traj, lidar, _, meta = episode_video.load_episode(episode_dir)
    step_log = episode_video.load_step_log(episode_dir)
    if len(traj) == 0:
        sys.exit(f"{episode_dir}: trajectory.csv has no rows, nothing to render.")
    if step_log is None:
        sys.exit(f"{episode_dir}: no steps.csv, cannot plot the sensing rate / free-space metric.")
    wall_specs = episode_video.load_walls(episode_dir)

    if episode_video.pick_timebase(traj, lidar) != "t_sim":
        sys.exit(f"{episode_dir}: simulation timestamps are incomplete; the metric is defined on sim time.")
    traj_t = traj["t_sim"].to_numpy()
    step_fps = step_log["current_fps"].to_numpy(dtype=float)
    episode_video.validate_fps_values(step_fps, os.path.join(episode_dir, "steps.csv"))

    # Ground-truth scene of every PPO step: exactly the inputs the metric uses.
    steps = free_space.load_steps(episode_dir)
    step_t = steps["t_sim"]
    walls = free_space.load_walls(eval_dir)
    obstacle_tracks = free_space.load_obstacle_tracks(episode_dir, step_t)
    origins, ends, blocked = compute_rays(steps, walls, obstacle_tracks)

    # The rays drawn must be the rays the plotted metric counted.
    episode = free_space.load_episode(eval_dir, n_ep)
    free_fraction = np.count_nonzero(~blocked, axis=1) / LIDAR_NUM_RAYS
    if not np.array_equal(free_fraction, episode["free_fraction"]):
        sys.exit(f"{episode_dir}: drawn rays do not reproduce the free-space metric.")
    # ... and the robot drawn (trajectory.csv) must be the robot they were cast from (steps.csv).
    # (compared to 1 nm / 1 ns: the two CSV readers may round the last digit differently)
    if not (len(traj_t) == len(step_t)
            and np.allclose(traj_t, step_t, rtol=0, atol=1e-9)
            and np.allclose(traj["x"].to_numpy(), steps["x"], rtol=0, atol=1e-9)
            and np.allclose(traj["y"].to_numpy(), steps["y"], rtol=0, atol=1e-9)
            and np.allclose(traj["yaw"].to_numpy(), steps["yaw"], rtol=0, atol=1e-9)):
        sys.exit(f"{episode_dir}: trajectory.csv and steps.csv do not hold the same poses.")

    frame_times, duration = base.frame_schedule(traj_t, lidar["t_sim"], fps, speed)
    step_idx = base.latest_index(step_t, frame_times)

    fig = plt.figure(figsize=(14, 7))
    # No obstacle streams: the obstacles are drawn by the overlay, at the PPO-step time.
    panel_title, update_episode_panel = base.draw_episode_panel(
        fig, traj, step_log, step_t, wall_specs, {}, meta)
    ax = fig.axes[0]
    rays, radius_circle, update_overlay = draw_ray_overlay(
        ax, origins, ends, blocked, obstacle_tracks, show_obstacles)
    fig.canvas.draw()  # instantiate the left panel's ticks under the default style
    with plt.rc_context(base.PLOT_STYLE):
        update_metric_panel = base.draw_metric_panel(fig, episode)
        fig.canvas.draw()  # ... and the right panel's under the static plots' style

    def update(frame_i):
        t_now = frame_times[frame_i]
        si = step_idx[frame_i]
        update_episode_panel(t_now, si)
        update_overlay(si)
        current_si = np.searchsorted(step_t, t_now, side="right") - 1
        current_fps_text = f"   sensing={step_fps[current_si]:g} Hz" if current_si >= 0 else ""
        panel_title.set_text(f"episode {n_ep}   t={t_now:.1f}s (sim)"
                             f"{current_fps_text}   [{speed:g}x]")
        update_metric_panel(si)
        return ()

    return {
        "fig": fig, "update": update, "frame_times": frame_times, "duration": duration,
        "step_idx": step_idx, "rays": rays, "radius_circle": radius_circle,
        "episode": episode, "origins": origins, "ends": ends, "blocked": blocked,
    }


# ══════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Render a side-by-side MP4: episode video with ground-truth LiDAR rays "
                    "+ animated free-space plots.")
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
                        help="Hide recorded moving-obstacle positions (default: shown); "
                             "the rays still account for them")
    args = parser.parse_args()
    if args.speed <= 0:
        sys.exit("--speed must be > 0")

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH -- install it (e.g. apt-get install ffmpeg) and retry.")

    eval_dir = args.eval_dir.rstrip("/")
    output = args.output or os.path.join(
        eval_dir, f"episode_{args.n_ep:04d}", f"episode_{args.n_ep:04d}_free_space_rays.mp4")

    video = build_video(eval_dir, args.n_ep, args.fps, args.speed, args.obstacles)
    num_frames = len(video["frame_times"])

    anim = animation.FuncAnimation(video["fig"], video["update"], frames=num_frames, blit=False)
    anim.save(output, fps=args.fps, dpi=args.dpi, writer="ffmpeg")
    plt.close(video["fig"])
    print(f"wrote {output} ({num_frames} frames, {video['duration']:.1f}s episode at {args.speed:g}x "
          f"-> {num_frames / args.fps:.1f}s video @ {args.fps} fps)")


if __name__ == "__main__":
    main()
