"""
plot_fps_vs_min_distance.py
───────────────────────────
Combines the recorded sensing FPS trace (top) and the reconstructed
minimum obstacle clearance (bottom) of one evaluation episode into one
figure with shared X axis.

Offline and read-only: works exclusively on the files eval.py already
recorded (steps.csv, obstacles.npz, metadata.json, world_geometry.json).
The only file written is <episode dir>/fps_vs_min_distance.png.

Usage:
    python AdaptiveFPS/scripts/plot_fps_vs_min_distance.py \
        --eval-dir AdaptiveFPS/eval_stage14_50ep_seed_1/<run> \
        --n-ep 10
"""

import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (registers the "science" style)

# ══════════════════════════════════════════════════════
# Defines
# ══════════════════════════════════════════════════════
OUTPUT_NAME = "fps_vs_min_distance.png"

# Moving obstacles: cylinder collision, radius from
# models/turtlebot3_drl_world/obstacle{1,2,3}/model.sdf (== settings.OBSTACLE_RADIUS).
OBSTACLE_RADIUS = 0.16  # meters

# Robot footprint: plan-view projection of the collision shapes in
# models/turtlebot3_burger/model.sdf, as (cx, cy, half_x, half_y) rectangles
# in the base_footprint frame (the frame the recorded x/y/yaw refer to).
# The lidar cylinder (r=0.0508 at x=-0.032) and the caster (r=0.005 at
# x=-0.081) lie inside the base box, so they do not extend the footprint.
ROBOT_FOOTPRINT = [
    (-0.032,  0.00, 0.070, 0.070),  # base_collision: box 0.140 x 0.140
    ( 0.000,  0.08, 0.033, 0.009),  # wheel_left_collision: cylinder r=0.033, length=0.018
    ( 0.000, -0.08, 0.033, 0.009),  # wheel_right_collision
]

# ── Style ──────────────────────────────────────────────
plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "figure.titlesize": 60,  # Figure title size.
    "font.size":        30,   # Default font size for all text elements unless overridden later.
    "axes.labelsize":  56,   # Controls axis label sizes: x-axis and y-axis labels.
    "xtick.labelsize": 50,   # Controls the size of tick numbers on the x-axis.
    "ytick.labelsize": 50,   # Controls the size of tick numbers on the y-axis.
    "legend.fontsize": 27,
    "figure.dpi":      300, # Controls figure resolution (Dots Per Inch).
})


# ══════════════════════════════════════════════════════
# Geometry
# ══════════════════════════════════════════════════════

def rect_corners(cx, cy, half_x, half_y, yaw):
    """World-frame corners (4, 2) of an oriented rectangle."""
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([[half_x, half_y], [-half_x, half_y], [-half_x, -half_y], [half_x, -half_y]])
    return local @ np.array([[c, s], [-s, c]]) + [cx, cy]


def points_to_segment(points, a, b):
    """Distance from each point (n, 2) to the segment a-b."""
    d = b - a
    t = np.clip(((points - a) @ d) / (d @ d), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * d), axis=1)


def points_inside(points, poly):
    """Which points (n, 2) lie inside the convex polygon poly (4, 2)."""
    cross = []
    for i in range(4):
        a, b = poly[i], poly[(i + 1) % 4]
        cross.append((b[0] - a[0]) * (points[:, 1] - a[1]) - (b[1] - a[1]) * (points[:, 0] - a[0]))
    cross = np.array(cross)
    return np.all(cross >= 0, axis=0) | np.all(cross <= 0, axis=0)


def rect_to_rect(a, b):
    """Boundary-to-boundary distance between two convex rectangles (0 if they overlap)."""
    if points_inside(a, b).any() or points_inside(b, a).any():
        return 0.0
    dist = np.inf
    for i in range(4):
        dist = min(dist,
                   points_to_segment(a, b[i], b[(i + 1) % 4]).min(),
                   points_to_segment(b, a[i], a[(i + 1) % 4]).min())
    return dist


def rect_to_circle(rect, center, radius):
    """Boundary-to-boundary distance between a rectangle and a circle (0 if they overlap)."""
    center = np.asarray(center)[None]
    if points_inside(center, rect)[0]:
        return 0.0
    dist = min(points_to_segment(center, rect[i], rect[(i + 1) % 4])[0] for i in range(4))
    return max(dist - radius, 0.0)


def robot_footprint(x, y, yaw):
    """World-frame corner arrays of the robot's footprint rectangles at one pose."""
    c, s = np.cos(yaw), np.sin(yaw)
    return [rect_corners(x + c * px - s * py, y + s * px + c * py, hx, hy, yaw)
            for px, py, hx, hy in ROBOT_FOOTPRINT]


# ══════════════════════════════════════════════════════
# Offline reconstruction
# ══════════════════════════════════════════════════════

def load_walls(eval_dir):
    with open(os.path.join(eval_dir, "world_geometry.json")) as f:
        geometry = json.load(f)
    return [rect_corners(w["cx"], w["cy"], w["size_x"] / 2.0, w["size_y"] / 2.0, w["yaw"])
            for w in geometry["walls"]]


def load_steps(episode_dir):
    with open(os.path.join(episode_dir, "steps.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        "ppo_step":    np.array([int(r["ppo_step"]) for r in rows]),
        "t_sim":       np.array([float(r["t_sim_s"]) for r in rows]),
        "x":           np.array([float(r["x"]) for r in rows]),
        "y":           np.array([float(r["y"]) for r in rows]),
        "yaw":         np.array([float(r["yaw"]) for r in rows]),
        "current_fps": np.array([float(r["current_fps"]) for r in rows]),
        "outcome_str": rows[-1]["outcome_str"],
    }


def load_obstacle_tracks(episode_dir, t_query):
    """Moving-obstacle centers (n_obstacles, n_steps, 2) at the steps.csv sim
    times. Obstacle poses are recorded at 50 Hz on the same sim clock and
    episode origin as steps.csv, but rarely on the exact same stamp, so each
    track is linearly interpolated (end values held)."""
    with np.load(os.path.join(episode_dir, "obstacles.npz")) as data:
        t_sim, frame_id, x, y = data["t_sim"], data["frame_id"], data["x"], data["y"]
    tracks = []
    for fid in sorted(set(frame_id)):
        mask = (frame_id == fid) & np.isfinite(t_sim)
        tracks.append(np.column_stack([np.interp(t_query, t_sim[mask], x[mask]),
                                       np.interp(t_query, t_sim[mask], y[mask])]))
    return np.array(tracks).reshape(len(tracks), len(t_query), 2)


def reconstruct_clearance(steps, walls, obstacle_tracks):
    """Per-step footprint clearance to the static walls and to the moving
    obstacles, measured boundary to boundary."""
    n_steps = len(steps["ppo_step"])
    wall_clearance = np.full(n_steps, np.inf)
    obstacle_clearance = np.full(n_steps, np.inf)
    for k in range(n_steps):
        for part in robot_footprint(steps["x"][k], steps["y"][k], steps["yaw"][k]):
            for wall in walls:
                wall_clearance[k] = min(wall_clearance[k], rect_to_rect(part, wall))
            for track in obstacle_tracks:
                obstacle_clearance[k] = min(obstacle_clearance[k],
                                            rect_to_circle(part, track[k], OBSTACLE_RADIUS))
    return wall_clearance, obstacle_clearance


def load_episode(eval_dir, episode_num):
    episode_dir = os.path.join(eval_dir, f"episode_{episode_num:04d}")
    if not os.path.isdir(episode_dir):
        raise SystemExit(f"Episode directory not found: {episode_dir}")

    with open(os.path.join(episode_dir, "metadata.json")) as f:
        fps_choices = json.load(f)["environment_fps_choices"]

    steps = load_steps(episode_dir)
    walls = load_walls(eval_dir)
    obstacle_tracks = load_obstacle_tracks(episode_dir, steps["t_sim"])
    wall_clearance, obstacle_clearance = reconstruct_clearance(steps, walls, obstacle_tracks)

    return {
        "episode_dir":        episode_dir,
        "fps_choices":        fps_choices,
        "timesteps":          steps["ppo_step"],
        "current_fps":        steps["current_fps"],
        "wall_clearance":     wall_clearance,
        "obstacle_clearance": obstacle_clearance,
        "min_clearance":      np.minimum(wall_clearance, obstacle_clearance),
        "outcome_str":        steps["outcome_str"],
    }


# ══════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════

def plot_combined(episode, episode_num, filename):

    timesteps = episode["timesteps"]
    fps_choices = episode["fps_choices"]

    fig, (ax_fps, ax_dist) = plt.subplots(
        2, 1, figsize=(15, 18), sharex=True,
        gridspec_kw={"hspace": 0.08}
    )

    ax_fps.plot(timesteps, episode["current_fps"], linewidth=7.0, drawstyle="steps-post",
                label="Current FPS", color="tab:blue")
    ax_fps.set_ylabel("Sens. Rate (Hz)", labelpad=16)
    ax_fps.grid(True, alpha=0.9)
    for fps in fps_choices:
        ax_fps.axhline(y=fps, linestyle=":", linewidth=3.5, alpha=0.9, color="gray",
                       label="Fixed FPS baselines" if fps == fps_choices[0] else None)
    # Log axis: the sensing rates span 0.2-10 Hz, so all of them get an evenly readable tick.
    ax_fps.set_yscale("log")
    ax_fps.set_yticks(fps_choices)
    ax_fps.set_yticklabels([f"{fps:g}" for fps in fps_choices], fontsize=40)
    ax_fps.minorticks_off()
    ax_fps.set_ylim(min(fps_choices) / 1.4, max(fps_choices) * 1.4)

    ax_dist.plot(timesteps, episode["min_clearance"], linewidth=7.0, color="tab:orange")
    ax_dist.set_xlabel("Timesteps", labelpad=10)
    ax_dist.set_ylabel("Min Obstacle Dist (m)", labelpad=30)
    ax_dist.grid(True, alpha=0.9)
    ax_dist.axhline(y=0, linestyle="--", linewidth=1.5, color="black",
                    alpha=0.8, label="Contact (clearance = 0)")

    fig.suptitle(f"Episode {episode_num} | {episode['outcome_str']}", y=0.94)

    plt.tight_layout(rect=[0, 0, 1, 0.86])

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


# ══════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir",   type=str, required=True,
                        help="Run directory holding episode_NNNN/ and world_geometry.json")
    parser.add_argument("--n-ep",       type=int, required=True,
                        help="Episode number to plot (N -> episode_NNNN only)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Save under <output-dir>/episode_NNNN/ instead of the episode directory")
    args = parser.parse_args()

    episode = load_episode(args.eval_dir, args.n_ep)

    if args.output_dir is None:
        output_dir = episode["episode_dir"]
    else:
        output_dir = os.path.join(args.output_dir, f"episode_{args.n_ep:04d}")
        os.makedirs(output_dir, exist_ok=True)

    print(f"Episode {args.n_ep}: {len(episode['timesteps'])} steps, outcome {episode['outcome_str']}, "
          f"min clearance {episode['min_clearance'].min():.3f} m "
          f"(walls {episode['wall_clearance'].min():.3f} m, "
          f"moving obstacles {episode['obstacle_clearance'].min():.3f} m)")

    plot_combined(episode, args.n_ep, os.path.join(output_dir, OUTPUT_NAME))
