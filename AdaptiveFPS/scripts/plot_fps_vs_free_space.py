"""
plot_fps_vs_free_space.py
─────────────────────────
Combines the recorded sensing FPS trace (top) and the ground-truth local
angular obstruction (bottom) of one evaluation episode into one figure
with shared X axis.

The angular obstruction is 1 - free-space fraction, in percent: the share of
LiDAR ray directions that are blocked within FREE_SPACE_RADIUS, i.e. how constrained
the robot's surroundings are. Rays follow the simulated
LiDAR's exact angular pattern and mounting pose, but their lengths are
noise-free intersections with the instantaneous ground-truth scene (static
walls + moving obstacles) at every PPO timestep -- recorded LiDAR ranges
and delayed/gated observations are never used.

Offline and read-only: works exclusively on the files eval.py already
recorded (steps.csv, obstacles.npz, metadata.json, world_geometry.json).
The only file written is <episode dir>/fps_vs_free_space.png.

Usage:
    python AdaptiveFPS/scripts/plot_fps_vs_free_space.py \
        --eval-dir AdaptiveFPS/eval_stage14_50ep_seed_1000/adaptive_zeroshot/<run> \
        --n-ep 1
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt

# Validated geometry / data loading of the minimum-clearance plot, reused as is.
# Importing it also applies the shared plotting style (SciencePlots + rcParams).
from plot_fps_vs_min_distance import OBSTACLE_RADIUS, load_obstacle_tracks, load_steps, load_walls

# ══════════════════════════════════════════════════════
# Defines
# ══════════════════════════════════════════════════════
OUTPUT_NAME = "fps_vs_free_space.png"

# A ray counts as free when nothing blocks it within this distance of the LiDAR.
FREE_SPACE_RADIUS = 0.8  # meters

# LiDAR: sensor "hls_lfcd_lds" in models/turtlebot3_burger/model.sdf.
# Gazebo spreads <samples> rays from <min_angle> to <max_angle> with both
# endpoints included, counter-clockwise, measured from the robot's heading --
# so the increment is 6.28/39 rad (not 2*pi/40) and beam 39 lands 0.0032 rad
# short of beam 0.
LIDAR_NUM_RAYS  = 40
LIDAR_ANGLE_MIN = 0.0     # rad, beam 0 points along the robot's +x (forward)
LIDAR_ANGLE_MAX = 6.28    # rad
# Sensor <pose> in the base_footprint frame (the frame the recorded x/y/yaw
# refer to): 0.032 m behind the pose origin, no yaw offset.
LIDAR_OFFSET_X  = -0.032  # meters


# ══════════════════════════════════════════════════════
# Ground-truth raycasting
# ══════════════════════════════════════════════════════

def lidar_rays(x, y, yaw):
    """World-frame LiDAR origin (2,) and unit ray directions (n_rays, 2) at one robot pose."""
    origin = np.array([x + LIDAR_OFFSET_X * np.cos(yaw), y + LIDAR_OFFSET_X * np.sin(yaw)])
    angles = yaw + np.linspace(LIDAR_ANGLE_MIN, LIDAR_ANGLE_MAX, LIDAR_NUM_RAYS)
    return origin, np.column_stack([np.cos(angles), np.sin(angles)])


def raycast(origin, directions, walls, obstacle_centers):
    """Distance along each ray to the first wall or moving obstacle (inf if none)."""
    distances = np.full(len(directions), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for wall in walls:
            for i in range(4):
                a, b = wall[i], wall[(i + 1) % 4]
                edge = b - a
                denom = directions[:, 0] * edge[1] - directions[:, 1] * edge[0]
                t = ((a[0] - origin[0]) * edge[1] - (a[1] - origin[1]) * edge[0]) / denom
                u = ((a[0] - origin[0]) * directions[:, 1] - (a[1] - origin[1]) * directions[:, 0]) / denom
                hit = (t > 0) & (u >= 0) & (u <= 1) & (t < distances)
                distances = np.where(hit, t, distances)
        for center in obstacle_centers:
            to_origin = origin - center
            b = directions @ to_origin
            disc = b ** 2 - (to_origin @ to_origin - OBSTACLE_RADIUS ** 2)
            t = -b - np.sqrt(np.where(disc >= 0, disc, np.nan))
            hit = (disc >= 0) & (t > 0) & (t < distances)
            distances = np.where(hit, t, distances)
    return distances


def free_space_fraction(steps, walls, obstacle_tracks):
    """Per-step fraction of LiDAR rays unobstructed within FREE_SPACE_RADIUS."""
    n_steps = len(steps["ppo_step"])
    fraction = np.empty(n_steps)
    for k in range(n_steps):
        origin, directions = lidar_rays(steps["x"][k], steps["y"][k], steps["yaw"][k])
        distances = raycast(origin, directions, walls, obstacle_tracks[:, k, :])
        fraction[k] = np.count_nonzero(distances >= FREE_SPACE_RADIUS) / LIDAR_NUM_RAYS
    return fraction


def load_episode(eval_dir, episode_num):
    episode_dir = os.path.join(eval_dir, f"episode_{episode_num:04d}")
    if not os.path.isdir(episode_dir):
        raise SystemExit(f"Episode directory not found: {episode_dir}")

    with open(os.path.join(episode_dir, "metadata.json")) as f:
        fps_choices = json.load(f)["environment_fps_choices"]

    steps = load_steps(episode_dir)
    walls = load_walls(eval_dir)
    obstacle_tracks = load_obstacle_tracks(episode_dir, steps["t_sim"])

    return {
        "episode_dir":   episode_dir,
        "fps_choices":   fps_choices,
        "timesteps":     steps["ppo_step"],
        "current_fps":   steps["current_fps"],
        "free_fraction": free_space_fraction(steps, walls, obstacle_tracks),
        "outcome_str":   steps["outcome_str"],
    }


# ══════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════

def plot_combined(episode, episode_num, filename):

    timesteps = episode["timesteps"]
    fps_choices = episode["fps_choices"]

    fig, (ax_fps, ax_free) = plt.subplots(
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

    # Plotted inverted (1 - free-space fraction), in percent: higher = more constrained surroundings.
    ax_free.plot(timesteps, 100.0 * (1.0 - episode["free_fraction"]), linewidth=7.0, color="tab:orange")
    ax_free.set_xlabel("Timesteps", labelpad=10)
    ax_free.set_ylabel("Angular\nObstruction (%)", labelpad=30)
    ax_free.grid(True, alpha=0.9)
    ax_free.set_yticks([0, 25, 50, 75, 100])
    ax_free.set_ylim(-2, 102)  # 0-100 range, padded so the thick line is not clipped at the limits

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

    obstruction = 100.0 * (1.0 - episode["free_fraction"])
    print(f"Episode {args.n_ep}: {len(episode['timesteps'])} steps, outcome {episode['outcome_str']}, "
          f"angular obstruction within {FREE_SPACE_RADIUS:g} m: "
          f"min {obstruction.min():.1f}%, mean {obstruction.mean():.1f}%, max {obstruction.max():.1f}%")

    plot_combined(episode, args.n_ep, os.path.join(output_dir, OUTPUT_NAME))
