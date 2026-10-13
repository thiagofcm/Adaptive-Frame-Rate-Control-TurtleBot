"""
plot_sensing_allocation.py
──────────────────────────
Grouped bar chart of Time Share (%) vs Sensing Share (%) per angular
obstruction level, from the one-row mean_sensing_alloc.csv written by
summarize_sensing_allocation.py (mean ± sample standard deviation over the
successful episodes).

The CSV is only read: nothing is recalculated and no per-episode data is
touched. The figure is saved next to it as plot_sensing_allocation.png.

Usage:
    python AdaptiveFPS/scripts/plot_sensing_allocation.py \
        --csv AdaptiveFPS/eval_stage14_50ep_seed_1000/adaptive_zeroshot/<run>/mean_sensing_alloc.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (registers the "science" style)

# ══════════════════════════════════════════════════════
# Defines
# ══════════════════════════════════════════════════════
OUTPUT_NAME = "plot_sensing_allocation.png"

# (CSV suffix, tick label) of each angular obstruction level.
LEVELS = [("low", "Low\n(0–25%)"), ("moderate", "Moderate\n(25–50%)"), ("high", "High\n(50–100%)")]

# (CSV prefix, legend label, color) of each bar series.
SERIES = [("pct_time", "Time Share", "tab:orange"), ("pct_frames", "Sensing Share", "tab:blue")]

REQUIRED_COLUMNS = [f"{prefix}_{level}{suffix}"
                    for prefix, _, _ in SERIES for level, _ in LEVELS for suffix in ("", "_std")]

# ── Style ──────────────────────────────────────────────
plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size":        18,
    "axes.labelsize":  22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 17,
})


# ══════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True,
                        help="mean_sensing_alloc.csv written by summarize_sensing_allocation.py")
    args = parser.parse_args()

    data = pd.read_csv(args.csv)
    if len(data) != 1:
        sys.exit(f"{args.csv}: expected exactly one data row, found {len(data)}.")
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        sys.exit(f"{args.csv}: missing required columns {missing}.")
    row = data.iloc[0]
    undefined = [column for column in REQUIRED_COLUMNS if pd.isna(row[column])]
    if undefined:
        sys.exit(f"{args.csv}: no value for {undefined}.")

    fig, ax = plt.subplots(figsize=(7, 5))

    positions = np.arange(len(LEVELS))
    bar_width = 0.36
    for i, (prefix, label, color) in enumerate(SERIES):
        means = [row[f"{prefix}_{level}"] for level, _ in LEVELS]
        stds = [row[f"{prefix}_{level}_std"] for level, _ in LEVELS]
        ax.bar(positions + (i - 0.5) * bar_width, means, bar_width, yerr=stds, label=label,
               color=color, edgecolor="black", linewidth=0.8,
               error_kw={"elinewidth": 1.3, "capsize": 4, "capthick": 1.3}, zorder=3)

    ax.set_xlabel("Angular Obstruction", labelpad=8)
    ax.set_ylabel("Share (%)")
    ax.set_xticks(positions)
    ax.set_xticklabels([label for _, label in LEVELS])
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.6, zorder=0)
    # Categories, not a scale: no minor ticks and no tick marks along x.
    ax.tick_params(axis="x", which="both", bottom=False, top=False)
    # Above the axes, so the legend can never cover a bar or an error bar.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(SERIES), frameon=False,
              columnspacing=1.5, handlelength=1.6)

    filename = os.path.join(os.path.dirname(os.path.abspath(args.csv)), OUTPUT_NAME)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")
