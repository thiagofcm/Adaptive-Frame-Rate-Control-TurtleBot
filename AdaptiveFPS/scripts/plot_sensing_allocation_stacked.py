"""
plot_sensing_allocation_stacked.py
──────────────────────────────────
Two horizontal 100% stacked bars -- Time Distribution and Sensing Allocation -- each split
by angular obstruction level, from the one-row mean_sensing_alloc.csv written
by summarize_sensing_allocation.py (means over the successful episodes).

The CSV is only read: nothing is recalculated and no per-episode data is
touched. The figure is saved next to it as plot_sensing_allocation_stacked.png.

Usage:
    python AdaptiveFPS/scripts/plot_sensing_allocation_stacked.py \
        --csv AdaptiveFPS/eval_stage14_50ep_seed_1000/adaptive_zeroshot/<run>/mean_sensing_alloc.csv
"""

import argparse
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (registers the "science" style)

# ══════════════════════════════════════════════════════
# Defines
# ══════════════════════════════════════════════════════
OUTPUT_NAME = "plot_sensing_allocation_stacked.png"

# (CSV suffix, legend label, segment color) of each angular obstruction level.
LEVELS = [
    ("low",      "Low",      "#96CBC7"),  # muted teal
    ("moderate", "Moderate", "#F6BC83"),  # soft orange
    ("high",     "High",     "#C69BB9"),  # muted purple
]
# Black reads well on all three fills (contrast ratio 6:1 or more); white does not.
VALUE_COLOR = "black"

# (CSV prefix, bar label) of each stacked bar, top to bottom.
BARS = [("pct_time", "Time\nDist."), ("pct_frames", "Sensing\nAllocation")]

REQUIRED_COLUMNS = [f"{prefix}_{level}" for prefix, _ in BARS for level, _, _ in LEVELS]

# The CSV holds means rounded to 4 decimals, so a bar may miss 100 % by a rounding step.
SUM_TOLERANCE = 0.01  # percentage points

# ── Style ──────────────────────────────────────────────
plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size":        18,
    "axes.labelsize":  22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 20,
    "legend.fontsize": 16,
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
    for prefix, label in BARS:
        total = sum(row[f"{prefix}_{level}"] for level, _, _ in LEVELS)
        if abs(total - 100.0) > SUM_TOLERANCE:
            sys.exit(f"{args.csv}: the {' '.join(label.split())} percentages sum to {total:.4f}, not 100.")

    fig, ax = plt.subplots(figsize=(7, 1.9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bar_height = 0.72
    for position, (prefix, _) in enumerate(BARS):
        left = 0.0
        for level, label, color in LEVELS:
            value = row[f"{prefix}_{level}"]
            ax.barh(position, value, bar_height, left=left, color=color, linewidth=0,
                    label=label if position == 0 else None)
            ax.text(left + value / 2.0, position, f"{value:.1f}%", ha="center", va="center",
                    color=VALUE_COLOR, fontsize=18)
            left += value

    # Minimalist frame: only the bars, their names and the legend.
    ax.set_xlim(0, 100)
    ax.set_ylim(len(BARS) - 1 + bar_height / 2.0, -bar_height / 2.0)  # first bar on top, no padding
    ax.xaxis.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    ax.set_yticks(range(len(BARS)))
    ax.set_yticklabels([label for _, label in BARS], ha="center", multialignment="center")
    # Both names centered on one common vertical line, left of the bars.
    fig.canvas.draw()
    label_width = max(label.get_window_extent().width for label in ax.get_yticklabels()) * 72.0 / fig.dpi
    ax.tick_params(axis="y", which="both", left=False, right=False, pad=8 + label_width / 2.0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(LEVELS), frameon=False,
              columnspacing=1.6, handlelength=1.2, handletextpad=0.5, borderaxespad=0.3)

    filename = os.path.join(os.path.dirname(os.path.abspath(args.csv)), OUTPUT_NAME)
    plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close()
    print(f"Saved: {filename}")
