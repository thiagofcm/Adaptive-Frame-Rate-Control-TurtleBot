"""
plot_success_vs_sensing.py
──────────────────────────
Scatter plot of navigation success rate against sensing consumption (mean
fresh LiDAR observations per episode) for the fixed-FPS baselines and the
most successful adaptive policies of an evaluation, from the summary.csv
written by eval.py (one row per run, statistics over all its episodes).

Fixed baselines are gray dots labelled F<rate> (F5 = fixed 5 Hz), adaptive policies
orange dots labelled with how they were obtained; the adaptive ones are
ranked by success rate (ties: fewer observations first). Policies toward the
upper left reach a higher success rate with fewer observations.

By default the policies are evenly spaced along x, in order of observations,
and each tick gives its policy's actual value -- that axis is not to scale.
--x-spacing log puts every policy at its true position on a log axis instead.

The CSV is only read: values are plotted as reported and nothing is
recalculated. The figure is saved next to the CSV
as plot_success_vs_sensing.png.

Usage:
    python AdaptiveFPS/scripts/plot_success_vs_sensing.py \
        --csv AdaptiveFPS/eval_stage14_50ep_seed_1000/summary.csv
    python AdaptiveFPS/scripts/plot_success_vs_sensing.py --csv <summary.csv> --n-adaptive 5
    python AdaptiveFPS/scripts/plot_success_vs_sensing.py --csv <summary.csv> --exclude-fixed   # all baselines

By default the 0.2 Hz and 0.5 Hz baselines are left out (--exclude-fixed): they
never succeed, and without a policy at 0 % the success axis can be logarithmic.
"""

import argparse
import os
import re
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator
import scienceplots  # noqa: F401  (registers the "science" style)

# ══════════════════════════════════════════════════════
# Defines
# ══════════════════════════════════════════════════════
OUTPUT_NAME = "plot_success_vs_sensing.png"

REQUIRED_COLUMNS = ["run_name", "success_rate", "mean_fresh_observations"]

FIXED_COLOR = "#B5B5B5"           # muted light gray
FIXED_LABEL_COLOR = "0.35"
ADAPTIVE_COLOR = "#F2A65A"        # soft orange
ADAPTIVE_LABEL_COLOR = "#C56A12"  # same hue, dark enough to read as text

MARKER_SIZE = 110       # points^2
ANNOTATION_SIZE = 14    # points

# Where each adaptive policy's label goes, by rank: (offset in points, ha, va).
# The best policies tend to sit almost on top of each other, so consecutive
# labels fan out to different sides and a thin line links each to its dot.
ADAPTIVE_LABEL_SPOTS = [
    ((-9, 0), "right", "center"),   # right beside the dot: no link needed
    ((0, -9), "center", "top"),     # right below the dot
    ((-16, -17), "right", "top"),
    ((16, 17), "left", "bottom"),
    ((-34, 0), "right", "center"),
    ((34, 0), "left", "center"),
]

# ── Style ──────────────────────────────────────────────
plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size":        16,
    "axes.labelsize":  20,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
})


# ══════════════════════════════════════════════════════
# Policy identification
# ══════════════════════════════════════════════════════

def fixed_fps(run_name):
    """FPS of a fixed-rate baseline (run folder 'fixed_<rate>Hz'), None for any other run."""
    match = re.fullmatch(r"fixed_(\d+(?:\.\d+)?)Hz", os.path.basename(run_name))
    return float(match.group(1)) if match else None


def adaptive_kind(run_name):
    """How an adaptive policy was obtained."""
    return "Transfer" if "transfer" in run_name else "Zero-shot" if "zeroshot" in run_name else "Adaptive"


def adaptive_labels(run_names):
    """Short label of each adaptive policy shown: its kind, plus its frame cost
    whenever the kind alone would not tell two of them apart."""
    kinds = [adaptive_kind(name) for name in run_names]
    labels = []
    for name, kind in zip(run_names, kinds):
        frame_cost = re.search(r"fc_?(\d+(?:\.\d+)?)", os.path.basename(name))
        labels.append(f"{kind} ({frame_cost.group(1)})" if kinds.count(kind) > 1 and frame_cost else kind)
    return labels


# ══════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True,
                        help="summary.csv written by eval.py (one row per evaluated run)")
    parser.add_argument("--n-adaptive", type=int, default=2,
                        help="Number of adaptive policies shown, most successful first (default: 2)")
    parser.add_argument("--exclude-fixed", type=float, nargs="*", default=[0.2, 0.5],
                        help="Fixed baselines left out, by rate in Hz (default: 0.2 0.5, which never "
                             "succeed; pass the flag alone to show every baseline)")
    parser.add_argument("--x-spacing", choices=["even", "log"], default="even",
                        help="even (default): policies evenly spaced in order of observations, each "
                             "tick showing its actual value; log: true positions on a log axis")
    args = parser.parse_args()
    if args.n_adaptive < 0:
        sys.exit("--n-adaptive must be >= 0")

    data = pd.read_csv(args.csv)
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        sys.exit(f"{args.csv}: missing required columns {missing}.")
    incomplete = data[data[REQUIRED_COLUMNS].isna().any(axis=1)]["run_name"].tolist()
    if incomplete:
        sys.exit(f"{args.csv}: missing values for {incomplete}.")

    # eval.py reports success_rate in percent. A file holding fractions instead
    # (nothing above 1) is converted, so the axis always reads 0-100 %.
    success = data["success_rate"].astype(float)
    if success.max() <= 1.0 and success.max() > 0.0:
        print("success_rate holds fractions (0-1): shown as percentages.")
        success = success * 100.0
    if success.min() < 0.0 or success.max() > 100.0:
        sys.exit(f"{args.csv}: success_rate outside 0-100 %.")
    data = data.assign(success_pct=success, fixed_fps=data["run_name"].map(fixed_fps))

    fixed = data[data["fixed_fps"].notna()].sort_values("fixed_fps")
    excluded = fixed[fixed["fixed_fps"].isin(args.exclude_fixed)]
    if len(excluded):
        print("Fixed baselines left out: "
              + ", ".join(f"{run['fixed_fps']:g} Hz ({run['success_pct']:g} % success)"
                          for _, run in excluded.iterrows()))
    fixed = fixed[~fixed["fixed_fps"].isin(args.exclude_fixed)]
    adaptive = (data[data["fixed_fps"].isna()]
                .sort_values(["success_pct", "mean_fresh_observations"], ascending=[False, True])
                .head(args.n_adaptive))
    if fixed.empty:
        sys.exit(f"{args.csv}: no fixed-FPS baseline (run named fixed_<rate>Hz) left to show.")
    if len(adaptive) < args.n_adaptive:
        print(f"Only {len(adaptive)} adaptive policies in the CSV.")

    # Evenly spaced policies need less width than a to-scale axis, which packs them closer.
    fig, ax = plt.subplots(figsize=(5.4 if args.x_spacing == "even" else 7, 2.5))

    labels = adaptive_labels(adaptive["run_name"].tolist())
    for rank, ((_, run), label) in enumerate(zip(adaptive.iterrows(), labels)):
        print(f"Adaptive #{rank + 1} ({label}): {run['run_name']}  "
              f"success {run['success_pct']:g} %, {run['mean_fresh_observations']:g} fresh observations")

    ax.set_xlabel("Mean Fresh Observations per Episode")
    ax.set_ylabel("Success\nRate (%)")
    if args.x_spacing == "even":
        # Policies in order of observations, one evenly spaced slot each: the
        # axis is NOT to scale, so every tick states its policy's actual value.
        shown = pd.concat([
            fixed.assign(label=[f"F{fps:g}" for fps in fixed["fixed_fps"]],
                         color=FIXED_COLOR, label_color=FIXED_LABEL_COLOR),
            adaptive.assign(label=labels, color=ADAPTIVE_COLOR, label_color=ADAPTIVE_LABEL_COLOR),
        ]).sort_values("mean_fresh_observations", kind="stable").reset_index(drop=True)
        for slot, run in shown.iterrows():
            ax.scatter(slot, run["success_pct"], s=MARKER_SIZE, marker="o", color=run["color"],
                       edgecolor="white", linewidth=0.8, zorder=3, clip_on=False)
            ax.annotate(run["label"], (slot, run["success_pct"]), xytext=(0, 10),
                        textcoords="offset points", ha="center", va="bottom",
                        color=run["label_color"], fontsize=ANNOTATION_SIZE)
        ax.set_xticks(range(len(shown)))
        ax.set_xticklabels([f"{observations:.0f}" for observations in shown["mean_fresh_observations"]])
        ax.set_xlim(-0.5, len(shown) - 0.5)
        ax.xaxis.set_minor_locator(NullLocator())
    else:
        # Fixed baselines: gray dots, rate written above.
        ax.scatter(fixed["mean_fresh_observations"], fixed["success_pct"], s=MARKER_SIZE, marker="o",
                   color=FIXED_COLOR, edgecolor="white", linewidth=0.8, zorder=3, clip_on=False)
        # Neighbours at the same success rate that sit too close for two centered
        # labels get theirs pushed apart (the dots themselves never move).
        fixed_x = fixed["mean_fresh_observations"].to_numpy()
        fixed_y = fixed["success_pct"].to_numpy()
        alignment = ["center"] * len(fixed)
        for i in range(len(fixed) - 1):
            if fixed_x[i + 1] / fixed_x[i] < 2.0 and abs(fixed_y[i + 1] - fixed_y[i]) < 10:
                alignment[i], alignment[i + 1] = "right", "left"
        label_shift = {"center": 0, "right": 9, "left": -9}
        for x, y, fps, ha in zip(fixed_x, fixed_y, fixed["fixed_fps"], alignment):
            ax.annotate(f"F{fps:g}", (x, y), xytext=(label_shift[ha], 8), textcoords="offset points",
                        ha=ha, va="bottom", color=FIXED_LABEL_COLOR, fontsize=ANNOTATION_SIZE)

        # Adaptive policies: orange dots at their true coordinates. The white rim
        # keeps near-coincident dots apart; each label is set off to its own side.
        ax.scatter(adaptive["mean_fresh_observations"], adaptive["success_pct"], s=MARKER_SIZE, marker="o",
                   color=ADAPTIVE_COLOR, edgecolor="white", linewidth=0.8, zorder=4, clip_on=False)
        for rank, ((_, run), label) in enumerate(zip(adaptive.iterrows(), labels)):
            offset, ha, va = ADAPTIVE_LABEL_SPOTS[rank % len(ADAPTIVE_LABEL_SPOTS)]
            ax.annotate(label, (run["mean_fresh_observations"], run["success_pct"]),
                        xytext=offset, textcoords="offset points", ha=ha, va=va,
                        color=ADAPTIVE_LABEL_COLOR, fontsize=ANNOTATION_SIZE,
                        arrowprops=None if max(abs(offset[0]), abs(offset[1])) < 12 else
                        {"arrowstyle": "-", "color": ADAPTIVE_COLOR, "linewidth": 1.0,
                         "shrinkA": 1, "shrinkB": 4.5})

        # Log axis: the baselines span almost two decades of observations, and the
        # low-rate ones would collapse onto each other on a linear axis.
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter("{x:g}")
        shown = pd.concat([fixed, adaptive])["mean_fresh_observations"]
        # Just enough room for the outermost labels (and, with baselines sharing a
        # success rate at the low end, for the one pushed to the left).
        ax.set_xlim(shown.min() / (1.9 if "right" in alignment else 1.3), shown.max() * 1.22)
        # A few round ticks: 1-2-5 steps over a short range, 1-3 steps over a long one.
        steps = (1, 2, 5) if ax.get_xlim()[1] / ax.get_xlim()[0] < 30 else (1, 3)
        ax.xaxis.set_major_locator(FixedLocator(
            [step * 10 ** exponent for exponent in range(0, 5) for step in steps
             if ax.get_xlim()[0] <= step * 10 ** exponent <= ax.get_xlim()[1]]))
    shown_success = pd.concat([fixed, adaptive])["success_pct"]
    if shown_success.min() > 0:
        # Log axis, possible once no policy shown sits at 0 %. The headroom
        # above 100 % is for the labels over the best policies.
        ax.set_yscale("log")
        ax.set_ylim(shown_success.min() / 1.22, 138)
        ax.yaxis.set_major_locator(FixedLocator([t for t in (1, 2, 5, 10, 25, 50, 100)
                                                 if t >= shown_success.min() / 1.22]))
        ax.yaxis.set_major_formatter("{x:g}")
    else:
        ax.set_ylim(-11, 124)                           # ... and for the ones above the 94 % dots
        ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(True, axis="y", alpha=0.35, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_bounds(ax.get_ylim()[0], 100)
    ax.tick_params(which="both", top=False, right=False)

    filename = os.path.join(os.path.dirname(os.path.abspath(args.csv)), OUTPUT_NAME)
    plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close()
    print(f"Saved: {filename}")
