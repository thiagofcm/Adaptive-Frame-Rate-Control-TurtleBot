import argparse
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--npz", required=True, help="Input LiDAR NPZ file")
parser.add_argument("--out", help="Output CSV; defaults to the same filename")
args = parser.parse_args()

input_path = Path(args.npz)
output_path = Path(args.out) if args.out else input_path.with_suffix(".csv")

with np.load(input_path, allow_pickle=False) as data:
    ranges = data["ranges"]
    t_sim = data["t_sim"]
    t_wall = data["t_wall"]

    if ranges.ndim != 2:
        raise ValueError(f"Expected a 2D ranges array, got {ranges.shape}")
    if not (len(t_sim) == len(t_wall) == len(ranges)):
        raise ValueError("Timestamp and scan counts do not match")

    scan_index = np.arange(len(ranges))
    table = np.column_stack((scan_index, t_sim, t_wall, ranges))

    header = ["scan_index", "t_sim_s", "t_wall_s"]
    header += [f"beam_{i:02d}" for i in range(ranges.shape[1])]

    np.savetxt(
        output_path,
        table,
        delimiter=",",
        header=",".join(header),
        comments="",
        fmt=["%d", "%.9f", "%.9f"] + ["%.6f"] * ranges.shape[1],
    )

print(f"Saved {len(ranges)} scans to {output_path}")