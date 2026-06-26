"""
Explore and visualise the .fcsv annotations (no CBCT needed).

Shows the 2D arch shape from annotations, useful for sanity-checking data.

Usage:
    python scripts/explore_annotations.py
    python scripts/explore_annotations.py --case ToothFairy2F_001_0000
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ANNOTATIONS_DIR
from spline.fcsv_io import load_fcsv
from spline.spline_utils import fit_spline, order_points_along_arch


def plot_annotation(ann: dict, ax: plt.Axes, title: str = "") -> None:
    pts_lps = ann["points_lps"]
    xy = pts_lps[:, :2]  # x=L→R, y=P→A

    ordered = order_points_along_arch(xy)
    curve, _ = fit_spline(ordered, n_eval=300)

    ax.plot(curve[:, 0], curve[:, 1], color="salmon", linewidth=2)
    ax.scatter(xy[:, 0], xy[:, 1], c="tomato", s=40, zorder=3)
    for i, (x, y) in enumerate(xy):
        ax.text(x, y + 1.5, str(i + 1), fontsize=6, ha="center", color="black")
    ax.set_title(title or ann["case_id"], fontsize=8)
    ax.set_aspect("equal")
    ax.invert_yaxis()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fcsv_dir", default=str(ANNOTATIONS_DIR))
    p.add_argument("--case", default=None, help="Specific case stem to display")
    p.add_argument("--n", type=int, default=9, help="Number of cases to show in grid")
    args = p.parse_args()

    fcsv_dir = Path(args.fcsv_dir)

    if args.case:
        path = fcsv_dir / f"{args.case}.fcsv"
        if not path.exists():
            print(f"Not found: {path}")
            sys.exit(1)
        ann = load_fcsv(path)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        plot_annotation(ann, ax, title=ann["case_id"])
        plt.tight_layout()
        plt.show()
        return

    # Show grid of N cases
    paths = sorted(fcsv_dir.glob("*.fcsv"))[: args.n]
    ncols = 3
    nrows = (len(paths) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax, path in zip(axes, paths):
        try:
            ann = load_fcsv(path)
            plot_annotation(ann, ax, title=ann["case_id"])
        except Exception as e:
            ax.set_title(f"Error: {e}", fontsize=7)

    for ax in axes[len(paths):]:
        ax.axis("off")

    plt.suptitle("Dental Arch Annotations (LPS x-y plane)", fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
