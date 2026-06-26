"""Spline fitting and evaluation utilities."""

from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.interpolate import splev, splprep
from scipy.spatial.distance import cdist


def fit_spline(
    points: npt.NDArray[np.float64],
    n_eval: int = 300,
    smoothing: float = 0.0,
    degree: int = 3,
) -> tuple[npt.NDArray[np.float64], tuple]:
    """
    Fit a parametric B-spline through ordered 2D/3D control points.

    Returns:
        curve_points: (n_eval, D) dense curve samples
        tck: scipy spline representation for further use
    """
    pts = np.array(points, dtype=np.float64)
    if len(pts) < degree + 1:
        raise ValueError(f"Need at least {degree + 1} points, got {len(pts)}")

    coords = [pts[:, i] for i in range(pts.shape[1])]
    tck, _ = splprep(coords, s=smoothing, k=degree)
    u_new = np.linspace(0, 1, n_eval)
    curve = np.array(splev(u_new, tck)).T
    return curve, tck


def order_points_along_arch(
    points: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Order unordered 2D keypoints into a smooth arch sequence.

    Uses a greedy nearest-neighbour walk starting from the leftmost point,
    which works well for the dental arch U-shape.
    """
    pts = np.array(points, dtype=np.float64)
    n = len(pts)
    if n <= 2:
        return pts

    # Start from the point with minimum x (left side of arch)
    start = int(np.argmin(pts[:, 0]))
    ordered = [start]
    remaining = set(range(n)) - {start}

    while remaining:
        current = ordered[-1]
        dists = {j: np.linalg.norm(pts[current] - pts[j]) for j in remaining}
        nearest = min(dists, key=dists.get)
        ordered.append(nearest)
        remaining.remove(nearest)

    return pts[ordered]


def resample_spline_to_n_points(
    curve_points: npt.NDArray[np.float64],
    n: int,
) -> npt.NDArray[np.float64]:
    """Uniformly resample a dense curve to exactly n control points."""
    dists = np.linalg.norm(np.diff(curve_points, axis=0), axis=1)
    arc_lengths = np.concatenate([[0], np.cumsum(dists)])
    total = arc_lengths[-1]
    target = np.linspace(0, total, n)

    resampled = np.zeros((n, curve_points.shape[1]))
    for dim in range(curve_points.shape[1]):
        resampled[:, dim] = np.interp(target, arc_lengths, curve_points[:, dim])
    return resampled


def compute_spline_length(points: npt.NDArray[np.float64]) -> float:
    """Total arc length along ordered points."""
    diffs = np.diff(points, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def mean_distance_to_spline(
    pred_points: npt.NDArray[np.float64],
    gt_curve: npt.NDArray[np.float64],
) -> float:
    """Mean minimum distance from predicted points to the GT dense curve."""
    dists = cdist(pred_points, gt_curve)
    return float(np.mean(np.min(dists, axis=1)))
