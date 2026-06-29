"""
Geometric (non-AI) arch spline methods.

Two complementary tools that need no trained model:

1. assisted_arch_from_clicks() — the RELIABLE path. The user clicks a handful
   of rough points along the arch; this orders them, fits a smooth B-spline,
   and resamples to N evenly-spaced control points. Optionally snaps each point
   toward the nearest bright structure (tooth / cortical bone).

2. auto_detect_arch() — a fully-automatic ROUGH starting guess. Detects bright
   blobs (teeth / bone), filters by size, drops the lower spine region, and
   fits an arch through the surviving centroids. Works well only when a clean
   tooth row is present in the slice; treat its output as a draft to refine.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from spline.spline_utils import (
    fit_spline,
    order_points_along_arch,
    resample_spline_to_n_points,
)


# ---------------------------------------------------------------------------
# 1. Assisted: a few clicks -> clean arch
# ---------------------------------------------------------------------------


def assisted_arch_from_clicks(
    click_points: npt.NDArray[np.float64],
    n_control: int = 24,
    slice_2d: npt.NDArray[np.float32] | None = None,
    snap_radius: int = 0,
) -> npt.NDArray[np.float64]:
    """
    Turn a few rough user clicks into a smooth, evenly-spaced arch.

    Args:
        click_points: (M, 2) array of (col, row) pixel clicks, M >= 4,
                      roughly along the arch (e.g. both rear molars + a few
                      points in between).
        n_control:    number of control points to output.
        slice_2d:     optional image for intensity snapping.
        snap_radius:  if > 0 and slice_2d given, nudge each output point toward
                      the brightest pixel within this radius.

    Returns:
        (n_control, 2) ordered, evenly-spaced control points (col, row).
    """
    pts = np.asarray(click_points, dtype=np.float64)
    if len(pts) < 4:
        raise ValueError(
            f"Need at least 4 clicks to fit an arch, got {len(pts)}. "
            "Click a few more points along the arch."
        )

    ordered = order_points_along_arch(pts)
    # Light smoothing so the curve doesn't kink through noisy clicks
    curve, _ = fit_spline(ordered, n_eval=400, smoothing=len(ordered) * 2.0)
    control = resample_spline_to_n_points(curve, n_control)

    if slice_2d is not None:
        # Keep control points inside the image (the smoothing spline can
        # overshoot slightly past the end clicks).
        h, w = slice_2d.shape
        control[:, 0] = np.clip(control[:, 0], 0, w - 1)
        control[:, 1] = np.clip(control[:, 1], 0, h - 1)
        if snap_radius > 0:
            control = snap_points_to_bright(slice_2d, control, snap_radius)

    return control


# ---------------------------------------------------------------------------
# Intensity snapping
# ---------------------------------------------------------------------------


def snap_points_to_bright(
    slice_2d: npt.NDArray[np.float32],
    points: npt.NDArray[np.float64],
    radius: int = 6,
) -> npt.NDArray[np.float64]:
    """
    Move each point to the brightest pixel within `radius` (local refinement).

    Useful to pull clicks/control points onto tooth enamel or cortical bone.
    Points whose neighbourhood is uniformly dark are left unchanged.
    """
    h, w = slice_2d.shape
    snapped = points.copy()
    for i, (col, row) in enumerate(points):
        c0, r0 = int(round(col)), int(round(row))
        # Skip points whose centre is outside the image — otherwise a negative
        # window bound would slice the wrong (huge) region and snap far away.
        if not (0 <= c0 < w and 0 <= r0 < h):
            continue
        c_lo, c_hi = max(0, c0 - radius), min(w, c0 + radius + 1)
        r_lo, r_hi = max(0, r0 - radius), min(h, r0 + radius + 1)
        patch = slice_2d[r_lo:r_hi, c_lo:c_hi]
        if patch.size == 0 or float(patch.max()) < 1e-3:
            continue
        dr, dc = np.unravel_index(int(np.argmax(patch)), patch.shape)
        snapped[i] = [c_lo + dc, r_lo + dr]
    return snapped


# ---------------------------------------------------------------------------
# 2. Fully-automatic rough detection
# ---------------------------------------------------------------------------


def auto_detect_arch(
    slice_2d: npt.NDArray[np.float32],
    n_control: int = 24,
    threshold: float | None = None,
    area_min: int = 12,
    area_max: int = 1200,
    drop_bottom_frac: float = 0.25,
) -> npt.NDArray[np.float64]:
    """
    Rough automatic arch guess from bright blob centroids.

    Args:
        slice_2d:       (H, W) float32 in [0, 1], HU-windowed.
        n_control:      number of output control points.
        threshold:      brightness cutoff; if None, uses the 90th percentile
                        of non-air pixels (adapts to per-scan brightness).
        area_min/max:   keep blobs whose area (px) is in this range (tooth /
                        bone sized), dropping noise and large metal blooms.
        drop_bottom_frac: ignore blobs whose centroid is in the bottom fraction
                        of the image (removes the spine / vertebra).

    Returns:
        (k, 2) control points (col, row). May return fewer than n_control if
        too few blobs are found. Raises ValueError if the arch can't be formed.
    """
    import cv2
    from skimage.measure import label, regionprops

    h, w = slice_2d.shape

    if threshold is None:
        non_air = slice_2d[slice_2d > 0.05]
        threshold = float(np.percentile(non_air, 90)) if non_air.size else 0.5

    mask = (slice_2d > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    labeled = label(mask)
    centroids = []
    row_cutoff = h * (1.0 - drop_bottom_frac)
    for region in regionprops(labeled):
        if not (area_min <= region.area <= area_max):
            continue
        cy, cx = region.centroid
        if cy > row_cutoff:  # drop spine / lower structures
            continue
        centroids.append([cx, cy])

    centroids = np.array(centroids, dtype=np.float64)
    if len(centroids) < 4:
        raise ValueError(
            f"Auto-detection found only {len(centroids)} usable blobs. "
            "This slice likely has no clean tooth row — use the assisted "
            "click method instead, or pick a slice through the tooth crowns."
        )

    ordered = order_points_along_arch(centroids)
    curve, _ = fit_spline(ordered, n_eval=400, smoothing=len(ordered) * 3.0)
    return resample_spline_to_n_points(curve, min(n_control, len(ordered) * 2))
