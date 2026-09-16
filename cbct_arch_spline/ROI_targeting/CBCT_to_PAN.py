"""
Usage:
    python standalone_artificial_px.py --cbct scan.nii --labels labels.nii.gz --out px.png
"""
import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy import interpolate
from scipy.spatial import cKDTree
from skimage.measure import EllipseModel

UPPER_SKULL, MANDIBLE, UPPER_TEETH, LOWER_TEETH = 1, 2, 3, 4

RESOLUTION_MM = 0.1                  # output pixel spacing (mm), along and across the arch
N_DEPTH = 240                        # ray-integration samples per output column
PROBE_REACH_MM = 45.0                # max half-reach searched when locating the band edge

BAND_DECAY_SIGMA_MM = 4.0            # Gaussian falloff (mm) of the focal band's soft edge
TIP_HALFWIDTH_MM = 10.0              # radius (mm) of the mask "sausage" extended toward the TMJ
SEMI_PAD_MM = 2.0                    # clearance (mm) added past the mask where the band is clipped
INNER_SIGMA_CANDIDATES = (0.12, 0.18, 0.28, 0.45)   # candidate widths for the inner-ellipse fit. Simulates real focal region templates PX machines use

TMJ_MARGIN_MM = 10.0                 # straight tangent run past each arch tip toward the TMJ
ARCH_APEX_R_FLOOR_MM = 30.0          # minimum radius of curvature required at the anterior apex

SOFT_TISSUE_RESPONSE_HU = 15.0       # HU-300 anchor of the attenuation transfer curve
DEFAULT_TRANSFER_XP = np.array([-200., 300., 800., 1600., 2800., 8000.], np.float32)
DEFAULT_TRANSFER_FP = np.array([0., 40., 300., 1100., 2200., 2500.], np.float32)

TONE_PERCENTILE_LOW, TONE_PERCENTILE_HIGH = 1.0, 99.5      # final display percentile clip
CORE_WEIGHT_FRAC = 0.5
CORE_WEIGHT_DECAY_FRAC = 0.08


# --------------------------------------------------------------------------- #
#  1. load
# --------------------------------------------------------------------------- #
def load_case(cbct_path, label_path):
    """Load the CBCT and its label volume, oriented so array index 0 (output row 0)
    is SUPERIOR. Returns (volume, labels, z_spacing_mm, ips_mm).

    MMDental/SD-Tooth-style label volumes are stored index-0-inferior, so the
    orientation is measured from the labels (the mandible must end up below the
    upper skull) and corrected automatically.
    """
    cbct_img = sitk.ReadImage(str(cbct_path))
    volume = sitk.GetArrayFromImage(cbct_img)                  # (Z, Y, X)
    labels = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))
    if labels.shape != volume.shape:
        raise ValueError(f"label shape {labels.shape} != volume shape {volume.shape}")

    if _needs_flip(labels):
        volume = np.flip(volume, axis=(0, 1)).copy()
        labels = np.flip(labels, axis=(0, 1)).copy()

    sx, sy, sz = cbct_img.GetSpacing()
    if not np.isclose(sx, sy):
        raise ValueError(f"expected equal X/Y spacing; got {sx} vs {sy}")
    return volume, labels, float(sz), float(sx)


def _needs_flip(labels):
    """True when the volume is stored index-0-inferior and must be flipped."""
    mandible, skull = labels == MANDIBLE, labels == UPPER_SKULL
    if mandible.any() and skull.any():
        return np.argwhere(mandible)[:, 0].mean() < np.argwhere(skull)[:, 0].mean()
    upper, lower = labels == UPPER_TEETH, labels == LOWER_TEETH
    if upper.any() and lower.any():
        return np.argwhere(lower)[:, 0].mean() < np.argwhere(upper)[:, 0].mean()
    raise ValueError("cannot determine orientation: need mandible+skull or both dentitions")


# --------------------------------------------------------------------------- #
#  2. dental-arch centreline: a polar fit to the label masks
# --------------------------------------------------------------------------- #
def arch_region(labels, teeth_weight=3.0):
    """Axial (Y, X) footprint of mandible + both dentitions, and a per-column
    weight that favours teeth `teeth_weight`x over bone -- so the arch curve rides
    the tooth ridge wherever teeth exist, falling back to jaw bone where they don't
    (e.g. an edentulous jaw)."""
    teeth_count = ((labels == UPPER_TEETH) | (labels == LOWER_TEETH)).sum(axis=0).astype(np.float32)
    mandible_count = (labels == MANDIBLE).sum(axis=0).astype(np.float32)
    region = (teeth_count > 0) | (mandible_count > 0)
    weight = (teeth_weight * teeth_count / (teeth_count.max() + 1e-6)
              + mandible_count / (mandible_count.max() + 1e-6))
    return region, weight


def _right_handed(anterior):
    """Lateral axis that makes (lateral, anterior) a proper rotation, not a mirror."""
    return np.array([anterior[1], -anterior[0]])


def _arch_anterior_axis(offsets, weights, n_bins=144, empty_frac=0.02,
                        smooth=1.5, min_gap_deg=25.0):
    """Anterior direction of the arch, as a (y, x) unit vector, found from the one
    large angular sector -- around the mask's centroid -- with no tissue in it: the
    mouth opening, which by construction faces posteriorly. Returns (anterior,
    gap_width_deg), or (None, 0.0) if no such gap stands out."""
    angle = np.arctan2(offsets[:, 0], offsets[:, 1])
    hist, edges = np.histogram(angle, bins=n_bins, range=(-np.pi, np.pi), weights=weights)
    hist = ndi.gaussian_filter1d(hist.astype(float), smooth, mode='wrap')
    if hist.max() <= 0:
        return None, 0.0
    empty = hist <= empty_frac * hist.max()
    if not empty.any():
        return None, 0.0
    idx = np.flatnonzero(empty)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    if len(runs) > 1 and empty[0] and empty[-1]:          # the run wraps at +/-pi
        runs = [np.concatenate([runs[-1], runs[0]])] + runs[1:-1]
    run = max(runs, key=len)
    width_deg = len(run) * 360.0 / n_bins
    if width_deg < min_gap_deg:                            # too narrow to be a mouth
        return None, width_deg
    centres = 0.5 * (edges[:-1] + edges[1:])[run % n_bins]
    g = float(np.arctan2(np.sin(centres).mean(), np.cos(centres).mean()))
    return -np.array([np.sin(g), np.cos(g)]), width_deg


def _anterior_sign(labels, point_frac=0.15):
    """+1 if the arch's anterior midline lies toward +y (array rows), else -1 --
    read off the mouth-opening axis (see `_arch_anterior_axis`).
    """
    region, weight = arch_region(labels)
    ridge = region & (weight > point_frac * weight.max())
    points = np.argwhere(ridge if int(ridge.sum()) >= 200 else region)
    w = np.clip(weight[points[:, 0], points[:, 1]], 0, None).astype(np.float32)
    centre = np.array([np.average(points[:, 0], weights=w), np.average(points[:, 1], weights=w)])
    anterior, _gap_deg = _arch_anterior_axis(points - centre, w)
    return 1.0 if anterior[0] > 0 else -1.0


def posterior_ends(labels, ips_mm, tol_mm=2.0, region=None):
    """The two posterior-most points of the arch mask (one per side), each averaged
    over the most posterior `tol_mm` so a single stray voxel cannot move it. Doubles
    as "the condyle" whether or not the scan's field of view actually reaches it."""
    if region is None:
        region, _ = arch_region(labels)
    points = np.argwhere(region).astype(float)                # (y, x)
    anterior_sign = _anterior_sign(labels)
    x_mid = points[:, 1].mean()
    tol_px = max(1.0, tol_mm / max(ips_mm, 1e-6))

    ends = []
    for side in (False, True):
        side_points = points[(points[:, 1] > x_mid) == side]
        y_extreme = side_points[:, 0].min() if anterior_sign > 0 else side_points[:, 0].max()
        near = side_points[np.abs(side_points[:, 0] - y_extreme) <= tol_px]
        ends.append((float(near[:, 0].mean()), float(near[:, 1].mean())))
    return ends                                                # [left, right], each (y, x)


def _weighted_median(values, weights):
    """Weighted median of `values` (used per angular bin to place the arch radius)."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    c = np.cumsum(w)
    if c[-1] <= 0:
        return float(np.median(v)) if len(v) else 0.0
    return float(v[min(int(np.searchsorted(c, 0.5 * c[-1])), len(v) - 1)])


def _catmull_rom_curve(points, n_samples=1000):
    """Uniform Catmull-Rom interpolating spline through `points` (N, 2): a dense
    (M, 2) curve passing exactly through every point, in order. Ends are handled by
    duplicating the first/last point."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return pts.copy()
    padded = np.vstack([pts[0], pts, pts[-1]])
    n_seg = n - 1
    per = max(2, n_samples // n_seg)
    out = []
    for i in range(n_seg):
        p0, p1, p2, p3 = padded[i], padded[i + 1], padded[i + 2], padded[i + 3]
        for t in np.linspace(0.0, 1.0, per, endpoint=(i == n_seg - 1)):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1)
                              + (-p0 + p2) * t
                              + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.asarray(out)


def manual_arch_tck(points, spline_smooth=3.0, n_dense=2000):
    """Fit a smoothing cubic B-spline through a poly-line of (x, y) pixel points, via
    a dense Catmull-Rom interpolation first -- this is what makes the spline track
    sparse or irregularly-spaced points instead of oscillating between them."""
    dense = _catmull_rom_curve(np.asarray(points, float), n_dense)
    tck, _u = interpolate.splprep([dense[:, 0], dense[:, 1]], k=3, s=len(dense) * spline_smooth)
    return tck


def _tip_extend(xs, ys, n_px, fit_px, guards):
    """Continue both ends of a poly-line STRAIGHT, along the chord from `fit_px` back
    along the curve to each endpoint -- straight by construction, so it carries none
    of the local curvature.

    `guards` is one (posterior_unit, outward_unit) pair per end, in (x, y): any
    medial component of the chord is removed against it, so the extension can run
    posteriorly or posterolaterally but never turns back toward the midline (near
    the joint the polar radius shrinks as the ramus climbs, which would otherwise
    aim a naive chord across the ramus instead of past it).
    """
    n_px = int(round(n_px))
    if n_px <= 0 or len(xs) < 4:
        return xs, ys
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])

    out_x, out_y = list(xs), list(ys)
    for k, end in enumerate((0, -1)):
        s_end = s[end]
        back = np.argmin(np.abs(s - (s_end + fit_px if end == 0 else s_end - fit_px)))
        dx, dy = xs[end] - xs[back], ys[end] - ys[back]
        norm = np.hypot(dx, dy)
        if norm < 1e-8:
            continue
        dx, dy = dx / norm, dy / norm
        post, outward = np.asarray(guards[k][0], float), np.asarray(guards[k][1], float)
        d = np.array([dx, dy])
        c_lat = float(d @ outward)
        if c_lat < 0.0:                                  # medial: drop the lateral component
            d = d - c_lat * outward
        c_post = float(d @ post)
        if c_post <= 0.0:                                 # degenerate: fall back to straight back
            d = post.copy()
        norm = np.hypot(*d)
        if norm < 1e-8:
            continue
        dx, dy = d / norm
        steps = np.arange(1, n_px + 1, dtype=float)
        nx, ny = xs[end] + dx * steps, ys[end] + dy * steps
        if end == 0:
            out_x, out_y = list(nx[::-1]) + out_x, list(ny[::-1]) + out_y
        else:
            out_x, out_y = out_x + list(nx), out_y + list(ny)
    return np.asarray(out_x), np.asarray(out_y)


def label_arch_curve(labels, ips_mm, n_bins=280, span_limit_deg=120.0,
                     smooth_deg=5.0, spline_smooth=3.0):
    """Dental-arch centreline measured directly from the labels: a polar fit around
    the midpoint of the two posterior mask ends, teeth-weighted (see `arch_region`)
    so the curve rides the tooth ridge and falls back to jaw bone only where a jaw is
    edentulous. Returns a scipy `splprep` tck.
    """
    region, weight = arch_region(labels)
    anterior_sign = _anterior_sign(labels)
    anterior_axis = np.array([anterior_sign, 0.0])         # (y, x)
    lateral_axis = np.array([0.0, 1.0])

    ends = posterior_ends(labels, ips_mm, region=region)
    origin = np.array([(ends[0][0] + ends[1][0]) / 2.0, (ends[0][1] + ends[1][1]) / 2.0])

    edges = np.linspace(-span_limit_deg, span_limit_deg, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    def polar(mask, is_3d):
        idx = np.argwhere(mask)[:, 1:] if is_3d else np.argwhere(mask)
        if len(idx) == 0:
            return None, None
        offset = idx.astype(float) - origin
        u, v = offset @ anterior_axis, offset @ lateral_axis
        return np.digitize(np.degrees(np.arctan2(v, u)), edges) - 1, np.hypot(u, v)

    # Radii are measured on 3-D tooth VOXELS (so bulky roots count proportionally to
    # their mass), falling back to the 2-D bone footprint in a bin with no teeth.
    idx_upper, r_upper = polar(labels == UPPER_TEETH, True)
    idx_lower, r_lower = polar(labels == LOWER_TEETH, True)
    idx_bone, r_bone = polar(region, False)
    bone_weight = weight[np.argwhere(region)[:, 0], np.argwhere(region)[:, 1]]

    min_voxels, min_pixels = 40, 12
    radius = np.full(n_bins, np.nan)
    for b in range(n_bins):
        per_jaw = []
        for idx, r in ((idx_upper, r_upper), (idx_lower, r_lower)):
            in_bin = idx == b
            if int(in_bin.sum()) >= min_voxels:
                per_jaw.append(float(np.median(r[in_bin])))
        if per_jaw:
            radius[b] = float(np.mean(per_jaw))            # each arch counts once
            continue
        in_bin = idx_bone == b
        if int(in_bin.sum()) >= min_pixels:
            radius[b] = _weighted_median(r_bone[in_bin], bone_weight[in_bin])

    known = ~np.isnan(radius)
    # Trim to the angular range the mask actually occupies, THEN fill the holes
    # inside it -- filling first would let the curve sail past the joint into air.
    first = int(np.argmax(known))
    last = int(len(known) - 1 - np.argmax(known[::-1]))
    centres, radius, known = centres[first:last + 1], radius[first:last + 1], known[first:last + 1]
    radius = np.interp(centres, centres[known], radius[known])
    bin_deg = 2.0 * span_limit_deg / n_bins
    radius = ndi.gaussian_filter1d(radius, max(1.0, smooth_deg / bin_deg), mode='nearest')

    theta = np.radians(centres)
    xs = origin[1] + radius * (np.cos(theta) * anterior_axis[1] + np.sin(theta) * lateral_axis[1])
    ys = origin[0] + radius * (np.cos(theta) * anterior_axis[0] + np.sin(theta) * lateral_axis[0])

    # Run a short straight tangent past each end so the joint sits inside the image
    # rather than exactly on its edge (see `_tip_extend`).
    posterior_dir = np.array([-anterior_axis[1], -anterior_axis[0]])    # (x, y)
    sign = 1.0 if xs[0] < xs[-1] else -1.0
    guards = [(posterior_dir, np.array([-sign, 0.0])), (posterior_dir, np.array([sign, 0.0]))]
    xs, ys = _tip_extend(xs, ys, TMJ_MARGIN_MM / max(ips_mm, 1e-6),
                        fit_px=8.0 / max(ips_mm, 1e-6), guards=guards)

    return manual_arch_tck(np.c_[xs, ys], spline_smooth=spline_smooth)


# --------------------------------------------------------------------------- #
#  3. arch-curve sampling, curvature, and anterior rounding
# --------------------------------------------------------------------------- #
def compute_panoramic_trajectory(tck, n_columns, n_dense=4000):
    """Equal-arc-length sampling of the arch spline: one output column per ray.
    Returns (points, normals, u_samples, arc_length) in pixel coordinates, with
    normals oriented outward (away from the arch centroid, i.e. toward buccal)."""
    u = np.linspace(0.0, 1.0, n_dense)
    x, y = interpolate.splev(u, tck)
    x, y = np.asarray(x, float), np.asarray(y, float)
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    arc_length = float(arc[-1])

    u_samples = np.interp(np.linspace(0.0, arc_length, n_columns), arc, u)
    xs, ys = interpolate.splev(u_samples, tck)
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    dx, dy = interpolate.splev(u_samples, tck, der=1)
    dx, dy = np.asarray(dx, float), np.asarray(dy, float)
    norm = np.hypot(dx, dy); norm[norm < 1e-8] = 1e-8
    tangent_x, tangent_y = dx / norm, dy / norm

    points = np.stack([xs, ys], axis=1)
    normals = np.stack([-tangent_y, tangent_x], axis=1)
    centroid = points.mean(axis=0)
    flip = ((points - centroid) * normals).sum(axis=1) < 0
    normals[flip] *= -1
    return points, normals, u_samples, arc_length


def compute_curve_radius(tck, u_samples, min_px=1.0, smooth_frac=0.02):
    """Local radius of curvature (px) of the arch at each sampled point -- the hard
    geometric limit on how deep a focal band may reach lingually there."""
    d1 = np.asarray(interpolate.splev(u_samples, tck, der=1), float)
    d2 = np.asarray(interpolate.splev(u_samples, tck, der=2), float)
    num = np.abs(d1[0] * d2[1] - d1[1] * d2[0])
    den = (d1[0] ** 2 + d1[1] ** 2) ** 1.5
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(num > 1e-9, den / np.maximum(num, 1e-12), np.inf)
    r = np.maximum(r, min_px)
    n = np.size(r)
    if smooth_frac > 0 and n > 8:
        k = np.reciprocal(np.maximum(r, min_px))           # smooth 1/R (bounded), not R (-> inf)
        k = ndi.gaussian_filter1d(k, max(1.0, smooth_frac * n), mode='nearest')
        r = np.reciprocal(np.maximum(k, 1e-12))
    return np.maximum(r, min_px)


def _resample_polyline(xs, ys, step):
    """Resample a poly-line to uniform arc-length spacing."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) < 3:
        return xs, ys
    d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
    if d[-1] <= step:
        return xs, ys
    t = np.arange(0.0, d[-1], float(step))
    return np.interp(t, d, xs), np.interp(t, d, ys)


def round_arch_anterior(tck, r_floor_px, max_shift_px=0.0, n=600, step_px=4.0,
                        max_rounds=14, s_fit=0.5, window=(0.30, 0.70),
                        test_window=(0.44, 0.56)):
    """Open out the arch's ANTERIOR (the arc-length `window` around the midline)
    until its radius of curvature at the apex clears `r_floor_px`, by raising the
    spline's own smoothing there -- pushing the apex posteriorly by at most
    `max_shift_px`. Returns a new tck; a curve already clearing the floor is
    returned unchanged.
    """
    u = np.linspace(0.0, 1.0, n)
    p0 = np.stack(interpolate.splev(u, tck), axis=1).astype(float)
    xs, ys = _resample_polyline(p0[:, 0], p0[:, 1], step_px)
    if len(xs) < 8:
        return tck
    p = np.stack([xs, ys], axis=1)
    t = np.linspace(0.0, 1.0, len(p))
    lo, hi = window
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    # Raised cosine: full displacement at the apex, tapering to zero (with zero
    # slope) at the window edges, so flattening the apex doesn't just move the
    # sharp turn out to the seam.
    bump = np.where(np.abs(t - mid) <= half,
                    0.5 * (1.0 + np.cos(np.pi * (t - mid) / max(half, 1e-9))), 0.0)

    d1 = np.gradient(p, axis=0)
    tl = np.hypot(d1[:, 0], d1[:, 1]); tl[tl < 1e-9] = 1e-9
    nx, ny = -d1[:, 1] / tl, d1[:, 0] / tl
    centroid = p.mean(axis=0)
    if ((centroid - p) * np.stack([nx, ny], 1)).sum() < 0:
        nx, ny = -nx, -ny                                   # point toward the concave side
    inward = np.stack([nx, ny], axis=1) * bump[:, None]

    def build(shift):
        pts = p + inward * shift
        try:
            candidate, _ = interpolate.splprep([pts[:, 0], pts[:, 1]], k=3, s=s_fit * len(pts))
        except Exception:
            return None
        radius = compute_curve_radius(candidate, u)
        apex = (u >= test_window[0]) & (u <= test_window[1])
        return candidate, float(np.median(radius[apex]))

    apex = (u >= test_window[0]) & (u <= test_window[1])
    best, best_r = tck, float(np.median(compute_curve_radius(tck, u)[apex]))
    if best_r >= r_floor_px:
        return tck

    # Bisect the displacement: radius grows monotonically with it, and the leash is
    # a hard bound, so the largest admissible push is the answer.
    lo_shift, hi_shift = 0.0, float(max_shift_px)
    for _ in range(max_rounds):
        shift = 0.5 * (lo_shift + hi_shift)
        got = build(shift)
        if got is None:
            break
        candidate, r = got
        if r >= r_floor_px:
            best, best_r, hi_shift = candidate, r, shift
        else:
            if r > best_r:
                best, best_r = candidate, r
            lo_shift = shift
    return best


# --------------------------------------------------------------------------- #
#  4. the "free" two-semi-ellipse focal band
# --------------------------------------------------------------------------- #
def condyle_extended_mask(labels, tck, ips_mm):
    """The mandible+teeth footprint, extended by a short "sausage" wherever the arch
    spline runs past it near the condyles -- so the outer ellipse below still has
    mask to follow all the way out to each ramus tip."""
    region, _ = arch_region(labels)
    H, W = region.shape
    u = np.linspace(0.0, 1.0, 5000)
    xs, ys = interpolate.splev(u, tck)
    xs, ys = np.asarray(xs), np.asarray(ys)
    xi = np.clip(np.round(xs).astype(int), 0, W - 1)
    yi = np.clip(np.round(ys).astype(int), 0, H - 1)
    outside = ~region[yi, xi]
    extended = region.copy()
    if outside.any():
        r = int(round(TIP_HALFWIDTH_MM / ips_mm))
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        disk = (xx ** 2 + yy ** 2) <= r ** 2
        stamp = np.zeros_like(region)
        stamp[yi[outside], xi[outside]] = True
        extended |= ndi.binary_dilation(stamp, disk)
    return extended


def _Qrot(px, py, cx, cy, a, b, theta):
    """Normalised squared radius of point (px, py) w.r.t. ellipse (cx, cy, a, b,
    theta): <= 1 inside, > 1 outside."""
    ct, st = np.cos(theta), np.sin(theta)
    dx, dy = px - cx, py - cy
    xr = ct * dx + st * dy
    yr = -st * dx + ct * dy
    return (xr / a) ** 2 + (yr / b) ** 2


def _fit_ellipse(points):
    """Least-squares ellipse through `points` (M, 2) -> (cx, cy, a, b, theta)."""
    try:
        em = EllipseModel()
        if em.estimate(np.asarray(points, float)):
            cx, cy, a, b, theta = em.params
            if a > 1 and b > 1 and np.isfinite([cx, cy, a, b, theta]).all():
                return float(cx), float(cy), float(a), float(b), float(theta)
    except Exception:
        pass
    p = np.asarray(points, float)                          # degenerate fallback
    return (float(p[:, 0].mean()), float(p[:, 1].mean()),
            float(np.ptp(p[:, 0]) / 2 + 1), float(np.ptp(p[:, 1]) / 2 + 1), 0.0)


def _mvee(points, tol=1e-4, max_iter=300):
    """Minimum-volume enclosing ellipse of a 2-D point set (Khachiyan's algorithm):
    the smallest ellipse that contains every point, as opposed to a least-squares fit
    (which minimises average residual, not area)."""
    pts = np.asarray(points, float)
    n, d = pts.shape
    Q = np.vstack([pts.T, np.ones(n)])
    u = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        AQ = np.linalg.solve((Q * u[None, :]) @ Q.T, Q)
        q_diag = np.einsum('ij,ij->j', Q, AQ)
        j = np.argmax(q_diag)
        step = (q_diag[j] - d - 1.0) / ((d + 1) * (q_diag[j] - 1.0))
        new_u = (1.0 - step) * u
        new_u[j] += step
        if np.linalg.norm(new_u - u) < tol:
            u = new_u
            break
        u = new_u
    centre = u @ pts
    A = np.linalg.inv(pts.T @ (pts * u[:, None]) - np.outer(centre, centre)) / d
    eigval, eigvec = np.linalg.eigh(A)
    if np.linalg.det(eigvec) < 0:
        eigvec[:, 1] *= -1                                  # force a proper rotation
    a, b = 1.0 / np.sqrt(np.maximum(eigval, 1e-12))
    theta = float(np.arctan2(eigvec[1, 0], eigvec[0, 0]))
    return float(centre[0]), float(centre[1]), float(a), float(b), theta


def _weighted_ellipse_fit(points, u_arc, sigma, min_weight=0.05, reps_scale=10):
    """Ellipse fit emphasising points near the anterior midline (u_arc=0.5), via a
    Gaussian arc-length weight approximated by point replication. Every point keeps
    some influence, so the fit stays well-conditioned instead of starving on a hard,
    narrow window."""
    w = np.exp(-0.5 * ((u_arc - 0.5) / sigma) ** 2)
    keep = w > min_weight
    if keep.sum() < 20:
        return None
    reps = np.maximum(1, np.round(w[keep] * reps_scale)).astype(int)
    return _fit_ellipse(np.repeat(points[keep], reps, axis=0))


def _march_halfwidths(point, normal, outer, k_out, inner, k_in, ips_mm, max_mm=80.0):
    """Band half-widths (mm) actually realised along one arch normal, by marching it
    and finding where it enters/leaves the (outer, inner) ellipse band."""
    t = np.linspace(-max_mm / ips_mm, max_mm / ips_mm, int(8 * max_mm) + 1)
    x, y = point[0] + t * normal[0], point[1] + t * normal[1]
    inside = (_Qrot(x, y, *outer) <= k_out ** 2) & (_Qrot(x, y, *inner) >= k_in ** 2)
    if not inside.any():
        return 0.0, 0.0
    idx = np.flatnonzero(inside)
    return max(-t[idx[0]], 0.0) * ips_mm, max(t[idx[-1]], 0.0) * ips_mm


def fit_free_ellipse_band(tissue_mask, tck, anterior_sign, ips_mm,
                          inner_sigma_candidates=INNER_SIGMA_CANDIDATES):
    """The "free" two-independent-semi-ellipse focal band. OUTER = the minimum-
    volume ellipse enclosing the whole tissue-mask boundary (hugs the mask as
    closely as an ellipse can). INNER = a Gaussian-arc-length-weighted fit of just
    the LINGUAL half of that boundary. Returns a dict of ellipse
    parameters consumed by `band_weight_field`.
    """
    ys, xs = np.nonzero(tissue_mask)
    gx, gy = xs.astype(float), ys.astype(float)
    posterior_dir = -anterior_sign
    y_cut = (ys.max() if posterior_dir > 0 else ys.min()) + posterior_dir * SEMI_PAD_MM / ips_mm

    # Split the mask boundary into buccal / lingual halves by the arch normal.
    boundary = tissue_mask ^ ndi.binary_erosion(tissue_mask)
    by, bx = np.nonzero(boundary)
    points, normals, u_arc_dense, _ = compute_panoramic_trajectory(tck, 2000)
    _dist, nearest = cKDTree(points).query(np.c_[bx, by])
    depth = ((bx - points[nearest, 0]) * normals[nearest, 0]
             + (by - points[nearest, 1]) * normals[nearest, 1])
    buccal = depth > 0
    u_arc = u_arc_dense[nearest]

    outer = _mvee(np.c_[bx, by])
    k_out = float(np.sqrt(_Qrot(gx, gy, *outer).max()))     # safety: exact, over ALL of the mask

    lingual_points = np.c_[bx[~buccal], by[~buccal]].astype(float)
    u_lingual = u_arc[~buccal]
    mid_point, mid_normal = points[len(points) // 2], normals[len(points) // 2]

    best = None
    for sigma in inner_sigma_candidates:
        candidate = _weighted_ellipse_fit(lingual_points, u_lingual, sigma)
        if candidate is None:
            continue
        k_in = float(np.sqrt(_Qrot(lingual_points[:, 0], lingual_points[:, 1], *candidate).min()))
        if (_Qrot(gx, gy, *candidate) < k_in ** 2).mean() > 0.01:
            k_in = float(np.sqrt(np.percentile(_Qrot(gx, gy, *candidate), 0.5)))
        anterior_halfwidth, _ = _march_halfwidths(mid_point, mid_normal, outer, k_out,
                                                   candidate, k_in, ips_mm)
        if best is None or anterior_halfwidth < best[0]:
            best = (anterior_halfwidth, candidate, k_in)
    _, inner, k_in = best

    return dict(outer=outer, inner=inner, k_out=k_out, k_in=k_in,
               y_cut=y_cut, post_dir=posterior_dir)


def band_weight_field(shape_yx, band, ips_mm, decay_sigma_mm=BAND_DECAY_SIGMA_MM):
    """(Y, X) float32 weight field: 1.0 inside the focal band, Gaussian decay
    (`decay_sigma_mm`) outside it."""
    H, W = shape_yx
    yy, xx = np.mgrid[0:H, 0:W]
    keep = (yy <= band['y_cut']) if band['post_dir'] > 0 else (yy >= band['y_cut'])
    inside = ((_Qrot(xx, yy, *band['outer']) <= band['k_out'] ** 2)
             & (_Qrot(xx, yy, *band['inner']) >= band['k_in'] ** 2) & keep)
    distance = ndi.distance_transform_edt(~inside).astype(np.float32)
    sigma_px = decay_sigma_mm / ips_mm
    return np.exp(-0.5 * (distance / sigma_px) ** 2).astype(np.float32)


# --------------------------------------------------------------------------- #
#  5. ray casting
# --------------------------------------------------------------------------- #
def denoise_volume(volume):
    """Edge-preserving 2-D median filter within each axial slice (panoramic-row
    resolution is untouched) -- removes sub-voxel CBCT speckle before it integrates
    into wispy bone texture along the ray."""
    return ndi.median_filter(volume, size=(1, 3, 3)).astype(np.float32)


def transfer_curve(soft_tissue_response=SOFT_TISSUE_RESPONSE_HU):
    """HU -> attenuation lookup curve. Lower `soft_tissue_response` (the response at
    HU 300) makes the nasal cavity / sinuses read more lucent."""
    fp = DEFAULT_TRANSFER_FP.copy()
    fp[1] = float(soft_tissue_response)
    return DEFAULT_TRANSFER_XP, fp


def cast_rays(volume, points, normals, wfield, ips_mm, z_rows, transfer_fp, n_depth=N_DEPTH):
    """Cast one ray per output column, perpendicular to the arch, weighted by
    `wfield` (the focal-band weight field): march each ray out to where the weight
    drops below 0.02 on both sides, sample `n_depth` points across that span at every
    output row (`z_rows`, a sub-voxel z pitch), map HU -> attenuation through
    `transfer_fp`, and sum weight * attenuation along the ray.

    The per-sample weight is additionally tapered by position within the local
    band half-width: full weight for the central `CORE_WEIGHT_FRAC` of the reach on
    each side (lingual/labial measured independently, since they usually differ),
    fast Gaussian falloff beyond it.
    """
    N = len(points)
    probe = np.linspace(-PROBE_REACH_MM / ips_mm, PROBE_REACH_MM / ips_mm, 601)
    xs = points[:, 0][:, None] + normals[:, 0][:, None] * probe[None, :]
    ys = points[:, 1][:, None] + normals[:, 1][:, None] * probe[None, :]
    w_probe = ndi.map_coordinates(wfield, [ys.ravel(), xs.ravel()], order=1,
                                  mode='constant', cval=0.0).reshape(N, len(probe))
    on = w_probe >= 0.02
    any_on = on.any(axis=1)
    first = np.argmax(on, axis=1)
    last = len(probe) - 1 - np.argmax(on[:, ::-1], axis=1)
    hw_lingual = np.where(any_on, np.maximum(-probe[first], 0.5), 0.5)
    hw_labial = np.where(any_on, np.maximum(probe[np.clip(last, 0, len(probe) - 1)], 0.5), 0.5)

    # Smooth column-to-column: a wobble in exactly where the weight crosses 0.02
    # (e.g. near an ellipse tangent point) would otherwise print as a vertical seam.
    smooth_px = max(1.0, 0.02 * N)
    hw_lingual = ndi.gaussian_filter1d(hw_lingual, smooth_px, mode='nearest')
    hw_labial = ndi.gaussian_filter1d(hw_labial, smooth_px, mode='nearest')

    depth_frac = np.linspace(0.0, 1.0, n_depth)
    depth = -hw_lingual[:, None] + (hw_lingual + hw_labial)[:, None] * depth_frac[None, :]
    step_mm = (hw_lingual + hw_labial) / (n_depth - 1)
    xc = points[:, 0][:, None] + depth * normals[:, 0][:, None]
    yc = points[:, 1][:, None] + depth * normals[:, 1][:, None]
    weight = ndi.map_coordinates(wfield, [yc.ravel(), xc.ravel()], order=1,
                                 mode='constant', cval=0.0).reshape(N, n_depth)

    side_hw = np.where(depth >= 0, hw_labial[:, None], hw_lingual[:, None])
    core_position = np.abs(depth) / np.maximum(side_hw, 1e-6)     # 0 at centerline, 1 at band edge
    core_taper = np.where(core_position <= CORE_WEIGHT_FRAC, 1.0,
                          np.exp(-0.5 * ((core_position - CORE_WEIGHT_FRAC)
                                        / CORE_WEIGHT_DECAY_FRAC) ** 2))
    weight = weight * core_taper

    Z = volume.shape[0]
    z_rows = np.clip(np.asarray(z_rows, float), 0.0, Z - 1)
    L = np.zeros((len(z_rows), N), np.float32)
    for row, z in enumerate(z_rows):
        z_lo = min(max(int(np.floor(z)), 0), Z - 1)
        frac = z - z_lo
        plane = (volume[z_lo] if frac <= 1e-9 or z_lo >= Z - 1
                else (1.0 - frac) * volume[z_lo] + frac * volume[z_lo + 1])
        sample = ndi.map_coordinates(plane, np.vstack([yc.ravel(), xc.ravel()]), order=1,
                                     mode='constant', cval=-1000.0).reshape(N, n_depth)
        attenuation = np.interp(sample, transfer_fp[0], transfer_fp[1]).astype(np.float32)
        L[row, :] = (attenuation * weight).sum(axis=1) * step_mm
    return L


# --------------------------------------------------------------------------- #
#  6. tone mapping
# --------------------------------------------------------------------------- #
def tone_map_raysum(L):
    """Linear tone map: divide by the 99th percentile of positive line-integral
    values (a robust gain normalization)."""
    positive = L[L > 0]
    reference = np.percentile(positive, 99) if positive.size else 1.0
    return L / (reference + 1e-8)


def normalize_display(image, p_low=TONE_PERCENTILE_LOW, p_high=TONE_PERCENTILE_HIGH):
    """Percentile-clip to [0, 1] and quantise to uint8 for display/saving."""
    lo, hi = np.percentile(image, [p_low, p_high])
    out = np.clip((image - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return (out * 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
#  top-level render
# --------------------------------------------------------------------------- #
def render(cbct_path, label_path, resolution_mm=RESOLUTION_MM):
    volume, labels, z_spacing_mm, ips_mm = load_case(cbct_path, label_path)

    # The arch the focal band is fitted to (the raw label fit)
    tck = label_arch_curve(labels, ips_mm)
    anterior_sign = _anterior_sign(labels)

    # Then the arch rays are actually cast along: the same curve, with its
    # anterior apex opened out to a minimum radius of curvature (a sharp apex
    # otherwise caps how deep the band may reach lingually there).
    render_tck = round_arch_anterior(tck, ARCH_APEX_R_FLOOR_MM / ips_mm)

    tissue_mask = condyle_extended_mask(labels, tck, ips_mm)
    band = fit_free_ellipse_band(tissue_mask, tck, anterior_sign, ips_mm)
    wfield = band_weight_field(labels.shape[1:], band, ips_mm)

    _, _, _, arc_length = compute_panoramic_trajectory(render_tck, 2000)
    n_columns = max(50, int(round(arc_length * ips_mm / resolution_mm)))
    points, normals, _u, _arc = compute_panoramic_trajectory(render_tck, n_columns)

    render_volume = denoise_volume(volume)
    transfer_fp = transfer_curve()
    z_rows = np.arange(0, volume.shape[0] - 1 + 1e-6, resolution_mm / z_spacing_mm)

    raw = cast_rays(render_volume, points, normals, wfield, ips_mm, z_rows, transfer_fp)
    return normalize_display(tone_map_raysum(raw))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cbct", type=Path, required=True, help="CBCT volume (.nii/.nii.gz)")
    parser.add_argument("--labels", type=Path, required=True, help="5-class label volume (.nii.gz)")
    parser.add_argument("--out", type=Path, required=True, help="output PNG path")
    parser.add_argument("--resolution-mm", type=float, default=RESOLUTION_MM,
                        help=f"output pixel spacing in mm (default {RESOLUTION_MM})")
    args = parser.parse_args()

    px = render(args.cbct, args.labels, args.resolution_mm)
    from PIL import Image
    Image.fromarray(px).save(args.out)
    print(f"Saved {args.out} ({px.shape[1]} x {px.shape[0]})")


if __name__ == "__main__":
    main()
