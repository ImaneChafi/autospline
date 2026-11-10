"""ALL-SCANS GROUPWISE CBCT ATLAS -- everything in one file.

Builds the average CBCTs (mean, median, std, teeth_prob) of every usable scan of a
dataset, exactly as groupwise_gpu.py did, but with no other project file needed: the scan
reader, the panoramic band, the GPU toolbox, the registrations and the rounds are all below.
Only standard libraries are imported: numpy, scipy, scikit-image, SimpleITK, torch (CUDA).

    python groupwise_atlas_standalone.py "D:/MMDental" "D:/MMDental Labels"              # masked atlas
    python groupwise_atlas_standalone.py "D:/MMDental" "D:/MMDental Labels" --unmasked   # afterwards: unmasked atlas, same recipes
    python groupwise_atlas_standalone.py "D:/SD-Tooth" "D:/SD-Tooth Labels" --n 12       # quick test on the first 12 scans
    (add --out <folder> to write somewhere other than OUT below)

Input  : a scan folder and its label folder (sub-folders included). Each label file is paired
         with the scan of the same file name; label codes 1 skull, 2 mandible, 3 upper teeth,
         4 lower teeth. Scans whose labels hold almost no mandible or teeth are skipped. Labels
         from Dental Segmentator from 3D Slicer
Output : OUT / <scan folder name> / atlas / {mean, median, std, teeth_prob}[_unmasked].nii.gz
         (0.25 mm, 640 x 640 x 400, float32). Every stage leaves files behind, so a stopped run
         resumes where it stopped.
Needs  : a GPU; disk for the 0.5 mm cache and one 328 MB file per scan (about 130 GB
         for MMDental's 398 scans); about 5 h (MMDental) / 2.5 h (SD-Tooth) on an RTX 4080.

The journey (sections below, in order):
  PART 1  settings
  PART 2  finding the scans; reading one, right way up
  PART 3  the panoramic band (top-down horseshoe the panoramic X-ray sees)
  PART 4  rulers and the GPU picture tools (read at any address, blur, shrink)
  PART 5  preparing one scan (cut-offs, jaw centre, 0.5 mm working copy)
  PART 6  scores (pattern match, squared difference, overlap)
  PART 7  recipes (stiff move, arrows, following a recipe)
  PART 8  lining scans up (stiff-move search, bending engine, fine bend, quality check)
  PART 9  the final 0.25 mm copy helpers
  PART 10 the rounds: Round 0, stiff rounds 1-2, bend rounds 3-5, fine bend, final copy, combine
  PART 11 command line
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from scipy import interpolate
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.measure import EllipseModel


# ===================================================================================== #
#  PART 1 -- SETTINGS
# ===================================================================================== #
# Where everything is written (--out replaces it).
OUT = Path(r"D:/Dataset Preparation/sharp_atlas/loop/groupwise_standalone")
DEV = torch.device("cuda")

# Label codes in the label files (code 5, the mandibular canal, is never used).
UPPER_SKULL, MANDIBLE, UPPER_TEETH, LOWER_TEETH = 1, 2, 3, 4

# Finding the valid scans
SCAN_EXTS = (".nii.gz", ".nii", ".nrrd", ".mha")
MIN_MANDIBLE_VOX = 20_000
MIN_TEETH_VOX = 5_000

# Panoramic-band geometry -> this controls the range and resolution of the panoramic
# focal region, which would then affect the coverage of the masked CBCT volumes
RESOLUTION_MM = 0.1                  # output pixel spacing (mm), along and across the arch
N_DEPTH = 240                        # ray-integration samples per output column
PROBE_REACH_MM = 45.0                # max half-reach searched when locating the band edge
BAND_DECAY_SIGMA_MM = 4.0            # Gaussian falloff (mm) of the focal band's soft edge
TIP_HALFWIDTH_MM = 10.0              # radius (mm) of the mask "sausage" extended toward the TMJ
SEMI_PAD_MM = 2.0                    # clearance (mm) added past the mask where the band is clipped
INNER_SIGMA_CANDIDATES = (0.12, 0.18, 0.28, 0.45)   # candidate widths for the inner-ellipse fit
TMJ_MARGIN_MM = 10.0                 # straight tangent run past each arch tip toward the TMJ
ARCH_APEX_R_FLOOR_MM = 30.0          # minimum radius of curvature required at the anterior apex
CORE_WEIGHT_FRAC = 0.5               # ray samples count fully over the inner half of each side
CORE_WEIGHT_DECAY_FRAC = 0.08        # ... then fade with this (bell-shaped) width

# The box every scan is brought into: 160 x 160 x 100 mm, centre (80, 80, 50) mm.
SP05 = 0.5                                                # working cube size (mm)
CANON = (200, 320, 320)                                   # box cube counts at 0.5 mm (slices, rows, columns)
CENTER = np.array([CANON[2], CANON[1], CANON[0]], float) * SP05 / 2.0   # (x, y, z) mm
HI_SP = 0.25                                              # final cube size (mm)
HI_CANON = tuple(2 * n for n in CANON)                    # (400, 640, 640): each 0.5 mm cube split in 8

# Brightness cut-offs: the 0.5th and 99.5th percentiles inside the band become 0 and 1.
PCT_LO, PCT_HI = 0.5, 99.5

# The rounds.
AFF_ITERS = 2                        # stiff-move rounds
MC_ROUNDS = 3                        # big-bend rounds
SHAPE_STEP = 0.2                     # shape update: target redrawn 1/5 of the way along the average arrow
FINE_MARGIN_MM = 10                  # mm added around the teeth to make the teeth box

# Label picture (stiff rounds): 1 on teeth, 0.3 on mandible.
TEETH_W, MAND_W = 1.0, 0.3
# The three label layers of the bend targets, and their mean-squares weights (x 20 below).
MC_LABELS = {"up": UPPER_TEETH, "lo": LOWER_TEETH, "mand": MANDIBLE}
MC_WEIGHTS = {"up": 0.5, "lo": 0.5, "mand": 0.25}

# Bending engine (PART 8). Calibrated against ANTs SyN[0.2, 3, 0] runs.
WEIGHTS = [20 * MC_WEIGHTS[k] for k in MC_LABELS]      # mean-squares weights of the mask channels (10, 10, 5)
STEP_VOX = 0.4                                         # biggest nudge per repeat, in pin spacings
UPDATE_SIGMA = np.sqrt(3.0)                            # SyN update-field variance 3 voxel^2
TOTAL_SIGMA = 0.5                                      # Gaussian on the whole field after each update (voxels)
CC_EPS = 1e-4                                          # added to each local variance (flat regions)
COARSE = [(2, 1.0, 80), (1, 0.0, 40)]                  # big bend: (shrink, smoothing voxels, iterations), CC radius 2
FINE = [(2, 1.0, 60), (1, 0.0, 40)]                    # fine bend: CC radius 3
AFFINE = [(4, 2.0, 150), (2, 1.0, 100)]                # stiff move: (shrink, smoothing mm, iterations)

# Packing of the final per-scan files: 15 switches = brightness, top switch = tooth.
VAL_MAX, MISSING = 32766, 32767                       # final files: 15-bit value (MISSING = outside band) | teeth << 15
Z_SLAB = 4                                            # slices read from every scan at a time when combining


# ===================================================================================== #
#  PART 2 -- FINDING THE SCANS. READING ONE, RIGHT WAY UP
# ===================================================================================== #
# A file's name without its ending ("114.nii.gz" -> "114"), or None if it is not a scan file.
def _scan_name(p):
    for e in SCAN_EXTS:
        if p.name.lower().endswith(e):
            return p.name[:-len(e)]
    return None


# THE SCAN LIST. Every label file in the label folder (and its sub-folders) is paired with the
# scan of the same name anywhere in the scan folder. When several scans share the name (the same
# scan stored as both .nii and .nii.gz), the one with exactly the label's file name is taken.
# Returns {scan name -> its two file paths} and {label file -> why it has no scan}.
def find_cases(images_dir, labels_dir):
    scans = {}
    for p in images_dir.rglob("*"):
        name = _scan_name(p)
        if name is not None:
            scans.setdefault(name, []).append(p)
    rows, unpaired = {}, {}
    for lp in sorted(labels_dir.rglob("*")):
        case = _scan_name(lp)
        if case is None:
            continue
        found = scans.get(case, [])
        if len(found) > 1:
            found = [p for p in found if p.name == lp.name] or found
        if len(found) != 1 or case in rows:
            unpaired[str(lp)] = (f"{len(found)} scans named {case}" if len(found) != 1
                                 else f"a second label file named {case}")
            continue
        rows[case] = {"case": case, "cbct_path": str(found[0]), "label_path": str(lp)}
    return rows, unpaired


# Raised for a scan whose label file is a failed segmentation: it is skipped, not retried.
class Unusable(Exception):
    pass


def check_usable(labels):
    n_mand = int(np.count_nonzero(labels == MANDIBLE))
    n_teeth = int(np.count_nonzero((labels == UPPER_TEETH) | (labels == LOWER_TEETH)))
    if n_mand < MIN_MANDIBLE_VOX or n_teeth < MIN_TEETH_VOX:
        raise Unusable(f"labels hold {n_mand} mandible / {n_teeth} tooth cubes "
                       f"(need {MIN_MANDIBLE_VOX} / {MIN_TEETH_VOX})")


# Reads the CBCT and its label file into two tables of the same size (a failed segmentation
# is rejected first, by check_usable). If the mandible sits
# ABOVE the skull in the table, the scan is stored upside down: both tables are turned over
# (slices and rows reversed together = a half-turn, so left/right are NOT mirrored).
# Returns the brightness table, the label table, the slice thickness and the in-slice cube size.
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
    check_usable(labels)                                       # added: replaces the usable-case list
    if labels.shape != volume.shape:
        raise ValueError(f"label shape {labels.shape} != volume shape {volume.shape}")

    if _needs_flip(labels):
        volume = np.flip(volume, axis=(0, 1)).copy()
        labels = np.flip(labels, axis=(0, 1)).copy()

    sx, sy, sz = cbct_img.GetSpacing()
    if not np.isclose(sx, sy):
        raise ValueError(f"expected equal X/Y spacing; got {sx} vs {sy}")
    return volume, labels, float(sz), float(sx)


# Compares the average slice number of the mandible with that of the skull
# (or of the lower teeth with the upper teeth). A smaller slice number = higher up.
def _needs_flip(labels):
    """True when the volume is stored index-0-inferior and must be flipped."""
    mandible, skull = labels == MANDIBLE, labels == UPPER_SKULL
    if mandible.any() and skull.any():
        return np.argwhere(mandible)[:, 0].mean() < np.argwhere(skull)[:, 0].mean()
    upper, lower = labels == UPPER_TEETH, labels == LOWER_TEETH
    if upper.any() and lower.any():
        return np.argwhere(lower)[:, 0].mean() < np.argwhere(upper)[:, 0].mean()
    raise ValueError("cannot determine orientation: need mandible+skull or both dentitions")


# ===================================================================================== #
#  PART 3 -- THE PANORAMIC BAND
# ===================================================================================== #
# The band: the horseshoe region of the jaw that the current CBCT->PAN pipeline uses, as a top-down yes/no
# map. Built in steps: footprint -> which way is front -> arch curve -> rounded front ->
# extension to the condyles -> two ellipses -> weight map -> imitated X-ray rays.

# Footprint seen from above: where there are teeth or mandible, plus a weight per pixel in 
# which teeth count 3x as much as bone (so the arch curve follows the teeth).
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


# Which way is front: looking around the middle of the footprint, the mouth opening is the
# widest direction with no jaw in it; it faces backward, so front is the opposite way.
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


# +1 if front is toward higher row numbers, -1 otherwise (from the mouth-opening direction).
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


# The two back ends of the jaw footprint (one per side), each averaged over its last 2 mm.
# The arch curve is traced from the midpoint between them.
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


# Weighted median: the middle value when each value counts as much as its weight.
def _weighted_median(values, weights):
    """Weighted median of `values` (used per angular bin to place the arch radius)."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    c = np.cumsum(w)
    if c[-1] <= 0:
        return float(np.median(v)) if len(v) else 0.0
    return float(v[min(int(np.searchsorted(c, 0.5 * c[-1])), len(v) - 1)])


# A smooth curve passing exactly through a list of points (used before the final smoothing).
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


# Fits the final smooth arch curve (stored as scipy's "tck" curve description).
def manual_arch_tck(points, spline_smooth=3.0, n_dense=2000):
    """Fit a smoothing cubic B-spline through a poly-line of (x, y) pixel points, via
    a dense Catmull-Rom interpolation first -- this is what makes the spline track
    sparse or irregularly-spaced points instead of oscillating between them."""
    dense = _catmull_rom_curve(np.asarray(points, float), n_dense)
    tck, _u = interpolate.splprep([dense[:, 0], dense[:, 1]], k=3, s=len(dense) * spline_smooth)
    return tck


# Extends both ends of the arch straight backward (never back toward the middle).
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


# THE ARCH CURVE. Stand at the midpoint of the two back ends and look in 280 directions over
# +-120 degrees. Per direction: the median distance of the 3-D tooth cubes (>= 40 needed; each
# jaw counts once), else the weighted median of the bone footprint (>= 12 pixels needed).
# Then trim to the occupied range, fill gaps, smooth over 5 degrees, turn back into x/y,
# extend the tips 10 mm, and fit a smooth curve.
# A jaw with NO teeth makes polar() return None -> `in_bin.sum()` fails with AttributeError:
# that is why the 8 toothless-jaw scans are excluded (stage_com catches it).
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


# Evenly spaced points along the arch curve, and at each one the across direction (pointing
# out toward the cheek). Also returns the curve's total length.
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


# How tightly the curve bends at each point (the radius of the circle that fits it there).
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


# Re-spaces the points of a line evenly along its length.
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


# Rounds the front of the arch if it bends more tightly than a 30 mm-radius circle.
# (Called with max_shift_px = 0: the curve is only re-smoothed there, never pushed back.)
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


# The footprint, extended by a 10 mm-wide "sausage" wherever the arch curve runs past it
# near the condyles (so the outer ellipse reaches out to each ramus tip).
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


# Is a point inside an ellipse? <= 1 inside, > 1 outside.
def _Qrot(px, py, cx, cy, a, b, theta):
    """Normalised squared radius of point (px, py) w.r.t. ellipse (cx, cy, a, b,
    theta): <= 1 inside, > 1 outside."""
    ct, st = np.cos(theta), np.sin(theta)
    dx, dy = px - cx, py - cy
    xr = ct * dx + st * dy
    yr = -st * dx + ct * dy
    return (xr / a) ** 2 + (yr / b) ** 2


# Best-fitting ellipse through a set of points (centre, two half-widths, tilt).
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


# Smallest ellipse that contains every point (the band's OUTER edge).
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


# Ellipse fit in which points near the front middle count more (by repeating them).
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


# How wide the band is along one across direction (used to pick the inner ellipse).
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


# THE TWO ELLIPSES. Outer = smallest ellipse around the whole extended footprint. Inner = an
# ellipse fitted to the tongue-side edge of the footprint (four widths tried; the one giving
# the narrowest band at the front middle wins). The band lies between them, cut 2 mm behind
# the back of the footprint.
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


# Weight map: 1 inside the band, fading (bell-shaped, 4 mm) with distance outside it.
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


# THE BAND ITSELF. Builds the curve, ellipses and weight map above, then imitates the X-ray:
# one point every 0.1 mm along the arch; at each, how far the band reaches on the tongue side
# and the cheek side (probing up to 45 mm, then smoothed along the arch); 240 samples placed
# across from edge to edge, each weighted by the band weight x a taper (1 over the inner half
# of each side, then a quick fade). Each sample marks its nearest pixel (a pixel keeps the
# largest weight it gets). Band = pixels with weight >= 0.05, largest connected piece only.
# All distances here are in pixels of the scan's own top-down map (mm / ips_mm).
def render_support(labels, ips_mm, resolution_mm=RESOLUTION_MM,
                   weight_threshold=0.05):
    """Axial PX support for a render-frame label volume.

    Returns (support, weight): both (Y, X). `weight` (float32) is the per-pixel
    maximum of wfield * core_taper, i.e. how much that pixel contributes to the
    PX. `support` (bool) is weight >= `weight_threshold`, largest connected
    component only (drops the near-zero Gaussian skirt and scattered far-reach
    samples). Reproduces standalone_artificial_px.render() + cast_rays() exactly.
    """
    # curve -> rounded front -> extended footprint -> two ellipses -> weight map
    tck = label_arch_curve(labels, ips_mm)
    anterior_sign = _anterior_sign(labels)
    render_tck = round_arch_anterior(tck, ARCH_APEX_R_FLOOR_MM / ips_mm)

    tissue_mask = condyle_extended_mask(labels, tck, ips_mm)
    band = fit_free_ellipse_band(tissue_mask, tck, anterior_sign, ips_mm)
    wfield = band_weight_field(labels.shape[1:], band, ips_mm)

    # one point every 0.1 mm along the (rounded) arch, with its across direction
    _, _, _, arc_length = compute_panoramic_trajectory(render_tck, 2000)
    n_columns = max(50, int(round(arc_length * ips_mm / resolution_mm)))
    points, normals, _u, _arc = compute_panoramic_trajectory(render_tck, n_columns)

    # --- cast_rays geometry (see standalone_artificial_px.cast_rays) --------------
    # how far the band reaches on the tongue side (hw_lingual) and cheek side (hw_labial)
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
    smooth_px = max(1.0, 0.02 * N)
    hw_lingual = ndi.gaussian_filter1d(hw_lingual, smooth_px, mode='nearest')
    hw_labial = ndi.gaussian_filter1d(hw_labial, smooth_px, mode='nearest')

    # the 240 samples across the band and their weights (band weight x core taper)
    depth_frac = np.linspace(0.0, 1.0, N_DEPTH)
    depth = -hw_lingual[:, None] + (hw_lingual + hw_labial)[:, None] * depth_frac[None, :]
    xc = points[:, 0][:, None] + depth * normals[:, 0][:, None]
    yc = points[:, 1][:, None] + depth * normals[:, 1][:, None]
    w_samp = ndi.map_coordinates(wfield, [yc.ravel(), xc.ravel()], order=1,
                                 mode='constant', cval=0.0).reshape(N, N_DEPTH)
    side_hw = np.where(depth >= 0, hw_labial[:, None], hw_lingual[:, None])
    core_position = np.abs(depth) / np.maximum(side_hw, 1e-6)
    core_taper = np.where(core_position <= CORE_WEIGHT_FRAC, 1.0,
                          np.exp(-0.5 * ((core_position - CORE_WEIGHT_FRAC)
                                         / CORE_WEIGHT_DECAY_FRAC) ** 2))
    w_samp = w_samp * core_taper

    # Splat the sampled positions onto the axial grid (nearest voxel; the sample
    # grid is sub-voxel dense, so this fills the band with no gaps).
    H, W = labels.shape[1:]
    xi = np.clip(np.round(xc.ravel()).astype(int), 0, W - 1)
    yi = np.clip(np.round(yc.ravel()).astype(int), 0, H - 1)
    flat = yi * W + xi
    weight = np.zeros(H * W, np.float32)
    np.maximum.at(weight, flat, w_samp.ravel().astype(np.float32))
    weight = weight.reshape(H, W)

    # yes where the weight >= 0.05, keeping only the biggest connected piece
    support = weight >= weight_threshold
    lbl, n = ndi.label(support)
    if n > 1:
        keep = 1 + int(np.argmax(ndi.sum(support, lbl, range(1, n + 1))))
        support = lbl == keep
    return support, weight


# ===================================================================================== #
#  PART 4 -- RULERS AND THE GPU PICTURE TOOLS
# ===================================================================================== #
# An empty SimpleITK image that only carries a ruler (start, cube size, counts).
def make_ref(origin, size_xyz, pixel=sitk.sitkFloat32, spacing=SP05):
    ref = sitk.Image([int(s) for s in size_xyz], pixel)
    ref.SetSpacing((spacing,) * 3)
    ref.SetOrigin(tuple(float(o) for o in origin))
    return ref


# The plan of a scan's 0.5 mm working copy: start, cube counts, and whether plain 2x2x2
# block averaging can be used (only when the scan has 0.25 mm cubes in every direction).
def grid05(shape_zyx, ips, zsp):
    """0.5 mm grid (origin, size xyz, use_block_mean) for a native volume. The
    2x block mean of a 0.25 mm volume puts voxel centres at 0.125 + 0.5 k."""
    Z, Y, X = shape_zyx
    if np.isclose(ips, 0.25) and np.isclose(zsp, 0.25):
        return (0.125,) * 3, (X // 2, Y // 2, Z // 2), True
    return (0.0,) * 3, (int(X * ips / SP05), int(Y * ips / SP05), int(Z * zsp / SP05)), False


# THE RULER. Turns cube numbers into addresses: address = start + cube number x cube size.
#   points(z0, z1) : the address of every cube of slices z0..z1
#   coarser(k)     : the ruler of the version with k x k x k blocks merged into one
#   inside(pts)    : yes for addresses that fall within the ruler's cubes
class Grid:
    """Regular grid: origin and spacing (x, y, z) in mm, shape (z, y, x)."""

    def __init__(self, origin, spacing, shape):
        self.origin = torch.tensor([float(v) for v in origin], device=DEV)
        self.spacing = torch.tensor([float(v) for v in spacing], device=DEV)
        self.shape = tuple(int(s) for s in shape)

    @classmethod
    def of(cls, img):
        return cls(img.GetOrigin(), img.GetSpacing(), img.GetSize()[::-1])

    def points(self, z0=0, z1=None):
        """Physical (x, y, z) of the voxel centres of slices z0..z1, (Dz, Y, X, 3)."""
        z1 = self.shape[0] if z1 is None else z1
        zz, yy, xx = torch.meshgrid(*(torch.arange(a, b, device=DEV, dtype=torch.float32)
                                      for a, b in ((z0, z1), (0, self.shape[1]), (0, self.shape[2]))), indexing="ij")
        return torch.stack([xx, yy, zz], -1) * self.spacing + self.origin

    def coarser(self, k):
        """The grid of a k-fold block shrink (sitk BinShrink geometry)."""
        return Grid((self.origin + (k - 1) / 2 * self.spacing).tolist(), (self.spacing * k).tolist(),
                    [n // k for n in self.shape])

    def inside(self, pts):
        idx = (pts - self.origin) / self.spacing
        hi = torch.tensor(self.shape[::-1], device=DEV, dtype=torch.float32) - 1
        return ((idx >= 0) & (idx <= hi)).all(-1)


# READ A PICTURE AT ANY ADDRESS: blend of the 8 nearest cubes ("bilinear", PyTorch's name
# even in 3-D) or the single nearest cube ("nearest", for labels). Outside the picture = 0.
def sample(vol, grid, pts, mode="bilinear"):
    """vol (C, Z, Y, X) on `grid` at physical points (..., 3) -> (C, ...); 0 outside."""
    size = torch.tensor(grid.shape[::-1], device=DEV, dtype=torch.float32)
    g = 2 * (pts - grid.origin) / grid.spacing / (size - 1) - 1
    out = F.grid_sample(vol[None].float(), g.reshape(1, -1, 1, 1, 3), mode=mode, padding_mode="zeros",
                        align_corners=True)
    return out.reshape(vol.shape[0], *pts.shape[:-1])


# BLUR: each cube becomes a bell-shaped weighted average of its neighbours (strength `sigma`
# in cubes, reach 3 x sigma), done along x, then y, then z. Edges are extended, not faded.
def blur(x, sigma):
    """Separable Gaussian of (C, Z, Y, X), sigma in voxels."""
    if sigma <= 0:
        return x
    r = int(np.ceil(3 * sigma))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=DEV, dtype=torch.float32) / sigma) ** 2)
    k = k / k.sum()
    y = x[None].float()
    for d in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + d] = 2 * r + 1
        pad = [0] * 6
        pad[2 * (2 - d)] = pad[2 * (2 - d) + 1] = r
        y = F.conv3d(F.pad(y, pad, mode="replicate"), k.reshape(shape).repeat(x.shape[0], 1, 1, 1, 1),
                     groups=x.shape[0])
    return y[0]


# SHRINK: each k x k x k block of cubes replaced by its average.
def shrink(x, k):
    return F.avg_pool3d(x[None].float(), k)[0] if k > 1 else x.float()


# The box's two rulers: 0.5 mm (all the rounds) and 0.25 mm (final copy and atlas). The
# 0.25 mm ruler starts at -0.125 mm so each 2x2x2 group of small cubes sits on one 0.5 mm cube.
HI_REF = make_ref((-HI_SP / 2,) * 3, (HI_CANON[2], HI_CANON[1], HI_CANON[0]), spacing=HI_SP)
C05 = Grid((0.0, 0.0, 0.0), (SP05,) * 3, CANON)           # the canonical 100 x 160 x 160 mm box at 0.5 mm
HI = Grid(HI_REF.GetOrigin(), HI_REF.GetSpacing(), HI_CANON)


# ===================================================================================== #
#  PART 5 -- PREPARING ONE SCAN
# ===================================================================================== #
# Brightness cut-offs: the 0.5th and 99.5th percentiles of the values inside the band (all
# slices). With more than 4 million values, every k-th one is used (same answer, faster).
def _norm_bounds(vol, support):
    """BCA.normalize_intensity's robust percentiles (same sampling), computed without
    its float64 copies of the whole volume."""
    vals = vol[:, support]
    if vals.size > 4_000_000:
        vals = vals.ravel()[:: vals.size // 4_000_000]
    return np.percentile(vals, [PCT_LO, PCT_HI])


# Everything done in main memory for one scan: read it (PART 2), draw its band (PART 3),
# its cut-offs, its jaw centre (average address of the mandible + tooth cubes, in mm), the
# plan of its 0.5 mm copy, and its band copied onto the 0.5 mm top-down map (nearest pixel).
def load_cpu(row):
    """CPU half of prepare(): native scan and labels, PX band (0.5 mm, as
    sharp_atlas.cache_case), normalisation bounds, mandible+teeth centroid. Raises for the
    cases whose arch cannot be traced (a jaw without teeth), as the sample pipeline does."""
    vol, lab, zsp, ips = load_case(row["cbct_path"], row["label_path"])
    support, _ = render_support(lab, ips)
    lo, hi = _norm_bounds(vol, support)
    m3 = np.isin(lab, (MANDIBLE, UPPER_TEETH, LOWER_TEETH))
    com = np.argwhere(m3).mean(axis=0)[::-1] * np.array([ips, ips, zsp])
    del m3
    origin, size, block = grid05(vol.shape, ips, zsp)
    s2 = sitk.GetImageFromArray(support.astype(np.uint8))
    s2.SetSpacing((ips, ips))
    r2 = sitk.Image([int(size[0]), int(size[1])], sitk.sitkUInt8)
    r2.SetSpacing((SP05, SP05))
    r2.SetOrigin(origin[:2])
    s05 = sitk.GetArrayFromImage(sitk.Resample(s2, r2, sitk.Transform(2, sitk.sitkIdentity),
                                               sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8))
    return {"vol": vol, "lab": lab, "ips": ips, "zsp": zsp, "lo": lo, "hi": hi, "com": com,
            "origin": origin, "size": size, "block": block, "s05": s05}


# On the GPU: the whole scan rescaled so lo -> 0 and hi -> 1 (clipped), the 0.5 mm working
# copy (2x2x2 block average, or read at the 0.5 mm addresses), the 0.5 mm labels (nearest
# cube) and a full-detail "tooth?" yes/no map (used only for the final copy).
def to_gpu(cpu):
    """What the chain needs, on the GPU: the scan normalised by its PX-band percentiles
    (native grid), the 0.5 mm image, labels and PX-band mask, the teeth indicator."""
    vol, lab, ips, zsp, lo, hi = (cpu.pop(k) for k in ("vol", "lab", "ips", "zsp", "lo", "hi"))
    origin, size, block, s05, com = cpu["origin"], cpu["size"], cpu["block"], cpu["s05"], cpu["com"]
    native = Grid((0.0, 0.0, 0.0), (ips, ips, zsp), vol.shape)
    V = torch.from_numpy(vol).to(DEV)
    del vol
    V = ((V.float() - float(lo)) / float(hi - lo + 1e-6)).clamp_(0.0, 1.0)[None]
    lab_g = torch.from_numpy(lab).to(DEV)
    del lab
    g05 = Grid(origin, (SP05,) * 3, size[::-1])
    if block:
        X, Y, Z = size
        img05 = shrink(V[:, :2 * Z, :2 * Y, :2 * X], 2)
    else:
        img05 = torch.cat([sample(V, native, g05.points(z, min(z + 50, g05.shape[0]))) for z in range(0, g05.shape[0], 50)], 1)
    lab05 = torch.cat([sample(lab_g[None].half(), native, g05.points(z, min(z + 50, g05.shape[0])), "nearest")
                       for z in range(0, g05.shape[0], 50)], 1)[0].round().to(torch.uint8)
    teeth = ((lab_g == UPPER_TEETH) | (lab_g == LOWER_TEETH)).float()[None]
    del lab_g
    return {"V": V, "native": native, "teeth": teeth, "img05": img05, "lab05": lab05, "g05": g05,
            "s05": torch.from_numpy(s05).to(DEV), "com": com, "lo": float(lo), "hi": float(hi)}


# Three 0.5 mm yes/no layers: upper teeth, lower teeth, mandible.
def masks05(prep):
    return torch.stack([(prep["lab05"] == v).float() for v in MC_LABELS.values()])


# ===================================================================================== #
#  PART 6 -- SCORES
# ===================================================================================== #
# PATTERN MATCH: for each cube, in the (2r+1)^3 window around it, how strongly the two
# pictures go up and down together (0 = unrelated, 1 = same pattern; overall brightness
# does not matter).
def lncc(a, b, r):
    """ANTs neighbourhood correlation: local cov^2 / (var_a var_b) in a (2r+1)^3 window."""
    box = lambda t: F.avg_pool3d(t[None, None], 2 * r + 1, stride=1, padding=r, count_include_pad=False)[0, 0]
    ma, mb = box(a), box(b)
    cab = box(a * b) - ma * mb
    return cab * cab / ((box(a * a) - ma * ma + CC_EPS) * (box(b * b) - mb * mb + CC_EPS))


# THE BEND SCORE (lower = better): - pattern match on the brightness layer
# + 10 x upper-teeth + 10 x lower-teeth + 5 x mandible squared differences, on `valid` cubes.
def cost(fix, mov, valid, r):
    """-local CC on the CBCT + weighted mean squares on the masks, over `valid` voxels."""
    loss = -lncc(fix[0], mov[0], r)[valid].mean()
    for c, w in enumerate(WEIGHTS, start=1):
        loss = loss + w * ((fix[c] - mov[c]) ** 2)[valid].mean()
    return loss


# Quality-check scores only: whole-region pattern match, and tooth overlap (1 = perfect).
def ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / torch.sqrt((a * a).sum() * (b * b).sum()))


def dice(a, b):
    return float(2 * (a & b).sum() / max(int(a.sum() + b.sum()), 1))


# ===================================================================================== #
#  PART 7 -- RECIPES: box address -> scan address
# ===================================================================================== #
# A recipe (`reg`) holds: M, o = the stiff move (3x3 table + slide = 12 numbers); u2, g2 =
# the big-bend arrows and their pin ruler; uf, gf = the fine-bend arrows and their pin ruler.

# STIFF MOVE: apply the table's rows to every address, then add the slide.
def apply_affine(M, o, pts):
    return pts @ M.T + o


# BEND: nudge every address by its arrow (arrows between pins are blended).
def displace(u, grid, pts):
    return pts + sample(u, grid, pts).movedim(0, -1)


# FOLLOW A RECIPE: fine arrow, then big arrow, then stiff move (newest correction first).
# The two `if`s skip the bends a recipe does not have yet (it grows over the rounds).
def to_case(reg, pts):
    """Template points -> case physical points through the whole chain."""
    if reg.get("uf") is not None:
        pts = displace(reg["uf"], reg["gf"], pts)
    if reg.get("u2") is not None:
        pts = displace(reg["u2"], reg["g2"], pts)
    return apply_affine(reg["M"], reg["o"], pts)


# ===================================================================================== #
#  PART 8 -- LINING SCANS UP
# ===================================================================================== #
# STIFF-MOVE SEARCH (rounds 1-2). The scan's label picture (1 teeth + 0.3 mandible, 1 mm,
# blurred 1 mm) is laid over the target's. Start: do-nothing table + the Round 0 slide. The
# table is applied around the box centre. Each repeat: score = average squared difference;
# the slope finder (loss.backward) says which way each of the 12 numbers should move; Adam
# takes the step; the step sizes (0.002 table, 0.2 mm slide) shrink smoothly to zero.
# 150 repeats at 4 mm with 2 mm extra blur, then 100 at 2 mm with 1 mm blur. The answer is
# rewritten in the plain "table x address + slide" form used by apply_affine.
def register_affine(T, prep):
    """Label affine (12 DOF about the box centre) from the COM shift, mean squares on
    1.0 x teeth + 0.3 x mandible (1 mm, sigma 1 mm), as sharp_atlas `affine`."""
    m = masks05(prep)
    limg = TEETH_W * (m[0] + m[1]) + MAND_W * m[2]
    g1 = prep["g05"].coarser(2)
    mov1 = blur(shrink(limg[None], 2), 1.0)
    c = torch.tensor(CENTER, dtype=torch.float32, device=DEV)
    A = torch.eye(3, device=DEV, requires_grad=True)
    t = torch.tensor(prep["com"] - CENTER, dtype=torch.float32, device=DEV).requires_grad_(True)
    for k, s_mm, iters in AFFINE:
        fix, g = shrink(blur(T["aff"], s_mm), k), T["g_aff"].coarser(k)
        mov = blur(mov1, s_mm)
        q = g.points()
        opt = torch.optim.Adam([{"params": [A], "lr": 2e-3}, {"params": [t], "lr": 0.2}])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
        for _ in range(iters):
            p = (q - c) @ A.T + c + t
            inside = g1.inside(p)
            loss = ((fix[0] - sample(mov, g1, p)[0]) ** 2)[inside].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    A, t = A.detach(), t.detach()
    return A, c + t - A @ c


# THE BENDING ENGINE (big and fine bends). Arrows start at zero. Per level (blurry first):
# each repeat, every pin looks at pin + nudge + old arrow; the bend score is computed; the
# slope finder gives, for every pin, the nudge that lowers it; the nudges are blurred so
# neighbours move together and scaled so the biggest is 0.4 pin spacing; new arrow = small
# nudge, then the old arrow from where it lands; blurred slightly; edge arrows set to zero.
# The next level starts from these arrows read at its pins.
def greedy(fix, mask, grid, moving, levels, r):
    """Greedy compositive diffeomorphic registration, ANTs SyN[0.2, 3, 0] settings.
    `moving(pts, s)` -> ((C, ...) channels at template points pts, inside flag), smoothed
    by s voxels of `grid`. Returns u (3, Z, Y, X) on `grid`: warped(q) = moving(q + u(q))."""
    u, gu = None, None
    for k, s, iters in levels:
        g = grid.coarser(k) if k > 1 else grid
        f = shrink(blur(fix, s), k)
        valid_t = shrink(mask[None].float(), k)[0] > 0.5
        q = g.points()
        u = torch.zeros((3, *g.shape), device=DEV) if u is None else sample(u, gu, q)
        sp = g.spacing.view(3, 1, 1, 1)
        for _ in range(iters):
            d = torch.zeros_like(u, requires_grad=True)
            qd = q + d.movedim(0, -1)
            x = qd + sample(u, g, qd).movedim(0, -1)
            mov, inside = moving(x, s)
            loss = cost(f, mov, valid_t & inside, r)
            grad, = torch.autograd.grad(loss, d)
            grad = blur(grad, UPDATE_SIGMA)
            nmax = (grad / sp).norm(dim=0).max()
            if nmax <= 0:
                break
            step = -STEP_VOX * grad / nmax                # largest step = STEP_VOX voxels
            with torch.no_grad():
                u = blur(step + sample(u, g, q + step.movedim(0, -1)), TOTAL_SIGMA)
                u[:, [0, -1]] = 0
                u[:, :, [0, -1]] = 0
                u[:, :, :, [0, -1]] = 0
        gu = g
    return u


# BIG BEND of one scan (rounds 3-5): the scan's 4 layers at 2 mm (brightness + 3 label layers
# blurred 1 mm), read through its stiff move; pins every 2 mm; 5x5x5 pattern window.
def register_mc(T, prep, M, o):
    """2 mm multi-channel deformable (mc_syn): case channels at 2 mm, through the affine."""
    g2c = prep["g05"].coarser(4)
    chans = torch.cat([shrink(prep["img05"], 4), blur(shrink(masks05(prep), 4), 0.5)])
    pyr = {s: blur(chans, s * T["g_mc"].spacing[0].item() / g2c.spacing[0].item()) for s in {lv[1] for lv in COARSE}}

    def moving2(x, s):
        p = apply_affine(M, o, x)
        return sample(pyr[s], g2c, p), g2c.inside(p)

    return greedy(T["mc"], T["mc_mask"], T["g_mc"], moving2, COARSE, 2)


# FINE BEND of one scan: the scan is first read through stiff move + big bend into the teeth
# box ("pre-bent copy"); then the engine with pins every 1 mm and a 7x7x7 pattern window.
def register_fine(T, prep, reg):
    """1 mm refinement in the teeth box (fine_r1): the case pre-warped through affine + u2."""
    qf = T["g_fine"].points()
    p0 = to_case(reg, qf)
    pre = torch.cat([sample(prep["img05"], prep["g05"], p0), blur(sample(masks05(prep), prep["g05"], p0), 0.5)])
    inside0 = prep["g05"].inside(p0)
    pyrf = {s: blur(pre, s) for s in {lv[1] for lv in FINE}}

    def movingf(x, s):
        return sample(pyrf[s], T["g_fine"], x), sample(inside0[None].float(), T["g_fine"], x)[0] > 0.5

    return greedy(T["fine"], T["fine_mask"], T["g_fine"], movingf, FINE, 3)


# Quality check only (changes no data): how much each bend locally squeezes or stretches
# (smallest and 1st-percentile volume change), the agreement with both targets, and the
# average arrow lengths.
def jacobian_det(u, grid, mask):
    """det d(q + u(q))/dq at the grid voxels inside mask."""
    sx, sy, sz = grid.spacing.tolist()
    J = torch.zeros(u.shape[1:] + (3, 3), device=DEV)
    for c in range(3):
        dz, dy, dx = torch.gradient(u[c], spacing=(sz, sy, sx))
        for j, d in enumerate((dx, dy, dz)):
            J[..., c, j] = d + (1.0 if c == j else 0.0)
    return torch.linalg.det(J)[mask]


def qa_scores(T, prep, reg):                      # "qa" in full_corpus_gpu.py (renamed: stage_fine has a dict named qa)
    """Agreement with the targets after each stage (the ANTs runs score the same way), and
    the smallest local volume change of each deformation (1st percentile, minimum)."""
    out = {}
    for name, u, g, m in (("2mm", reg["u2"], reg["g2"], T["mc_mask"]), ("fine", reg["uf"], reg["gf"], T["fine_mask"])):
        det = jacobian_det(u, g, m)
        out[f"jac_p1_{name}"] = float(torch.quantile(det, 0.01))
        out[f"jac_min_{name}"] = float(det.min())
    g2c = prep["g05"].coarser(4)
    chans = torch.cat([shrink(prep["img05"], 4), blur(shrink(masks05(prep), 4), 0.5)])
    q = T["g_mc"].points()
    w = sample(chans, g2c, apply_affine(reg["M"], reg["o"], displace(reg["u2"], reg["g2"], q)))
    m = T["mc_mask"]
    out["ncc_2mm"] = ncc(T["mc"][0][m], w[0][m])
    out["dice_2mm"] = dice(((T["mc"][1] + T["mc"][2]) > 0.5) & m, ((w[1] + w[2]) > 0.5) & m)
    qf = T["g_fine"].points()
    p = to_case(reg, qf)
    w = torch.cat([sample(prep["img05"], prep["g05"], p), blur(sample(masks05(prep), prep["g05"], p), 0.5)])
    m = T["fine_mask"]
    out["ncc_fine"] = ncc(T["fine"][0][m], w[0][m])
    out["dice_fine"] = dice(((T["fine"][1] + T["fine"][2]) > 0.5) & m, ((w[1] + w[2]) > 0.5) & m)
    out["u2_mm"] = float(reg["u2"].norm(dim=0)[T["mc_mask"]].mean())
    out["uf_mm"] = float(reg["uf"].norm(dim=0)[T["fine_mask"]].mean())
    return out


# ===================================================================================== #
#  PART 9 -- FINAL 0.25 mm COPY HELPERS
# ===================================================================================== #
# Slices z0..z1 of the 0.25 mm box through the complete recipe: brightness read from the
# ORIGINAL rescaled scan, "tooth?" from the full-detail tooth map (nearest cube), band check.
def resample_hi(T, prep, reg, z0, z1):
    """Slices z0..z1 of the 0.25 mm box: normalised CBCT, PX-band mask (0.5 mm band, as the
    sample's warp cache), teeth indicator."""
    p = to_case(reg, T["hi"].points(z0, z1))
    v = sample(prep["V"], prep["native"], p)[0]
    t = sample(prep["teeth"], prep["native"], p, "nearest")[0]
    return v, band_mask(prep, p), t


# BAND CHECK: nearest 0.5 mm cube of each scan address; yes if it is inside the working copy
# and its (row, column) is in the band (the band is the same for every slice).
def band_mask(prep, p):
    """Case points p inside the case's PX band (0.5 mm footprint, nearest) and its 0.5 mm box."""
    g = prep["g05"]
    idx = torch.round((p - g.origin) / g.spacing).long()
    ok = ((idx >= 0) & (idx < torch.tensor(g.shape[::-1], device=DEV))).all(-1)
    return prep["s05"][idx[..., 1].clamp(0, g.shape[1] - 1), idx[..., 0].clamp(0, g.shape[2] - 1)].bool() & ok


# ===================================================================================== #
#  PART 10 -- THE ROUNDS
# ===================================================================================== #
RUN = None                                            # output folder name: the dataset, or <dataset>_test<n>


# ---- the 0.5 mm cache: written once in Round 0, read by every later round ------------ #
def cache_p(ds, case):
    return OUT / "cache" / ds / f"{case}.npz"             # shared by the test and full runs


# Saved under a temporary ".part" name, renamed when complete.
def write_cache(ds, case, prep):
    g = prep["g05"]
    cache_p(ds, case).parent.mkdir(parents=True, exist_ok=True)   # added: groupwise_gpu.py assumed the folder existed
    part = cache_p(ds, case).with_suffix(".part.npz")
    np.savez(part, img05=prep["img05"][0].half().cpu().numpy(), lab05=prep["lab05"].cpu().numpy(),
             s05=prep["s05"].cpu().numpy(), com=prep["com"], origin=g.origin.cpu().numpy(), shape=np.array(g.shape))
    part.replace(cache_p(ds, case))


def read_cache(ds, case):
    with np.load(cache_p(ds, case)) as z:
        return {k: z[k] for k in z.files}


def cache_to_gpu(c):
    return {"img05": torch.from_numpy(c["img05"]).to(DEV).float()[None], "lab05": torch.from_numpy(c["lab05"]).to(DEV),
            "s05": torch.from_numpy(c["s05"]).to(DEV), "com": c["com"],
            "g05": Grid(c["origin"].tolist(), (SP05,) * 3, c["shape"].tolist())}


# Hands out scans one at a time while a background helper already loads the next one.
# By default it reads the cache; with `load` it can read the original scans instead.
def each_case(ds, cases, load=None):
    """Yield (case, prep on the GPU), the next case loading on the CPU meanwhile."""
    load = load or (lambda c: read_cache(ds, c))
    pool = ThreadPoolExecutor(1)
    nxt = pool.submit(load, cases[0]) if cases else None
    for i, c in enumerate(cases):
        cpu = nxt.result()
        nxt = pool.submit(load, cases[i + 1]) if i + 1 < len(cases) else None
        prep = (to_gpu if "vol" in cpu else cache_to_gpu)(cpu)
        if "fov2d" in cpu:
            prep["fov2d"], prep["zok"] = torch.from_numpy(cpu["fov2d"]).to(DEV), torch.from_numpy(cpu["zok"]).to(DEV)
        del cpu
        yield c, prep
        del prep
        torch.cuda.empty_cache()
    pool.shutdown()


# One recipe file per scan (the rulers are shared, so left out and added back on loading).
def save_reg(p, reg):
    torch.save({k: v for k, v in reg.items() if not isinstance(v, Grid)}, p)


def load_reg(p, grids):
    return {**torch.load(p, map_location=DEV, weights_only=True), **grids}


# ---- building an average in a round: running totals in the 0.5 mm box ------------------- #
# Six totals start at zero: sum, count and sum of squares of the brightness (counted only
# inside each scan's band), and the totals of the three label layers (counted everywhere).
def new_acc():
    return {k: torch.zeros(C05.shape, device=DEV) for k in ("sum", "cnt", "sq", "up", "lo", "mand")}


# Adds one scan: for every box cube (50 slices at a time), follow the recipe, read the
# brightness, check the band, read the 3 label layers (blended), add to the totals.
def accumulate(acc, prep, reg, chunk=50):
    """sharp_atlas.build_average on the GPU: the 0.5 mm cache through the case's chain,
    PX band as mask, plus the fraction of cases with each structure."""
    masks = masks05(prep)
    for z0 in range(0, C05.shape[0], chunk):
        z1 = min(z0 + chunk, C05.shape[0])
        p = to_case(reg, C05.points(z0, z1))
        v = sample(prep["img05"], prep["g05"], p)[0]
        m = band_mask(prep, p).float()
        lm = sample(masks, prep["g05"], p)
        acc["sum"][z0:z1] += v * m
        acc["cnt"][z0:z1] += m
        acc["sq"][z0:z1] += v * v * m
        for k, x in zip(("up", "lo", "mand"), lm):
            acc[k][z0:z1] += x


# From the totals: mean, spread, the compared region (cubes covered by more than half the
# maximum count), and the quality numbers (average spread, teeth agreement per jaw).
def stats(acc, n):
    cnt = acc["cnt"]
    avg = acc["sum"] / cnt.clamp(min=1)
    std = (acc["sq"] / cnt.clamp(min=1) - avg ** 2).clamp(min=0).sqrt()
    core = cnt > 0.5 * cnt.max()
    pu, pl = acc["up"] / n, acc["lo"] / n
    m = {"std_mean": float(std[core].mean()),
         "teeth_consistency_upper": float((pu ** 2).sum() / pu.sum().clamp(min=1e-6)),
         "teeth_consistency_lower": float((pl ** 2).sum() / pl.sum().clamp(min=1e-6))}
    return avg, core, m


# ---- targets ---------------------------------------------------------------------------- #
# Stiff-round target: label picture from the fractions (1 x teeth + 0.3 x mandible), 1 mm,
# blurred 1 mm.
def aff_template(acc, n):
    """1 mm label template 1.0 x teeth + 0.3 x mandible, smoothed 1 mm (save_templates)."""
    limg = TEETH_W * (acc["up"] + acc["lo"]) / n + MAND_W * acc["mand"] / n
    return {"aff": blur(shrink(limg[None], 2), 1.0), "g_aff": C05.coarser(2)}


# Bend-round target: 4 layers at 2 mm (average brightness + the 3 label fractions blurred
# 1 mm) and the compared region. With `ubar` (the average arrow of the round) the target is
# redrawn: each pin copies the old target from (pin - 0.2 x average arrow) = shape update.
def mc_template(acc, n, ubar=None):
    """2 mm template channels (save_mc_templates): CBCT average, one smoothed fraction per
    structure, core mask; with `ubar` moved by -SHAPE_STEP x the mean warp."""
    avg, core, _ = stats(acc, n)
    g = C05.coarser(4)
    chans = torch.cat([shrink(avg[None], 4),
                       blur(shrink(torch.stack([acc[k] / n for k in ("up", "lo", "mand")]), 4), 0.5)])
    mask = shrink(core[None].float(), 4)
    if ubar is not None:
        x = displace(-SHAPE_STEP * ubar, g, g.points())
        chans = sample(chans, g, x)
        mask = sample(mask, g, x, "nearest")
    return {"mc": chans, "mc_mask": mask[0] > 0.5, "g_mc": g}


# MEDIAN across scans that skips missing values: missing -> "infinity" so they sort last;
# then halfway between the values at positions (k-1)//2 and k//2 (k = number of real values).
def nanmedian(x):
    """np.nanmedian over dim 0 (mean of the two middle values), NaN = missing."""
    k = (~torch.isnan(x)).sum(0, keepdim=True)
    s = torch.nan_to_num(x, nan=float("inf")).sort(dim=0).values
    a = s.gather(0, ((k - 1) // 2).clamp(min=0))
    b = s.gather(0, (k // 2).clamp(max=x.shape[0] - 1))
    return torch.where(k > 0, (a + b) / 2, torch.zeros_like(a))[0]


# TEETH BOX: the 1 mm cubes where more than 5% of scans have a tooth, widened by 10 mm on each
# side (kept inside the box).
def fine_box(acc, n):
    """1 mm teeth box (+ margin) from the teeth fraction (write_fine_template)."""
    g1 = C05.coarser(2)
    pz, py, px = torch.nonzero(shrink(((acc["up"] + acc["lo"]) / n)[None], 2)[0] > 0.05, as_tuple=True)
    lo = [max(int(a.min()) - FINE_MARGIN_MM, 0) for a in (pz, py, px)]
    hi = [min(int(a.max()) + FINE_MARGIN_MM + 1, s) for a, s in zip((pz, py, px), g1.shape)]
    origin = (g1.origin + torch.tensor(lo[::-1], device=DEV, dtype=torch.float32) * g1.spacing).tolist()
    return Grid(origin, g1.spacing.tolist(), [b - a for a, b in zip(lo, hi)])


# ---- round bookkeeping: one folder per round; a round is done when its template.pt exists -- #
def stage_dir(ds, name):
    d = OUT / RUN / name
    (d / "reg").mkdir(parents=True, exist_ok=True)
    return d


def done(d):
    return (d / "template.pt").exists()


def log(ds, name, msg, t0):
    print(f"  {ds}/{name}: {msg}  [{(time.time() - t0) / 60:.1f} min]", flush=True)


# ---- ROUND 0 ------------------------------------------------------------------------------ #
# Every scan prepared once (PART 5) and cached (up to 3 tries per scan for flaky disk reads;
# AttributeError = a toothless jaw whose arch cannot be traced, Unusable = a failed
# segmentation -> recorded as failed, not retried). Then
# the average with slide-only recipes (do-nothing table, slide = jaw centre - box centre),
# and the first label-picture target.
def stage_com(ds, rows, cases):
    """Native scans -> 0.5 mm cache (the later rounds read only this) and the COM average."""
    d = stage_dir(ds, "com")
    if done(d):
        return torch.load(d / "template.pt", map_location=DEV, weights_only=False)
    failed = json.loads((d / "failed.json").read_text()) if (d / "failed.json").exists() else {}
    todo = [c for c in cases if not cache_p(ds, c).exists() and c not in failed]
    t0 = time.time()

    def load(c):
        for attempt in range(3):                          # the USB disk occasionally fails a read; retry
            try:
                return load_cpu(rows[c])
            except (AttributeError, Unusable) as e:       # a jaw without teeth (no arch) / a failed segmentation
                return {"failed": repr(e)[:200]}
            except Exception as e:
                err = e
                time.sleep(5)
        return {"failed": repr(err)[:200]}
    pool = ThreadPoolExecutor(1)
    nxt = pool.submit(load, todo[0]) if todo else None
    for i, c in enumerate(todo):
        cpu = nxt.result()
        nxt = pool.submit(load, todo[i + 1]) if i + 1 < len(todo) else None
        if "failed" in cpu:
            failed[c] = cpu["failed"]
            (d / "failed.json").write_text(json.dumps(failed, indent=1))
            log(ds, "com", f"{c} FAILED {failed[c]}", t0)
            continue
        prep = to_gpu(cpu)
        del cpu
        write_cache(ds, c, prep)
        del prep
        torch.cuda.empty_cache()
        log(ds, "com", f"cached {i + 1}/{len(todo)} {c}", t0)
    pool.shutdown()
    ok = [c for c in cases if c not in failed]
    acc = new_acc()
    for c, prep in each_case(ds, ok):
        accumulate(acc, prep, {"M": torch.eye(3, device=DEV),
                               "o": torch.tensor(prep["com"] - CENTER, dtype=torch.float32, device=DEV)})
    _, _, m = stats(acc, len(ok))
    out = {"cases": ok, "failed": failed, "T": aff_template(acc, len(ok)), "metrics": m}
    torch.save(out, d / "template.pt")
    log(ds, "com", f"{len(ok)} cases, {len(failed)} failed  {m}", t0)
    return out


# ---- ROUNDS 1-2: stiff moves ---------------------------------------------------------------- #
# FAIRNESS STEP: average all scans' tables, then combine each table with the undo of that
# average (slides kept), so the group's average move is "do nothing".
def unbias(regs):
    """sharp_atlas.unbias: remove the mean matrix about the box centre (keeps translations)."""
    c = torch.tensor(CENTER, dtype=torch.float32, device=DEV)
    inv = torch.linalg.inv(torch.stack([r["M"] for r in regs.values()]).mean(0))
    out = {}
    for k, r in regs.items():
        t = r["o"] - c + r["M"] @ c
        M = r["M"] @ inv
        out[k] = {"M": M, "o": c + t - M @ c}
    return out


# One stiff round: search every scan's stiff move (always from the Round 0 slide), apply the
# fairness step, average with the fair moves, and make the next target (label picture after
# round 1, the 4-layer bend target after round 2). The fair moves are kept under "affine".
def stage_aff(ds, it, prev):
    name = f"aff{it}"
    d = stage_dir(ds, name)
    if done(d):
        return torch.load(d / "template.pt", map_location=DEV, weights_only=False)
    cases, T, t0 = prev["cases"], prev["T"], time.time()
    todo = [c for c in cases if not (d / "reg" / f"{c}.pt").exists()]
    for i, (c, prep) in enumerate(each_case(ds, todo)):
        M, o = register_affine(T, prep)
        save_reg(d / "reg" / f"{c}.pt", {"M": M, "o": o})
        if (i + 1) % 50 == 0:
            log(ds, name, f"registered {i + 1}/{len(todo)}", t0)
    regs = unbias({c: load_reg(d / "reg" / f"{c}.pt", {}) for c in cases})
    acc = new_acc()
    for c, prep in each_case(ds, cases):
        accumulate(acc, prep, regs[c])
    _, _, m = stats(acc, len(cases))
    last = it == AFF_ITERS
    out = {"cases": cases, "affine": {c: {k: v.cpu() for k, v in r.items()} for c, r in regs.items()},
           "T": mc_template(acc, len(cases)) if last else aff_template(acc, len(cases)), "metrics": m}
    torch.save(out, d / "template.pt")
    log(ds, name, f"{m}", t0)
    return out


# ---- ROUNDS 3-5: big bends ----------------------------------------------------------------- #
# Every scan gets a new big bend on top of Round 2's stiff move (bends are never stacked).
# Rounds 3-4: average + average arrow -> shape-updated target; teeth totals kept for the
# teeth box. Round 5: builds the sharp median teeth target instead.
def stage_mc(ds, r, prev, aff):
    name = f"mc{r}"
    d = stage_dir(ds, name)
    if done(d):
        return torch.load(d / "template.pt", map_location=DEV, weights_only=False)
    cases, T, t0 = prev["cases"], prev["T"], time.time()
    A = {c: {k: v.to(DEV) for k, v in a.items()} for c, a in aff.items()}
    todo = [c for c in cases if not (d / "reg" / f"{c}.pt").exists()]
    for i, (c, prep) in enumerate(each_case(ds, todo)):
        save_reg(d / "reg" / f"{c}.pt", {**A[c], "u2": register_mc(T, prep, A[c]["M"], A[c]["o"])})
        if (i + 1) % 50 == 0:
            log(ds, name, f"registered {i + 1}/{len(todo)}", t0)
    grids = {"g2": T["g_mc"]}
    out = {"cases": cases, "T_registered_to": T}
    if r < MC_ROUNDS:
        acc, ubar = new_acc(), torch.zeros_like(T["mc"][:3])
        for c, prep in each_case(ds, cases):
            reg = load_reg(d / "reg" / f"{c}.pt", grids)
            accumulate(acc, prep, reg)
            ubar += reg["u2"]
        ubar /= len(cases)
        _, _, m = stats(acc, len(cases))
        m["mean_shape_warp_mm"] = float(ubar.norm(dim=0)[T["mc_mask"]].mean())
        out["T"] = mc_template(acc, len(cases), ubar)
        out["acc_teeth"] = (acc["up"] + acc["lo"]).cpu()          # for the fine box
    else:
        out["T"], m = fine_template(ds, cases, d, grids, prev["acc_teeth"].to(DEV))
    out["metrics"] = m
    torch.save(out, d / "template.pt")
    log(ds, name, f"{m}", t0)
    return out


# SHARP TEETH TARGET (end of round 5): every scan's teeth box read through its round-5 recipe
# at 0.5 mm, only inside its band, averaged per 2x2x2 group into 1 mm (missing if no in-band
# cube); all scans held at once; MEDIAN per cube; plus the 3 label fractions (blurred 0.5 mm)
# and the compared region.
def fine_template(ds, cases, d, grids, teeth05):
    """fine_r1's target: per voxel of the 1 mm teeth box, the MEDIAN over the cases of
    their 0.5 mm values through affine + mc3 (box-averaged to 1 mm), plus the fraction of
    cases with each structure (smoothed 0.5 mm) and the core mask."""
    n = len(cases)
    gf = fine_box({"up": teeth05, "lo": torch.zeros_like(teeth05)}, n)
    g05 = Grid((gf.origin - 0.25).tolist(), (SP05,) * 3, [2 * s for s in gf.shape])
    q = g05.points()
    stack = torch.empty((n, *gf.shape), dtype=torch.float16, device=DEV)
    fr = torch.zeros((3, *gf.shape), device=DEV)
    cov = torch.zeros(gf.shape, device=DEV)
    for i, (c, prep) in enumerate(each_case(ds, cases)):
        p = to_case(load_reg(d / "reg" / f"{c}.pt", grids), q)
        v = sample(prep["img05"], prep["g05"], p)[0]
        m = band_mask(prep, p).float()
        lab = sample(prep["lab05"][None].float(), prep["g05"], p, "nearest")[0]
        ms = F.avg_pool3d(m[None, None], 2)[0, 0]
        vs = F.avg_pool3d((v * m)[None, None], 2)[0, 0]
        stack[i] = torch.where(ms > 0, vs / ms.clamp(min=1e-6), torch.full_like(vs, float("nan"))).half()
        cov += ms
        for j, lv in enumerate(MC_LABELS.values()):
            fr[j] += F.avg_pool3d((lab == lv).float()[None, None], 2)[0, 0]
    med = torch.cat([nanmedian(stack[:, z:z + 4].float()) for z in range(0, gf.shape[0], 4)])
    del stack
    T = {"fine": torch.cat([med[None], blur(fr / n, 0.5)]), "fine_mask": cov > 0.5 * cov.max(), "g_fine": gf}
    return T, {"fine_box_vox": list(gf.shape), "fine_box_origin": gf.origin.tolist()}


# ---- FINE BEND ------------------------------------------------------------------------------ #
# Every scan's round-5 recipe + a fine bend = its complete recipe; quality check recorded.
def stage_fine(ds, mc3):
    d = stage_dir(ds, "fine")
    if done(d):
        return torch.load(d / "template.pt", map_location=DEV, weights_only=False)
    cases, Tf, t0 = mc3["cases"], mc3["T"], time.time()
    Tq = {**mc3["T_registered_to"], **Tf}                # qa: agreement with the mc2 template and the fine target
    grids = {"g2": Tq["g_mc"], "gf": Tf["g_fine"]}
    qa_p = d / "qa.json"
    qa = json.loads(qa_p.read_text()) if qa_p.exists() else {}
    todo = [c for c in cases if not (d / "reg" / f"{c}.pt").exists()]
    for i, (c, prep) in enumerate(each_case(ds, todo)):
        reg = load_reg(stage_dir(ds, "mc3") / "reg" / f"{c}.pt", grids)
        reg["uf"] = register_fine(Tf, prep, reg)
        qa[c] = qa_scores(Tq, prep, reg)
        save_reg(d / "reg" / f"{c}.pt", reg)
        if (i + 1) % 20 == 0 or i + 1 == len(todo):
            qa_p.write_text(json.dumps(qa, indent=0))
            log(ds, "fine", f"registered {i + 1}/{len(todo)}", t0)
    m = {k: float(np.mean([q[k] for q in qa.values()])) for k in next(iter(qa.values()))}
    out = {"cases": cases, "grids": grids, "metrics": m}
    torch.save(out, d / "template.pt")
    log(ds, "fine", f"{m}", t0)
    return out


# ---- FINAL COPY ----------------------------------------------------------------------------- #
def final_p(ds, case, unmasked=False):
    return OUT / RUN / ("final_unmasked" if unmasked else "final") / f"{case}.npy"


# Unmasked version only: what the scanner really measured. Columns (seen from above) holding
# anything other than the corner padding value, filled out to their rubber-band shape (the
# scanner's disc), and the slices that are not all padding.
def with_fov(cpu):
    """The scan's real field of view: axial columns holding any value other than the padding
    (the corner value), filled to their convex hull (the FOV cylinder), minus all-padding slices."""
    from skimage.morphology import convex_hull_image
    vol = cpu["vol"]
    real = vol != vol[vol.shape[0] // 2, 0, 0]
    cpu["fov2d"], cpu["zok"] = convex_hull_image(real.any(0)), real.any((1, 2))
    return cpu


# Unmasked check: yes if the address is inside the scan, inside its disc and on a real slice.
def fov_mask(prep, p):
    """Case points p inside the native volume, its FOV cylinder and a non-padding slice."""
    g = prep["native"]
    idx = torch.round((p - g.origin) / g.spacing).long()
    x, y, z = (idx[..., k].clamp(0, g.shape[2 - k] - 1) for k in range(3))
    return g.inside(p) & prep["fov2d"][y, x] & prep["zok"][z]


# Each ORIGINAL scan read once, through its complete recipe, into the 0.25 mm box (40 slices
# at a time), packed into one uint16 per cube: brightness x 32766 inside the band (or the
# field of view, unmasked), 32767 = missing, + 32768 if tooth. One 328 MB file per scan.
def stage_final(ds, rows, fine, unmasked=False):
    """Each native scan resampled once at 0.25 mm through affine + mc3 + fine. Values outside
    the scan's PX band are MISSING, or with `unmasked` only those outside its field of view."""
    final_p(ds, "x", unmasked).parent.mkdir(parents=True, exist_ok=True)
    cases, t0 = fine["cases"], time.time()
    todo = [c for c in cases if not final_p(ds, c, unmasked).exists()]
    T = {"hi": HI}
    load = (lambda c: with_fov(load_cpu(rows[c]))) if unmasked else (lambda c: load_cpu(rows[c]))
    for i, (c, prep) in enumerate(each_case(ds, todo, load)):
        reg = load_reg(OUT / RUN / "fine" / "reg" / f"{c}.pt", fine["grids"])
        vol = np.empty(HI.shape, np.uint16)
        for z0 in range(0, HI.shape[0], 40):
            z1 = min(z0 + 40, HI.shape[0])
            v, m, t = resample_hi(T, prep, reg, z0, z1)
            if unmasked:
                m = fov_mask(prep, to_case(reg, HI.points(z0, z1)))
            code = torch.where(m, torch.round(v * VAL_MAX), torch.full_like(v, MISSING)) + 32768 * (t > 0.5)
            vol[z0:z1] = code.to(torch.int32).cpu().numpy()
        part = final_p(ds, c, unmasked).with_suffix(".part.npy")
        np.save(part, vol)
        part.replace(final_p(ds, c, unmasked))
        del vol
        log(ds, "final", f"resampled {i + 1}/{len(todo)} {c}", t0)


# ---- COMBINE -------------------------------------------------------------------------------- #
# 4 slices of every scan's final file at a time: unpack (top switch = tooth, lower 15 =
# brightness, 32767 = missing), then per cube: mean, median, std, teeth_prob (= tooth count /
# ALL scans). The number of scans with a value in the cube is used inside but not saved. Results grow in on-disk tables (resumable via _z_done.txt),
# then are written as .nii.gz with the 0.25 mm ruler. PyTorch has no uint16, so the same 16
# switches are read as int16 and turned back with "& 0xFFFF".
def stage_reduce(ds, cases, unmasked=False):
    """Mean, median (np.nanmedian convention), std, teeth fraction per voxel
    (`unmasked`: written as <name>_unmasked.nii.gz, its per-case files deleted afterwards)."""
    d = OUT / RUN / "atlas"
    d.mkdir(parents=True, exist_ok=True)
    sfx = "_unmasked" if unmasked else ""
    names = ("mean", "median", "std", "teeth_prob")
    res = {k: np.lib.format.open_memmap(d / f"_{k}{sfx}.npy", mode="r+" if (d / f"_{k}{sfx}.npy").exists() else "w+",
                                        dtype=np.float32, shape=HI.shape) for k in names}
    prog_p = d / f"_z_done{sfx}.txt"
    z_start = int(prog_p.read_text()) if prog_p.exists() else 0
    files = [np.load(final_p(ds, c, unmasked), mmap_mode="r") for c in cases]
    n, t0 = len(cases), time.time()
    buf = np.empty((n, Z_SLAB, *HI.shape[1:]), np.uint16)
    for z0 in range(z_start, HI.shape[0], Z_SLAB):
        z1 = min(z0 + Z_SLAB, HI.shape[0])
        for i, a in enumerate(files):
            buf[i, :z1 - z0] = a[z0:z1]
        xs = torch.from_numpy(buf[:, :z1 - z0].view(np.int16)).to(DEV)
        for dz in range(z1 - z0):
            x = xs[:, dz].int() & 0xFFFF
            teeth = (x >> 15).float().sum(0)
            v = (x & 0x7FFF).float()
            ok = v != MISSING
            v = torch.where(ok, v / VAL_MAX, torch.full_like(v, float("nan")))
            cnt = ok.float().sum(0)
            s = torch.nan_to_num(v).sum(0)
            mean = s / cnt.clamp(min=1)
            sq = torch.nan_to_num(v * v).sum(0)
            out = {"mean": mean, "median": nanmedian(v), "std": (sq / cnt.clamp(min=1) - mean ** 2).clamp(min=0).sqrt(),
                   "teeth_prob": teeth / n}
            for k in names:
                res[k][z0 + dz] = out[k].cpu().numpy()
        del xs
        for a in res.values():
            a.flush()
        prog_p.write_text(str(z1))
        if z0 % 40 == 0 or z1 == HI.shape[0]:
            log(ds, "reduce", f"z {z1}/{HI.shape[0]}", t0)
    del files, buf, a                                    # a: the last memmap of the flush loop (Windows keeps it open)
    for k in names:
        im = sitk.GetImageFromArray(np.asarray(res[k]))
        im.CopyInformation(HI_REF)
        sitk.WriteImage(im, str(d / f"{k}{sfx}.nii.gz"))
        del im
    del res
    for k in names:
        (d / f"_{k}{sfx}.npy").unlink()
    prog_p.unlink()
    if unmasked:                                          # ~120 GB per dataset; the atlas is what is kept
        for c in cases:
            final_p(ds, c, True).unlink()
    log(ds, "reduce", f"wrote {d}", t0)


# ---- THE CONDUCTOR ---------------------------------------------------------------------------- #
# Round 0 -> stiff rounds 1-2 -> bend rounds 3-5 (all from round 2's stiff moves) -> fine bend
# -> final copy -> combine. A stage whose files exist is skipped, so a stopped run resumes.
# The dataset's name (output sub-folder) is its scan folder's name.
def main(images, labels, n=None, unmasked=False):
    global RUN
    ds = Path(images).name
    RUN = f"{ds}_test{n}" if n else ds
    rows, unpaired = find_cases(Path(images), Path(labels))
    print(f"{ds}: {len(rows)} label files paired with a scan, {len(unpaired)} not paired", flush=True)
    for p, why in unpaired.items():
        print(f"    not paired: {p} ({why})")
    cases = sorted(rows)[:n] if n else sorted(rows)
    t0 = time.time()
    prev = stage_com(ds, rows, cases)
    for it in range(1, AFF_ITERS + 1):
        prev = stage_aff(ds, it, prev)
    aff = prev["affine"]
    for r in range(1, MC_ROUNDS + 1):
        prev = stage_mc(ds, r, prev, aff)
    fine = stage_fine(ds, prev)
    sfx = "_unmasked" if unmasked else ""
    if not (OUT / RUN / "atlas" / f"teeth_prob{sfx}.nii.gz").exists():
        stage_final(ds, rows, fine, unmasked)
        stage_reduce(ds, fine["cases"], unmasked)
    log(ds, "all", "done", t0)


# ===================================================================================== #
#  PART 11 -- COMMAND LINE
# ===================================================================================== #
#   python groupwise_atlas_standalone.py <scan folder> <label folder> [--out DIR] [--n N] [--unmasked]
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="All-scans groupwise CBCT atlas (mean, median, std, teeth_prob).")
    ap.add_argument("images", help="folder holding the CBCT scans (sub-folders included)")
    ap.add_argument("labels", help="folder holding the label files, named like their scans")
    ap.add_argument("--out", default=str(OUT), help="output folder (default: %(default)s)")
    ap.add_argument("--n", type=int, help="quick test on the first N scans (alphabetical order)")
    ap.add_argument("--unmasked", action="store_true", help="the unmasked atlas (run after the masked one)")
    a = ap.parse_args()
    OUT = Path(a.out)
    main(a.images, a.labels, a.n, a.unmasked)
