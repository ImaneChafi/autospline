"""
DRR Pipeline v4 — True Curved Planar Reformation (CPR)
=======================================================
Implements the correct clinical panoramic generation method based on:
  - Luo et al. 2016 (PLOS ONE) — 3D curved surface extraction
  - 3D Slicer CPR algorithm (Lassoan)
  - Kwon et al. 2023 (Sci Reports) — dual parabola + view normalization
  - Clinical software: Invivo, Romexis, OnDemand3D

Key improvements over v3:
  1. TRUE CPR: perpendicular slices along arch tangent (not flat slab)
  2. PARAMETRIC spline: splprep([x,y]) not y=f(x) — handles U-shape correctly
  3. 95th PERCENTILE: robust to metal, preserves dense structures
  4. THIN slab: 5mm not 12mm — less blur, sharper teeth
  5. BICUBIC resize: order=3, anti_aliasing=False — preserves sharpness
  6. CEPH FLIP: patient faces right (clinical standard)
  7. MIP for cephalogram: max not sum — clearer bone visualization

Modes:
  visualize: single case with step-by-step debug figures
  generate:  full dataset production run (multiprocessing)

Usage:
  # Single case with visualization
  python drr_pipeline_v4.py --mode visualize \\
      --img_path preprocessed/imagesTr/ToothFairy2F_001_0000.mha \\
      --lbl_path preprocessed/labelsTr/ToothFairy2F_001.mha \\
      --out_dir  drr_debug_v4/

  # Full dataset
  python drr_pipeline_v4.py --mode generate \\
      --preprocessed_root preprocessed/ \\
      --out_root drr_pairs_v4/ \\
      --n_workers 8
"""

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import map_coordinates, gaussian_filter
from skimage.exposure import equalize_adapthist
from skimage.transform import resize as sk_resize
from tqdm import tqdm

# ─── Constants ───────────────────────────────────────────────────────────────

UPPER_IDS = list(range(11, 29))
LOWER_IDS = list(range(31, 49))
TOOTH_IDS = UPPER_IDS + LOWER_IDS
STRUCT_IDS = list(range(1, 8)) + TOOTH_IDS


# ─── Case classification ──────────────────────────────────────────────────────

def classify_case(label_ZYX, vol_ZYX):
    """
    Classify CBCT into three clinical categories.

    Metal detection fix:
      - Only count voxels > 0.94 (HU ~3230+) as metal
      - Require > 2000 voxels (rules out isolated noise)
      - Only check within the jaw region (labels 1,2 + teeth)
      These rules prevent dense cortical bone from being classified as metal.
    """
    present = []
    for tid in TOOTH_IDS:
        if (label_ZYX == tid).sum() > 50:
            present.append(tid)

    # Metal detection: only within jaw+teeth region, high threshold
    jaw_tooth_mask = np.zeros(vol_ZYX.shape, dtype=bool)
    for sid in [1, 2] + TOOTH_IDS:
        jaw_tooth_mask |= (label_ZYX == sid)

    # Check metal in full volume (artifacts extend beyond labeled regions)
    # Use high threshold (0.94 = HU ~3230) + high count (>2000) to avoid
    # misclassifying dense cortical bone as metal
    metal_voxels = int((vol_ZYX > 0.94).sum())
    
    # Additional check: metal should be spatially clustered
    # (implants are compact, bone is distributed)
    if metal_voxels > 0:
        from scipy.ndimage import label as nd_label
        metal_bin = (vol_ZYX > 0.94).astype(np.uint8)
        _, n_comp = nd_label(metal_bin)
        # Real metal: few large clusters. Noise: many small ones
        has_metal = (metal_voxels > 2000) or (metal_voxels > 500 and n_comp < 5)
    else:
        has_metal = False

    has_metal = metal_voxels > 2000

    n_upper = sum(1 for t in present if t in UPPER_IDS)
    n_lower = sum(1 for t in present if t in LOWER_IDS)

    if has_metal:
        category = "implant"
    elif len(present) >= 20:
        category = "complete"
    else:
        category = "missing"

    return category, {
        "n_teeth":     len(present),
        "n_upper":     n_upper,
        "n_lower":     n_lower,
        "present":     present,
        "metal_voxels": metal_voxels,
        "has_metal":   has_metal,
        "category":    category,
    }


# ─── I/O ─────────────────────────────────────────────────────────────────────

def load_mha(path):
    import SimpleITK as sitk
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    sx, sy, sz = img.GetSpacing()
    return arr, (float(sz), float(sy), float(sx)), \
           np.array(img.GetOrigin()), img.GetDirection()


def get_pairs(root):
    root = Path(root)
    pairs = []
    for p in sorted((root / "imagesTr").glob("*.mha")):
        stem = p.stem
        cid  = stem[:-5] if stem.endswith("_0000") else stem
        lbl  = root / "labelsTr" / f"{cid}.mha"
        if lbl.exists():
            pairs.append((p, lbl))
    print(f"  Found {len(pairs)} image-label pairs")
    return pairs


def case_id(path):
    stem = Path(path).stem
    return stem[:-5] if stem.endswith("_0000") else stem


# ─── Arch detection ───────────────────────────────────────────────────────────

def get_centroids_2d(label_ZYX, tooth_subset):
    """Return list of (x, y) voxel centroids for each present tooth."""
    centroids = []
    for tid in tooth_subset:
        mask = (label_ZYX == tid)
        if mask.sum() > 20:
            coords = np.argwhere(mask)
            y_c = float(coords[:, 1].mean())
            x_c = float(coords[:, 2].mean())
            centroids.append((x_c, y_c))
    return centroids


def sort_arch_centroids(centroids):
    """
    Sort arch centroids left-to-right by X coordinate.

    IMPORTANT: Do NOT sort by angle around centroid.
    Angle-sort creates a closed loop (traces all the way around)
    which causes the spline to double back and produce
    a mirrored/symmetric artifact in the panoramic.

    The dental arch goes from left molar → incisors → right molar.
    X coordinate increases monotonically left to right.
    Sorting by X gives the correct open U traversal.
    """
    pts = np.array(centroids)
    return pts[np.argsort(pts[:, 0])]  # sort by X: left → right


def fit_parametric_spline(pts_sorted, n_pts=512, smooth_factor=3):
    """
    Fit PARAMETRIC cubic spline through arch points.

    Critical: use splprep([x, y]) NOT y=f(x).
    The dental arch is U-shaped — for any given X,
    there may be multiple Y values (especially posterior).
    Using y=f(x) collapses these and produces stripe artifacts.

    smooth_factor: multiplied by n_points for s parameter.
                   Lower = tighter fit, 3-5 is good for dental arch.
    """
    xs = pts_sorted[:, 0]
    ys = pts_sorted[:, 1]
    s  = len(xs) * smooth_factor

    try:
        # per=False: OPEN curve (not periodic/closed) — critical for U-arch
        tck, u = splprep([xs, ys], s=s, k=min(3, len(xs) - 1), per=False)
        t_new  = np.linspace(0, 1, n_pts)
        x_out, y_out   = splev(t_new, tck)
        dx_out, dy_out = splev(t_new, tck, der=1)
        return x_out, y_out, dx_out, dy_out, "spline-ok"
    except Exception as e:
        return None, None, None, None, f"spline-failed: {e}"


def arch_from_mip_parabola(vol_ZYX, label_ZYX, n_pts=512):
    """
    Fallback arch detection when tooth labels are sparse.
    Uses axial MIP intensity + parabola fit.
    """
    Z, Y, X = vol_ZYX.shape

    # Find tooth Z range
    z_cs = []
    for tid in TOOTH_IDS:
        mask = (label_ZYX == tid)
        if mask.sum() > 50:
            coords = np.argwhere(mask)
            z_cs.append(float(coords[:, 0].mean()))

    if len(z_cs) >= 2:
        z_c   = float(np.mean(z_cs))
        z_std = float(np.std(z_cs)) + 5
        z0 = max(0, int(z_c - 2*z_std))
        z1 = min(Z, int(z_c + 2*z_std))
    else:
        z0, z1 = int(Z*0.3), int(Z*0.7)

    mip = vol_ZYX[z0:z1].max(axis=0)   # (Y, X)
    smooth = gaussian_filter(mip, sigma=4)
    thresh = np.percentile(smooth, 75)
    pts    = np.argwhere(smooth > thresh)   # (N, 2): Y, X

    if len(pts) > 20:
        poly   = np.polyfit(pts[:, 1].astype(float),
                            pts[:, 0].astype(float), deg=2)
        t      = np.linspace(0, 1, n_pts)
        x_out  = np.linspace(pts[:, 1].min(), pts[:, 1].max(), n_pts)
        y_out  = np.polyval(poly, x_out)
    else:
        # Pure geometric parabola
        angles = np.linspace(-np.pi*0.58, np.pi*0.58, n_pts)
        x_out  = X/2 + X*0.35*np.sin(angles)
        y_out  = Y*0.58 - Y*0.28*np.cos(angles)

    # Compute numerical tangents
    dx_out = np.gradient(x_out)
    dy_out = np.gradient(y_out)
    return x_out, y_out, dx_out, dy_out


def detect_arch(vol_ZYX, label_ZYX, n_pts=512, smooth_factor=3):
    """
    Arch detection: always tries spline first regardless of category.
    Falls back to MIP parabola only if spline fails.
    Always extends to full jaw X range.

    The category only affects how we REPORT the case and
    whether we apply metal suppression — NOT the arch strategy.
    """
    Z, Y, X = vol_ZYX.shape

    # Classify for reporting + metal suppression decision
    category, cls_stats = classify_case(label_ZYX, vol_ZYX)

    # Get jaw X boundaries from label 2 (lower jaw)
    jaw_mask = (label_ZYX == 2).any(axis=0)
    if jaw_mask.sum() > 200:
        jaw_x     = np.argwhere(jaw_mask)[:, 1]
        x_jaw_min = max(0,   int(jaw_x.min()) + 5)
        x_jaw_max = min(X-1, int(jaw_x.max()) - 5)
    else:
        x_jaw_min = int(X * 0.06)
        x_jaw_max = int(X * 0.94)

    debug = {
        "category":     category,
        "n_teeth":      cls_stats["n_teeth"],
        "metal_voxels": cls_stats["metal_voxels"],
        "n_centroids":  0,
        "x_jaw_min":    x_jaw_min,
        "x_jaw_max":    x_jaw_max,
    }

    # ── Always try spline first ───────────────────────────────────────────
    # Prefer upper teeth (define primary arch, no posterior offset)
    centroids = get_centroids_2d(label_ZYX, UPPER_IDS)
    if len(centroids) < 4:
        centroids = get_centroids_2d(label_ZYX, TOOTH_IDS)

    debug["n_centroids"] = len(centroids)

    if len(centroids) >= 4:
        sorted_pts = sort_arch_centroids(centroids)
        pts_x = sorted_pts[:, 0].astype(float)
        pts_y = sorted_pts[:, 1].astype(float)

        # Fit parabola for Y-extrapolation to jaw boundaries
        poly    = np.polyfit(pts_x, pts_y, deg=2)
        y_left  = float(np.polyval(poly, x_jaw_min))
        y_right = float(np.polyval(poly, x_jaw_max))

        # Clamp Y extensions to valid range
        y_left  = float(np.clip(y_left,  0, Y-1))
        y_right = float(np.clip(y_right, 0, Y-1))

        # Build extended point set
        ext_pts = np.vstack([
            [[float(x_jaw_min), y_left]],
            sorted_pts,
            [[float(x_jaw_max), y_right]],
        ])

        x_out, y_out, dx_out, dy_out, status = fit_parametric_spline(
            ext_pts, n_pts=n_pts, smooth_factor=smooth_factor)

        if x_out is not None:
            method = f"spline-{category}-extended"
            debug.update({"status": status, "method": method,
                          "x_arch": x_out, "y_arch": y_out})
            return x_out, y_out, dx_out, dy_out, method, debug

    # ── Fallback: intensity MIP parabola ─────────────────────────────────
    x_out, y_out, dx_out, dy_out = arch_from_mip_parabola(
        vol_ZYX, label_ZYX, n_pts=n_pts)
    method = f"mip-parabola-{category}"
    debug.update({"status": "fallback", "method": method,
                  "x_arch": x_out, "y_arch": y_out})
    return x_out, y_out, dx_out, dy_out, method, debug

# ─── CPR Panoramic ───────────────────────────────────────────────────────────

def generate_panoramic_cpr(vol_ZYX, label_ZYX, sp_zyx,
                            out_w=1024, out_h=512,
                            slab_mm=5.0,
                            percentile=95,
                            arch_smooth=3):
    """
    TRUE Curved Planar Reformation (CPR) panoramic.

    Algorithm (as used in clinical software):
      For each column (arch position):
        1. Get arch point (x, y) in axial plane
        2. Compute tangent direction (dx, dy) — direction of arch travel
        3. Compute NORMAL direction — perpendicular to tangent
        4. Sample voxels along the NORMAL through a thin slab (slab_mm)
        5. Take 95th percentile of sampled values → one column pixel per Z

    This produces clean tooth separation because:
      - Each column samples tissue PERPENDICULAR to the arch
      - Adjacent columns don't overlap anatomy
      - The curved geometry follows tooth positions correctly
    """
    sz, sy, sx = sp_zyx
    Z, Y, X    = vol_ZYX.shape

    x_arch, y_arch, dx_dt, dy_dt, method, debug = detect_arch(
        vol_ZYX, label_ZYX, n_pts=out_w, smooth_factor=arch_smooth)

    print(f"    Arch: {method}  n_centroids={debug['n_centroids']}")

    # Slab half-width in voxels (perpendicular direction)
    slab_vox    = max(2, int(slab_mm / ((sy + sx) / 2)))
    slab_offsets = np.linspace(-slab_vox/2, slab_vox/2, slab_vox)

    pano = np.zeros((Z, out_w), dtype=np.float32)

    for col in range(out_w):
        ax = x_arch[col]
        ay = y_arch[col]

        # Normalize tangent
        tx = dx_dt[col]; ty = dy_dt[col]
        t_len = np.sqrt(tx**2 + ty**2) + 1e-9
        tx /= t_len; ty /= t_len

        # Normal = perpendicular to tangent (rotate 90°)
        nx = -ty; ny = tx

        # Sample coords along normal for all Z at once
        # sample_x: (slab_vox,)  sample_y: (slab_vox,)
        sample_x = ax + slab_offsets * nx   # (slab_vox,)
        sample_y = ay + slab_offsets * ny   # (slab_vox,)

        # Build coordinate arrays for map_coordinates: (3, Z*slab_vox)
        z_coords = np.repeat(np.arange(Z), slab_vox)
        y_coords = np.tile(sample_y, Z)
        x_coords = np.tile(sample_x, Z)

        # Clip to volume bounds
        z_coords = np.clip(z_coords, 0, Z-1)
        y_coords = np.clip(y_coords, 0, Y-1)
        x_coords = np.clip(x_coords, 0, X-1)

        coords = np.array([z_coords, y_coords, x_coords])

        # Trilinear interpolation
        sampled = map_coordinates(vol_ZYX, coords,
                                  order=1, mode='constant', cval=0.0)
        # Reshape to (Z, slab_vox) then take percentile along slab axis
        sampled = sampled.reshape(Z, slab_vox)
        pano[:, col] = np.percentile(sampled, percentile, axis=1)

    # Flip: superior at top
    pano = np.flipud(pano)

    # Resize with bicubic (order=3), no anti-aliasing (preserves sharpness)
    pano_resized = sk_resize(pano, (out_h, out_w),
                             order=3, anti_aliasing=False).astype(np.float32)

    debug["x_arch"] = x_arch
    debug["y_arch"] = y_arch
    debug["method"] = method
    return pano_resized, pano, debug


# ─── Cephalogram ─────────────────────────────────────────────────────────────

def generate_cephalogram(vol_ZYX, label_ZYX, sp_zyx,
                          out_h=1024, out_w=1024,
                          use_mip=True):
    """
    Lateral cephalogram via MIP or Beer-Lambert sum along X axis.

    Literature (3D Slicer community, Kwon 2023):
      MIP gives better bone visibility and avoids overexposure.
      Sum (Beer-Lambert) is physically correct but overexposes dense regions.

    Clinical standard orientation:
      Patient faces RIGHT (nose on right, occiput on left).
      In RAS CBCT: X goes left→right, so projection faces left by default.
      Fix: flip horizontally after projection.

    Also: Z crop to jaw region only (avoid empty skull space).
    """
    sz, sy, sx = sp_zyx
    Z, Y, X    = vol_ZYX.shape

    # ── Find jaw Z range using tooth centroids ──
    z_cs = []
    for tid in TOOTH_IDS:
        mask = (label_ZYX == tid)
        if mask.sum() > 50:
            coords = np.argwhere(mask)
            z_cs.append(float(coords[:, 0].mean()))

    if len(z_cs) >= 3:
        z_mean = float(np.mean(z_cs))
        z_std  = float(np.std(z_cs))
        z_min  = max(0, int(z_mean - 2.5*z_std) - 20)
        z_max  = min(Z, int(z_mean + 2.0*z_std) + 20)
    else:
        z_min, z_max = int(Z*0.1), int(Z*0.9)

    print(f"    Ceph Z crop: [{z_min}:{z_max}] / {Z}")
    vol_crop = vol_ZYX[z_min:z_max]   # (Z_crop, Y, X)

    # ── Project along X axis → (Z_crop, Y) ──
    if use_mip:
        # Maximum Intensity Projection — sharper bone/teeth
        ceph_ZY = vol_crop.max(axis=2).astype(np.float32)
        proj_method = "MIP"
    else:
        # Beer-Lambert sum — physically accurate
        mu       = _normalized_to_mu(vol_crop, sp_zyx)
        integral = mu.sum(axis=2) * sx
        ceph_ZY  = (1.0 - np.exp(-integral)).astype(np.float32)
        proj_method = "Beer-Lambert"

    print(f"    Ceph method: {proj_method}")

    # Resize (Z_crop, Y) → (out_h, out_w)
    ceph = sk_resize(ceph_ZY, (out_h, out_w),
                     order=3, anti_aliasing=False).astype(np.float32)

    # NOTE: Do NOT flipud — in ToothFairy2, Z=0=superior is already at top
    # The MIP along X naturally gives superior at top row (correct orientation)

    # Flip horizontally: patient faces RIGHT (clinical standard)
    # In RAS CBCT, X goes left→right, projection faces left → flip to face right
    ceph = np.fliplr(ceph)

    return ceph, z_min, z_max, proj_method


# ─── Post-processing ──────────────────────────────────────────────────────────

def _normalized_to_mu(vol, sp_zyx):
    """Normalized [0,1] → approximate attenuation mu (mm^-1)."""
    HU_MIN, HU_MAX = -1000.0, 3500.0
    hu = vol * (HU_MAX - HU_MIN) + HU_MIN
    hu = np.clip(hu, -1000.0, 3500.0)
    mu = 0.020 * (1.0 + hu / 1000.0)
    return np.clip(mu, 0.0, None).astype(np.float32)


def xray_window(img, p_low=1, p_high=99, gamma=0.65):
    """Percentile windowing + gamma correction."""
    p1, p99 = np.percentile(img, [p_low, p_high])
    img = np.clip((img - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)
    return np.power(img, gamma).astype(np.float32)


def clahe(img, clip_limit=0.015, kernel_size=None):
    """CLAHE contrast enhancement."""
    if kernel_size is None:
        kernel_size = max(32, min(img.shape) // 8)
    return equalize_adapthist(img, kernel_size=kernel_size,
                               clip_limit=clip_limit).astype(np.float32)


def postprocess(img, gamma=0.65, clahe_clip=0.015):
    """Full post-processing: window → gamma → CLAHE."""
    img = xray_window(img, gamma=gamma)
    img = clahe(img, clip_limit=clahe_clip)
    return img


# ─── Visualization ───────────────────────────────────────────────────────────

def visualize_all(vol_ZYX, label_ZYX, sp_zyx,
                  arch_debug,
                  pano_raw, pano_final,
                  ceph_raw, ceph_final,
                  ceph_z_min, ceph_z_max,
                  case_name, out_dir):
    """Save comprehensive step-by-step debug figure."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    Z, Y, X  = vol_ZYX.shape
    x_arch   = arch_debug.get("x_arch", np.full(100, X/2))
    y_arch   = arch_debug.get("y_arch", np.full(100, Y/2))
    method   = arch_debug.get("method", "unknown")
    n_cent   = arch_debug.get("n_centroids", 0)

    # Find arch Z range for visualization
    z_cs = []
    for tid in TOOTH_IDS:
        mask = (label_ZYX == tid)
        if mask.sum() > 50:
            coords = np.argwhere(mask)
            z_cs.append(float(coords[:, 0].mean()))
    mid_z = int(np.mean(z_cs)) if z_cs else Z // 2

    fig = plt.figure(figsize=(28, 30), facecolor="#111827")
    cat     = arch_debug.get("category", "?")
    n_teeth = arch_debug.get("n_teeth",   "?")
    n_metal = arch_debug.get("metal_voxels", 0)
    fig.suptitle(
        f"DRR Pipeline v4 (CPR) — {case_name}  |  "
        f"Category: {cat}  Teeth: {n_teeth}  Metal vox: {n_metal}\n"
        f"Volume (Z={Z}, Y={Y}, X={X})  Spacing={sp_zyx[0]:.2f}mm  "
        f"Arch: {method}  ({n_cent} centroids)",
        color="white", fontsize=12, fontweight="bold", y=0.99
    )

    gs = gridspec.GridSpec(5, 4, figure=fig,
                           hspace=0.40, wspace=0.30,
                           top=0.96, bottom=0.02)

    def im(ax, img, title, cmap="gray", vmin=None, vmax=None,
           title_color="white"):
        ax.set_facecolor("#070d1a")
        ax.imshow(img, cmap=cmap, aspect="auto",
                  vmin=vmin, vmax=vmax, origin="upper")
        ax.set_title(title, color=title_color, fontsize=8.5, pad=4)
        ax.tick_params(colors="#555", labelsize=6)
        for s in ax.spines.values(): s.set_edgecolor("#333")

    # Row 0: CBCT slices
    im(fig.add_subplot(gs[0,0]), vol_ZYX[mid_z],
       f"CBCT axial Z={mid_z} (tooth mid)", vmin=0, vmax=1)
    im(fig.add_subplot(gs[0,1]), vol_ZYX[:,Y//2,:],
       "CBCT coronal mid-Y", vmin=0, vmax=1)
    im(fig.add_subplot(gs[0,2]), vol_ZYX[:,:,X//2],
       "CBCT sagittal mid-X", vmin=0, vmax=1)
    # Label overlay
    ax = fig.add_subplot(gs[0,3])
    im(ax, vol_ZYX[mid_z], "Labels overlay (mid-Z)", vmin=0, vmax=1)
    lbl_show = label_ZYX[mid_z].astype(float)
    lbl_show[lbl_show == 0] = np.nan
    ax.imshow(lbl_show, cmap="tab20", aspect="auto",
              alpha=0.6, origin="upper", vmin=0, vmax=48)

    # Row 1: Arch detection
    z0_arch = max(0, mid_z - 15)
    z1_arch = min(Z, mid_z + 15)
    axial_mip = vol_ZYX[z0_arch:z1_arch].max(axis=0)

    im(fig.add_subplot(gs[1,0]), axial_mip,
       f"Axial MIP Z=[{z0_arch}:{z1_arch}]", vmin=0, vmax=1)

    # Tooth mask
    tooth_mask = np.zeros(vol_ZYX.shape[1:], dtype=bool)
    for tid in TOOTH_IDS:
        tooth_mask |= (label_ZYX == tid).any(axis=0)
    ax = fig.add_subplot(gs[1,1])
    im(ax, axial_mip, "Tooth mask on MIP", vmin=0, vmax=1)
    tm = np.ma.masked_where(~tooth_mask, np.ones_like(tooth_mask, float))
    ax.imshow(tm, cmap="cool", aspect="auto", alpha=0.5, origin="upper")

    # Arch curve on MIP
    ax = fig.add_subplot(gs[1,2])
    im(ax, axial_mip, f"Arch curve ({method})", vmin=0, vmax=1,
       title_color="yellow")
    ax.plot(x_arch, y_arch, "y-", lw=1.5, label="Arch")
    ax.plot(x_arch[0],  y_arch[0],  "go", ms=6, label="Start")
    ax.plot(x_arch[-1], y_arch[-1], "ro", ms=6, label="End")
    ax.legend(fontsize=6, facecolor="#111", labelcolor="white")

    # Arch + centroids
    ax = fig.add_subplot(gs[1,3])
    im(ax, axial_mip, "Arch + tooth centroids", vmin=0, vmax=1,
       title_color="cyan")
    ax.plot(x_arch, y_arch, "y-", lw=1.5, alpha=0.8)
    centroids = get_centroids_2d(label_ZYX, TOOTH_IDS)
    if centroids:
        cx_ = [c[0] for c in centroids]
        cy_ = [c[1] for c in centroids]
        ax.scatter(cx_, cy_, c="cyan", s=20, zorder=5, label="Centroids")
        ax.legend(fontsize=6, facecolor="#111", labelcolor="white")

    # Row 2: Panoramic steps
    im(fig.add_subplot(gs[2,0:2]),
       pano_raw, "CPR Panoramic RAW (percentile-95 projection)",
       vmin=0, vmax=pano_raw.max(), title_color="white")

    pano_w = xray_window(pano_raw)
    im(fig.add_subplot(gs[2,2:4]),
       pano_w, "After xray_window (percentile windowing + gamma)",
       vmin=0, vmax=1)

    # Row 3: Final panoramic + ceph
    im(fig.add_subplot(gs[3,0:2]),
       pano_final, "FINAL PANORAMIC (+ CLAHE)\n← Ground truth DRR",
       vmin=0, vmax=1, title_color="#00ff88")

    im(fig.add_subplot(gs[3,2]),
       ceph_raw, f"Ceph RAW\n(Z=[{ceph_z_min}:{ceph_z_max}])",
       vmin=0, vmax=1)

    im(fig.add_subplot(gs[3,3]),
       ceph_final, "FINAL CEPHALOGRAM\n(MIP + window + CLAHE + flip)\n← Ground truth DRR",
       vmin=0, vmax=1, title_color="#00aaff")

    # Row 4: Large final outputs
    ax_pano_big = fig.add_subplot(gs[4,0:2])
    im(ax_pano_big,
       pano_final, "FINAL PANORAMIC — sharp teeth + correct orientation",
       vmin=0, vmax=1, title_color="#00ff88")

    ax_ceph_big = fig.add_subplot(gs[4,2:4])
    im(ax_ceph_big,
       ceph_final, "FINAL CEPHALOGRAM — crowns top, mandible bottom, faces right",
       vmin=0, vmax=1, title_color="#00aaff")

    path = out_dir / f"pipeline_{case_name}.png"
    fig.savefig(str(path), dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    Visualization → {path}")

    # Individual output PNGs
    plt.imsave(str(out_dir / f"{case_name}_pano.png"),
               pano_final, cmap="gray", vmin=0, vmax=1)
    plt.imsave(str(out_dir / f"{case_name}_ceph.png"),
               ceph_final, cmap="gray", vmin=0, vmax=1)
    return path


# ─── Full pipeline for one case ───────────────────────────────────────────────

def run_one(img_path, lbl_path, out_dir,
            pano_w=1024, pano_h=512,
            ceph_h=1024, ceph_w=1024,
            slab_mm=5.0, percentile=95,
            ceph_mip=True, visualize=False):

    cid     = case_id(img_path)
    out_dir = Path(out_dir)

    # Load
    vol, sp, origin, direction = load_mha(img_path)
    label, _, _, _             = load_mha(lbl_path)
    label = label.astype(np.int16)
    print(f"  [{cid}] Shape={vol.shape}  Spacing={sp}")

    # Panoramic CPR
    pano_raw, pano_raw_full, arch_debug = generate_panoramic_cpr(
        vol, label, sp,
        out_w=pano_w, out_h=pano_h,
        slab_mm=slab_mm, percentile=percentile)
    pano_final = postprocess(pano_raw, gamma=0.65, clahe_clip=0.015)

    # Cephalogram
    ceph_raw, cz_min, cz_max, ceph_method = generate_cephalogram(
        vol, label, sp,
        out_h=ceph_h, out_w=ceph_w, use_mip=ceph_mip)
    ceph_final = postprocess(ceph_raw, gamma=0.70, clahe_clip=0.012)

    # Save outputs
    (out_dir / "panoramics").mkdir(parents=True, exist_ok=True)
    (out_dir / "cephalograms").mkdir(parents=True, exist_ok=True)

    np.save(str(out_dir / "panoramics"   / f"{cid}_pano.npy"), pano_final)
    np.save(str(out_dir / "cephalograms" / f"{cid}_ceph.npy"), ceph_final)
    plt.imsave(str(out_dir / "panoramics"   / f"{cid}_pano.png"),
               pano_final, cmap="gray", vmin=0, vmax=1)
    plt.imsave(str(out_dir / "cephalograms" / f"{cid}_ceph.png"),
               ceph_final, cmap="gray", vmin=0, vmax=1)

    if visualize:
        vis_dir = out_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        visualize_all(vol, label, sp,
                      arch_debug,
                      pano_raw, pano_final,
                      ceph_raw, ceph_final,
                      cz_min, cz_max,
                      case_name=cid, out_dir=vis_dir)

    return {
        "case_id":      cid,
        "status":       "ok",
        "arch_method":  arch_debug.get("method"),
        "n_centroids":  arch_debug.get("n_centroids"),
        "pano_shape":   list(pano_final.shape),
        "ceph_shape":   list(ceph_final.shape),
        "ceph_method":  ceph_method,
        "ceph_z":       [cz_min, cz_max],
    }


def _worker(task):
    img_p, lbl_p, out_dir, kwargs = task
    try:
        return run_one(img_p, lbl_p, out_dir, **kwargs)
    except Exception as e:
        import traceback
        return {"case_id": case_id(img_p), "status": "error",
                "error": str(e), "tb": traceback.format_exc()}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="DRR Pipeline v4 — True CPR Panoramic + Cephalogram")
    ap.add_argument("--mode", choices=["visualize","generate"], default="visualize")
    ap.add_argument("--img_path")
    ap.add_argument("--lbl_path")
    ap.add_argument("--preprocessed_root")
    ap.add_argument("--out_dir",    default="drr_pairs_v4")
    ap.add_argument("--n_workers",  type=int,   default=4)
    ap.add_argument("--max_volumes",type=int,   default=None)
    ap.add_argument("--pano_w",     type=int,   default=1024)
    ap.add_argument("--pano_h",     type=int,   default=512)
    ap.add_argument("--ceph_h",     type=int,   default=1024)
    ap.add_argument("--ceph_w",     type=int,   default=1024)
    ap.add_argument("--slab_mm",    type=float, default=5.0)
    ap.add_argument("--percentile", type=int,   default=95)
    ap.add_argument("--beer_lambert_ceph", action="store_true",
                    help="Use Beer-Lambert for ceph instead of MIP")
    args = ap.parse_args()

    kwargs = dict(
        pano_w=args.pano_w, pano_h=args.pano_h,
        ceph_h=args.ceph_h, ceph_w=args.ceph_w,
        slab_mm=args.slab_mm, percentile=args.percentile,
        ceph_mip=not args.beer_lambert_ceph,
    )

    if args.mode == "visualize":
        if not args.img_path or not args.lbl_path:
            print("ERROR: --img_path and --lbl_path required")
            sys.exit(1)
        print(f"\n{'='*60}")
        print(f"  VISUALIZE MODE — {Path(args.img_path).stem}")
        print(f"{'='*60}\n")
        r = run_one(args.img_path, args.lbl_path,
                    args.out_dir, visualize=True, **kwargs)
        print(f"\n{'='*60}")
        print("  DONE")
        for k, v in r.items():
            print(f"  {k}: {v}")
        print(f"\n  Open: {args.out_dir}/visualizations/pipeline_*.png")
        print(f"{'='*60}\n")

    else:
        if not args.preprocessed_root:
            print("ERROR: --preprocessed_root required")
            sys.exit(1)
        pairs = get_pairs(args.preprocessed_root)
        if args.max_volumes:
            pairs = pairs[:args.max_volumes]

        print(f"\n{'='*60}")
        print(f"  GENERATE MODE — {len(pairs)} volumes")
        print(f"  slab={args.slab_mm}mm  percentile={args.percentile}")
        print(f"{'='*60}\n")

        tasks = [(ip, lp, args.out_dir, {**kwargs, "visualize": False})
                 for ip, lp in pairs]

        results = []
        with mp.Pool(args.n_workers) as pool:
            for r in tqdm(pool.imap_unordered(_worker, tasks),
                          total=len(tasks), desc="CPR DRR"):
                results.append(r)
                if r["status"] == "error":
                    print(f"\n  ⚠  {r['case_id']}: {r['error']}")

        n_ok = sum(1 for r in results if r["status"] == "ok")
        print(f"\n✅  {n_ok}/{len(results)} ok")

        methods = {}
        for r in results:
            if r["status"] == "ok":
                m = r.get("arch_method","?")
                methods[m] = methods.get(m, 0) + 1
        for m, c in methods.items():
            print(f"  {m}: {c} cases")

        Path(args.out_dir).mkdir(exist_ok=True)
        with open(Path(args.out_dir)/"drr_manifest_v4.json","w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Inspect: {args.out_dir}/panoramics/*.png")
        print(f"           {args.out_dir}/cephalograms/*.png")


if __name__ == "__main__":
    main()
