"""
Per-case data preparation for the arch-spline DL model.

For each case with a manual .fcsv spline, builds:
  - mip            : axial MIP of the relevant jaw's Z range, resized to
                      a canonical CANVAS x CANVAS resolution
  - label_mip       : same projection of the binary tooth/bone label mask
  - geo_channel     : the EXISTING geometric centroid-spline rendered as a
                      thin blurred line on the same canvas (hybrid input —
                      network learns the correction, not the arch from
                      scratch)
  - jaw             : 0 = lower, 1 = upper (determined from FDI label
                      proximity to the manual spline points, not filename)
  - curve_cp        : (N_CONTROL, 2) Catmull-Rom control points resampled
                      spline, in the SAME canonical canvas coordinate
                      space as the images (so loss can be computed
                      directly in pixel space)
  - case_id, category (from classify_case)

Canonical canvas size and Bezier degree are fixed constants below so
every case in the dataset is dimensionally consistent for batching.
"""
import importlib.util
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import resize as sk_resize
from skimage.draw import line as sk_line

from catmull_rom_utils import fit_catmull_rom_target, catmull_rom_eval

CANVAS = 256          # canonical output resolution (CANVAS x CANVAS)
N_CONTROL = 24        # Catmull-Rom resample points; ~0.12mm mean / ~0.46mm max
                      # fidelity vs the TRUE annotated curve, with NO global
                      # oscillation risk (unlike the degree-10 Bezier this
                      # replaces -- Catmull-Rom is piecewise/local, matching
                      # the actual curve type drawn in 3D Slicer)
Z_MARGIN_VOX = 8      # padding (voxels) around the relevant jaw's Z extent

FDI_UPPER = set(range(11, 29))   # 11-28
FDI_LOWER = set(range(31, 49))   # 31-48


def load_pipeline_module(pipeline_path):
    spec = importlib.util.spec_from_file_location("drr_pipeline_v4", pipeline_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_fcsv_xy(fcsv_path):
    pts = []
    with open(fcsv_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split(',')
            pts.append((float(parts[1]), float(parts[2])))
    return np.array(pts)


def determine_jaw(spline_xy_idx, label_ZYX, z_idx, search_radius=3):
    """Look up FDI label near each spline point (small cube search since
    points often sit on the inter-tooth jawbone label, not directly on a
    tooth voxel) and majority-vote upper vs lower."""
    votes = {"upper": 0, "lower": 0}
    Z, Y, X = label_ZYX.shape
    for x, y in spline_xy_idx:
        xi, yi = int(round(x)), int(round(y))
        found = None
        for r in range(0, search_radius + 1):
            x0, x1 = max(0, xi - r), min(X, xi + r + 1)
            y0, y1 = max(0, yi - r), min(Y, yi + r + 1)
            z0, z1 = max(0, z_idx - r), min(Z, z_idx + r + 1)
            patch = label_ZYX[z0:z1, y0:y1, x0:x1]
            vals = set(np.unique(patch)) - {0, 1}  # 0=bg, 1=jawbone label
            tooth_vals = [v for v in vals if v in FDI_UPPER or v in FDI_LOWER]
            if tooth_vals:
                found = tooth_vals[0]
                break
        if found is not None:
            if found in FDI_UPPER:
                votes["upper"] += 1
            else:
                votes["lower"] += 1
    if votes["upper"] == 0 and votes["lower"] == 0:
        return "unknown"
    return "upper" if votes["upper"] >= votes["lower"] else "lower"


def jaw_z_range(label_ZYX, jaw, pad=Z_MARGIN_VOX):
    fdi_set = FDI_UPPER if jaw == "upper" else FDI_LOWER
    mask = np.isin(label_ZYX, list(fdi_set))
    if not mask.any():
        return 0, label_ZYX.shape[0]
    z_idx = np.where(mask.any(axis=(1, 2)))[0]
    z0, z1 = max(0, z_idx.min() - pad), min(label_ZYX.shape[0], z_idx.max() + pad + 1)
    return z0, z1


def render_line_channel(canvas_size, x_pts, y_pts, blur_sigma=2.0):
    img = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    pts = np.stack([x_pts, y_pts], axis=1)
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        x0, y0, x1, y1 = (int(round(v)) for v in (x0, y0, x1, y1))
        x0 = np.clip(x0, 0, canvas_size - 1); x1 = np.clip(x1, 0, canvas_size - 1)
        y0 = np.clip(y0, 0, canvas_size - 1); y1 = np.clip(y1, 0, canvas_size - 1)
        rr, cc = sk_line(y0, x0, y1, x1)
        img[rr, cc] = 1.0
    return gaussian_filter(img, sigma=blur_sigma)


def prepare_for_inference(mod, img_path, lbl_path, jaw):
    """Same image/geometric-arch preparation as prepare_case(), but for
    NEW cases with no manual spline available. jaw must be given
    explicitly ('lower' or 'upper') rather than inferred from manual
    points, since there's nothing to infer it from at inference time --
    and given training only had 6 upper-jaw examples vs 155 lower, the
    caller should be making a deliberate choice here, not relying on an
    automatic guess that the model itself can't back up reliably."""
    if jaw not in ("lower", "upper"):
        raise ValueError(f"jaw must be 'lower' or 'upper', got {jaw!r}")

    vol, sp, origin, direction = mod.load_mha(img_path)
    label, _, _, _ = mod.load_mha(lbl_path)
    label = label.astype(np.int16)
    Z, Y, X = vol.shape

    z0, z1 = jaw_z_range(label, jaw)
    mip = vol[z0:z1].max(axis=0)
    fdi_set = FDI_UPPER if jaw == "upper" else FDI_LOWER
    label_mask = np.isin(label[z0:z1], list(fdi_set) + [1]).astype(np.float32)
    label_mip = label_mask.max(axis=0)

    x_geo, y_geo, dx_geo, dy_geo, geo_method, geo_debug = mod.detect_arch(
        vol, label, n_pts=256)

    size = max(X, Y)
    pad_y_total, pad_x_total = size - Y, size - X
    pad_top, pad_left = pad_y_total // 2, pad_x_total // 2
    pad_bottom, pad_right = pad_y_total - pad_top, pad_x_total - pad_left

    def pad_square(img2d):
        return np.pad(img2d, ((pad_top, pad_bottom), (pad_left, pad_right)),
                     mode='constant', constant_values=0)

    mip_sq, label_mip_sq = pad_square(mip), pad_square(label_mip)
    scale = CANVAS / size

    mip_resized = sk_resize(mip_sq, (CANVAS, CANVAS), order=1,
                             anti_aliasing=True).astype(np.float32)
    label_mip_resized = sk_resize(label_mip_sq, (CANVAS, CANVAS), order=1,
                                   anti_aliasing=True).astype(np.float32)

    geo_x_canvas = (x_geo + pad_left) * scale
    geo_y_canvas = (y_geo + pad_top) * scale
    geo_channel = render_line_channel(CANVAS, geo_x_canvas, geo_y_canvas)
    geo_cp = fit_catmull_rom_target(
        np.stack([geo_x_canvas, geo_y_canvas], axis=1), n_control=N_CONTROL)

    category, cls_stats = mod.classify_case(label, vol)

    return {
        "mip": mip_resized, "label_mip": label_mip_resized,
        "geo_channel": geo_channel, "geo_cp": geo_cp, "jaw": jaw,
        "category": category, "geo_method": geo_method,
        "z_range": (int(z0), int(z1)),
        "pad_left": pad_left, "pad_top": pad_top, "scale": scale,
        "vol_shape": vol.shape, "spacing": sp,
    }


def prepare_case(mod, img_path, lbl_path, fcsv_path):
    vol, sp, origin, direction = mod.load_mha(img_path)   # vol: Z,Y,X
    label, _, _, _ = mod.load_mha(lbl_path)
    label = label.astype(np.int16)
    sx, sy, sz = sp[2], sp[1], sp[0]
    Z, Y, X = vol.shape

    pts_phys = load_fcsv_xy(fcsv_path)               # (N,2) in mm (LPS)
    pts_idx = pts_phys / np.array([sx, sy])          # -> voxel index (x,y)

    with open(fcsv_path) as f:
        for line in f:
            if not line.startswith('#'):
                z_mm = float(line.strip().split(',')[3])
                break
    z_idx = int(round(z_mm / sz))

    jaw = determine_jaw(pts_idx, label, z_idx)
    z0, z1 = jaw_z_range(label, jaw)

    mip = vol[z0:z1].max(axis=0)                     # (Y,X)
    fdi_set = FDI_UPPER if jaw == "upper" else FDI_LOWER
    label_mask = np.isin(label[z0:z1], list(fdi_set) + [1]).astype(np.float32)
    label_mip = label_mask.max(axis=0)               # (Y,X)

    # geometric baseline arch (hybrid input channel)
    x_geo, y_geo, dx_geo, dy_geo, geo_method, geo_debug = mod.detect_arch(
        vol, label, n_pts=256)

    # --- pad to SQUARE before resizing (preserves true physical aspect
    # ratio -- the previous version used independent scale_x/scale_y to
    # force a square canvas directly, which visibly distorted the arch
    # shape for any non-square crop, e.g. 463x310 cases) ---
    size = max(X, Y)
    pad_y_total, pad_x_total = size - Y, size - X
    pad_top, pad_left = pad_y_total // 2, pad_x_total // 2
    pad_bottom, pad_right = pad_y_total - pad_top, pad_x_total - pad_left

    def pad_square(img2d):
        return np.pad(img2d, ((pad_top, pad_bottom), (pad_left, pad_right)),
                     mode='constant', constant_values=0)

    mip_sq = pad_square(mip)
    label_mip_sq = pad_square(label_mip)
    scale = CANVAS / size   # single uniform scale -- no distortion

    mip_resized = sk_resize(mip_sq, (CANVAS, CANVAS), order=1,
                             anti_aliasing=True).astype(np.float32)
    label_mip_resized = sk_resize(label_mip_sq, (CANVAS, CANVAS), order=1,
                                   anti_aliasing=True).astype(np.float32)

    geo_x_canvas = (x_geo + pad_left) * scale
    geo_y_canvas = (y_geo + pad_top) * scale
    geo_channel = render_line_channel(CANVAS, geo_x_canvas, geo_y_canvas)
    geo_cp = fit_catmull_rom_target(
        np.stack([geo_x_canvas, geo_y_canvas], axis=1), n_control=N_CONTROL)

    pts_canvas = (pts_idx + np.array([pad_left, pad_top])) * scale
    control_points = fit_catmull_rom_target(pts_canvas, n_control=N_CONTROL)

    # fidelity check: reconstructed Catmull-Rom curve from the resampled
    # control points, vs. the TRUE Catmull-Rom curve through the raw
    # annotated points (not vs. the raw points themselves, since the raw
    # points are sparse/uneven -- this measures resampling fidelity, the
    # analogue of the old Bezier fit_error_mm check)
    true_dense = catmull_rom_eval(pts_canvas, n_samples=1000)
    reconstructed = catmull_rom_eval(control_points, n_samples=1000)
    d = np.linalg.norm(reconstructed[:, None, :] - true_dense[None, :, :],
                       axis=2).min(axis=1)
    mean_err_canvas_px, max_err_canvas_px = d.mean(), d.max()

    category, cls_stats = mod.classify_case(label, vol)

    return {
        "mip": mip_resized,
        "label_mip": label_mip_resized,
        "geo_channel": geo_channel,
        "geo_cp": geo_cp,                       # (N_CONTROL, 2) numeric geometric
                                                 # control points, same representation
                                                 # as curve_cp -- for residual learning
        "jaw": jaw,
        "curve_cp": control_points,            # (N_CONTROL, 2) in canvas px
                                                # -- Catmull-Rom control points
        "canvas": CANVAS,
        "category": category,
        "fit_err_px": mean_err_canvas_px,
        "geo_method": geo_method,
        "z_range": (int(z0), int(z1)),
    }
