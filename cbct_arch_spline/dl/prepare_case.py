"""
Per-case data preparation for the arch-spline DL model (v1 -- 3-channel).

Builds per case:
  - mip         : axial MIP (jaw Z-range), square-padded, 256x256
  - label_mip   : binary tooth+jawbone label MIP, same canvas
  - geo_channel : geometric arch rendered as blurred line
  - geo_cp      : (N_CONTROL, 2) geometric arch control points (numeric)
  - curve_cp    : (N_CONTROL, 2) manual spline control points (target)
  - jaw, category, fit_err_px, geo_method, z_range
"""
import importlib.util
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import resize as sk_resize
from skimage.draw import line as sk_line

from catmull_rom_utils import fit_catmull_rom_target, catmull_rom_eval

CANVAS = 256
N_CONTROL = 24
Z_MARGIN_VOX = 8

FDI_UPPER = set(range(11, 29))
FDI_LOWER = set(range(31, 49))


def load_label_matched(img_path, lbl_path):
    """
    Load the label as a (Z, Y, X) array on the SAME grid as the CBCT.

    If the label volume has a different size than the CBCT (e.g. a label from a
    differently-cropped/resampled source), resample it onto the image grid with
    nearest-neighbour interpolation (physical-space correct) so the two arrays
    are element-wise compatible. Prevents shape-broadcast errors like
    "(268,410,410) vs (274,410,410)" downstream.
    """
    import SimpleITK as sitk

    img = sitk.ReadImage(str(img_path))
    lbl = sitk.ReadImage(str(lbl_path))
    if lbl.GetSize() != img.GetSize():
        rs = sitk.ResampleImageFilter()
        rs.SetReferenceImage(img)
        rs.SetInterpolator(sitk.sitkNearestNeighbor)
        rs.SetDefaultPixelValue(0)
        lbl = rs.Execute(lbl)
    return sitk.GetArrayFromImage(lbl).astype(np.int16)


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


def determine_jaw(spline_xy_idx, label_ZYX, z_idx, search_radius=10):
    votes = {"upper": 0, "lower": 0}
    Z, Y, X = label_ZYX.shape
    for x, y in spline_xy_idx:
        xi, yi = int(round(x)), int(round(y))
        found = None
        for r in range(0, search_radius + 1, 2):
            x0, x1 = max(0, xi - r), min(X, xi + r + 1)
            y0, y1 = max(0, yi - r), min(Y, yi + r + 1)
            z0, z1 = max(0, z_idx - r), min(Z, z_idx + r + 1)
            patch = label_ZYX[z0:z1, y0:y1, x0:x1]
            vals = set(np.unique(patch)) - {0, 1}
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
    if jaw == "unknown":
        fdi_set = FDI_UPPER | FDI_LOWER
    else:
        fdi_set = FDI_UPPER if jaw == "upper" else FDI_LOWER
    mask = np.isin(label_ZYX, list(fdi_set))
    if not mask.any():
        return 0, label_ZYX.shape[0]
    z_idx = np.where(mask.any(axis=(1, 2)))[0]
    z0 = max(0, z_idx.min() - pad)
    z1 = min(label_ZYX.shape[0], z_idx.max() + pad + 1)
    return z0, z1


def render_line_channel(canvas_size, x_pts, y_pts, blur_sigma=2.0):
    img = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    pts = np.stack([x_pts, y_pts], axis=1)
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        x0, y0, x1, y1 = (int(round(v)) for v in (x0, y0, x1, y1))
        x0 = np.clip(x0, 0, canvas_size - 1)
        x1 = np.clip(x1, 0, canvas_size - 1)
        y0 = np.clip(y0, 0, canvas_size - 1)
        y1 = np.clip(y1, 0, canvas_size - 1)
        rr, cc = sk_line(y0, x0, y1, x1)
        img[rr, cc] = 1.0
    return gaussian_filter(img, sigma=blur_sigma)


def _pad_and_resize(vol, label, jaw, x_geo, y_geo, pts_idx=None):
    """Shared pad/resize logic for both prepare_case and prepare_for_inference."""
    Z, Y, X = vol.shape
    z0, z1 = jaw_z_range(label, jaw)
    mip = vol[z0:z1].max(axis=0)
    fdi_set = (FDI_UPPER if jaw == "upper"
               else FDI_LOWER if jaw == "lower"
               else FDI_UPPER | FDI_LOWER)
    label_mask = np.isin(label[z0:z1], list(fdi_set) + [1]).astype(np.float32)
    label_mip = label_mask.max(axis=0)

    size = max(X, Y)
    pad_y_total, pad_x_total = size - Y, size - X
    pad_top = pad_y_total // 2
    pad_left = pad_x_total // 2
    pad_bottom = pad_y_total - pad_top
    pad_right = pad_x_total - pad_left

    def pad_sq(img2d):
        return np.pad(img2d,
                      ((pad_top, pad_bottom), (pad_left, pad_right)),
                      mode='constant', constant_values=0)

    scale = CANVAS / size
    mip_r = sk_resize(pad_sq(mip), (CANVAS, CANVAS),
                      order=1, anti_aliasing=True).astype(np.float32)
    lbl_r = sk_resize(pad_sq(label_mip), (CANVAS, CANVAS),
                      order=1, anti_aliasing=True).astype(np.float32)

    geo_x_c = (x_geo + pad_left) * scale
    geo_y_c = (y_geo + pad_top) * scale
    geo_ch = render_line_channel(CANVAS, geo_x_c, geo_y_c)
    geo_cp = fit_catmull_rom_target(
        np.stack([geo_x_c, geo_y_c], axis=1), n_control=N_CONTROL)

    out = dict(mip=mip_r, label_mip=lbl_r, geo_channel=geo_ch,
               geo_cp=geo_cp, z_range=(int(z0), int(z1)),
               pad_left=pad_left, pad_top=pad_top, scale=scale)

    if pts_idx is not None:
        pts_c = (pts_idx + np.array([pad_left, pad_top])) * scale
        cp = fit_catmull_rom_target(pts_c, n_control=N_CONTROL)
        true_dense = catmull_rom_eval(pts_c, n_samples=1000)
        recon = catmull_rom_eval(cp, n_samples=1000)
        d = np.linalg.norm(
            recon[:, None, :] - true_dense[None, :, :], axis=2).min(axis=1)
        out["curve_cp"] = cp
        out["fit_err_px"] = d.mean()

    return out


def prepare_for_inference(mod, img_path, lbl_path, jaw):
    """For new cases without a manual spline. jaw must be explicit."""
    if jaw not in ("lower", "upper"):
        raise ValueError(f"jaw must be 'lower' or 'upper', got {jaw!r}")
    vol, sp, *_ = mod.load_mha(img_path)
    label = load_label_matched(img_path, lbl_path)
    x_geo, y_geo, *_, geo_method, _ = mod.detect_arch(vol, label, n_pts=256)
    category, cls_stats = mod.classify_case(label, vol)
    r = _pad_and_resize(vol, label, jaw, x_geo, y_geo)
    r.update(jaw=jaw, category=category, geo_method=geo_method,
             vol_shape=vol.shape, spacing=sp)
    return r


def prepare_case(mod, img_path, lbl_path, fcsv_path):
    """For training cases with a manual spline."""
    vol, sp, *_ = mod.load_mha(img_path)
    label = load_label_matched(img_path, lbl_path)
    sx, sy, sz = sp[2], sp[1], sp[0]

    pts_phys = load_fcsv_xy(fcsv_path)
    pts_idx = pts_phys / np.array([sx, sy])

    with open(fcsv_path) as f:
        for line in f:
            if not line.startswith('#'):
                z_mm = float(line.strip().split(',')[3])
                break
    z_idx = int(round(z_mm / sz))

    jaw = determine_jaw(pts_idx, label, z_idx)
    x_geo, y_geo, *_, geo_method, _ = mod.detect_arch(vol, label, n_pts=256)
    category, cls_stats = mod.classify_case(label, vol)

    r = _pad_and_resize(vol, label, jaw, x_geo, y_geo, pts_idx=pts_idx)
    r.update(jaw=jaw, category=category, geo_method=geo_method,
             canvas=CANVAS, z_range=r["z_range"])
    return r
