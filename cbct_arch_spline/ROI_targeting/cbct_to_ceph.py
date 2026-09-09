#!/usr/bin/env python
"""Lateral cephalogram synthesis from a CBCT volume (face-RIGHT orientation)."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy.ndimage
from skimage import exposure

TRANSFER_XP = np.array([-200., 300., 800., 1600., 2800., 8000.], np.float32)
TRANSFER_FP = np.array([0., 40., 300., 1100., 2200., 2500.], np.float32)

SOFT_TISSUE_GAIN = 0.25
SOFT_TISSUE_HU = 200.0
AIR_HU = -300.0
OUT_HEIGHT = 1400


def load_volume_ras(path):
    img = nib.as_closest_canonical(nib.load(str(path)))
    volume = np.transpose(np.asanyarray(img.dataobj), (2, 1, 0)).astype(np.float32)
    zooms = img.header.get_zooms()[:3]
    spacing = np.array([zooms[2], zooms[1], zooms[0]], np.float32)
    extent = spacing * np.array(volume.shape, np.float32)
    return volume, spacing, extent


def denoise_volume(volume):
    return scipy.ndimage.median_filter(volume, size=(1, 3, 3)).astype(np.float32)


def hu_to_attenuation(samples):
    atten = np.interp(samples, TRANSFER_XP, TRANSFER_FP).astype(np.float32)
    atten += SOFT_TISSUE_GAIN * (np.clip(samples, AIR_HU, SOFT_TISSUE_HU) - AIR_HU)
    return atten


def detector_grid(volume, spacing, out_height):
    n_s, n_a = volume.shape[0], volume.shape[1]
    extent_s, extent_a = n_s * spacing[0], n_a * spacing[1]
    px_mm = extent_s / out_height
    out_width = max(1, int(round(extent_a / px_mm)))
    ds = -(np.arange(out_height, dtype=np.float32) - (out_height - 1) / 2.0) * px_mm / spacing[0]
    da = (np.arange(out_width, dtype=np.float32) - (out_width - 1) / 2.0) * px_mm / spacing[1]
    return ds, da, px_mm, (float(extent_s), float(extent_a))


def cast_lateral_rays(volume, spacing, out_height=OUT_HEIGHT):
    ds, da, px_mm, extent = detector_grid(volume, spacing, out_height)
    c_s = (volume.shape[0] - 1) / 2.0
    c_a = (volume.shape[1] - 1) / 2.0
    atten = hu_to_attenuation(volume)
    L_native = atten.sum(axis=2) * float(spacing[2])
    rr, cc = np.meshgrid(c_s + ds, c_a + da, indexing="ij")
    L = scipy.ndimage.map_coordinates(L_native, [rr, cc], order=1, mode="constant", cval=0.0)
    return L.astype(np.float32), px_mm, extent


def postprocess(L):
    ref = np.percentile(L[L > 0], 99) if np.any(L > 0) else 1.0
    px = L / (ref + 1e-8)

    lo, hi = np.percentile(px, [1.0, 99.5])
    out = np.clip((px - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    out = 0.06 + (1.0 - 0.06) * out
    px = (out * 255).astype(np.uint8)

    eq = exposure.equalize_adapthist(
        np.clip(px.astype(np.float32) / 255.0, 0, 1),
        clip_limit=0.01, kernel_size=None)
    px = np.clip(eq.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
    return px


def check_aspect(px, extent):
    aspect_img = px.shape[1] / px.shape[0]
    aspect_mm = extent[1] / extent[0]
    assert abs(aspect_img - aspect_mm) < 0.02, (
        f"aspect mismatch: image {aspect_img:.4f} vs mm {aspect_mm:.4f}")


def render_cephalogram(in_path, out_path, out_height=OUT_HEIGHT):
    volume, spacing, _ = load_volume_ras(in_path)
    volume = denoise_volume(volume)
    L, _, extent = cast_lateral_rays(volume, spacing, out_height=out_height)
    px = postprocess(L)
    check_aspect(px, extent)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(str(out_path), px, cmap="gray", vmin=0, vmax=255)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--out-height", type=int, default=OUT_HEIGHT)
    args = parser.parse_args()
    out = render_cephalogram(args.input, args.output, out_height=args.out_height)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# HOW TO RUN
#
#   python cbct_to_ceph.py <input_cbct.nii[.gz]> <output.png> [--out-height 1400]
#
# Inputs
#   input        path to a CBCT volume readable by nibabel (.nii or .nii.gz).
#                Any patient orientation is fine: it is canonicalised to RAS
#                internally, so MMDental ('L','A','S') and SD-Tooth ('L','P','S')
#                both render with the profile facing RIGHT (anterior on the right
#                of the image, cervical spine on the left).
#   output       path to write the grayscale PNG cephalogram to. Parent folders
#                are created automatically.
#   --out-height output image height in pixels (default 1400); width is derived
#                from the volume's real mm extents to keep the aspect ratio true.
#
# Example (reproduces D:\cephalograms\MMDental_faceR\41.png):
#
#   python cbct_to_ceph.py D:\MMDental\41\41.nii.gz D:\cephalograms\MMDental_faceR\41.png
#
# Requires: numpy, scipy, nibabel, scikit-image, matplotlib.
# ---------------------------------------------------------------------------
