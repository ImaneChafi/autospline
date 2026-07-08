import os
import csv
from typing import Union
from collections import deque

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.ndimage
import scipy.interpolate
from skimage.morphology import skeletonize as _skeletonize
from skimage.morphology import binary_closing as _binary_closing, disk as _disk


def apply_bone_window(volume: np.ndarray, bone_min: int = 200, bone_max: int = 800) -> np.ndarray:
    """
    Clips the volume to the cortical bone HU range and shifts the floor to zero.

    Tissue mapping after clipping:
        Air/soft tissue  (<= bone_min)  →  0
        Cortical bone      (bone_min–bone_max)  →  1 … (bone_max − bone_min)
        Enamel/metal     (>= bone_max)  →  bone_max − bone_min  (same as densest bone)

    All hard tissue is therefore capped at the same ceiling value, making the
    MeIP profile reflect bone presence rather than peak intensity.
    """
    return (np.clip(volume, bone_min, bone_max) - bone_min).astype(np.float32)


def find_MIPs(image_obj, axis: Union[str, int]='axial', show: bool=True):
    """Returns the MIP array along a given axis and np_image. Possible axis are the axial, coronal and
    sagittal axis. The 'show' variable determines whether matplotlib visualizes the 2D image."""
    ax = {'axial': 0, 'coronal': 1, 'sagittal': 2}
    if isinstance(axis, str):
        axis = ax.get(axis.lower(), 0)
    elif axis not in (0, 1, 2):
        axis = 0

    mip = np.max(image_obj, axis=axis)

    if show:
        plt.imshow(mip, cmap="gray")
        plt.title("MIP")
        plt.axis("off")
        plt.show()
    return mip


def find_MeIPs(image_obj, axis: Union[str, int]='axial', show: bool=True,
               bone_min: int = 200, bone_max: int = 800):
    """Returns the Mean Intensity Projection along a given axis. Possible axes are the axial, coronal and
    sagittal axis. The 'show' variable determines whether matplotlib visualizes the 2D image."""
    ax = {'axial': 0, 'coronal': 1, 'sagittal': 2}
    if isinstance(axis, str):
        axis = ax.get(axis.lower(), 0)
    elif axis not in (0, 1, 2):
        axis = 0

    mip = np.mean(apply_bone_window(image_obj, bone_min, bone_max), axis=axis)

    if show:
        plt.imshow(mip, cmap="gray")
        plt.title("MeIP")
        plt.axis("off")
        plt.show()

    return mip


def flip_volume_sagittal(volume: np.ndarray) -> np.ndarray:
    """
    Rotates the volume 180° around the sagittal axis.

    Parameters
    ----------
    volume: ndarray (Z, Y, X)

    Returns
    -------
    ndarray, same shape as input, C-contiguous, with corrected orientation
    """
    return np.flip(volume, axis=(0, 1)).copy()


def _localize_jaws_edentulous(volume, meip_coronal, z_spacing_mm, cortical_s,
                              air_hu, open_max_mm, sup_margin_mm, inf_margin_mm, show,
                              jaw_half_mm=45.0):
    """
    Edentulous fallback for jaw Z-localization. Triggered by
    _find_coronal_roi_enamel when the enamel signal is negligible.

    Strategy:
    anchor on the oral-cavity air band then bracket it with the nearest flanking
    cortical-bone peaks (the maxillary palate/alveolus above and the mandibular
    body below).
    """
    Z, Y, X = volume.shape
    y0, y1 = Y // 4, 3 * Y // 4
    x0, x1 = X // 3, 2 * X // 3
    air_frac = (volume[:, y0:y1, x0:x1] < air_hu).mean(axis=(1, 2))
    air_s = scipy.ndimage.gaussian_filter1d(air_frac, 3)

    z_lo, z_hi = int(0.20 * Z), int(0.85 * Z)
    z_occ = z_lo + int(np.argmax(air_s[z_lo:z_hi]))   # oral-cavity air anchor

    jaw_half = int(round(jaw_half_mm/z_spacing_mm))
    distance = max(5, int(round(8.0/z_spacing_mm)))
    peaks, _ = scipy.signal.find_peaks(cortical_s, distance=distance)
    above = peaks[(peaks < z_occ) & (peaks >= z_occ - jaw_half)]
    below = peaks[(peaks > z_occ) & (peaks <= z_occ + jaw_half)]
    fallback_off = int(round(10.0/z_spacing_mm))
    z_sup = int(above[-1]) if len(above) else max(0, z_occ - fallback_off)
    z_inf = int(below[0]) if len(below) else min(Z - 1, z_occ + fallback_off)

    z_top    = max(0, z_sup - int(round(sup_margin_mm/z_spacing_mm)))
    z_bottom = min(Z - 1, z_inf + int(round(inf_margin_mm/z_spacing_mm)))

    if show:
        _, (ax_img, ax_prof) = plt.subplots(1, 2, figsize=(13, 6))
        ax_img.imshow(meip_coronal, cmap='gray', aspect='auto')
        for z, c, ls, lab in [(z_top, 'cyan', '--', 'ROI top'),
                              (z_sup, 'yellow', '-', 'Maxilla bone'),
                              (z_occ, 'red', ':', 'Oral-cavity air'),
                              (z_inf, 'orange', '-', 'Mandible bone'),
                              (z_bottom, 'cyan', '--', 'ROI bottom')]:
            ax_img.axhline(z, color=c, linestyle=ls, linewidth=1, label=lab)
        ax_img.legend(fontsize=7, loc='upper right')
        ax_img.set_title('Coronal MeIP — edentulous (air-anchored) jaw ROI')
        ax_img.axis('off')
        cn = cortical_s/(cortical_s.max() + 1e-6)
        an = air_s/(air_s.max() + 1e-6)
        ax_prof.plot(cn, np.arange(Z), color='black', linewidth=1.2, label='cortical bone')
        ax_prof.plot(an, np.arange(Z), color='royalblue', linewidth=1.0, label='central air')
        for z, c, ls in [(z_sup, 'yellow', '-'), (z_occ, 'red', ':'), (z_inf, 'orange', '-')]:
            ax_prof.axhline(z, color=c, linestyle=ls, linewidth=1)
        ax_prof.invert_yaxis()
        ax_prof.set_xlabel('Normalised count')
        ax_prof.set_ylabel('Z slice index')
        ax_prof.set_title('Edentulous fallback profiles')
        ax_prof.legend(fontsize=7)
        plt.tight_layout()
        plt.show()

    return {
        'z_top': z_top,
        'z_top_raw': z_sup,
        'z_maxilla_peak': z_sup,
        'z_gap': z_occ,
        'z_mandible_peak': z_inf,
        'z_bottom_raw': z_inf,
        'z_bottom': z_bottom,
        'margin_slices': int(round(sup_margin_mm/z_spacing_mm)),
        'z_spacing_mm': z_spacing_mm,
        'method': 'edentulous_air_anchor',
    }


def _find_coronal_roi_enamel(volume, meip_coronal, z_spacing_mm,
                             enamel_hu, enamel_hu_max,
                             open_max_mm, sup_margin_mm, inf_margin_mm, show,
                             cortical_hu=350, air_hu=-400, min_enamel_ratio=0.04):
    """
    Enamel-band jaw localization. Counts enamel-HU voxels per
    Z-slice → dentition profile, picks the two tallest occlusal peaks within
    open_max_mm of the global max (one when the bite is closed), and brackets them
    with anatomical margins. Returns the same dict shape as find_coronal_roi.
    """
    band = (volume > enamel_hu) & (volume < enamel_hu_max)
    prof = scipy.ndimage.gaussian_filter1d(band.sum(axis=(1, 2)).astype(float), 3)

    cortical = ((volume > cortical_hu) & (volume < enamel_hu_max)).sum(axis=(1, 2)).astype(float)
    cortical_s = scipy.ndimage.gaussian_filter1d(cortical, 3)

    enamel_ratio = float(prof.max()/(cortical_s.max() + 1e-6))
    if prof.max() <= 0 or enamel_ratio < min_enamel_ratio:
        return _localize_jaws_edentulous(
            volume, meip_coronal, z_spacing_mm, cortical_s, air_hu,
            open_max_mm, sup_margin_mm, inf_margin_mm, show)

    pn = prof/prof.max()

    z_occ = int(np.argmax(pn))
    distance = max(5, int(round(8.0/z_spacing_mm)))
    peaks, _ = scipy.signal.find_peaks(pn, prominence=0.12, distance=distance)
    if len(peaks) == 0:
        peaks = np.array([z_occ])

    # The opposing arch is the next-tallest peak within a plausible bite opening.
    open_max = int(round(open_max_mm/z_spacing_mm))
    cand = peaks[np.abs(peaks - z_occ) <= open_max]
    if len(cand) >= 2:
        two = cand[np.argsort(pn[cand])[-2:]]
        z_sup, z_inf = int(min(two)), int(max(two))   # superior/inferior occlusal
    else:
        z_sup = z_inf = z_occ                           # closed bite → fused arches

    z_top    = max(0, z_sup - int(round(sup_margin_mm/z_spacing_mm)))
    z_bottom = min(volume.shape[0] - 1, z_inf + int(round(inf_margin_mm/z_spacing_mm)))

    if show:
        _, (ax_img, ax_prof) = plt.subplots(1, 2, figsize=(13, 6))
        ax_img.imshow(meip_coronal, cmap='gray', aspect='auto')
        for z, c, ls, lab in [(z_top, 'cyan', '--', 'ROI top'),
                              (z_sup, 'yellow', '-', 'Maxilla occlusal'),
                              (z_occ, 'red', ':', 'Densest enamel'),
                              (z_inf, 'orange', '-', 'Mandible occlusal'),
                              (z_bottom, 'cyan', '--', 'ROI bottom')]:
            ax_img.axhline(z, color=c, linestyle=ls, linewidth=1, label=lab)
        ax_img.legend(fontsize=7, loc='upper right')
        ax_img.set_title('Coronal MeIP — enamel-band jaw ROI')
        ax_img.axis('off')

        ax_prof.plot(pn, np.arange(len(pn)), color='black', linewidth=1.2)
        ax_prof.plot(pn[peaks], peaks, 'rx', ms=7, label='enamel peaks')
        for z, c, ls in [(z_top, 'cyan', '--'), (z_sup, 'yellow', '-'),
                         (z_inf, 'orange', '-'), (z_bottom, 'cyan', '--')]:
            ax_prof.axhline(z, color=c, linestyle=ls, linewidth=1)
        ax_prof.invert_yaxis()
        ax_prof.set_xlabel('Normalised enamel-voxel count')
        ax_prof.set_ylabel('Z slice index')
        ax_prof.set_title('Dentition profile (enamel band)')
        ax_prof.legend(fontsize=7)
        plt.tight_layout()
        plt.show()

    return {
        'z_top': z_top,
        'z_top_raw': z_sup,
        'z_maxilla_peak': z_sup,
        'z_gap': z_occ,
        'z_mandible_peak': z_inf,
        'z_bottom_raw': z_inf,
        'z_bottom': z_bottom,
        'margin_slices': int(round(sup_margin_mm/z_spacing_mm)),
        'z_spacing_mm': z_spacing_mm,
        'enamel_hu': enamel_hu,
        'enamel_hu_max': enamel_hu_max,
    }


def find_coronal_roi(meip_coronal, volume=None, z_spacing_mm=1.0,
                     enamel_hu=1800, enamel_hu_max=5000,
                     open_max_mm=20.0, sup_margin_mm=20.0, inf_margin_mm=30.0,
                     cortical_hu=350, air_hu=-400, min_enamel_ratio=0.04,
                     smooth_sigma=5, threshold_fraction=0.15, min_margin_slices=5, show=True):
    """
    Finds the jaw Z-extent.

    Two methods:

    * Enamel-band (recommended; used when  'volume ' is given) — FOV-independent and
      robust to a closed mouth.  The mean-intensity method below keys on the two
      tallest *bone-mass* peaks, which fails on whole-skull scans (the cranial vault
      outweighs the jaws) and on a closed mouth (the arches fuse into one peak).
      Tooth enamel is the densest consistent structure, so counting voxels in an
      enamel HU band ([enamel_hu, enamel_hu_max], the upper cap rejecting surgical
      metal) per Z-slice gives a clean dentition profile: the cranium vanishes and
      each arch's occlusal plane is a sharp peak.  The two tallest peaks within
      open_max_mm of the global maximum are the maxillary/mandibular occlusal
      planes (a single peak when the bite is closed and the arches coincide).  The
      ROI brackets them with anatomical margins (sup_margin_mm up toward the sinus
      floor, inf_margin_mm down toward the mandible base).

    * Mean-intensity (fallback;  'volume=None ') — collapse the coronal MeIP to a 1D
      intensity-vs-Z profile, take the two tallest peaks as maxilla/mandible (or a
      single fused peak), scan outward to per-peak thresholds, and apply an adaptive
      margin from the mandible body height.

    Parameters
    ----------
    meip_coronal: ndarray, shape (Z, X) — used for visualization (and the
                         fallback method's profile)
    volume: (Z, Y, X) raw CBCT (HU). Pass it to use the enamel method.
    z_spacing_mm: axial slice spacing, for the mm-based enamel margins
    enamel_hu: lower HU bound of the enamel band
    enamel_hu_max: upper HU bound (rejects extreme metal/surgical hardware)
    open_max_mm: max inter-arch separation searched for the opposing arch
    sup_margin_mm: margin above the maxillary occlusal plane
    inf_margin_mm: margin below the mandibular occlusal plane
    smooth_sigma: (fallback) median filter size and Savitzky-Golay window
    threshold_fraction: (fallback) fraction of each jaw's own peak used as stop threshold
    min_margin_slices: floor for the adaptive margin
    show: plot the MeIP with ROI overlays and the 1D profile

    Returns
    -------
    dict with keys: z_top, z_top_raw, z_maxilla_peak, z_gap,
                    z_mandible_peak, z_bottom_raw, z_bottom, margin_slices
    """
    if volume is not None:
        return _find_coronal_roi_enamel(
            volume, meip_coronal, z_spacing_mm, enamel_hu, enamel_hu_max,
            open_max_mm, sup_margin_mm, inf_margin_mm, show,
            cortical_hu=cortical_hu, air_hu=air_hu, min_enamel_ratio=min_enamel_ratio)
    profile = meip_coronal.mean(axis=1)
    profile_denoised = scipy.ndimage.median_filter(profile, size=smooth_sigma)
    profile_smooth = scipy.signal.savgol_filter(profile_denoised, window_length=smooth_sigma * 4 + 1, polyorder=3)

    p_max = profile_smooth.max()

    peaks, _ = scipy.signal.find_peaks(
        profile_smooth,
        prominence=threshold_fraction * p_max,
        distance=20,
    )
    if len(peaks) == 0:
        raise ValueError("No jaw peak found. Try reducing threshold_fraction or smooth_sigma.")

    if len(peaks) >= 2:
        # The two tallest peaks: maxilla (lower z = superior), mandible (higher z = inferior)
        top2 = peaks[np.argsort(profile_smooth[peaks])[-2:]]
        z_maxilla_peak, z_mandible_peak = np.sort(top2)
    else:
        # Closed-mouth/fused arches: when the upper and lower teeth intersect
        # vertically there is no inter-arch air gap, so both arches project to a
        # single intensity peak. Treat that peak as the centre of the combined
        # dental complex similarly to Zotero article methods.
        z_single = int(peaks[np.argmax(profile_smooth[peaks])])
        z_maxilla_peak = z_mandible_peak = z_single

    # Inter-arch gap: deepest point between the two peaks
    z_gap = z_maxilla_peak + int(np.argmin(profile_smooth[z_maxilla_peak:z_mandible_peak + 1]))

    # Per-peak thresholds: each jaw's extension is measured against its own peak,
    # not a global value. This handles asymmetric jaw densities (e.g. edentulous
    # patients where the mandible peak is much lower than the maxilla peak).
    threshold_sup = threshold_fraction * float(profile_smooth[z_maxilla_peak])
    threshold_inf = threshold_fraction * float(profile_smooth[z_mandible_peak])

    # Raw threshold crossings, without any margin
    z_top_raw = 0
    for z in range(z_maxilla_peak, -1, -1):
        if profile_smooth[z] < threshold_sup:
            z_top_raw = z
            break

    z_bottom_raw = len(profile_smooth) - 1
    for z in range(z_mandible_peak, len(profile_smooth)):
        if profile_smooth[z] < threshold_inf:
            z_bottom_raw = z
            break

    # Margin = half the distance from the mandible peak (yellow line) to its
    # lower threshold crossing (cyan dotted line). The final ROI bounds are
    # this margin above/below the jaw peaks themselves, not the raw crossings.
    # We estimate half the height of the mandible corresponds to the height of
    # the alveolar bone.
    mandible_height = z_bottom_raw - z_mandible_peak
    margin = max(min_margin_slices, mandible_height // 2)

    z_top = max(0, z_maxilla_peak - margin)
    z_bottom = min(len(profile_smooth) - 1, z_mandible_peak + margin)

    if show:
        _, (ax_img, ax_prof) = plt.subplots(1, 2, figsize=(13, 6))

        ax_img.imshow(meip_coronal, cmap='gray', aspect='auto')
        # Shade the margin contribution (jaw peak → final ROI bound)
        ax_img.axhspan(z_top,          z_maxilla_peak,  color='orange', alpha=0.18, label=f'Margin (±{margin} slices)')
        ax_img.axhspan(z_mandible_peak, z_bottom,        color='orange', alpha=0.18)
        # Shade the inter-peak region (maxilla peak → mandible peak, includes air gap)
        ax_img.axhspan(z_maxilla_peak, z_mandible_peak,  color='cyan',   alpha=0.10, label='Inter-peak region')
        for z, color, ls, label in [
            (z_top, 'cyan', '--', f'ROI top'),
            (z_top_raw, 'cyan', ':', 'Maxilla crossing'),
            (z_maxilla_peak, 'yellow', '-', 'Maxilla peak'),
            (z_gap, 'red', '-', 'Inter-arch gap'),
            (z_mandible_peak, 'yellow', '-', 'Mandible peak'),
            (z_bottom_raw, 'cyan', ':', 'Mandible crossing'),
            (z_bottom, 'cyan', '--', 'ROI bottom'),
        ]:
            ax_img.axhline(z, color=color, linestyle=ls, linewidth=1, label=label)
        ax_img.legend(fontsize=7, loc='upper right')
        ax_img.set_title('Coronal MeIP — jaw ROI')
        ax_img.axis('off')

        ax_prof.set_facecolor('black')
        # Shade margin regions on profile (jaw peak → final ROI bound)
        ax_prof.axhspan(z_top, z_maxilla_peak, color='orange', alpha=0.25, label=f'Margin (±{margin} slices)')
        ax_prof.axhspan(z_mandible_peak, z_bottom, color='orange', alpha=0.25)
        ax_prof.axhspan(z_maxilla_peak, z_mandible_peak, color='cyan', alpha=0.10, label='Inter-peak region')
        ax_prof.plot(profile_smooth, np.arange(len(profile_smooth)), color='white', linewidth=1.5)
        ax_prof.axvline(threshold_sup, color='yellow', linestyle=':', linewidth=1,
                        label=f'Threshold sup = {threshold_sup:.1f}')
        ax_prof.axvline(threshold_inf, color='orange', linestyle=':', linewidth=1,
                        label=f'Threshold inf = {threshold_inf:.1f}')
        for z, color, ls in [
            (z_top, 'cyan', '--'),
            (z_top_raw, 'cyan', ':'),
            (z_maxilla_peak, 'yellow', '-'),
            (z_gap, 'red', '-'),
            (z_mandible_peak, 'yellow', '-'),
            (z_bottom_raw, 'cyan', ':'),
            (z_bottom, 'cyan', '--'),
        ]:
            ax_prof.axhline(z, color=color, linestyle=ls, linewidth=1)
        ax_prof.invert_yaxis()
        ax_prof.set_xlabel('Mean intensity')
        ax_prof.set_ylabel('Z slice index')
        ax_prof.set_title('Vertical profile — threshold + margin composition')
        ax_prof.legend(fontsize=7)

        plt.tight_layout()
        plt.show()

    return {
        'z_top': z_top,
        'z_top_raw': z_top_raw,
        'z_maxilla_peak': z_maxilla_peak,
        'z_gap': z_gap,
        'z_mandible_peak': z_mandible_peak,
        'z_bottom_raw': z_bottom_raw,
        'z_bottom': z_bottom,
        'margin_slices': margin,
        'threshold_sup': threshold_sup,
        'threshold_inf': threshold_inf,
    }


def _enamel_arch_mask(sub_volume, z_spacing_mm, enamel_hu=1800, enamel_hu_max=5000,
                      frac=0.06, close_mm=4.0):
    """
    Arch footprint from the enamel signal alone, for whole-skull/closed-bite scans
    where the bone window fills the whole facial cross-section.

    Counts enamel-HU voxels along Z → an (x,y) tooth-density map on which only the
    teeth light up (facial bone does not reach the enamel band). The teeth project as
    a dotted arch, so a morphological closing (radius close_mm) bridges the
    inter-tooth gaps into one connected curve before the largest component is kept.
    """
    band = ((sub_volume > enamel_hu) & (sub_volume < enamel_hu_max)).mean(axis=0)
    if band.max() <= 0:
        return np.zeros(band.shape, dtype=bool)
    mask = band > frac * band.max()
    r = max(2, int(round(close_mm/z_spacing_mm)))
    mask = _binary_closing(mask, _disk(r))
    lbl = scipy.ndimage.label(mask)[0]      # type: ignore[index]
    sizes = np.bincount(lbl.ravel()); sizes[0] = 0
    if sizes.max() == 0:
        return mask
    return lbl == int(np.argmax(sizes))


def find_arch_footprint(volume, roi, bone_min=200, bone_max=800, threshold_fraction=0.15,
                        min_patch_size=200, blob_extent=0.55, show=True):
    """
    Projects the coronal ROI along Z (mean intensity) to reveal the dental arch
    footprint in the (Y, X) plane.

    The arch stacks bone across many Z-slices (teeth, alveolar bone, cortical
    plates), so its mean value is high. The palate is a thin plate spanning only a
    few slices, so its mean value is lower. Thresholding the MeIP separates them.

    Parameters
    ----------
    volume: ndarray (Z, Y, X)
    roi: dict from find_coronal_roi
    bone_min/bone_max: bone window applied before averaging
    threshold_fraction: fraction of the MeIP peak used as the binary threshold
    min_patch_size: 2D connected components smaller than this (pixels) are removed
    blob_extent: bounding-box fill ratio above which the bone-window mask is
                         treated as a whole-skull blob and the enamel fallback is used
    show: display the MeIP and the arch mask

    Returns
    -------
    meip: ndarray (Y, X) — mean intensity projection along Z within the ROI
    mask2d: bool ndarray (Y, X) — arch footprint mask
    """
    z_top    = roi['z_top']
    z_bottom = roi['z_bottom']

    # Detect the arch SHAPE from only the tooth-bearing occlusal band, not the full render
    # ROI. Over a tall ROI the mean projection superimposes the dental arch with the
    # mandibular rami and skull base, which flare outward inferiorly and join at the back —
    # closing the open horseshoe into a ring whose skeleton is a loop (→ a looped spline).
    # A band centred on the two occlusal planes keeps just the teeth/alveolar arch; the
    # render still uses the full z_top..z_bottom extent for the roots.
    if 'z_maxilla_peak' in roi and 'z_mandible_peak' in roi:
        zc0, zc1 = sorted((int(roi['z_maxilla_peak']), int(roi['z_mandible_peak'])))
        half = max(int(round((zc1 - zc0) * 0.6)), 8)
        z_top    = int(max(z_top, zc0 - half))
        z_bottom = int(min(z_bottom, zc1 + half))

    sub_vol = apply_bone_window(volume[z_top:z_bottom + 1], bone_min, bone_max)
    meip = sub_vol.mean(axis=0)          # (Y, X)

    threshold = threshold_fraction * float(meip.max())
    mask2d    = meip > threshold

    # Remove small islands
    lbl   = scipy.ndimage.label(mask2d)[0]  # type: ignore[index]
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    mask2d = sizes[lbl] >= min_patch_size

    # Keep only the single largest component
    lbl   = scipy.ndimage.label(mask2d)[0]  # type: ignore[index]
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    mask2d = lbl == int(np.argmax(sizes))

    # Blob guard: a true arch is a thin curve that fills little of its bounding box;
    # a whole-skull bone blob fills most of it. Fall back to the enamel-only arch.
    ys, xs = np.where(mask2d)
    if ys.size:
        bbox_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
        if mask2d.sum()/bbox_area > blob_extent:
            mask2d = _enamel_arch_mask(
                volume[z_top:z_bottom + 1], roi.get('z_spacing_mm', 0.3),
                roi.get('enamel_hu', 1800), roi.get('enamel_hu_max', 5000))

    if show:
        overlay = np.zeros(meip.shape + (4,), dtype=np.float32)
        overlay[mask2d] = [1.0, 0.45, 0.0, 0.6]

        fig, (ax_mip, ax_mask) = plt.subplots(1, 2, figsize=(12, 5))

        ax_mip.imshow(meip, cmap='gray', aspect='auto')
        ax_mip.set_title(f'Axial MeIP  (z={z_top}–{z_bottom})', fontsize=9)
        ax_mip.axis('off')

        ax_mask.imshow(meip, cmap='gray', aspect='auto')
        ax_mask.imshow(overlay, aspect='auto')
        ax_mask.set_title(f'Arch footprint  (threshold={threshold:.1f})', fontsize=9)
        ax_mask.axis('off')

        fig.suptitle(f'threshold_fraction={threshold_fraction}', fontsize=9)
        plt.tight_layout()
        plt.show()

    return meip, mask2d


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component in a binary mask.

    Parameters:
        mask (np.ndarray): binary mask (bool or 0/1)

    Returns:
        np.ndarray: binary mask with only the largest connected object
    """

    # 1. Ensure binary
    mask = mask.astype(bool)

    # 2. Label connected components (8-connectivity for 2D)
    structure = np.ones((3, 3), dtype=int)  # defines connectivity

    labeled, num_features = scipy.ndimage.label(mask, structure=structure)

    if num_features == 0:
        return mask.copy()  # empty input

    # 3. Count size of each component
    sizes = np.bincount(labeled.ravel())

    # ignore background (label 0)
    sizes[0] = 0

    # 4. Find largest component label
    largest_label = sizes.argmax()

    # 5. Keep only that component
    output = (labeled == largest_label)

    return output


def _smooth_arch_adaptive(mask):
    """
    Light, thickness-adaptive arch smoothing for the one-call entry point.

     'smooth_arch_footprint' opens with a fixed 10 px disk, which is right for the
    thick bone-window arch but erodes a thin enamel-derived arch 
    away entirely, leaving only a lump and a degenerate spline.
    """
    edt = scipy.ndimage.distance_transform_edt(mask)
    med = float(np.median(edt[mask])) if mask.any() else 3.0
    r = int(np.clip(round(med * 0.6), 2, 10))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    disk = (xx ** 2 + yy ** 2) <= r ** 2
    s = scipy.ndimage.binary_closing(mask, structure=disk)
    return scipy.ndimage.binary_fill_holes(s)


def find_dental_arch(arch_mask2d, arch_thickness_factor=1.4,
                     posterior_extend=140.0, spline_smooth=3.0,
                     background=None, show=True):
    """
    Detects the dental arch centreline and focal trough from a 2D arch footprint mask.

    Centreline method: skeletonize → prune branches → BFS to get the ordered medial
    path → resample densely, re-centre each sample on the mask cross-section, smooth,
    and fit a *smoothing* cubic B-spline (s > 0) so the curve hugs the arch closely
    without wobbling.  The arch is then continued past each ramus tip
    (posterior_extend) so the back of the ramus/condyle is captured.
      - EDT median radius × factor × 2 → arch thickness Td.
      - Distance-from-spline ≤ Td/2 → focal trough.

    Parameters
    ----------
    arch_mask2d: bool ndarray (Y, X) — smoothed arch footprint
    arch_thickness_factor: multiplier on median EDT radius (paper: 1.2)
    posterior_extend: px to continue the arch past each ramus tip (0 = none)
    spline_smooth: B-spline smoothing budget per point in px² (higher = smoother)
    background: optional (Y, X) image shown behind the focal trough in
                           popup 3 (pass footprint_meip for anatomical context)
    show: three sequential popups

    Returns
    -------
    tck: scipy spline tuple (use scipy.interpolate.splev to evaluate)
    Td: float — arch thickness in pixels
    arch_region: bool ndarray (Y, X) — focal trough mask
    """
    _kernel = np.ones((3, 3), dtype=np.int32)
    _kernel[1, 1] = 0

    # --- 1. Skeletonize -------------------------------------------------------
    pruned = _skeletonize(arch_mask2d).copy()

    # --- 2. Branch pruning: farthest-first, remove shortest branch at each junction ---
    def _tip_mask(sk: np.ndarray) -> np.ndarray:
        """Boolean mask of skeleton pixels with exactly 1 neighbour."""
        nc = scipy.ndimage.convolve(
            sk.astype(np.int32), _kernel, mode='constant', cval=0
        ) * sk.astype(np.int32)
        return sk & (nc == 1)

    pruned = keep_largest_component(pruned)
    H, W = pruned.shape

    # Find junction pixels, then cluster adjacent ones into a single logical
    # junction.  Multiple touching pixels with nc ≥ 3 are all part of the same
    # thick junction region and must be treated as one unit so that each
    # conceptual fork is processed exactly once.
    nc_init = (scipy.ndimage.convolve(
        pruned.astype(np.int32), _kernel, mode='constant', cval=0
    ) * pruned.astype(np.int32))
    junction_mask  = pruned & (nc_init >= 3)
    cluster_labels, n_clusters = scipy.ndimage.label(
        junction_mask, structure=np.ones((3, 3), dtype=bool))

    if n_clusters > 0:
        skel_yx = np.argwhere(pruned)
        cy_c = (float(skel_yx[:, 0].min()) + float(skel_yx[:, 0].max()))/2.0
        cx_c = (float(skel_yx[:, 1].min()) + float(skel_yx[:, 1].max()))/2.0

        # Sort clusters farthest-from-centre first
        cluster_ids   = list(range(1, n_clusters + 1))
        cluster_cents = [np.argwhere(cluster_labels == cid).mean(axis=0)
                         for cid in cluster_ids]
        cluster_dists = [float(np.hypot(c[0] - cy_c, c[1] - cx_c))
                         for c in cluster_cents]
        cluster_ids   = [cid for _, cid in
                         sorted(zip(cluster_dists, cluster_ids), reverse=True)]

        for cid in cluster_ids:
            # All pixels of this cluster that are still in the skeleton
            cluster_set: set = {
                (int(p[0]), int(p[1]))
                for p in np.argwhere(cluster_labels == cid)
                if pruned[p[0], p[1]]
            }
            if not cluster_set:
                continue  # entire cluster removed by an earlier step

            # Border pixels: skeleton pixels adjacent to the cluster but outside it
            border: set = set()
            for cy2, cx2 in cluster_set:
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if (dy, dx) == (0, 0):
                            continue
                        ny, nx = cy2 + dy, cx2 + dx
                        if (0 <= ny < H and 0 <= nx < W
                                and pruned[ny, nx]
                                and (ny, nx) not in cluster_set):
                            border.add((ny, nx))

            # BFS from each unvisited border pixel, blocked at the entire cluster.
            #  'seen' merges border pixels that are 8-connected to each other
            # (without crossing the cluster) into one component.
            components: list = []
            seen: set = set()
            for ny, nx in border:
                if (ny, nx) in seen:
                    continue
                comp: set = {(ny, nx)}
                q: deque = deque([(ny, nx)])
                while q:
                    cy2, cx2 = q.popleft()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if (dy, dx) == (0, 0):
                                continue
                            ey, ex = cy2 + dy, cx2 + dx
                            if (0 <= ey < H and 0 <= ex < W
                                    and pruned[ey, ex]
                                    and (ey, ex) not in comp
                                    and (ey, ex) not in cluster_set):
                                comp.add((ey, ex))
                                q.append((ey, ex))
                seen |= comp
                components.append(comp)

            if len(components) < 2:
                continue

            # Wide check: erase the cluster and all its border pixels, then
            # count remaining skeleton segments.  ≤ 2 segments means the cluster
            # sits on the main arch path with no real side branch — skip it.
            pruned_tmp = pruned.copy()
            for cy2, cx2 in cluster_set:
                pruned_tmp[cy2, cx2] = False
            for ny, nx in border:
                pruned_tmp[ny, nx] = False
            _, n_wide = scipy.ndimage.label(
                pruned_tmp, structure=np.ones((3, 3), dtype=bool))
            if n_wide <= 2:
                continue

            # Length metric: max BFS depth from the component's border entry
            # pixel, blocked at the cluster.  Handles sub-junctions correctly.
            def _branch_depth(comp_set: set) -> int:
                entry = next((p for p in border if p in comp_set), None)
                if entry is None:
                    return 0
                vis: set = cluster_set | {entry}
                frontier = [entry]
                depth = 0
                while frontier:
                    nxt = [
                        (cy2 + dy, cx2 + dx)
                        for cy2, cx2 in frontier
                        for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                        if (dy, dx) != (0, 0)
                        and (cy2 + dy, cx2 + dx) in comp_set
                        and (cy2 + dy, cx2 + dx) not in vis
                    ]
                    for p in nxt:
                        vis.add(p)
                    if nxt:
                        frontier = nxt
                        depth += 1
                    else:
                        break
                return depth

            for py, px in min(components, key=_branch_depth):
                pruned[py, px] = False


    # --- 3. Endpoint selection: farthest-apart tip pair ----------------------
    # Using max Euclidean distance instead of leftmost/rightmost X avoids a
    # stray branch pixel hijacking an endpoint.  The two ramus tips are always
    # the farthest apart in a dental arch.
    coord_set = {tuple(p) for p in np.argwhere(pruned)}

    def _nbrs(p: tuple) -> list:
        y, x = p
        return [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if (dy, dx) != (0, 0) and (y + dy, x + dx) in coord_set]

    final_tips = np.argwhere(_tip_mask(pruned))
    if len(final_tips) >= 2:
        best_d, start, end = 0.0, tuple(final_tips[0]), tuple(final_tips[-1])
        for i in range(len(final_tips)):
            for j in range(i + 1, len(final_tips)):
                d = float(np.hypot(
                    final_tips[i, 0] - final_tips[j, 0],
                    final_tips[i, 1] - final_tips[j, 1],
                ))
                if d > best_d:
                    best_d, start, end = d, tuple(final_tips[i]), tuple(final_tips[j])
        if start[1] > end[1]:          # ensure left-to-right order
            start, end = end, start
    else:
        arr   = np.argwhere(pruned)
        start = tuple(arr[arr[:, 1].argmin()])
        end   = tuple(arr[arr[:, 1].argmax()])

    # --- 4. BFS from start to end -------------------------------------------
    prev: dict = {start: None}
    q = deque([start])
    while q:
        node = q.popleft()
        if node == end:
            break
        for nb in _nbrs(node):
            if nb not in prev:
                prev[nb] = node
                q.append(nb)

    path_nodes: list = []
    node = end
    while node is not None:
        path_nodes.append(node)
        node = prev.get(node)
    path    = np.array(path_nodes[::-1])
    ys_path = path[:, 0].astype(float)
    xs_path = path[:, 1].astype(float)

    # --- 5. Arch thickness via distance transform ----------------------------
    D: np.ndarray = np.asarray(scipy.ndimage.distance_transform_edt(arch_mask2d))
    D_norm = D/float(D.max()) * 255.0
    core   = D_norm > 245.0
    radius = float(np.median(D[core])) if core.any() else float(D.max())
    Td     = arch_thickness_factor * 2.0 * radius

    # --- 6. Smooth, well-centred arch spline --------------------------------
    # The skeleton is the medial axis (roughly centred) but can be jagged and
    # pulled off-centre where teeth bulge.  Resample it densely by arc length,
    # re-centre each sample on the mask cross-section (midpoint of the LONGEST
    # in-mask run along the local perpendicular — robust to sockets/gaps that
    # a first-to-last midpoint would straddle), smooth, then fit a *smoothing*
    # B-spline (s > 0) so the curve hugs the arch without wobbling through every
    # noisy point.  A poorly centred or wobbly arch is exactly what tilts and
    # distorts the teeth in the panoramic, so both properties matter.
    H, W = arch_mask2d.shape

    ds_path  = np.hypot(np.diff(xs_path), np.diff(ys_path))
    arc_path = np.concatenate([[0.0], np.cumsum(ds_path)])
    n_dense  = max(40, int(arc_path[-1]/2.0))           # ~1 sample per 2 px
    u_arc    = np.linspace(0.0, arc_path[-1], n_dense)
    xs_b     = np.interp(u_arc, arc_path, xs_path)
    ys_b     = np.interp(u_arc, arc_path, ys_path)

    base_sigma = max(2.0, n_dense/20.0)
    xs_b = scipy.ndimage.gaussian_filter1d(xs_b, base_sigma)
    ys_b = scipy.ndimage.gaussian_filter1d(ys_b, base_sigma)

    # Re-centre each sample on the mask cross-section
    dxb = np.gradient(xs_b); dyb = np.gradient(ys_b)
    nb  = np.hypot(dxb, dyb); nb[nb < 1e-8] = 1e-8
    perp_x, perp_y = -dyb/nb, dxb/nb
    t_scan = np.arange(-Td/2.0, Td/2.0 + 0.5, 0.5)
    xs_c, ys_c = xs_b.copy(), ys_b.copy()
    for i in range(n_dense):
        xi = np.round(xs_b[i] + t_scan * perp_x[i]).astype(int)
        yi = np.round(ys_b[i] + t_scan * perp_y[i]).astype(int)
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        m  = np.zeros(len(t_scan), dtype=bool)
        m[ok] = arch_mask2d[yi[ok], xi[ok]]
        if not m.any():
            continue
        idx  = np.flatnonzero(m)
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        run  = max(runs, key=len)                         # longest contiguous chord
        t_mid = (t_scan[run[0]] + t_scan[run[-1]])/2.0
        xs_c[i] = xs_b[i] + t_mid * perp_x[i]
        ys_c[i] = ys_b[i] + t_mid * perp_y[i]
    xs_c = scipy.ndimage.gaussian_filter1d(xs_c, base_sigma)
    ys_c = scipy.ndimage.gaussian_filter1d(ys_c, base_sigma)

    # Posterior extension: continue each ramus end along its local trend so the
    # back of the ramus/condyle falls inside the trough.  A line fit to the
    # last segment keeps the direction the arch was heading (so it "seems to
    # continue") without curving back toward the spine.
    def _continue(xs, ys, length, n_fit=22, step=2.0):
        xseg, yseg = xs[-n_fit:], ys[-n_fit:]
        s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xseg), np.diff(yseg)))])
        cx, cy = np.polyfit(s, xseg, 1), np.polyfit(s, yseg, 1)
        s_ext = s[-1] + np.arange(step, length + step, step)
        return np.polyval(cx, s_ext), np.polyval(cy, s_ext)

    if posterior_extend > 0:
        xe_end, ye_end = _continue(xs_c, ys_c, posterior_extend)
        xe_st,  ye_st  = _continue(xs_c[::-1], ys_c[::-1], posterior_extend)
        xs_c = np.concatenate([xe_st[::-1], xs_c, xe_end])
        ys_c = np.concatenate([ye_st[::-1], ys_c, ye_end])

    # Smoothing B-spline (s > 0 → hugs the arch closely, no wobble)
    tck, _ = scipy.interpolate.splprep([xs_c, ys_c], k=3,
                                       s=len(xs_c) * spline_smooth)

    # --- 6. Focal trough: pixels within Td/2 of the spline ------------------
    t_dense  = np.linspace(0.0, 1.0, 3000)
    _spl     = scipy.interpolate.splev(t_dense, tck)
    xs_dense = np.asarray(_spl[0], dtype=float)
    ys_dense = np.asarray(_spl[1], dtype=float)

    spline_img = np.zeros(arch_mask2d.shape, dtype=bool)
    for xd, yd in zip(xs_dense, ys_dense):
        xi, yi = int(round(float(xd))), int(round(float(yd)))
        if 0 <= yi < arch_mask2d.shape[0] and 0 <= xi < arch_mask2d.shape[1]:
            spline_img[yi, xi] = True

    dist_from_spline: np.ndarray = np.asarray(
        scipy.ndimage.distance_transform_edt(~spline_img)
    )
    arch_region = dist_from_spline <= (Td/2.0)

    # Smooth the focal trough boundary: Gaussian blur on the binary mask then
    # re-threshold at 0.5.  This rounds all sharp corners and end-caps into
    # smooth curves without changing the overall width or position.
    smooth_sigma = Td/8.0
    arch_region = scipy.ndimage.gaussian_filter(
        arch_region.astype(np.float32), sigma=smooth_sigma
    ) > 0.5

    # --- 7. Visualisation ---------------------------------------------------
    if show:
        t_plot  = np.linspace(0.0, 1.0, 1000)
        _plt    = scipy.interpolate.splev(t_plot, tck)
        xs_plot = np.asarray(_plt[0], dtype=float)
        ys_plot = np.asarray(_plt[1], dtype=float)

        # Popup 1: pruned skeleton + control points + spline
        _, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.imshow(arch_mask2d, cmap='gray', aspect='auto')
        ax.scatter(xs_path, ys_path, s=0.5, c='orange', label='Pruned skeleton')
        ax.plot(xs_plot, ys_plot, 'b-', linewidth=1.5, label='Smoothing spline')
        ax.scatter(xs_c, ys_c, s=4, c='red', zorder=5,
                   label='re-centred points (+ extension)')
        ax.set_xlim(0, W); ax.set_ylim(H, 0)
        ax.set_title('Skeleton + re-centred points → smoothing spline (extended)', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.axis('off')
        plt.tight_layout()
        plt.show()

        # Popup 2: distance transform heatmap + spline
        _, ax = plt.subplots(1, 1, figsize=(8, 6))
        im = ax.imshow(D, cmap='hot', aspect='auto')
        ax.plot(xs_plot, ys_plot, 'c-', linewidth=1.5,
                label=f'Spline  Td={Td:.1f} px')
        plt.colorbar(im, ax=ax, fraction=0.03)
        ax.set_title(f'Distance transform — arch thickness Td = {Td:.1f} px',
                     fontsize=9)
        ax.legend(fontsize=7)
        ax.axis('off')
        plt.tight_layout()
        plt.show()

        # Popup 3: focal trough over MeIP (or mask if no background supplied)
        if background is not None:
            bg3: np.ndarray = background
        else:
            bg3 = arch_mask2d.astype(np.float32)

        trough_ov = np.zeros(arch_mask2d.shape + (4,), dtype=np.float32)
        trough_ov[arch_region] = [1.0, 0.45, 0.0, 0.5]

        _, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 10))

        ax_top.imshow(bg3, cmap='gray', aspect='auto')
        ax_top.set_title('Axial MeIP', fontsize=9)
        ax_top.axis('off')

        ax_bot.imshow(bg3, cmap='gray', aspect='auto')
        ax_bot.imshow(trough_ov, aspect='auto')
        ax_bot.plot(xs_plot, ys_plot, 'b-', linewidth=1.5, label='Arch spline')
        ax_bot.set_title(
            f'Focal trough  Td={Td:.1f} px',
            fontsize=9,
        )
        ax_bot.legend(fontsize=7)
        ax_bot.axis('off')

        plt.tight_layout()
        plt.show()

    return tck, Td, arch_region


def load_cbct_volume(file_path, flip=False):
    """
    Loads a CBCT (.mha/.nii/.nrrd) file into a (Z, Y, X) HU numpy array,
    plus the axial slice spacing in mm.

    Parameters
    ----------
    file_path: path to the CBCT file
    flip: pass True when the mandible appears at the top of the volume

    Returns
    -------
    volume       : ndarray (Z, Y, X)
    z_spacing_mm : float
    """
    img = sitk.ReadImage(file_path)
    z_spacing_mm = img.GetSpacing()[2]
    volume = sitk.GetArrayFromImage(img)
    if flip:
        volume = flip_volume_sagittal(volume)
    return volume, z_spacing_mm


def compute_arch_spline(volume, z_spacing_mm, posterior_extend_mm=22.0, show=False):
    """
    Runs the full detection pipeline from a loaded CBCT volume to a single,
    smooth dental-arch spline: coronal jaw ROI -> axial arch footprint ->
    adaptive smoothing -> skeleton-based smoothing B-spline.

    Mirrors the arch-detection stage of synthesize_panoramic_from_volume in
    cbct_to_panoramic.py.

    Parameters
    ----------
    volume: ndarray (Z, Y, X), raw CBCT HU values
    z_spacing_mm: axial slice spacing (mm)
    posterior_extend_mm: how far past each ramus tip the spline is extended (mm)
    show: forward to each stage's own diagnostic plots

    Returns
    -------
    tck: scipy spline tuple (use scipy.interpolate.splev to evaluate)
    roi: dict from find_coronal_roi
    arch_mask2d: bool ndarray (Y, X) — smoothed arch footprint used to fit the spline
    footprint_meip: ndarray (Y, X) — axial MeIP the footprint was detected from
    """
    meip_coronal = find_MeIPs(volume, axis='coronal', show=False)
    roi = find_coronal_roi(meip_coronal, volume=volume, z_spacing_mm=z_spacing_mm, show=show)
    footprint_meip, arch_mask2d = find_arch_footprint(volume, roi, show=show)
    arch_mask2d = _smooth_arch_adaptive(arch_mask2d)
    pe = posterior_extend_mm/z_spacing_mm
    tck, _Td, _arch_region = find_dental_arch(arch_mask2d, posterior_extend=pe,
                                              background=footprint_meip, show=show)
    return tck, roi, arch_mask2d, footprint_meip


def save_spline_to_csv(tck, out_path, n_samples=500):
    """
    Samples the arch spline at n_samples equally spaced parameter values and
    writes the (x, y) pixel coordinates to a CSV file (columns: u, x, y).

    Parameters
    ----------
    tck: spline tuple from compute_arch_spline/find_dental_arch
    out_path: destination .csv path
    n_samples: number of points sampled along the spline

    Returns
    -------
    out_path
    """
    u = np.linspace(0.0, 1.0, n_samples)
    x, y = scipy.interpolate.splev(u, tck)
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['u', 'x', 'y'])
        for ui, xi, yi in zip(u, x, y):
            writer.writerow([ui, xi, yi])
    return out_path


def save_spline_to_fcsv(tck, roi, file_path, out_path, flip=False, n_samples=40):
    """
    Samples the arch spline and writes it as a 3D Slicer Markups Fiducial
    (.fcsv) file, in the physical (LPS) coordinates of the source volume, for
    loading/QA in 3D Slicer.

    The spline lives in the 2D (Y, X) pixel plane of the axial projection; the
    Z coordinate is fixed at the midpoint between the maxillary and mandibular
    occlusal planes (roi['z_maxilla_peak']/roi['z_mandible_peak']), i.e. the
    centre of the z-band the arch footprint was detected from.

    Parameters
    ----------
    tck: spline tuple from compute_arch_spline/find_dental_arch
    roi: dict from find_coronal_roi
    file_path: path to the source CBCT file (re-read for its geometry only:
               origin, spacing, direction — not the pixel data)
    out_path: destination .fcsv path
    flip: must match the  'flip ' passed to load_cbct_volume for this
               volume, so pixel indices map back to the file's own index space
    n_samples: number of points sampled along the spline

    Returns
    -------
    out_path
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(file_path)
    reader.ReadImageInformation()
    size = reader.GetSize()  # (X, Y, Z)
    origin = np.array(reader.GetOrigin())
    spacing = np.array(reader.GetSpacing())
    direction = np.array(reader.GetDirection()).reshape(3, 3)

    u = np.linspace(0.0, 1.0, n_samples)
    x, y = scipy.interpolate.splev(u, tck)
    z = np.full(n_samples, (roi['z_maxilla_peak'] + roi['z_mandible_peak'])/2.0)

    if flip:
        # flip_volume_sagittal reverses the Z and Y axes of the (Z, Y, X)
        # array; undo that here so indices map back to the file's own space.
        y = (size[1] - 1) - y
        z = (size[2] - 1) - z

    with open(out_path, 'w', newline='') as f:
        f.write("# Markups fiducial file version = 5.10\n")
        f.write("# CoordinateSystem = LPS\n")
        f.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")
        for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
            px, py, pz = origin + direction @ (spacing * np.array([xi, yi, zi]))
            f.write(f"{i + 1},{px},{py},{pz},0,0,0,1,1,1,0,arch-{i + 1},,\n")
    return out_path


def visualize_spline_on_mip(volume, roi, tck, n_samples=1000, show=True, save_path=None):
    """
    Overlays the dental-arch spline on the axial MIP of the coronal jaw ROI
    (z_top..z_bottom from  'roi ').

    Parameters
    ----------
    volume: ndarray (Z, Y, X), raw CBCT HU values
    roi: dict from find_coronal_roi
    tck: spline tuple from compute_arch_spline/find_dental_arch
    n_samples: number of points sampled along the spline for plotting
    show: call plt.show()
    save_path: optional path to save the figure as a PNG

    Returns
    -------
    fig, ax
    """
    z_top, z_bottom = roi['z_top'], roi['z_bottom']
    axial_mip = find_MIPs(volume[z_top:z_bottom + 1], axis='axial', show=False)

    u = np.linspace(0.0, 1.0, n_samples)
    x, y = scipy.interpolate.splev(u, tck)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(axial_mip, cmap='gray', aspect='auto')
    ax.plot(x, y, 'r-', linewidth=1.5, label='Dental arch spline')
    ax.set_title(f'Axial MIP  (z={z_top}–{z_bottom})  with dental arch spline', fontsize=9)
    ax.axis('off')
    ax.legend(fontsize=7, loc='upper right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()

    return fig, ax


if __name__ == "__main__":
    OUTPUT_DIR = r"ouput_link_goes_here"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    INPUT_FILE = r'link_leading to the .nii, .mha, .nrrd, ..., file'

    # ONLY THE .NII FILE LINKS FROM THE MCSTU DATASET NEED TO BE FLIPPED along the sagittal axis, NOT THE OTHER LINKS
    FLIP = True

    volume, z_spacing_mm = load_cbct_volume(INPUT_FILE, flip=FLIP)
    tck, roi, arch_mask2d, footprint_meip = compute_arch_spline(volume, z_spacing_mm, show=False)

    csv_path = save_spline_to_csv(tck, os.path.join(OUTPUT_DIR, "arch_spline.csv"))
    print("saved:", csv_path)

    fcsv_path = save_spline_to_fcsv(tck, roi, INPUT_FILE,
                                    os.path.join(OUTPUT_DIR, "arch_spline.fcsv"), flip=FLIP)
    print("saved:", fcsv_path)

    visualize_spline_on_mip(volume, roi, tck,
                            save_path=os.path.join(OUTPUT_DIR, "arch_spline_on_mip.png"))
