import os
from typing import Union
from collections import deque
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal
import scipy.ndimage
import scipy.interpolate
from skimage.morphology import skeletonize as _skeletonize
from skimage.morphology import binary_closing as _binary_closing, disk as _disk

# Optional: only needed for the interactive Fiji/ImageJ viewer used by
# show_focal_trough_overlay() (sitk.Show). Point it at your ImageJ/Fiji
# executable to enable it; harmless to leave unset otherwise.
# os.environ["SITK_SHOW_COMMAND"] = r"<path-to>\fiji-windows-x64.exe"

# Sample paths used only by the _run_pipeline() helper below. The real, user
# facing run configuration lives in the CONFIG block at the very bottom of this
# file (under `if __name__ == "__main__"`) -- edit it there.
test_mha_file = ""   # e.g. r"C:\data\ToothFairy2F_001_0000.mha"
test_nii_file = ""   # e.g. r"C:\data\scan.nii"

def load_np_mha(mha_link):
    """Returns  a numpy array from a mha file from the mha file link. The array is in the shape of (z, y, x),
    where z is the number of slices (axial), y is antero-posterior, and x is for left and right.
    up_lim and low_lim determines above or below which HU does the voxel get adjusted to new_value
    """
    image = sitk.ReadImage(mha_link)
    return sitk.GetArrayFromImage(image)
def apply_bone_window(volume: np.ndarray, bone_min: int = 200, bone_max: int = 800) -> np.ndarray:
    """
    Clips the volume to the cortical bone HU range and shifts the floor to zero.

    Tissue mapping after clipping:
        Air / soft tissue  (<= bone_min)  →  0
        Cortical bone      (bone_min–bone_max)  →  1 … (bone_max − bone_min)
        Enamel / metal     (>= bone_max)  →  bone_max − bone_min  (same as densest bone)

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
        plt.title("Coronal MeIP")
        plt.axis("off")
        plt.show()

    return mip

def flip_volume_sagittal(volume: np.ndarray) -> np.ndarray:
    """
    Rotates the volume 180° around the sagittal (left–right) axis.

    Use when the CBCT was acquired with the patient inverted and the mandible
    appears at the top of the volume. After the rotation:
      - superior ↔ inferior  (z axis reverses, mandible moves to the bottom)
      - anterior ↔ posterior (y axis reverses)
      - left / right unchanged (the axis of rotation)

    Parameters
    ----------
    volume : ndarray (Z, Y, X)

    Returns
    -------
    ndarray, same shape as input, C-contiguous, with corrected orientation
    """
    return np.flip(volume, axis=(0, 1)).copy()

def _localize_jaws_edentulous(volume, meip_coronal, z_spacing_mm, cortical_s,
                              air_hu, open_max_mm, sup_margin_mm, inf_margin_mm, show,
                              jaw_half_mm=45.0):
    """
    No-enamel (edentulous) fallback for jaw Z-localization. Triggered by
    _find_coronal_roi_enamel when the enamel signal is negligible.

    Strategy (the air-anchor idea, scoped to where enamel is unavailable):
    anchor on the oral-cavity air band — the air maximum within a central XY
    window and the central Z range (which excludes the nasal cavity above and
    the sub-mental airway below) — then bracket it with the nearest flanking
    cortical-bone peaks (the maxillary palate / alveolus above and the mandibular
    body below). The cranial vault is far superior of the oral air, so picking the
    *nearest* bone peak above the anchor selects the maxilla, not the vault.

    Heuristic; on the current (all-dentate) data this path is reached only on a
    synthetically enamel-stripped volume, so it is validated only synthetically.
    Returns the same dict shape as _find_coronal_roi_enamel.
    """
    Z, Y, X = volume.shape
    y0, y1 = Y // 4, 3 * Y // 4
    x0, x1 = X // 3, 2 * X // 3
    air_frac = (volume[:, y0:y1, x0:x1] < air_hu).mean(axis=(1, 2))
    air_s = scipy.ndimage.gaussian_filter1d(air_frac, 3)

    z_lo, z_hi = int(0.20 * Z), int(0.85 * Z)
    z_occ = z_lo + int(np.argmax(air_s[z_lo:z_hi]))   # oral-cavity air anchor

    # Flanking cortical-bone peaks, restricted to a physiological jaw half-height
    # of the oral-cavity anchor so a far cortical peak (cervical spine, neck) on a
    # tall scan cannot be mistaken for the jaw.
    jaw_half = int(round(jaw_half_mm / z_spacing_mm))
    distance = max(5, int(round(8.0 / z_spacing_mm)))
    peaks, _ = scipy.signal.find_peaks(cortical_s, distance=distance)
    above = peaks[(peaks < z_occ) & (peaks >= z_occ - jaw_half)]
    below = peaks[(peaks > z_occ) & (peaks <= z_occ + jaw_half)]
    fallback_off = int(round(10.0 / z_spacing_mm))
    z_sup = int(above[-1]) if len(above) else max(0, z_occ - fallback_off)
    z_inf = int(below[0]) if len(below) else min(Z - 1, z_occ + fallback_off)

    z_top    = max(0, z_sup - int(round(sup_margin_mm / z_spacing_mm)))
    z_bottom = min(Z - 1, z_inf + int(round(inf_margin_mm / z_spacing_mm)))

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
        cn = cortical_s / (cortical_s.max() + 1e-6)
        an = air_s / (air_s.max() + 1e-6)
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
        'z_top':           z_top,
        'z_top_raw':       z_sup,
        'z_maxilla_peak':  z_sup,
        'z_gap':           z_occ,
        'z_mandible_peak': z_inf,
        'z_bottom_raw':    z_inf,
        'z_bottom':        z_bottom,
        'margin_slices':   int(round(sup_margin_mm / z_spacing_mm)),
        'z_spacing_mm':    z_spacing_mm,
        'method':          'edentulous_air_anchor',
    }


def _find_coronal_roi_enamel(volume, meip_coronal, z_spacing_mm,
                             enamel_hu, enamel_hu_max,
                             open_max_mm, sup_margin_mm, inf_margin_mm, show,
                             cortical_hu=350, air_hu=-400, min_enamel_ratio=0.04):
    """
    Enamel-band jaw localization (see find_coronal_roi). Counts enamel-HU voxels per
    Z-slice → dentition profile, picks the two tallest occlusal peaks within
    open_max_mm of the global max (one when the bite is closed), and brackets them
    with anatomical margins. Returns the same dict shape as find_coronal_roi.

    Robustness:
    * Metal: the profile is a *binary* voxel count, so a single very dense metal
      voxel counts the same as one enamel voxel and cannot dominate the occlusal
      peak; enamel_hu_max additionally caps both the enamel and cortical bands so
      bright restorations / streaks above the cap are excluded outright.
    * Edentulous: when the enamel signal is negligible relative to cortical bone
      (enamel_ratio < min_enamel_ratio, e.g. no teeth), dispatch to the
      air-anchored fallback _localize_jaws_edentulous instead of failing.
    """
    band = (volume > enamel_hu) & (volume < enamel_hu_max)
    prof = scipy.ndimage.gaussian_filter1d(band.sum(axis=(1, 2)).astype(float), 3)

    cortical = ((volume > cortical_hu) & (volume < enamel_hu_max)).sum(axis=(1, 2)).astype(float)
    cortical_s = scipy.ndimage.gaussian_filter1d(cortical, 3)

    enamel_ratio = float(prof.max() / (cortical_s.max() + 1e-6))
    if prof.max() <= 0 or enamel_ratio < min_enamel_ratio:
        return _localize_jaws_edentulous(
            volume, meip_coronal, z_spacing_mm, cortical_s, air_hu,
            open_max_mm, sup_margin_mm, inf_margin_mm, show)

    pn = prof / prof.max()

    z_occ = int(np.argmax(pn))
    distance = max(5, int(round(8.0 / z_spacing_mm)))
    peaks, _ = scipy.signal.find_peaks(pn, prominence=0.12, distance=distance)
    if len(peaks) == 0:
        peaks = np.array([z_occ])

    # The opposing arch is the next-tallest peak within a plausible bite opening.
    open_max = int(round(open_max_mm / z_spacing_mm))
    cand = peaks[np.abs(peaks - z_occ) <= open_max]
    if len(cand) >= 2:
        two = cand[np.argsort(pn[cand])[-2:]]
        z_sup, z_inf = int(min(two)), int(max(two))   # superior / inferior occlusal
    else:
        z_sup = z_inf = z_occ                           # closed bite → fused arches

    z_top    = max(0, z_sup - int(round(sup_margin_mm / z_spacing_mm)))
    z_bottom = min(volume.shape[0] - 1, z_inf + int(round(inf_margin_mm / z_spacing_mm)))

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
        'z_top':           z_top,
        'z_top_raw':       z_sup,
        'z_maxilla_peak':  z_sup,
        'z_gap':           z_occ,
        'z_mandible_peak': z_inf,
        'z_bottom_raw':    z_inf,
        'z_bottom':        z_bottom,
        'margin_slices':   int(round(sup_margin_mm / z_spacing_mm)),
        'z_spacing_mm':    z_spacing_mm,
        'enamel_hu':       enamel_hu,
        'enamel_hu_max':   enamel_hu_max,
    }


def find_coronal_roi(meip_coronal, volume=None, z_spacing_mm=1.0,
                     enamel_hu=1800, enamel_hu_max=5000,
                     open_max_mm=20.0, sup_margin_mm=20.0, inf_margin_mm=30.0,
                     cortical_hu=350, air_hu=-400, min_enamel_ratio=0.04,
                     smooth_sigma=5, threshold_fraction=0.15, min_margin_slices=5, show=True):
    """
    Finds the jaw Z-extent.

    Two methods:

    * Enamel-band (recommended; used when `volume` is given) — FOV-independent and
      robust to a closed mouth.  The mean-intensity method below keys on the two
      tallest *bone-mass* peaks, which fails on whole-skull scans (the cranial vault
      outweighs the jaws) and on a closed mouth (the arches fuse into one peak).
      Tooth enamel is the densest consistent structure, so counting voxels in an
      enamel HU band ([enamel_hu, enamel_hu_max], the upper cap rejecting surgical
      metal) per Z-slice gives a clean dentition profile: the cranium vanishes and
      each arch's occlusal plane is a sharp peak.  The two tallest peaks within
      open_max_mm of the global maximum are the maxillary / mandibular occlusal
      planes (a single peak when the bite is closed and the arches coincide).  The
      ROI brackets them with anatomical margins (sup_margin_mm up toward the sinus
      floor, inf_margin_mm down toward the mandible base).

    * Mean-intensity (fallback; `volume=None`) — collapse the coronal MeIP to a 1D
      intensity-vs-Z profile, take the two tallest peaks as maxilla / mandible (or a
      single fused peak), scan outward to per-peak thresholds, and apply an adaptive
      margin from the mandible body height.

    Parameters
    ----------
    meip_coronal       : ndarray, shape (Z, X) — used for visualization (and the
                         fallback method's profile)
    volume             : (Z, Y, X) raw CBCT (HU). Pass it to use the enamel method.
    z_spacing_mm       : axial slice spacing, for the mm-based enamel margins
    enamel_hu          : lower HU bound of the enamel band
    enamel_hu_max      : upper HU bound (rejects extreme metal / surgical hardware)
    open_max_mm        : max inter-arch separation searched for the opposing arch
    sup_margin_mm      : margin above the maxillary occlusal plane
    inf_margin_mm      : margin below the mandibular occlusal plane
    smooth_sigma       : (fallback) median filter size and Savitzky-Golay window
    threshold_fraction : (fallback) fraction of each jaw's own peak used as stop threshold
    min_margin_slices  : floor for the adaptive margin
    show               : plot the MeIP with ROI overlays and the 1D profile

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
    # 1D vertical profile: mean intensity at each Z level
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
        # Closed-mouth / fused arches: when the upper and lower teeth intersect
        # vertically there is no inter-arch air gap, so both arches project to a
        # single intensity peak. Treat that peak as the centre of the combined
        # dental complex (maxilla and mandible coincide); the outward threshold
        # scan + margin below still bracket the whole jaw block correctly.
        z_single = int(peaks[np.argmax(profile_smooth[peaks])])
        z_maxilla_peak = z_mandible_peak = z_single

    # Inter-arch gap: deepest point between the two peaks (== the peak itself when fused)
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
            (z_top,           'cyan',   '--', f'ROI top'),
            (z_top_raw,       'cyan',   ':',  'Maxilla crossing'),
            (z_maxilla_peak,  'yellow', '-',  'Maxilla peak'),
            (z_gap,           'red',    '-',  'Inter-arch gap'),
            (z_mandible_peak, 'yellow', '-',  'Mandible peak'),
            (z_bottom_raw,    'cyan',   ':',  'Mandible crossing'),
            (z_bottom,        'cyan',   '--', 'ROI bottom'),
        ]:
            ax_img.axhline(z, color=color, linestyle=ls, linewidth=1, label=label)
        ax_img.legend(fontsize=7, loc='upper right')
        ax_img.set_title('Coronal MeIP — jaw ROI')
        ax_img.axis('off')

        ax_prof.set_facecolor('black')
        # Shade margin regions on profile (jaw peak → final ROI bound)
        ax_prof.axhspan(z_top,          z_maxilla_peak,  color='orange', alpha=0.25, label=f'Margin (±{margin} slices)')
        ax_prof.axhspan(z_mandible_peak, z_bottom,        color='orange', alpha=0.25)
        ax_prof.axhspan(z_maxilla_peak, z_mandible_peak,  color='cyan',   alpha=0.10, label='Inter-peak region')
        ax_prof.plot(profile_smooth, np.arange(len(profile_smooth)), color='white', linewidth=1.5)
        ax_prof.axvline(threshold_sup, color='yellow', linestyle=':', linewidth=1,
                        label=f'Threshold sup = {threshold_sup:.1f}')
        ax_prof.axvline(threshold_inf, color='orange', linestyle=':', linewidth=1,
                        label=f'Threshold inf = {threshold_inf:.1f}')
        for z, color, ls in [
            (z_top,           'cyan',   '--'),
            (z_top_raw,       'cyan',   ':'),
            (z_maxilla_peak,  'yellow', '-'),
            (z_gap,           'red',    '-'),
            (z_mandible_peak, 'yellow', '-'),
            (z_bottom_raw,    'cyan',   ':'),
            (z_bottom,        'cyan',   '--'),
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
        'z_top':           z_top,
        'z_top_raw':       z_top_raw,
        'z_maxilla_peak':  z_maxilla_peak,
        'z_gap':           z_gap,
        'z_mandible_peak': z_mandible_peak,
        'z_bottom_raw':    z_bottom_raw,
        'z_bottom':        z_bottom,
        'margin_slices':   margin,
        'threshold_sup':   threshold_sup,
        'threshold_inf':   threshold_inf,
    }


def segment_mandible_3d(volume, roi, bone_min=500, min_patch_size=50, show=True):
    """
    Extracts the mandible as a 3D binary mask using connected component analysis.

    Seeds from the base of the mandible (z_bottom in the coronal ROI), where
    the mandible body is cleanest and fully separated from the maxilla. The 3D
    connected component with 26-connectivity then propagates upward through the
    body, ramus, and condyles without requiring per-slice decisions.

    Parameters
    ----------
    volume         : ndarray (Z, Y, X) — raw CBCT values
    roi            : dict from find_coronal_roi
    bone_min       : lower intensity threshold for binarisation
    min_patch_size : 2D connected components smaller than this (in pixels) are
                     removed slice-by-slice before 3D labelling, preventing
                     noise or partial-volume artifacts from bridging structures
    show           : if True, display coronal MIP of mask and representative axial slices

    Returns
    -------
    mask : bool ndarray, shape == volume.shape
    """
    z_top    = roi['z_top']
    z_bottom = roi['z_bottom']

    # Binarise using only a lower threshold — no upper clipping, so metal
    # and enamel are treated identically to dense bone for connectivity purposes
    binary = (volume[z_top:z_bottom + 1] > bone_min).copy()

    # Remove small 2D patches slice-by-slice before 3D labelling.
    # Each isolated region smaller than min_patch_size pixels is zeroed out,
    # preventing noise or partial-volume speckle from bridging bone structures
    # that are not genuinely connected.
    for i in range(binary.shape[0]):
        slc_lbl        = scipy.ndimage.label(binary[i])[0]  # type: ignore[index]
        sizes          = np.bincount(slc_lbl.ravel())
        sizes[0]       = 0                                   # ignore background
        binary[i]      = sizes[slc_lbl] >= min_patch_size

    # 26-connectivity: full 3×3×3 neighbourhood preserves thin structures
    # like the condylar neck across adjacent slices
    labeled = scipy.ndimage.label(binary, structure=np.ones((3, 3, 3), dtype=bool))[0]  # type: ignore[index]

    # Seed: largest bone component at the mandible peak slice.
    # Using z_mandible_peak directly avoids depending on which end of the
    # sub-volume is inferior — it is always the identified mandible arch.
    seed_local_z   = roi['z_mandible_peak'] - z_top
    seed_slice_bin = binary[seed_local_z]

    if not seed_slice_bin.any():
        raise ValueError(
            "No bone found at the mandible peak slice. "
            "Verify that bone_min matches the volume HU scale."
        )

    slice_labeled = scipy.ndimage.label(seed_slice_bin)[0]  # type: ignore[index]
    sizes            = np.bincount(slice_labeled.ravel())
    sizes[0]         = 0                          # exclude background label
    seed_component   = int(np.argmax(sizes))
    coords           = np.argwhere(slice_labeled == seed_component)
    seed_y           = int(round(coords[:, 0].mean()))
    seed_x           = int(round(coords[:, 1].mean()))

    component_id = int(labeled[seed_local_z, seed_y, seed_x])
    if component_id == 0:
        # centroid landed on background (rare concave shape): nearest bone voxel
        dists        = np.hypot(coords[:, 0] - coords[:, 0].mean(),
                                coords[:, 1] - coords[:, 1].mean())
        nearest      = coords[int(np.argmin(dists))]
        component_id = int(labeled[seed_local_z, nearest[0], nearest[1]])

    mandible_sub = (labeled == component_id).copy()

    # --- Pass 2: break thin bridges (e.g. to the spine) ----------------------
    # Re-apply the per-slice area filter to the extracted component.  Any
    # spurious connection that survives 3D CC but has a small 2D cross-section
    # (thin bridge, partial-volume artifact) gets severed here.
    for i in range(mandible_sub.shape[0]):
        slc_lbl        = scipy.ndimage.label(mandible_sub[i])[0]  # type: ignore[index]
        sizes          = np.bincount(slc_lbl.ravel())
        sizes[0]       = 0
        mandible_sub[i] = sizes[slc_lbl] >= min_patch_size

    # After severing bridges, pick the single largest 3D component (mandible).
    # Any off-anatomy fragment (spine, hyoid, artifact) that was disconnected
    # by the pass above becomes its own smaller component and is dropped.
    relabeled   = scipy.ndimage.label(mandible_sub, structure=np.ones((3, 3, 3), dtype=bool))[0]  # type: ignore[index]
    comp_sizes  = np.bincount(relabeled.ravel())
    comp_sizes[0] = 0
    mandible_sub  = relabeled == int(np.argmax(comp_sizes))

    # --- Hole filling ---------------------------------------------------------
    # Voxels completely enclosed by mask in the 2D axial plane are filled in.
    # This closes the inner void of the arch cross-section and small cortical
    # gaps, without affecting the 3D outer boundary.
    for i in range(mandible_sub.shape[0]):
        mandible_sub[i] = scipy.ndimage.binary_fill_holes(mandible_sub[i])

    # --- Per-slice single-object enforcement ----------------------------------
    # After hole filling, each axial slice is reduced to its single largest
    # 2D connected component.  This prevents the two rami or condyles from
    # appearing as separate islands in the visualisation; only the dominant
    # cross-section is kept at every level.
    for i in range(mandible_sub.shape[0]):
        slc_lbl = scipy.ndimage.label(mandible_sub[i])[0]  # type: ignore[index]
        sizes   = np.bincount(slc_lbl.ravel())
        sizes[0] = 0
        if sizes.max() > 0:
            mandible_sub[i] = slc_lbl == int(np.argmax(sizes))
        else:
            mandible_sub[i] = False

    mask = np.zeros(volume.shape, dtype=bool)
    mask[z_top:z_bottom + 1] = mandible_sub

    if show:
        step  = 15
        # Walk from z_bottom (base) upward toward the mandible peak
        z_sample = np.arange(z_bottom-50, roi['z_maxilla_peak'] +50, -step)

        coronal_orig = find_MeIPs(volume, axis='coronal', show=False)
        coronal_mask = mask.max(axis=1).astype(np.float32)
        axial_mip    = np.where(mask, volume, 0).max(axis=0).astype(np.float32)

        # Popup 1: coronal overview
        fig, (ax_cor_top, ax_cor_bot) = plt.subplots(2, 1, figsize=(5, 8))
        ax_cor_top.imshow(coronal_orig, cmap='gray', aspect='auto')
        ax_cor_top.set_title('Coronal MeIP', fontsize=9)
        ax_cor_top.axis('off')
        ax_cor_bot.imshow(coronal_mask, cmap='hot', aspect='auto')
        ax_cor_bot.set_title('Mask (coronal MIP)', fontsize=9)
        ax_cor_bot.axis('off')
        fig.suptitle(f'ROI z={z_top}–{z_bottom}', fontsize=9)
        plt.tight_layout()
        plt.show()

        # One popup per step: axial slice at z with cumulative mask overlay.
        # The cumulative mask is the axial MIP of all mask slices from z_bottom
        # down to the current z, so it grows as we move upward.
        for z in z_sample:
            slc             = volume[z]
            cumulative_mask = mask[z:z_bottom + 1].any(axis=0)

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(slc, cmap='gray')
            overlay = np.zeros(slc.shape + (4,), dtype=np.float32)
            overlay[cumulative_mask] = [1.0, 0.45, 0.0, 0.6]
            ax.imshow(overlay)
            ax.set_title(f'z={z}  (accumulated z={z}–{z_bottom})', fontsize=9)
            ax.axis('off')
            plt.tight_layout()
            plt.show()

        # Final popup: full axial MIP of mandible with mask overlay
        axial_mask_proj = mask.max(axis=0)
        axial_overlay   = np.zeros(axial_mip.shape + (4,), dtype=np.float32)
        axial_overlay[axial_mask_proj] = [1.0, 0.45, 0.0, 0.6]

        fig, ax_axial = plt.subplots(1, 1, figsize=(6, 5))
        ax_axial.imshow(axial_mip, cmap='gray', aspect='auto')
        ax_axial.imshow(axial_overlay, aspect='auto')
        ax_axial.set_title(
            f'Axial MIP — mandible  |  seed z={roi["z_mandible_peak"]}  |  ROI z={z_top}–{z_bottom}',
            fontsize=9,
        )
        ax_axial.axis('off')
        plt.tight_layout()
        plt.show()


    return mask


def _enamel_arch_mask(sub_volume, z_spacing_mm, enamel_hu=1800, enamel_hu_max=5000,
                      frac=0.06, close_mm=4.0):
    """
    Arch footprint from the enamel signal alone, for whole-skull / closed-bite scans
    where the bone window fills the whole facial cross-section.

    Counts enamel-HU voxels along Z → a per-(Y,X) tooth-density map on which only the
    teeth light up (facial bone does not reach the enamel band). The teeth project as
    a dotted arch, so a morphological closing (radius close_mm) bridges the
    inter-tooth gaps into one connected curve before the largest component is kept.
    """
    band = ((sub_volume > enamel_hu) & (sub_volume < enamel_hu_max)).mean(axis=0)
    if band.max() <= 0:
        return np.zeros(band.shape, dtype=bool)
    mask = band > frac * band.max()
    r = max(2, int(round(close_mm / z_spacing_mm)))
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

    The arch has bone stacked across many Z-slices (teeth, alveolar bone, cortical
    plates), so its mean value is high.  The palate is a thin plate spanning only a
    few slices, so its mean value is lower.  Thresholding the MeIP separates them.

    On a jaw-cropped scan this gives a clean curved arch.  On a whole-skull scan the
    bone window fills the entire facial cross-section, so the "arch" comes back as a
    solid blob filling its bounding box.  When that happens (mask extent within its
    bounding box > blob_extent) the footprint is recomputed from the enamel signal
    (`_enamel_arch_mask`), which isolates the teeth from the surrounding facial bone.

    Parameters
    ----------
    volume             : ndarray (Z, Y, X)
    roi                : dict from find_coronal_roi
    bone_min / bone_max: bone window applied before averaging
    threshold_fraction : fraction of the MeIP peak used as the binary threshold
    min_patch_size     : 2D connected components smaller than this (pixels) are removed
    blob_extent        : bounding-box fill ratio above which the bone-window mask is
                         treated as a whole-skull blob and the enamel fallback is used
    show               : display the MeIP and the arch mask

    Returns
    -------
    meip   : ndarray (Y, X) — mean intensity projection along Z within the ROI
    mask2d : bool ndarray (Y, X) — arch footprint mask
    """
    z_top    = roi['z_top']
    z_bottom = roi['z_bottom']

    # Detect the arch SHAPE from only the tooth-bearing occlusal band, not the full render
    # ROI. Over a tall ROI the mean projection superimposes the dental arch with the
    # mandibular rami and skull base, which flare outward inferiorly and join at the back —
    # closing the open horseshoe into a ring whose skeleton is a loop (→ a looped spline).
    # A band centred on the two occlusal planes keeps just the teeth / alveolar arch; the
    # render still uses the full z_top..z_bottom extent for the roots.
    if 'z_maxilla_peak' in roi and 'z_mandible_peak' in roi:
        zc0, zc1 = sorted((int(roi['z_maxilla_peak']), int(roi['z_mandible_peak'])))
        half = max(int(round((zc1 - zc0) * 0.6)), 8)
        z_top    = int(max(z_top, zc0 - half))
        z_bottom = int(min(z_bottom, zc1 + half))

    sub_vol = apply_bone_window(volume[z_top:z_bottom + 1], bone_min, bone_max)
    meip    = sub_vol.mean(axis=0)          # (Y, X)

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
        if mask2d.sum() / bbox_area > blob_extent:
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

def smooth_arch_footprint(mask2d, smooth_radius=10, background=None, show=True):
    """
    Smooths the arch footprint mask by closing small holes then opening (eroding +
    dilating) with a disk, which removes protrusions and blobs from the contour.
    A final hole-fill rounds out any internal voids left by the opening step.

    Parameters
    ----------
    mask2d        : bool ndarray (Y, X)
    smooth_radius : radius in pixels of the disk structuring element
    background    : optional (Y, X) image shown behind the mask in the visualisation
                    (pass the MeIP so anatomy is visible for spatial reference)
    show          : display original vs smoothed mask

    Returns
    -------
    smoothed : bool ndarray (Y, X)
    """
    r = smooth_radius
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    disk = (xx ** 2 + yy ** 2) <= r ** 2

    smoothed = scipy.ndimage.binary_closing(mask2d, structure=disk)
    smoothed = scipy.ndimage.binary_opening(smoothed, structure=disk)
    smoothed = scipy.ndimage.binary_fill_holes(smoothed)

    if show:
        if background is not None:
            bg: np.ndarray = background
        else:
            bg = mask2d.astype(np.float32)

        orig_overlay   = np.zeros(mask2d.shape + (4,), dtype=np.float32)
        orig_overlay[mask2d] = [1.0, 0.45, 0.0, 0.6]

        smooth_overlay = np.zeros(smoothed.shape + (4,), dtype=np.float32)
        smooth_overlay[smoothed] = [1.0, 0.45, 0.0, 0.6]

        _, (ax_orig, ax_smooth) = plt.subplots(1, 2, figsize=(12, 5))

        ax_orig.imshow(bg, cmap='gray', aspect='auto')
        ax_orig.imshow(orig_overlay, aspect='auto')
        ax_orig.set_title('Arch mask — before smoothing', fontsize=9)
        ax_orig.axis('off')

        ax_smooth.imshow(bg, cmap='gray', aspect='auto')
        ax_smooth.imshow(smooth_overlay, aspect='auto')
        ax_smooth.set_title(f'Arch mask — after smoothing  (radius={smooth_radius}px)', fontsize=9)
        ax_smooth.axis('off')

        plt.tight_layout()
        plt.show()

    return smoothed


def find_dental_arch(arch_mask2d, arch_thickness_factor=1.4,
                     posterior_extend=140.0, spline_smooth=3.0,
                     background=None, show=True):
    """
    Detects the dental arch centreline and focal trough from a 2D arch footprint mask.

    Centreline method: skeletonize → prune branches → BFS to get the ordered medial
    path → resample densely, re-centre each sample on the mask cross-section, smooth,
    and fit a *smoothing* cubic B-spline (s > 0) so the curve hugs the arch closely
    without wobbling.  The arch is then continued past each ramus tip
    (posterior_extend) so the back of the ramus / condyle is captured.
      - EDT median radius × factor × 2 → arch thickness Td.
      - Distance-from-spline ≤ Td/2 → focal trough.

    Parameters
    ----------
    arch_mask2d          : bool ndarray (Y, X) — smoothed arch footprint
    arch_thickness_factor: multiplier on median EDT radius (paper: 1.2)
    posterior_extend     : px to continue the arch past each ramus tip (0 = none)
    spline_smooth        : B-spline smoothing budget per point in px² (higher = smoother)
    background           : optional (Y, X) image shown behind the focal trough in
                           popup 3 (pass footprint_meip for anatomical context)
    show                 : three sequential popups

    Returns
    -------
    tck        : scipy spline tuple (use scipy.interpolate.splev to evaluate)
    Td         : float — arch thickness in pixels
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
        junction_mask, structure=np.ones((3, 3), dtype=bool)
    )

    if n_clusters > 0:
        skel_yx = np.argwhere(pruned)
        cy_c = (float(skel_yx[:, 0].min()) + float(skel_yx[:, 0].max())) / 2.0
        cx_c = (float(skel_yx[:, 1].min()) + float(skel_yx[:, 1].max())) / 2.0

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
            # `seen` merges border pixels that are 8-connected to each other
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
                pruned_tmp, structure=np.ones((3, 3), dtype=bool)
            )
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
    D_norm = D / float(D.max()) * 255.0
    core   = D_norm > 245.0
    radius = float(np.median(D[core])) if core.any() else float(D.max())
    Td     = arch_thickness_factor * 2.0 * radius

    # --- 6. Smooth, well-centred arch spline --------------------------------
    # The skeleton is the medial axis (roughly centred) but can be jagged and
    # pulled off-centre where teeth bulge.  Resample it densely by arc length,
    # re-centre each sample on the mask cross-section (midpoint of the LONGEST
    # in-mask run along the local perpendicular — robust to sockets / gaps that
    # a first-to-last midpoint would straddle), smooth, then fit a *smoothing*
    # B-spline (s > 0) so the curve hugs the arch without wobbling through every
    # noisy point.  A poorly centred or wobbly arch is exactly what tilts and
    # distorts the teeth in the panoramic, so both properties matter.
    H, W = arch_mask2d.shape

    ds_path  = np.hypot(np.diff(xs_path), np.diff(ys_path))
    arc_path = np.concatenate([[0.0], np.cumsum(ds_path)])
    n_dense  = max(40, int(arc_path[-1] / 2.0))           # ~1 sample per 2 px
    u_arc    = np.linspace(0.0, arc_path[-1], n_dense)
    xs_b     = np.interp(u_arc, arc_path, xs_path)
    ys_b     = np.interp(u_arc, arc_path, ys_path)

    base_sigma = max(2.0, n_dense / 20.0)
    xs_b = scipy.ndimage.gaussian_filter1d(xs_b, base_sigma)
    ys_b = scipy.ndimage.gaussian_filter1d(ys_b, base_sigma)

    # Re-centre each sample on the mask cross-section
    dxb = np.gradient(xs_b); dyb = np.gradient(ys_b)
    nb  = np.hypot(dxb, dyb); nb[nb < 1e-8] = 1e-8
    perp_x, perp_y = -dyb / nb, dxb / nb
    t_scan = np.arange(-Td / 2.0, Td / 2.0 + 0.5, 0.5)
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
        t_mid = (t_scan[run[0]] + t_scan[run[-1]]) / 2.0
        xs_c[i] = xs_b[i] + t_mid * perp_x[i]
        ys_c[i] = ys_b[i] + t_mid * perp_y[i]
    xs_c = scipy.ndimage.gaussian_filter1d(xs_c, base_sigma)
    ys_c = scipy.ndimage.gaussian_filter1d(ys_c, base_sigma)

    # Posterior extension: continue each ramus end along its local trend so the
    # back of the ramus / condyle falls inside the trough.  A line fit to the
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
    arch_region = dist_from_spline <= (Td / 2.0)

    # Smooth the focal trough boundary: Gaussian blur on the binary mask then
    # re-threshold at 0.5.  This rounds all sharp corners and end-caps into
    # smooth curves without changing the overall width or position.
    smooth_sigma = Td / 8.0
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


def show_focal_trough_overlay(volume, arch_region, bone_min=200, bone_max=1500, opacity=0.5):
    """
    Overlays the focal trough on the CBCT volume and opens it in Fiji for
    interactive inspection.

    The 2D arch_region (Y, X) is extruded along Z — the same trough footprint is
    copied onto every axial slice — producing a curved 3D "curtain" the height of
    the volume. The CBCT is windowed to a bone range and the trough is drawn over
    it as a colour label, so scrolling through the Fiji stack shows whether the
    trough hugs the teeth and jaw at every level.

    Parameters
    ----------
    volume      : ndarray (Z, Y, X) — raw CBCT values
    arch_region : bool ndarray (Y, X) — focal trough mask from find_dental_arch
    bone_min    : lower window edge for the grayscale CBCT display
    bone_max    : upper window edge (above 800 so enamel stays visible without
                  extreme-metal saturation washing out the bone)
    opacity     : trough overlay opacity in Fiji (0–1)
    """
    base  = np.clip(volume, bone_min, bone_max)
    base  = ((base - bone_min) / (bone_max - bone_min) * 255).astype(np.uint8)
    label = np.broadcast_to(arch_region, volume.shape).astype(np.uint8)

    overlay = sitk.LabelOverlay(sitk.GetImageFromArray(base),
                                sitk.GetImageFromArray(label),
                                opacity=opacity)
    sitk.Show(overlay, "Focal trough over CBCT")


# ============================================================================
# Panoramic X-ray synthesis (simulated ray casting)
# ============================================================================

# Tissue-aware HU -> attenuation transfer (novel #2). The single clipped bone
# window (apply_bone_window) is slope-1 and caps enamel at the metal ceiling, so
# pulp / soft tissue is lost (floored to 0) and the brightest enamel merges with
# metal fillings. This piecewise-linear curve instead: gives pulp / soft tissue a
# faint floor (pulp chambers and oral soft tissue read as low gray), keeps a steep
# dentin->enamel ramp (enamel stays distinctly brighter than dentin), and
# compresses metal above the enamel band (restorations / streaks no longer blow
# out or dominate the line integral). Calibrated on the MCSTU 0.18mm HU histogram.
# Applied per sample before integration; np.interp clamps to fp[0]/fp[-1] outside.
DEFAULT_TRANSFER_XP = np.array([-200.,  300.,  800., 1600., 2800., 8000.], np.float32)
DEFAULT_TRANSFER_FP = np.array([   0.,   40.,  300., 1100., 2200., 2500.], np.float32)


def denoise_volume(volume, size=(1, 3, 3)):
    """
    Edge-preserving speckle denoise for the render copy of the volume (novel #1).

    Because each output pixel integrates hundreds of samples along the beam,
    per-voxel CBCT noise accumulates into the wispy / cloudy bone texture that
    gives a synthetic panoramic away. A small median filter removes sub-voxel
    speckle while preserving tooth / cortical edges. The default size (1, 3, 3) is
    a 2-D median within each axial slice, so axial (vertical panoramic) resolution
    is untouched. Only the render copy is denoised; detection keeps the raw volume.
    """
    return scipy.ndimage.median_filter(volume, size=size).astype(np.float32)


def compute_panoramic_trajectory(tck, n_columns=800, n_dense=4000):
    """
    Equal-arc-length sampling of the dental-arch spline → one ray per output column.

    Returns the focal points (on the arch) and the unit normals along which each
    ray integrates (perpendicular to the local arch tangent = the buccolingual
    direction). Equal-arc-length spacing gives uniform tooth spacing in the
    panoramic, avoiding the anterior crowding of uniform-angle sampling.

    Parameters
    ----------
    tck       : spline tuple from find_dental_arch
    n_columns : number of rays / output columns (panoramic width)
    n_dense   : dense samples used to measure arc length

    Returns
    -------
    points    : (N, 2) float — focal point [x, y] per column
    normals   : (N, 2) float — unit normal [nx, ny] per column
    u_samples : (N,)  float — spline parameter at each column (0=one ramus, 1=other)
    arc_length: float — total arch length in pixels
    """
    u = np.linspace(0.0, 1.0, n_dense)
    x, y = scipy.interpolate.splev(u, tck)
    x = np.asarray(x, float); y = np.asarray(y, float)

    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    arc_length = float(s[-1])

    u_samples = np.interp(np.linspace(0.0, arc_length, n_columns), s, u)
    xs, ys = scipy.interpolate.splev(u_samples, tck)
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)

    dx, dy = scipy.interpolate.splev(u_samples, tck, der=1)
    dx = np.asarray(dx, float); dy = np.asarray(dy, float)
    nrm = np.hypot(dx, dy); nrm[nrm < 1e-8] = 1e-8
    tx, ty = dx / nrm, dy / nrm

    points  = np.stack([xs, ys], axis=1)
    normals = np.stack([-ty, tx], axis=1)          # perpendicular to tangent
    # Orient every normal outward (away from the arch centroid) so +normal points
    # labially/buccally and −normal lingually.  This lets the focal trough be
    # extended asymmetrically — further labially at the incisors to catch forward /
    # spread anterior teeth, while staying shallow lingually.
    centroid = points.mean(axis=0)
    flip = ((points - centroid) * normals).sum(axis=1) < 0
    normals[flip] *= -1
    return points, normals, u_samples, arc_length


def compute_trough_halfwidths(Td, n_columns, anterior_boost=0.0, anterior_sigma=0.28,
                              posterior_boost=0.0, posterior_sigma=0.16):
    """
    Per-column focal-trough half-widths, returned separately for the lingual
    (inward, −normal) and labial (outward, +normal) sides.

    The lingual side stays at the uniform Td/2 — extending it deeper buys nothing
    (tongue / air) and, at the sharp incisor apex where the curvature radius is
    small, makes neighbouring columns' normals cross, smearing the anterior out of
    focus.  The labial side is widened toward the incisors (arc-length midpoint) by
    anterior_boost so forward-projecting and spread anterior teeth are captured.
    Set anterior_boost=0 for a uniform symmetric trough.

    posterior_boost widens BOTH sides toward the two arch ends (u→0 and u→1).  The
    maxillary molars / third molars sit slightly off the mandible-biased arch curve
    and have a wide bucco-lingual spread, so a deeper trough at the posterior keeps
    them in view without affecting the rest of the arch. posterior_sigma sets how
    far in from each end the widening reaches.

    Returns
    -------
    hw_lingual, hw_labial : (N,) arrays — inward / outward half-widths per column
    """
    u = np.linspace(0.0, 1.0, n_columns)
    bump = np.exp(-((u - 0.5) ** 2) / (2.0 * anterior_sigma ** 2))
    post = (np.exp(-(u ** 2) / (2.0 * posterior_sigma ** 2)) +
            np.exp(-((u - 1.0) ** 2) / (2.0 * posterior_sigma ** 2)))
    base = Td / 2.0
    hw_lingual = base * (1.0 + posterior_boost * post)
    hw_labial  = base * (1.0 + anterior_boost * bump + posterior_boost * post)
    return hw_lingual, hw_labial


def cast_panoramic_rays(volume, points, normals, hw_lingual, hw_labial, z_range,
                        n_depth=140, soft_tissue_hu=200, metal_ceiling_hu=2500,
                        vertical_tilt_deg=0.0, focal_sigma_frac=0.0,
                        render_scale=1.0, focal_floor=0.0,
                        soft_tissue_gain=0.0, air_hu=-300.0,
                        depth_tilt=0.0, ray_angle_deg=0.0, focal_offset=None,
                        root_focus=0.0, occ_rows=None, transfer=None):
    """
    Core ray caster. For each column (arch focal point + normal) and each output
    row (axial height z), integrate windowed attenuation along the normal across
    the focal trough → a line-integral panoramic (before tone mapping).

    Attenuation per sample = apply_bone_window(HU, soft_tissue_hu, metal_ceiling_hu):
    air/soft tissue below soft_tissue_hu contribute 0; metal/enamel above
    metal_ceiling_hu are capped (metal-artifact mitigation, novel #7 — raise the
    ceiling to disable). Sampling is trilinear; out-of-volume = air.

    Parameters
    ----------
    volume            : (Z, Y, X) raw CBCT (HU)
    points, normals       : from compute_panoramic_trajectory (normals point labially)
    hw_lingual, hw_labial : (N,) inward / outward trough half-widths per column
    z_range           : (z_top, z_bottom) axial rows to render
    n_depth           : samples along each ray
    vertical_tilt_deg : downward beam tilt; shears z with depth (smile-line). 0 = flat.
    focal_sigma_frac  : focal-trough depth weighting. The real focal layer is not a
                        hard slab: in-layer structures dominate and off-layer ones
                        fade out (motion blur). A Gaussian weight centred on the arch
                        layer (t=0), sigma = focal_sigma_frac · half-width, emphasises
                        on-layer teeth/roots and suppresses off-layer ghosting that
                        otherwise superimposes and distorts them, while keeping the
                        trough wide enough to still capture spread anterior teeth.
                        0 = uniform (hard slab, no weighting).
    render_scale      : multiplies the buccolingual *integration* extent beyond the
                        focal trough (the rays are cast over ±render_scale·half-width).
                        A real panoramic superimposes the whole beam path, not just the
                        focal slab, so the sinus walls, nasal septum, hard palate and
                        zygoma above the teeth are only rendered when the integration
                        reaches them. render_scale=1 = trough-only (old behaviour).
    focal_floor       : background weight floor for the off-trough extent, in [0,1].
                        With focal_sigma_frac>0 the depth weight is
                        focal_floor + (1−focal_floor)·Gaussian, so the focal layer keeps
                        weight≈1 (sharp teeth) while structures out to ±render_scale·hw
                        contribute at ≥focal_floor (a soft, present background instead of
                        the empty band the hard trough produced). 0 = off-trough fades to
                        nothing (pure focal layer).
    soft_tissue_gain  : weight for a faint soft-tissue attenuation term (HU in
                        [air_hu, soft_tissue_hu]). The bone window discards everything
                        below soft_tissue_hu, leaving the oral cavity pitch black; a
                        small gain adds the tongue / cheek / lip / air-gap soft tissue
                        back as a low gray so the mouth interior looks like a real
                        panoramic. 0 = off (black mouth).
    air_hu            : HU floor of the soft-tissue ramp; below this is treated as air
                        and contributes nothing.

    Returns
    -------
    L : (H, N) float — line integral of windowed attenuation
    """
    z0 = max(0, int(z_range[0]))
    z1 = min(volume.shape[0] - 1, int(z_range[1]))
    zs = np.arange(z0, z1 + 1)
    H, N = len(zs), len(points)
    L = np.zeros((H, N), dtype=np.float32)
    tan_t = np.tan(np.deg2rad(vertical_tilt_deg))

    # Z-dependent buccolingual offset of the focal centre (depth_tilt). The maxillary
    # teeth and their roots flare buccally and sit higher than the mandibular teeth,
    # so a single fixed-depth trough cannot stay centred on both. off_row shifts the
    # focal centre along the normal as a linear function of height — total shift
    # `depth_tilt` px from the top render row to the bottom — so the focal layer
    # tracks the flare instead of slicing across it (less root/crown distortion).
    z_ref = 0.5 * (float(zs[0]) + float(zs[-1]))
    zspan = max(1.0, float(zs[-1] - zs[0]))
    off_row = depth_tilt * (zs - z_ref) / zspan        # (H,)
    # ray_angle_deg may be a scalar (uniform — skews the whole occlusal plane) or a
    # per-column array. The useful form is anti-symmetric about the arch midline so the
    # two posterior ends rotate outward while the anterior stays at the normal: that
    # reduces interproximal overlap on both sides without skewing the arch.
    ang = np.deg2rad(np.broadcast_to(np.asarray(ray_angle_deg, float), (N,)))
    ca_a, sa_a = np.cos(ang), np.sin(ang)

    # ---- Fast path (no vertical beam tilt) --------------------------------------
    # With tan_t == 0 every ray sample sits in an integer axial slice (Zc == zs[i]),
    # so the 3-D trilinear interpolation reduces *exactly* to a 2-D bilinear one on
    # that single slice. Interpolating per slice (small, cache-resident) instead of
    # per column (the whole volume, cache-thrashing) is ~1.5-2x faster and bit-for-bit
    # identical. The per-column loop below is kept for the vertical-tilt case.
    if tan_t == 0.0:
        nx_all = ca_a * normals[:, 0] - sa_a * normals[:, 1]      # (N,) ray dirs
        ny_all = sa_a * normals[:, 0] + ca_a * normals[:, 1]
        x0_all, y0_all = points[:, 0], points[:, 1]
        hw_l = np.asarray(hw_lingual, float) * render_scale       # (N,)
        hw_o = np.asarray(hw_labial, float) * render_scale
        s = np.linspace(0.0, 1.0, n_depth)
        T = -hw_l[:, None] + (hw_l + hw_o)[:, None] * s[None, :]   # (N, n_depth) == per-col linspace
        dt = (hw_l + hw_o) / (n_depth - 1)                        # (N,)
        if focal_sigma_frac > 0.0:
            sig = focal_sigma_frac * np.maximum(hw_l, hw_o) / render_scale
            g = np.exp(-0.5 * (T / (sig[:, None] + 1e-8)) ** 2)   # (N, n_depth) pure focal core
            W = focal_floor + (1.0 - focal_floor) * g
        else:
            g = W = None
        # Root-band focal tightening (#2): lower the background floor away from the
        # occlusal plane, so the off-layer haze that fogs the apical roots is suppressed
        # while the crowns keep their normal superimposition. floor_row multiplies the
        # background term per output row; 0 at the occlusal band → focal_floor, ramping
        # to focal_floor·(1−root_focus) at the apices.
        floor_row = None
        if root_focus > 0.0 and g is not None:
            if occ_rows is not None:
                lo, hi = sorted((int(occ_rows[0]), int(occ_rows[1])))
                d = np.where(zs < lo, lo - zs, np.where(zs > hi, zs - hi, 0.0)).astype(float)
                dmax = max(lo - zs[0], zs[-1] - hi, 1.0)
            else:
                c = 0.5 * (zs[0] + zs[-1]); d = np.abs(zs - c).astype(float)
                dmax = max(zs[-1] - c, 1.0)
            apical = np.clip(d / dmax, 0.0, 1.0)                  # 0 crowns … 1 apices
            floor_row = focal_floor * (1.0 - root_focus * apical) # (H,)
        shifts = bool(depth_tilt) or (focal_offset is not None)
        for i in range(H):
            if shifts:
                off = 0.0
                if depth_tilt:
                    off = off + off_row[i]
                if focal_offset is not None:
                    off = off + focal_offset[i, :]               # (N,)
                td = T + np.asarray(off).reshape(-1, 1)          # (N, n_depth)
            else:
                td = T
            Xc = x0_all[:, None] + td * nx_all[:, None]
            Yc = y0_all[:, None] + td * ny_all[:, None]
            samp = scipy.ndimage.map_coordinates(
                volume[zs[i]], np.vstack([Yc.ravel(), Xc.ravel()]),
                order=1, mode='constant', cval=-1000.0).reshape(N, n_depth)
            if transfer is not None:
                atten = np.interp(samp, transfer[0], transfer[1]).astype(np.float32)
            else:
                atten = apply_bone_window(samp, soft_tissue_hu, metal_ceiling_hu)
                if soft_tissue_gain > 0.0:
                    atten = atten + soft_tissue_gain * (np.clip(samp, air_hu, soft_tissue_hu) - air_hu)
            if floor_row is not None:
                fr = floor_row[i]
                L[i, :] = (fr * atten.sum(axis=1) + (1.0 - fr) * (atten * g).sum(axis=1)) * dt
            elif W is not None:
                L[i, :] = (atten * W).sum(axis=1) * dt
            else:
                L[i, :] = atten.sum(axis=1) * dt
        return L

    for j in range(N):
        x0, y0 = points[j]
        nx, ny = normals[j]
        # In-plane ray angle: rotate the cast direction off the strict arch normal.
        # A real panoramic beam is not 90° to the arch (Anusree §4.3); a small rotation
        # changes which buccolingual planes line up and can reduce interproximal overlap.
        if sa_a[j] != 0.0:
            nx, ny = ca_a[j] * nx - sa_a[j] * ny, sa_a[j] * nx + ca_a[j] * ny
        hw_l, hw_o = float(hw_lingual[j]) * render_scale, float(hw_labial[j]) * render_scale
        t = np.linspace(-hw_l, hw_o, n_depth)      # −lingual … +labial
        dt = (hw_l + hw_o) / (n_depth - 1)
        if focal_sigma_frac > 0.0:
            sig = focal_sigma_frac * max(hw_l, hw_o) / render_scale
            g = np.exp(-0.5 * (t / (sig + 1e-8)) ** 2)
            w = focal_floor + (1.0 - focal_floor) * g
        else:
            w = None

        if depth_tilt or focal_offset is not None:
            # focal centre shifts along the normal at each height. off_row is the
            # linear depth_tilt term; focal_offset[:, j] adds the data-driven smooth
            # axial focal-trough surface (compute_focal_offset_surface) so the focal
            # layer continuously tracks the dentition at every level without snapping.
            off = off_row if depth_tilt else 0.0
            if focal_offset is not None:
                off = off + focal_offset[:, j]
            td = t[None, :] + np.asarray(off).reshape(-1, 1)   # (H, n_depth)
            Xc = x0 + td * nx
            Yc = y0 + td * ny
        else:
            xs = x0 + t * nx                       # (n_depth,)
            ys = y0 + t * ny
            Xc = np.broadcast_to(xs, (H, n_depth))
            Yc = np.broadcast_to(ys, (H, n_depth))
        if tan_t:
            Zc = zs[:, None] + (t * tan_t)[None, :]
        else:
            Zc = np.broadcast_to(zs[:, None], (H, n_depth))

        coords = np.vstack([Zc.ravel(), Yc.ravel(), Xc.ravel()])
        samp = scipy.ndimage.map_coordinates(
            volume, coords, order=1, mode='constant', cval=-1000.0
        ).reshape(H, n_depth)

        if transfer is not None:
            atten = np.interp(samp, transfer[0], transfer[1]).astype(np.float32)
        else:
            atten = apply_bone_window(samp, soft_tissue_hu, metal_ceiling_hu)
            if soft_tissue_gain > 0.0:
                # Faint soft-tissue contribution so the beam path is never pure air:
                # the tongue, cheeks, lips and air-gap soft tissue render as a low gray,
                # filling the oral cavity (otherwise pitch black) like a real panoramic.
                # Ramps 0 (air at air_hu) → soft_tissue_hu−air_hu (soft tissue), where the
                # bone window takes over; metal/enamel are unaffected.
                soft = np.clip(samp, air_hu, soft_tissue_hu) - air_hu
                atten = atten + soft_tissue_gain * soft
        if w is not None:
            L[:, j] = (atten * w[None, :]).sum(axis=1) * dt
        else:
            L[:, j] = atten.sum(axis=1) * dt

    return L


def compute_focal_offset_surface(volume, points, normals, z_range, Td,
                                 soft_tissue_hu=200, metal_ceiling_hu=2500,
                                 search_scale=1.6, smooth_z_frac=0.12,
                                 smooth_col_frac=0.05, max_offset_frac=0.9,
                                 show=False):
    """
    Smooth, axially-varying focal-trough centre surface — `offset[row, column]` in px.

    A single fixed-depth focal trough is the same buccolingual slab at every height,
    but the dentition is not: the maxillary teeth and their roots flare buccally and
    sit higher than the mandibular teeth. This computes a per-(height, column) shift of
    the focal centre along the arch normal so the focal layer continuously follows the
    teeth at every level — a *continuous* axial focal trough, each slice growing/
    shrinking smoothly from its neighbours rather than snapping.

    Two stages, exactly as a "crude then smoothed" construction:

      1. Raw (crude) centre — for every column and every axial row, probe windowed
         attenuation along the normal over ±search_scale·(Td/2) and take the bone
         centre-of-mass. Where bone is present the centre sits on the teeth/alveolus;
         where the probe is mostly air the estimate is unreliable (low confidence).
      2. Coherence smoothing — a *normalised* (confidence-weighted) Gaussian blur over
         the (row, column) surface. Dividing a blurred confidence-weighted raw by a
         blurred confidence both (a) fills the low-confidence air gaps from neighbours
         and (b) enforces C-infinity continuity across rows and columns, so no sharp
         transition can appear. The result is clamped to ±max_offset_frac·(Td/2).

    Parameters
    ----------
    Td               : focal-trough depth in px (sets probe window and clamp scale)
    search_scale     : probe half-window as a multiple of Td/2
    smooth_z_frac    : Gaussian σ over rows, as a fraction of the row count (large =
                       more axial coherence)
    smooth_col_frac  : Gaussian σ over columns, as a fraction of the column count
    max_offset_frac  : clamp on |offset| as a fraction of Td/2 (keeps the layer from
                       chasing distant structures like the spine)
    show             : unused here (visualisation is done by the caller)

    Returns
    -------
    surface : (H, N) float — smooth focal-centre offset per row/column
    raw     : (H, N) float — pre-smoothing centre (for visualisation/debug)
    conf    : (H, N) float — per-sample bone confidence used in the smoothing
    """
    z0 = max(0, int(z_range[0]))
    z1 = min(volume.shape[0] - 1, int(z_range[1]))
    zs = np.arange(z0, z1 + 1)
    H, N = len(zs), len(points)

    half = search_scale * (Td / 2.0)
    n_probe = max(31, int(round(2 * half)))
    t = np.linspace(-half, half, n_probe)

    raw  = np.zeros((H, N), dtype=np.float32)
    conf = np.zeros((H, N), dtype=np.float32)
    for j in range(N):
        x0, y0 = points[j]
        nx, ny = normals[j]
        xs = x0 + t * nx
        ys = y0 + t * ny
        Xc = np.broadcast_to(xs, (H, n_probe))
        Yc = np.broadcast_to(ys, (H, n_probe))
        Zc = np.broadcast_to(zs[:, None], (H, n_probe))
        samp = scipy.ndimage.map_coordinates(
            volume, np.vstack([Zc.ravel(), Yc.ravel(), Xc.ravel()]),
            order=1, mode='constant', cval=-1000.0).reshape(H, n_probe)
        wbone = apply_bone_window(samp, soft_tissue_hu, metal_ceiling_hu)  # (H, n_probe)
        wsum = wbone.sum(axis=1)
        good = wsum > 1e-6
        cen = np.zeros(H, dtype=np.float32)
        cen[good] = (wbone[good] * t[None, :]).sum(axis=1) / wsum[good]
        raw[:, j]  = cen
        conf[:, j] = wsum

    # Normalised (confidence-weighted) Gaussian smoothing → fills air gaps + C-inf coherence
    sig_z   = max(1.0, smooth_z_frac * H)
    sig_col = max(1.0, smooth_col_frac * N)
    num = scipy.ndimage.gaussian_filter(raw * conf, (sig_z, sig_col), mode='nearest')
    den = scipy.ndimage.gaussian_filter(conf,       (sig_z, sig_col), mode='nearest')
    surface = num / (den + 1e-6)

    clamp = max_offset_frac * (Td / 2.0)
    surface = np.clip(surface, -clamp, clamp).astype(np.float32)
    return surface, raw, conf


def tone_map(L, method='beer_lambert', strength=4.0):
    """
    Map the line integral to displayable intensity (dense tissue → bright).
      beer_lambert : 1 − exp(−strength · L/L99)   (physical; compresses metal/enamel)
      raysum       : L / L99                        (linear; conventional fallback)
    L99 = 99th percentile of positive L (robust gain normalization).
    """
    ref = np.percentile(L[L > 0], 99) if np.any(L > 0) else 1.0
    Ln = L / (ref + 1e-8)
    if method == 'beer_lambert':
        return 1.0 - np.exp(-strength * Ln)
    return Ln


def correct_column_intensity(L, sigma=25.0):
    """
    Flatten slow left-right brightness drift (e.g. denser molar columns) by
    dividing each column by a smoothed version of the per-column mean. Analogue of
    Kwon's view weighting. Optional — can erase real density differences.
    """
    col_mean = L.mean(axis=0)
    smooth = scipy.ndimage.gaussian_filter1d(col_mean, sigma=sigma)
    smooth[smooth < 1e-8] = 1e-8
    return L * (smooth.mean() / smooth)[None, :]


def apply_scatter(px, fraction=0.0, sigma=12.0):
    """Add a broad low-frequency scatter floor (novel #4, simple form). 0 disables."""
    if fraction <= 0:
        return px
    return px + fraction * scipy.ndimage.gaussian_filter(px, sigma=sigma)


def enhance_edges(px, alpha=0.0, sigma=1.2):
    """Unsharp masking (Yun): px + alpha·(px − G*px). alpha=0 disables."""
    if alpha <= 0:
        return px
    return px + alpha * (px - scipy.ndimage.gaussian_filter(px, sigma=sigma))


def apply_detector_blur(px, sigma_v=0.0, sigma_h=0.0):
    """
    Focal-spot + detector PSF blur (novel #5). Anisotropic Gaussian (rows=vertical,
    cols=horizontal) removes unrealistic pixel-sharpness. 0,0 disables.
    """
    if sigma_v <= 0 and sigma_h <= 0:
        return px
    return scipy.ndimage.gaussian_filter(px, sigma=(sigma_v, sigma_h))


def apply_smile_curve(px, u_samples, amount=0.0):
    """
    Cosmetic occlusal 'smile': raise each column by a parabola of arc position
    (ends lifted relative to the middle). amount in pixels; 0 disables.
    """
    if amount == 0:
        return px
    shift = (amount * (2.0 * (u_samples - 0.5)) ** 2).astype(int)
    out = np.zeros_like(px)
    H = px.shape[0]
    for j, sh in enumerate(shift):
        if sh <= 0:
            out[:, j] = px[:, j]
        else:
            out[:H - sh, j] = px[sh:, j]
    return out


def normalize_panoramic(px, p_low=1.0, p_high=99.5, gamma=1.0,
                        base_fog=0.0, noise_std=0.0, seed=0):
    """
    Percentile-clip → [0,1] → gamma → uint8 grayscale for display/saving.

    Real panoramic backgrounds are not pitch black: Compton scatter, soft tissue
    always in the beam, and detector base fog + electronic noise lift the empty
    space to a low gray. base_fog raises the floor *after* the [0,1] mapping
    (out = base_fog + (1−base_fog)·out) so the background becomes gray instead of
    being clipped to 0; noise_std adds faint Gaussian detector noise on top. Both
    default to 0 (clean black background).
    """
    lo, hi = np.percentile(px, [p_low, p_high])
    out = np.clip((px - lo) / (hi - lo + 1e-8), 0.0, 1.0) ** gamma
    if base_fog > 0.0:
        out = base_fog + (1.0 - base_fog) * out
    if noise_std > 0.0:
        rng = np.random.default_rng(seed)
        out = np.clip(out + rng.normal(0.0, noise_std, out.shape), 0.0, 1.0)
    return (out * 255).astype(np.uint8)


def enhance_panoramic(px, clahe=False, multiscale=False, unsharp=False,
                      sharpen=False, contrast=False):
    """
    Post-synthesis contrast / detail enhancement on a uint8 panoramic. Returns uint8.

    Each effect is an independent on/off flag and they compose in this order —
    CLAHE → contrast → multiscale → unsharp → sharpen (local contrast first, broad
    then fine detail, strong high-boost last):

      clahe      : Contrast-Limited Adaptive Histogram Equalization — flattens the
                   strong dark-background / bright-tooth imbalance so mid-density bone
                   (sinus walls, mandibular canal, roots) becomes readable. The heavier
                   change in look; also lifts the off-focal background.
      contrast   : global sigmoid contrast stretch (skimage adjust_sigmoid, cutoff 0.5,
                   gain 7) — deepens blacks and brightens enamel for more "pop" without
                   the local-detail amplification CLAHE introduces.
      multiscale : Kwon multi-scale unsharp — adds detail bands at increasing scales,
                   α0·I0 + α1·(I0−G1) + α2·(G1−G2) + α3·(G2−G3) with α=(1.0,1.0,1.5,1.5)
                   and σ=(2.4,4.8,19.2) (Kwon edge-enhancement formula). Lifts both fine
                   tooth detail and broad bone contrast.
      unsharp    : Yun single-scale unsharp mask — px + 0.9·(px − G_0.8·px), 3×3
                   Gaussian, σ=0.8 (Yun §2.2.3, the recommended parameters).
      sharpen    : stronger high-boost unsharp — px + 1.5·(px − G_1.6·px) — for a
                   crisper edge than `unsharp` when the image still looks soft.

    All False → pass-through.
    """
    from skimage import exposure

    out = px.astype(np.float32)

    if clahe:
        eq = exposure.equalize_adapthist(
            np.clip(out / 255.0, 0, 1), clip_limit=0.01)
        out = eq.astype(np.float32) * 255.0
    if contrast:
        sig = exposure.adjust_sigmoid(
            np.clip(out / 255.0, 0, 1), cutoff=0.5, gain=7)
        out = sig.astype(np.float32) * 255.0
    if multiscale:
        g1 = scipy.ndimage.gaussian_filter(out, 2.4)
        g2 = scipy.ndimage.gaussian_filter(out, 4.8)
        g3 = scipy.ndimage.gaussian_filter(out, 19.2)
        out = 1.0 * out + 1.0 * (out - g1) + 1.5 * (g1 - g2) + 1.5 * (g2 - g3)
    if unsharp:
        out = out + 0.9 * (out - scipy.ndimage.gaussian_filter(out, 0.8))
    if sharpen:
        out = out + 1.5 * (out - scipy.ndimage.gaussian_filter(out, 1.6))

    return np.clip(out, 0, 255).astype(np.uint8)


def synthesize_panoramic(volume, tck, Td, roi,
                         n_columns=800, n_depth=140, top_margin=90, bottom_margin=35,
                         soft_tissue_hu=200, metal_ceiling_hu=2500,
                         tone='beer_lambert', strength=4.0,
                         vertical_tilt_deg=0.0, trough_scale=0.5,
                         anterior_boost=0.0, anterior_sigma=0.28,
                         posterior_boost=0.0, posterior_sigma=0.16,
                         focal_sigma_frac=0.0, render_scale=1.0, focal_floor=0.0,
                         soft_tissue_gain=0.0, air_hu=-300.0,
                         ray_angle_deg=0.0, ray_angle_curvature=False,
                         depth_tilt=0.0, focal_offset=None, root_focus=0.0,
                         maxilla_post_bias=0.0, maxilla_post_sigma=0.16,
                         source_traj=0.0,
                         column_correct=False, scatter_fraction=0.0,
                         edge_alpha=0.0, edge_sigma=1.2,
                         blur_v=0.0, blur_h=0.0, smile_amount=0.0,
                         gamma=1.0, base_fog=0.0, noise_std=0.0, transfer=None):
    """
    Full simulated-ray-casting panoramic synthesis, one knob per step so each stage
    can be tweaked independently. Returns a uint8 (H, N) grayscale image.

    trough_scale shrinks the detected Td (which find_dental_arch calibrates for MIP
    synthesis) to a tighter ray-casting focal depth; ~0.4 fits the jaw on the test
    data. anterior_boost then widens the trough labially (outward) at the incisors
    so forward / spread anterior teeth stay captured; anterior_sigma sets how much
    of the arch around the midline is treated as "anterior".

    focal_sigma_frac applies a Gaussian focal-depth weight inside the trough to
    suppress off-layer ghosting and reduce distortion of teeth/roots (see
    cast_panoramic_rays). base_fog / noise_std lift the empty space to a realistic
    gray and add faint detector noise (see normalize_panoramic); scatter_fraction
    adds a broad Compton-scatter halo around dense structures.

    ray_angle_deg is the *amplitude* of an anti-symmetric in-plane ray rotation: it is
    expanded to an outward rotation at the two posterior ends ramping to 0 at the
    anterior (ray_angle_deg·2(u−0.5)). This reduces interproximal overlap and improves
    the posterior / 3rd-molar separation symmetrically, without skewing the occlusal
    plane the way a uniform rotation would. depth_tilt tilts the focal layer with
    height (see cast_panoramic_rays). focal_offset is an optional (H, N) smooth
    focal-centre surface (compute_focal_offset_surface) for a continuous axial focal
    trough; it supersedes the linear depth_tilt term where supplied.

    Pipeline: trajectory → trough widths → ray cast (focal-weighted line integral) →
    [column correction] → tone map → [scatter] → [edge enhance] →
    [detector blur] → [smile] → normalize (+ base fog / noise).
    """
    points, normals, u_samples, _ = compute_panoramic_trajectory(tck, n_columns)
    if source_traj:
        # Virtual source-trajectory geometry (Article 7): instead of casting straight
        # across the arch (⟂ normal), cast along the beam from a moving X-ray source.
        # Model the source positions as an ellipse about the arch centroid, so the
        # imaging ray at each focal point runs radially through the centroid — oblique
        # to the arch away from the axes (canine / premolar), like a real panoramic
        # beam that is not 90° to the arch. source_traj in [0,1] blends ⟂ normal (0) →
        # radial source-trajectory ray (1).
        centroid = points.mean(axis=0)
        radial = points - centroid
        radial = radial / (np.hypot(radial[:, 0], radial[:, 1])[:, None] + 1e-8)
        flip = (radial * normals).sum(axis=1) < 0          # keep outward orientation
        radial[flip] *= -1
        normals = normals + source_traj * (radial - normals)
        normals = normals / (np.hypot(normals[:, 0], normals[:, 1])[:, None] + 1e-8)
    hw_lingual, hw_labial = compute_trough_halfwidths(
        Td * trough_scale, n_columns, anterior_boost, anterior_sigma,
        posterior_boost, posterior_sigma)
    z_range = (roi['z_top'] - top_margin, roi['z_bottom'] + bottom_margin)

    if ray_angle_deg and ray_angle_curvature:
        # Concentrate the beam angulation where the arch is FLAT (low curvature =
        # premolars/molars), the region where adjacent-tooth contacts run parallel to
        # the beam and overlap most. Anti-symmetric (outward at each side) so the
        # occlusal plane is not skewed; magnitude = (1 − normalised curvature).
        dx, dy = scipy.interpolate.splev(u_samples, tck, der=1)
        ddx, ddy = scipy.interpolate.splev(u_samples, tck, der=2)
        dx = np.asarray(dx, float); dy = np.asarray(dy, float)
        ddx = np.asarray(ddx, float); ddy = np.asarray(ddy, float)
        kappa = np.abs(dx * ddy - dy * ddx) / (dx * dx + dy * dy) ** 1.5
        kn = kappa / (kappa.max() + 1e-9)
        ray_ang = ray_angle_deg * np.sign(u_samples - 0.5) * (1.0 - kn)
    elif ray_angle_deg:
        ray_ang = ray_angle_deg * 2.0 * (u_samples - 0.5)
    else:
        ray_ang = 0.0

    if maxilla_post_bias:
        # Maxilla-biased posterior focal path: the maxillary molars sit more buccal and
        # higher than the mandible-biased arch, so shift the focal centre outward (+normal
        # = buccal) in the upper-posterior region to bring them onto the sharp core. The
        # shift ramps up toward the maxilla rows (above the inter-arch gap) and toward the
        # two posterior arch ends, and is zero elsewhere (so it never disturbs the rest).
        z0b = max(0, int(z_range[0])); z1b = min(volume.shape[0] - 1, int(z_range[1]))
        zs_b = np.arange(z0b, z1b + 1)
        z_gap = roi['z_gap']
        mw = np.clip((z_gap - zs_b) / max(z_gap - zs_b[0], 1.0), 0.0, 1.0)     # (H,) 1=maxilla top, 0=gap/below
        pw = (np.exp(-(u_samples ** 2) / (2.0 * maxilla_post_sigma ** 2)) +
              np.exp(-((u_samples - 1.0) ** 2) / (2.0 * maxilla_post_sigma ** 2)))  # (N,) posterior ends
        bias = (maxilla_post_bias * (Td / 2.0) * mw[:, None] * pw[None, :]).astype(np.float32)
        focal_offset = bias if focal_offset is None else (focal_offset + bias)

    occ_rows = (roi['z_maxilla_peak'], roi['z_mandible_peak']) if root_focus > 0 else None
    L = cast_panoramic_rays(volume, points, normals, hw_lingual, hw_labial, z_range,
                            n_depth, soft_tissue_hu, metal_ceiling_hu,
                            vertical_tilt_deg, focal_sigma_frac,
                            render_scale, focal_floor, soft_tissue_gain, air_hu,
                            depth_tilt, ray_ang, focal_offset,
                            root_focus=root_focus, occ_rows=occ_rows, transfer=transfer)
    if column_correct:
        L = correct_column_intensity(L)
    px = tone_map(L, tone, strength)
    px = apply_scatter(px, scatter_fraction)
    px = enhance_edges(px, edge_alpha, edge_sigma)
    px = apply_detector_blur(px, blur_v, blur_h)
    px = apply_smile_curve(px, u_samples, smile_amount)
    return normalize_panoramic(px, gamma=gamma, base_fog=base_fog, noise_std=noise_std)


def _smooth_arch_adaptive(mask):
    """
    Light, thickness-adaptive arch smoothing for the one-call entry point.

    `smooth_arch_footprint` opens with a fixed 10 px disk, which is right for the
    thick bone-window arch but erodes a thin enamel-derived arch (whole-skull /
    closed-bite scans) away entirely, leaving only a lump and a degenerate spline.
    Here the structuring radius follows the arch's own median half-thickness and
    only a closing + hole-fill is applied (no aggressive opening), so thin arches
    survive intact while inter-tooth gaps are still bridged.
    """
    edt = scipy.ndimage.distance_transform_edt(mask)
    med = float(np.median(edt[mask])) if mask.any() else 3.0
    r = int(np.clip(round(med * 0.6), 2, 10))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    disk = (xx ** 2 + yy ** 2) <= r ** 2
    s = scipy.ndimage.binary_closing(mask, structure=disk)
    return scipy.ndimage.binary_fill_holes(s)


def synthesize_panoramic_from_volume(volume, z_spacing_mm,
                                     n_depth=200, trough_depth_mm=14.0,
                                     posterior_extend_mm=22.0,
                                     sup_margin_mm=38.0, inf_margin_mm=16.0,
                                     anterior_boost=1.0, anterior_sigma=0.28,
                                     posterior_boost=1.0, posterior_sigma=0.20,
                                     render_scale=3.0, focal_floor=0.12,
                                     focal_sigma_frac=0.5, soft_tissue_gain=0.25,
                                     ray_angle_deg=6.0, ray_angle_curvature=False,
                                     depth_tilt=0.0, adaptive_focal=True, root_focus=0.5,
                                     maxilla_post_bias=0.0,
                                     source_traj=0.0, smile_frac=0.10,
                                     tone='raysum', strength=4.0, gamma=1.0, clahe=True,
                                     denoise=True, transfer=True,
                                     column_correct=True, base_fog=0.06,
                                     noise_std=0.0, blur_v=0.0, blur_h=0.0,
                                     display_height=560, show=False, tck_override=None):
    """
    One-call panoramic synthesis, robust across field-of-view, resolution and bite.

    Wraps the whole pipeline with resolution-independent settings so the same call
    works on a tight jaw CBCT and a whole-skull scan, open or closed bite:

      * enamel-band ROI (`find_coronal_roi(volume=...)`) — FOV / bite robust;
      * `_smooth_arch_adaptive` — keeps thin enamel arches alive;
      * focal-trough depth, posterior extension and vertical margins given in **mm**
        (converted with z_spacing_mm) so they don't scale with voxel size — the old
        pixel defaults over-extended low-resolution scans, inflating the arch and
        shrinking the upper structures;
      * columns sampled at the native arch length → correct panoramic aspect ratio
        (fixed n_columns horizontally stretched short low-res arches);
      * cubic upsample to display_height for a smooth, non-grainy output.

    Parameters
    ----------
    volume         : (Z, Y, X) raw CBCT (HU), already in upright orientation
    z_spacing_mm   : axial slice thickness (mm)
    trough_depth_mm: bucco-lingual focal-trough depth (the *sharp* focal layer)
    posterior_extend_mm / sup_margin_mm / inf_margin_mm : arch + render extents (mm)
    posterior_boost / posterior_sigma : widen the sharp focal layer toward the two
        arch ends so the most posterior / 3rd molars stay in focus (issue: far teeth
        out of focus on a single uniform-depth trough)
    render_scale / focal_floor / focal_sigma_frac : full-beam superimposition
        controls (see cast_panoramic_rays). render_scale>1 casts each ray well beyond
        the focal trough so the mid-face (sinus walls, nasal septum, hard palate,
        zygoma) is rendered like a real panoramic instead of leaving an empty band;
        focal_sigma_frac keeps the focal layer sharp and focal_floor sets how strongly
        the off-layer background shows through. render_scale=1, focal_floor=0 restores
        the old trough-only behaviour.
    soft_tissue_gain : fills the oral cavity / air gap with a faint soft-tissue gray
        (tongue, cheeks, lips) so the mouth is not pitch black, like a real panoramic
        (see cast_panoramic_rays). 0 = black mouth.
    ray_angle_deg : amplitude of the anti-symmetric in-plane ray rotation (outward at
        the posterior ends, 0 at the anterior) — reduces interproximal overlap and
        sharpens the posterior / 3rd-molar separation without skewing the arch. 0 = off.
    depth_tilt : px the focal layer shifts buccolingually from the top to the bottom
        render row, to track the maxilla-vs-mandible flare (see cast_panoramic_rays).
        Subtle on the test data; 0 by default.
    ray_angle_curvature : if True, distribute the ray_angle_deg beam angulation by local
        arch curvature (max where the arch is flat = premolars/molars) instead of a linear
        arc-position ramp, to open interproximal contacts where teeth overlap most, without
        changing tooth orientation. Off by default (subtle on the test data).
    adaptive_focal : if True (default), build a smooth axially-varying focal trough from the
        anatomy (compute_focal_offset_surface) so the focal layer tracks the dentition —
        crowns AND the buccally/lingually flaring roots — at every height, sharpening
        molar roots and posterior teeth a fixed-depth trough leaves off-plane. Adds one
        probe pass (~half a render); set False for speed.
    tone / strength / gamma : tone mapping of the line integral. Default tone='raysum'
        (linear) keeps dense structures separable. tone='beer_lambert' is the physical
        1-exp(-strength*L/L99) law; its FORM is correct but strength=4 with L99
        normalisation saturates the whole dense half of the range into near-white (teeth
        + cortical bone merge), which is why raysum is preferred here — drop strength to
        ~1.5 if the physical model is wanted. gamma>1 darkens mid-tones.
    source_traj : virtual source-trajectory ray geometry in [0,1] (see synthesize_panoramic).
        0 = rays ⟂ to the arch (default); 1 = rays follow the beam from an elliptical
        moving source (radial through the arch centroid), oblique to the arch off-axis
        like a real panoramic. Experimental — evaluate per case.
    smile_frac : parabolic occlusal "smile" correction, as a fraction of the render
        height. The flat unwrapping otherwise drops the posterior below the anterior (a
        "sad" frown); this lifts the arch ends into the upward smile line of a real
        panoramic. 0.10 by default; 0 disables.
    denoise : edge-preserving median denoise of the render volume (novel #1) so
        accumulated CBCT noise does not integrate into wispy bone. Detection always
        uses the raw volume. True by default.
    transfer : use the tissue-aware HU->attenuation transfer curve (novel #2,
        DEFAULT_TRANSFER_*) instead of the single clipped bone window — gives pulp /
        soft tissue a faint floor and keeps enamel separable from metal. True by default.
    column_correct / base_fog / noise_std / blur_v / blur_h : detector-realism knobs
        forwarded to synthesize_panoramic (Kwon column-intensity flattening to reduce
        vertical banding; gray background fog; faint detector noise; focal-spot / detector
        PSF blur). column_correct on and base_fog=0.06 by default; blur / noise off to
        keep structure differentiation.
    display_height : output is cubically resampled to this many rows (None = native)

    Returns
    -------
    px  : uint8 (H, N) panoramic
    roi : the ROI dict
    tck : the arch spline
    """
    meip = find_MeIPs(volume, axis='coronal', show=False)
    roi = find_coronal_roi(meip, volume=volume, z_spacing_mm=z_spacing_mm, show=show)
    footprint_meip, arch_mask2d = find_arch_footprint(volume, roi, show=show)
    arch_mask2d = _smooth_arch_adaptive(arch_mask2d)
    pe = posterior_extend_mm / z_spacing_mm
    tck, _Td, _region = find_dental_arch(arch_mask2d, posterior_extend=pe,
                                         background=footprint_meip, show=show)
    # Substitute an externally supplied arch spline (e.g. a deliberately truncated /
    # translated / misaligned one) so every downstream stage — trajectory, adaptive
    # focal surface and ray cast — uses it consistently.
    if tck_override is not None:
        tck = tck_override

    _, _, _, arc_length = compute_panoramic_trajectory(tck, 2000)
    Td = trough_depth_mm / z_spacing_mm        # fixed-depth focal trough (mm-based)
    n_columns = max(200, int(round(arc_length)))
    top_margin = int(round(sup_margin_mm / z_spacing_mm))
    bottom_margin = int(round(inf_margin_mm / z_spacing_mm))

    # Render on an edge-preserving denoised copy (novel #1) so accumulated CBCT
    # noise does not integrate into wispy bone; detection above used the raw
    # volume. transfer=True selects the tissue-aware HU->attenuation curve (#2).
    render_vol = denoise_volume(volume) if denoise else volume
    tf = (DEFAULT_TRANSFER_XP, DEFAULT_TRANSFER_FP) if transfer else None

    focal_offset = None
    if adaptive_focal:
        # Smooth axially-varying focal trough: derive a per-(row, column) focal-centre
        # surface from the anatomy and pass it to the caster (see
        # compute_focal_offset_surface). Uses the same trajectory / z_range the caster
        # rebuilds internally, so the column indexing matches.
        pts, nrm, _u, _ = compute_panoramic_trajectory(tck, n_columns)
        z_range = (roi['z_top'] - top_margin, roi['z_bottom'] + bottom_margin)
        focal_offset, _raw, _conf = compute_focal_offset_surface(
            render_vol, pts, nrm, z_range, Td)

    # Smile-line correction: the flat unwrapping drops the posterior occlusal below the
    # anterior (a "sad" frown); a parabolic vertical lift of the arch ends restores the
    # gentle upward "smile" expected of a real panoramic. Sized as a fraction of the
    # native render height so it is resolution-independent.
    native_H = max(1, (roi['z_bottom'] + bottom_margin) - max(0, roi['z_top'] - top_margin) + 1)
    smile_amount = int(round(smile_frac * native_H))

    px = synthesize_panoramic(
        render_vol, tck, Td, roi,
        n_columns=n_columns,
        n_depth=int(round(n_depth * render_scale)),   # keep sample density over the wider extent
        top_margin=top_margin, bottom_margin=bottom_margin,
        tone=tone, trough_scale=1.0,
        anterior_boost=anterior_boost, anterior_sigma=anterior_sigma,
        posterior_boost=posterior_boost, posterior_sigma=posterior_sigma,
        focal_sigma_frac=focal_sigma_frac, render_scale=render_scale,
        focal_floor=focal_floor, soft_tissue_gain=soft_tissue_gain,
        ray_angle_deg=ray_angle_deg, ray_angle_curvature=ray_angle_curvature,
        depth_tilt=depth_tilt, root_focus=root_focus,
        maxilla_post_bias=maxilla_post_bias,
        focal_offset=focal_offset, source_traj=source_traj,
        strength=strength, gamma=gamma,
        column_correct=column_correct, base_fog=base_fog,
        noise_std=noise_std, blur_v=blur_v, blur_h=blur_h,
        transfer=tf,
        smile_amount=smile_amount)

    if display_height and px.shape[0] != display_height:
        z = display_height / px.shape[0]
        px = np.clip(scipy.ndimage.zoom(px.astype(np.float32), (z, z), order=3),
                     0, 255).astype(np.uint8)
    if clahe:
        # Local contrast equalisation: spreads each tooth's density range so the
        # enamel / dentin / pulp gradient and the root/bone trabeculae become visible,
        # while flattening the harsh global black-to-white contrast of the raw raysum.
        px = enhance_panoramic(px, clahe=True)
    return px, roi, tck


def save_focal_overlay(volume, z_spacing_mm, roi, tck, out_path,
                       trough_depth_mm=14.0, render_scale=3.0,
                       sup_margin_mm=38.0, inf_margin_mm=16.0):
    """
    Static diagnostic PNG: the focal core (Td/2) and the full integration extent
    (render_scale·Td/2) drawn on axial slices at the maxillary occlusal plane, the
    inter-arch gap and the mandibular occlusal plane, plus the render z-window on the
    coronal MeIP. Reveals whether a tooth that renders faint is outside the sharp focal
    core (in-plane) or outside the vertical render window.
    """
    Td = trough_depth_mm / z_spacing_mm
    Z, H, W = volume.shape
    u = np.linspace(0.0, 1.0, 3000)
    xs, ys = scipy.interpolate.splev(u, tck)
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    spl = np.zeros((H, W), bool)
    xi = np.round(xs).astype(int); yi = np.round(ys).astype(int)
    ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    spl[yi[ok], xi[ok]] = True
    dist = np.asarray(scipy.ndimage.distance_transform_edt(~spl))
    core = dist <= Td / 2.0
    extent = dist <= render_scale * Td / 2.0

    meip = find_MeIPs(volume, axis='coronal', show=False)
    rtop = max(0, roi['z_top'] - int(round(sup_margin_mm / z_spacing_mm)))
    rbot = min(Z - 1, roi['z_bottom'] + int(round(inf_margin_mm / z_spacing_mm)))
    levels = [('maxilla occ', roi['z_maxilla_peak']), ('inter-arch gap', roi['z_gap']),
              ('mandible occ', roi['z_mandible_peak'])]

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    ax = axs[0, 0]; ax.imshow(meip, cmap='gray', aspect='auto')
    for zz, c, l in [(rtop, 'cyan', 'render top'), (roi['z_maxilla_peak'], 'yellow', 'maxilla occ'),
                     (roi['z_gap'], 'red', 'gap'), (roi['z_mandible_peak'], 'orange', 'mandible occ'),
                     (rbot, 'cyan', 'render bot')]:
        ax.axhline(zz, color=c, lw=1, label=l)
    ax.legend(fontsize=6, loc='upper right')
    ax.set_title('coronal MeIP + render z-window', fontsize=9); ax.axis('off')
    for ax, (name, zz) in zip([axs[0, 1], axs[1, 0], axs[1, 1]], levels):
        zz = int(np.clip(zz, 0, Z - 1))
        ax.imshow(volume[zz], cmap='gray', vmin=-200, vmax=2200, aspect='auto')
        ov = np.zeros((H, W, 4), np.float32)
        ov[extent & ~core] = [0, 1, 1, 0.30]
        ov[core] = [1, 0.5, 0, 0.5]
        ax.imshow(ov, aspect='auto')
        ax.plot(xs, ys, 'b-', lw=0.7)
        ax.set_title(f'axial z={zz} ({name})  orange=focal core, cyan=integration extent', fontsize=8)
        ax.axis('off')
    fig.suptitle(os.path.basename(out_path), fontsize=9)
    plt.tight_layout(); plt.savefig(out_path, dpi=110); plt.close()


def batch_process(input_dir, output_dir, pattern='*.nii', min_size_kb=0,
                  flip=False, overlay=True, **synth_kwargs):
    """
    Render a panoramic (and optional focal-overlay) for every scan in input_dir → output_dir.

    Walks input_dir for files matching `pattern` (and larger than min_size_kb), writes
    `<stem>_PX.png` (and `<stem>_overlay.png` when overlay=True) to output_dir. Resumable
    (skips files whose _PX.png already exists) and robust (a per-file error is logged and
    the batch continues). Extra keyword args are forwarded to synthesize_panoramic_from_volume.

    Returns (n_ok, n_fail, n_skip).
    """
    import glob
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(f for f in glob.glob(os.path.join(input_dir, pattern))
                   if os.path.getsize(f) > min_size_kb * 1024)
    print(f"batch: {len(files)} file(s) {input_dir} -> {output_dir}", flush=True)
    n_ok = n_fail = n_skip = 0
    for i, f in enumerate(files, 1):
        stem = os.path.splitext(os.path.basename(f))[0]
        if os.path.exists(os.path.join(output_dir, stem + "_PX.png")):
            n_skip += 1
            continue
        try:
            img = sitk.ReadImage(f)
            z = img.GetSpacing()[2]
            vol = sitk.GetArrayFromImage(img)
            if flip:
                vol = flip_volume_sagittal(vol)
            px, roi, tck = synthesize_panoramic_from_volume(vol, z, **synth_kwargs)
            plt.imsave(os.path.join(output_dir, stem + "_PX.png"), px, cmap='gray', vmin=0, vmax=255)
            if overlay:
                save_focal_overlay(vol, z, roi, tck, os.path.join(output_dir, stem + "_overlay.png"))
            n_ok += 1
            print(f"[{i}/{len(files)}] OK {stem}", flush=True)
        except Exception as e:
            n_fail += 1
            print(f"[{i}/{len(files)}] FAIL {stem}: {repr(e)[:140]}", flush=True)
    print(f"batch done: ok={n_ok} fail={n_fail} skip={n_skip}", flush=True)
    return n_ok, n_fail, n_skip


# --- Execution / verification ---
def _run_pipeline(nii_path=test_mha_file, show_steps=False):
    """
    Load → detect arch + focal trough. Returns (volume, tck, Td, roi, arch_region,
    footprint_meip). Detection-step plots are off by default so the pipeline runs
    straight through to synthesis; pass show_steps=True to inspect each stage.
    """
    # Comment the first line if you are running the mha file.
    # volume = flip_volume_sagittal(load_np_mha(nii_path))
    volume = load_np_mha(nii_path)
    z_spacing_mm = sitk.ReadImage(nii_path).GetSpacing()[2]   # axial slice thickness (mm)
    meip_cor = find_MeIPs(volume, axis='coronal', show=False)
    # Pass the volume so the FOV- and bite-robust enamel-band ROI is used.
    roi = find_coronal_roi(meip_cor, volume=volume, z_spacing_mm=z_spacing_mm, show=show_steps)
    footprint_meip, arch_mask2d = find_arch_footprint(volume, roi, show=show_steps)
    arch_mask2d_smooth = smooth_arch_footprint(arch_mask2d, background=footprint_meip, show=show_steps)
    tck, Td, arch_region = find_dental_arch(arch_mask2d_smooth, background=footprint_meip, show=show_steps)
    return volume, tck, Td, roi, arch_region, footprint_meip


# ============================================================================
# Manual arch-spline input
# ============================================================================
# Drive the panoramic from a hand-drawn arch spline instead of the automatic
# footprint -> skeleton -> smoothing detection.  The manual control points ARE
# the arch skeleton, so with `automatic=False` the whole detection chain
# (find_arch_footprint / smooth_arch_footprint / find_dental_arch) is skipped
# and replaced by a Catmull-Rom interpolating spline through the points -- the
# same curve type 3D Slicer draws through markup points -- fed to the existing
# render tail unchanged.  None of the functions above are modified.


def _catmull_rom_curve(points, n_samples=1000):
    """
    Uniform Catmull-Rom interpolating spline through `points` (N, 2): returns a
    dense (M, 2) curve passing exactly through every control point, in order.

    This is the curve type 3D Slicer uses for markup curves, so a spline drawn
    in Slicer is reproduced faithfully -- unlike a single global least-squares
    fit, which oscillates between sparse / sharp-turning points. The ends are
    handled by duplicating the first and last point (standard boundary handling)
    so the endpoint segments are well defined.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return pts.copy()
    padded = np.vstack([pts[0], pts, pts[-1]])            # N + 2 points
    n_seg = n - 1
    per = max(2, n_samples // n_seg)
    out = []
    for i in range(n_seg):
        p0, p1, p2, p3 = padded[i], padded[i + 1], padded[i + 2], padded[i + 3]
        for t in np.linspace(0.0, 1.0, per, endpoint=(i == n_seg - 1)):
            t2 = t * t
            t3 = t2 * t
            out.append(0.5 * ((2 * p1)
                              + (-p0 + p2) * t
                              + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.asarray(out)


def load_arch_spline_csv(path, with_z=False):
    """
    Load manual arch control points from a Slicer .fcsv or a plain .csv.

    * .fcsv  -- lines starting with '#' are skipped; the id,x,y,z,... data rows
                are parsed and columns x, y, z (indices 1, 2, 3) are taken.
    * .csv   -- rows of x,y or x,y,z (';' or ',' separated); a non-numeric header
                row is skipped automatically. Columns x, y (indices 0, 1) are
                taken, and z (index 2) if present.

    The file type is auto-detected: an fcsv header comment or rows with the full
    fcsv column count select the fcsv column layout, otherwise the plain layout.

    Parameters
    ----------
    with_z : if True, return (N, 3) points including the z coordinate (needed for
             the origin/direction-aware physical->voxel transform in
             manual_arch_tck). If False (default), return (N, 2) x,y only. A plain
             csv with no z column defaults its z to 0.0 when with_z=True.

    Returns
    -------
    points : (N, 2) or (N, 3) float -- control points in their source coordinate
             frame, in drawing order.
    """
    is_fcsv = False
    data_rows = []
    with open(path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith('#'):
                if 'Markups' in s or 'columns = id' in s:
                    is_fcsv = True
                continue
            data_rows.append(s)

    pts = []
    for s in data_rows:
        parts = s.replace(';', ',').split(',')
        try:
            if is_fcsv or len(parts) >= 14:
                x, y = float(parts[1]), float(parts[2])       # fcsv: id,x,y,z,...
                z = float(parts[3])
            else:
                x, y = float(parts[0]), float(parts[1])       # plain csv: x,y[,z]
                z = float(parts[2]) if len(parts) > 2 else 0.0
        except (ValueError, IndexError):
            continue                                           # header / malformed row
        pts.append((x, y, z))

    if len(pts) < 2:
        raise ValueError(f"{path}: found {len(pts)} usable point(s); need at least 2.")
    pts = np.asarray(pts, dtype=float)
    return pts if with_z else pts[:, :2]


def manual_arch_tck(points, coords='pixel', in_plane_spacing_mm=1.0, image=None,
                    flip=False, y_size=None, n_dense=2000, spline_smooth=0.5):
    """
    Build a pipeline-compatible arch spline (scipy `tck`) from manual control points.

    The points are interpolated with a Catmull-Rom curve (Slicer's markup-curve
    type, see _catmull_rom_curve), then a smoothing cubic B-spline is fit through
    the dense curve so the result plugs into compute_panoramic_trajectory /
    cast_panoramic_rays exactly like an automatically detected arch spline.

    Parameters
    ----------
    points              : (N, 2) or (N, 3) manual control points from
                          load_arch_spline_csv. (N, 3) LPS points are required for
                          the image-based transform below.
    coords              : 'pixel'       -- points are already axial voxel indices
                          (x = column, y = row), used directly.
                          'physical_mm' -- points are Slicer LPS mm; converted to
                          voxel indices (see `image`).
    image               : a SimpleITK image (the CBCT the spline was drawn on). When
                          given with coords='physical_mm', each LPS point is mapped to
                          voxel indices with image.TransformPhysicalPointToContinuousIndex,
                          which uses the image's ORIGIN, SPACING and DIRECTION and so
                          handles a non-zero (e.g. centred) origin and oblique
                          acquisitions correctly. Requires (N, 3) points. This is the
                          robust path and the recommended one for Slicer .fcsv files.
                          When image is None, coords='physical_mm' falls back to the
                          simple divide-by-spacing shortcut (pts / in_plane_spacing_mm),
                          which assumes an identity direction and ZERO origin -- valid
                          only for zero-origin isotropic volumes.
    in_plane_spacing_mm : axial (x, y) voxel spacing in mm; only used for the
                          shortcut (coords='physical_mm' and image is None).
    flip                : set True when the spline was drawn on the ORIGINAL volume
                          but the renderer is fed a flip_volume_sagittal'd volume
                          (the FLIP=True case for the .nii / .nrrd scans).
                          flip_volume_sagittal mirrors the Z and Y axes, so in the
                          axial (Y, X) plane the row coordinate is mirrored while
                          the column is unchanged. The mirror is applied in voxel
                          space, AFTER the mm->voxel conversion: y -> (y_size-1) - y.
    y_size              : the volume's Y (row) dimension in voxels (volume.shape[1]);
                          required when flip=True, since the mirror needs the axis
                          extent. Ignored when flip=False.
    n_dense             : samples on the Catmull-Rom curve before the B-spline fit
    spline_smooth       : B-spline smoothing budget per point (px^2); small keeps
                          the fit faithful to the hand-drawn curve.

    Returns
    -------
    tck : scipy spline tuple (evaluate with scipy.interpolate.splev)
    """
    pts = np.asarray(points, dtype=float)
    if coords == 'physical_mm':
        if image is not None:
            if pts.shape[1] < 3:
                raise ValueError(
                    "coords='physical_mm' with image= needs (N, 3) LPS points; load them "
                    "with load_arch_spline_csv(path, with_z=True).")
            idx = np.array([image.TransformPhysicalPointToContinuousIndex(
                (float(p[0]), float(p[1]), float(p[2]))) for p in pts])
            pts = idx[:, :2]                       # (x = column i, y = row j)
        else:
            pts = pts[:, :2] / float(in_plane_spacing_mm)
    elif coords == 'pixel':
        pts = pts[:, :2]
    else:
        raise ValueError("coords must be 'pixel' or 'physical_mm'")

    if flip:
        if y_size is None:
            raise ValueError("flip=True requires y_size (the volume's Y/row dimension, "
                             "i.e. volume.shape[1]).")
        pts = pts.copy()
        pts[:, 1] = (y_size - 1) - pts[:, 1]   # mirror rows to match flip_volume_sagittal

    dense = _catmull_rom_curve(pts, n_dense)
    tck, _ = scipy.interpolate.splprep([dense[:, 0], dense[:, 1]], k=3,
                                       s=len(dense) * spline_smooth)
    return tck


def synthesize_panoramic_from_volume_manual(volume, z_spacing_mm, tck,
                                             n_depth=200, trough_depth_mm=14.0,
                                             sup_margin_mm=38.0, inf_margin_mm=16.0,
                                             anterior_boost=1.0, anterior_sigma=0.28,
                                             posterior_boost=1.0, posterior_sigma=0.20,
                                             render_scale=3.0, focal_floor=0.12,
                                             focal_sigma_frac=0.5, soft_tissue_gain=0.25,
                                             ray_angle_deg=6.0, ray_angle_curvature=False,
                                             depth_tilt=0.0, adaptive_focal=True, root_focus=0.5,
                                             maxilla_post_bias=0.0,
                                             source_traj=0.0, smile_frac=0.10,
                                             tone='raysum', strength=4.0, gamma=1.0, clahe=True,
                                             denoise=True, transfer=True,
                                             column_correct=True, base_fog=0.06,
                                             noise_std=0.0, blur_v=0.0, blur_h=0.0,
                                             display_height=560, show=False):
    """
    Manual-arch counterpart of synthesize_panoramic_from_volume.

    Identical rendering pipeline, but the arch spline is supplied (`tck`, e.g. from
    manual_arch_tck) instead of detected: find_arch_footprint / smooth_arch_footprint
    / find_dental_arch are skipped entirely. Only the coronal jaw ROI (the render
    z-window) is still found automatically with find_coronal_roi, since the manual
    input specifies only the in-plane arch, not the vertical extent.

    All render parameters have the same meaning and defaults as
    synthesize_panoramic_from_volume; see that function's docstring.

    Returns
    -------
    px  : uint8 (H, N) panoramic
    roi : the ROI dict
    tck : the (manual) arch spline
    """
    meip = find_MeIPs(volume, axis='coronal', show=False)
    roi = find_coronal_roi(meip, volume=volume, z_spacing_mm=z_spacing_mm, show=show)
    # Manual arch: `tck` already defines the arch skeleton, so every automatic
    # footprint / skeleton / smoothing stage is skipped. Downstream stages
    # (trajectory, adaptive focal surface, ray cast) use it exactly as they would
    # an automatically detected spline.

    _, _, _, arc_length = compute_panoramic_trajectory(tck, 2000)
    Td = trough_depth_mm / z_spacing_mm        # fixed-depth focal trough (mm-based)
    n_columns = max(200, int(round(arc_length)))
    top_margin = int(round(sup_margin_mm / z_spacing_mm))
    bottom_margin = int(round(inf_margin_mm / z_spacing_mm))

    render_vol = denoise_volume(volume) if denoise else volume
    tf = (DEFAULT_TRANSFER_XP, DEFAULT_TRANSFER_FP) if transfer else None

    focal_offset = None
    if adaptive_focal:
        pts, nrm, _u, _ = compute_panoramic_trajectory(tck, n_columns)
        z_range = (roi['z_top'] - top_margin, roi['z_bottom'] + bottom_margin)
        focal_offset, _raw, _conf = compute_focal_offset_surface(
            render_vol, pts, nrm, z_range, Td)

    native_H = max(1, (roi['z_bottom'] + bottom_margin) - max(0, roi['z_top'] - top_margin) + 1)
    smile_amount = int(round(smile_frac * native_H))

    px = synthesize_panoramic(
        render_vol, tck, Td, roi,
        n_columns=n_columns,
        n_depth=int(round(n_depth * render_scale)),
        top_margin=top_margin, bottom_margin=bottom_margin,
        tone=tone, trough_scale=1.0,
        anterior_boost=anterior_boost, anterior_sigma=anterior_sigma,
        posterior_boost=posterior_boost, posterior_sigma=posterior_sigma,
        focal_sigma_frac=focal_sigma_frac, render_scale=render_scale,
        focal_floor=focal_floor, soft_tissue_gain=soft_tissue_gain,
        ray_angle_deg=ray_angle_deg, ray_angle_curvature=ray_angle_curvature,
        depth_tilt=depth_tilt, root_focus=root_focus,
        maxilla_post_bias=maxilla_post_bias,
        focal_offset=focal_offset, source_traj=source_traj,
        strength=strength, gamma=gamma,
        column_correct=column_correct, base_fog=base_fog,
        noise_std=noise_std, blur_v=blur_v, blur_h=blur_h,
        transfer=tf,
        smile_amount=smile_amount)

    if display_height and px.shape[0] != display_height:
        z = display_height / px.shape[0]
        px = np.clip(scipy.ndimage.zoom(px.astype(np.float32), (z, z), order=3),
                     0, 255).astype(np.uint8)
    if clahe:
        px = enhance_panoramic(px, clahe=True)
    return px, roi, tck


def synthesize_panoramic_pipeline(volume, z_spacing_mm, automatic=True,
                                  spline_csv=None, spline_coords='pixel',
                                  in_plane_spacing_mm=None, image=None, flip=False,
                                  **kwargs):
    """
    Single entry point switching between the fully automatic pipeline and the
    manual-arch pipeline.

    automatic=True  -> synthesize_panoramic_from_volume (arch detected from the CBCT).
    automatic=False -> load the manual spline from `spline_csv`, build a Catmull-Rom
                       arch spline (manual_arch_tck) and render with
                       synthesize_panoramic_from_volume_manual (detection skipped).

    Parameters
    ----------
    spline_csv          : path to the manual .fcsv / .csv (required when automatic=False)
    spline_coords       : 'pixel' or 'physical_mm' -- coordinate frame of the CSV
                          points (see manual_arch_tck)
    image               : the SimpleITK image the spline was drawn on. Pass it when
                          spline_coords='physical_mm' so the LPS points are mapped to
                          voxels with the image's origin / spacing / direction (the
                          correct, assumption-free path -- required for scans with a
                          non-zero / centred origin, e.g. Slicer .fcsv with negative
                          coordinates). Without it, 'physical_mm' uses the
                          divide-by-spacing shortcut (zero-origin assumption).
    in_plane_spacing_mm : axial voxel spacing in mm for the shortcut ('physical_mm'
                          and image is None); defaults to z_spacing_mm.
    flip                : set True when `volume` has been flip_volume_sagittal'd
                          (FLIP=True) but the manual spline was drawn on the ORIGINAL
                          orientation, so the spline's rows are mirrored to match (see
                          manual_arch_tck). The volume's Y size is supplied automatically.
                          Note: pass the ORIGINAL (unflipped) `image` for the transform;
                          the flip is applied afterwards in voxel space.
    **kwargs            : forwarded to whichever render function is selected (both
                          share the same render parameters).

    Returns (px, roi, tck).
    """
    if automatic:
        return synthesize_panoramic_from_volume(volume, z_spacing_mm, **kwargs)

    if spline_csv is None:
        raise ValueError("automatic=False requires spline_csv (the manual arch spline).")
    points = load_arch_spline_csv(spline_csv, with_z=(image is not None))
    tck = manual_arch_tck(points, coords=spline_coords,
                          in_plane_spacing_mm=in_plane_spacing_mm or z_spacing_mm,
                          image=image, flip=flip, y_size=volume.shape[1])

    # Sanity check: a spline that lands mostly outside the axial (Y, X) plane means
    # the coordinate frame is wrong (physical mm read as pixels, a non-zero origin
    # with no `image=` given, or the wrong volume) -- which renders as a blank/black
    # panoramic. Fail loudly with the likely cause instead.
    xs, ys = scipy.interpolate.splev(np.linspace(0.0, 1.0, 200), tck)
    inside = ((xs >= 0) & (xs < volume.shape[2]) & (ys >= 0) & (ys < volume.shape[1])).mean()
    if inside < 0.5:
        raise ValueError(
            f"Manual spline lands mostly outside the volume "
            f"(only {inside*100:.0f}% of it is within the {volume.shape[2]}x{volume.shape[1]} "
            f"axial plane; x={xs.min():.0f}..{xs.max():.0f}, y={ys.min():.0f}..{ys.max():.0f}). "
            f"Check spline_coords (is the CSV in mm or pixels?), pass image= for a non-zero "
            f"origin, and confirm the CSV matches this volume.")

    return synthesize_panoramic_from_volume_manual(volume, z_spacing_mm, tck, **kwargs)


# ============================================================================
# Deep-learning arch-spline input (alternative to the geometric detection)
# ============================================================================
# Predict the dental-arch spline with the HeatmapNet model shipped in
# 'autospline/Deep_learning model' instead of detecting it geometrically
# (find_arch_footprint -> smooth_arch_footprint -> find_dental_arch).  The model
# outputs arch control points that ARE the arch skeleton, so - exactly like the
# manual-spline path - they are turned into a Catmull-Rom / B-spline `tck`
# (manual_arch_tck) and fed to synthesize_panoramic_from_volume_manual
# unchanged.  The DL model files are used as a library and left unmodified.

ML_DL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "autospline", "Deep_learning model")
ML_CHECKPOINT = os.path.join(ML_DL_DIR, "final_model.pt")
_ML_CACHE = {}


def _load_ml_model(dl_dir, checkpoint):
    """Lazily load and cache (drr-pipeline module, model, device, N_CONTROL).
    torch and the DL package are imported only here, so the geometric / manual
    paths never require them."""
    import sys
    key = (dl_dir, checkpoint)
    if key not in _ML_CACHE:
        if dl_dir not in sys.path:
            sys.path.insert(0, dl_dir)
        import torch
        from prepare_case import load_pipeline_module, N_CONTROL
        from architectures import HeatmapNetArch
        mod = load_pipeline_module(os.path.join(dl_dir, "drr_pipeline_v4.py"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HeatmapNetArch(n_control_points=N_CONTROL, use_pretrained=False)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.to(device).eval()
        _ML_CACHE[key] = (mod, model, device, N_CONTROL)
    return _ML_CACHE[key]


def _smooth_ml_control_points(cp, sigma=1.5):
    """
    De-kink the ML arch control points before the spline fit (toggled by
    ml_arch_tck(smooth=True)).

    The HeatmapNet control points occasionally jut out at a single point; because
    the downstream Catmull-Rom passes exactly through every control point, that
    kink becomes a sharp curvature spike, and the perpendicular ray casting turns
    it into a column of distortion in the panoramic.  Gaussian-smoothing the
    ordered control points (per axis, endpoints preserved with mode='nearest')
    relaxes the kink while keeping the overall arch shape.  `sigma` is in
    control-point units; sigma <= 0 is a no-op.
    """
    cp = np.asarray(cp, dtype=float)
    if sigma <= 0:
        return cp
    out = cp.copy()
    out[:, 0] = scipy.ndimage.gaussian_filter1d(cp[:, 0], sigma, mode='nearest')
    out[:, 1] = scipy.ndimage.gaussian_filter1d(cp[:, 1], sigma, mode='nearest')
    return out


def ml_arch_tck(cbct_path, label_path, jaw='lower',
                dl_dir=ML_DL_DIR, checkpoint=ML_CHECKPOINT,
                hu_window=(-500.0, 2000.0), spline_smooth=0.5,
                smooth=False, smooth_sigma=1.5):
    """
    Deep-learning alternative to the geometric arch detection.

    Runs the HeatmapNet arch-spline model (Payer et al., MICCAI 2016) from the
    'Deep_learning model' folder on a CBCT + tooth/bone label pair and returns a
    pipeline-compatible arch spline (scipy `tck`, in axial voxel coordinates)
    that plugs into synthesize_panoramic_from_volume_manual exactly like a
    hand-drawn spline.

    Windowing fix (required): the model was trained on HU-windowed [0,1] MIPs,
    but prepare_for_inference builds the MIP channel in raw HU (~ -1000..3200).
    Feeding raw HU is out-of-distribution and makes the U-Net emit confident-
    but-wrong heatmaps (control points scatter off the canvas).  The MIP channel
    is therefore clipped to `hu_window` and rescaled to [0,1] before inference,
    which restores correct predictions.  The shipped DL files are untouched --
    the fix lives here in the integration glue.

    Parameters
    ----------
    cbct_path, label_path : CBCT and matching tooth/bone label (SimpleITK-readable).
                            In ToothFairy2 the CBCT is the *_0000.mha file and the
                            label is the same id without the _0000 suffix.
    jaw           : 'lower' (default; reliable, 155 training cases) or 'upper'
                    (6 cases, preliminary).  The predicted arch is used as the
                    panoramic trajectory.
    hu_window     : (lo, hi) HU window mapped to [0,1] for the MIP channel.
    spline_smooth : B-spline smoothing budget per point (px^2) for manual_arch_tck.
    smooth        : if True, de-kink the predicted control points before the
                    spline fit (see _smooth_ml_control_points) to remove sharp
                    turns that cast a distorted column into the panoramic.
                    False (default) leaves the raw prediction unchanged.
    smooth_sigma  : Gaussian sigma (control-point units) used when smooth=True.

    Returns
    -------
    tck : scipy spline tuple (evaluate with scipy.interpolate.splev), in the same
          axial voxel (x=column, y=row) frame the renderer expects -- no volume
          flip is applied, matching the ToothFairy2 .mha convention.
    """
    import torch
    mod, model, device, _n = _load_ml_model(dl_dir, checkpoint)

    # prepare: jaw-cropped MIP, label MIP, geometric-arch channel + geo control pts
    from prepare_case import prepare_for_inference
    r = prepare_for_inference(mod, cbct_path, label_path, jaw)

    # --- windowing fix: raw-HU MIP -> [0,1], the distribution the model expects ---
    lo, hi = hu_window
    mip_win = ((np.clip(r["mip"], lo, hi) - lo) / (hi - lo)).astype(np.float32)

    stack = np.stack([mip_win, r["label_mip"], r["geo_channel"]], axis=0)
    img_t = torch.from_numpy(stack).float().unsqueeze(0).to(device)
    jaw_t = torch.tensor([0 if jaw == "lower" else 1], dtype=torch.long).to(device)
    geo_cp_t = torch.from_numpy(r["geo_cp"]).float().unsqueeze(0).to(device)
    with torch.no_grad():
        cp_canvas = model(img_t, jaw_t, geo_cp_t).cpu().numpy()[0]

    # canvas px -> full-volume voxel (x=col, y=row): reverse prepare's pad + scale
    cp_voxel = cp_canvas / r["scale"] - np.array([r["pad_left"], r["pad_top"]])

    # Optional de-kinking of the arch (toggle: smooth=True), applied after the
    # model prediction and before the spline fit.
    if smooth:
        cp_voxel = _smooth_ml_control_points(cp_voxel, smooth_sigma)

    # Same pixel-space Catmull-Rom -> smoothing B-spline used for manual splines.
    return manual_arch_tck(cp_voxel, coords='pixel', spline_smooth=spline_smooth)


if __name__ == "__main__":
    # =====================================================================
    #  CONFIG  --  edit everything here, then run:  python alter_version.py
    # ---------------------------------------------------------------------
    #  Pick a spline METHOD, set the paths, tune the knobs.  A full how-to
    #  for each method is in the USAGE notes at the very bottom of this file.
    # =====================================================================

    # --- which arch-spline source to use ---------------------------------
    #   'geometric' : detect the dental arch automatically from the CBCT
    #                 (no extra inputs -- the usual starting point)
    #   'manual'    : use a hand-drawn arch spline from a .fcsv / .csv file
    #   'ml'        : predict the arch with the deep-learning model
    #                 (needs PyTorch + a matching tooth/bone LABEL file)
    METHOD = 'geometric'

    # --- input / output --------------------------------------------------
    # One run handles any .mha / .nii / .nrrd CBCT: the slice spacing is read
    # automatically and the focal-trough depth, posterior extension and image
    # proportions all adapt to it (see synthesize_panoramic_from_volume).
    INPUT_FILE  = r"...\scan.nii"     # CBCT volume (.mha / .nii / .nrrd)
    OUTPUT_DIR  = r"...\output"       # folder for the result (created if missing)
    OUTPUT_NAME = "px_panoramic.png"  # output file name

    # Set FLIP = True when the mandible appears at the TOP of the volume (the
    # .nii exports from MCSTU usually need this; leave False for ToothFairy .mha).
    # switch. The 'ml' method is trained apex-up -- keep FLIP = False for it.
    # Flips the entire CBCT volume along the sagittal axis
    FLIP = False

    JAW = 'lower'                     # 'lower' or 'upper' (used by 'ml')

    # --- 'manual' method inputs (ignored by the other methods) -----------
    MANUAL_CSV    = r"...\arch.fcsv"  # hand-drawn arch markup (3D Slicer .fcsv / .csv)
    MANUAL_COORDS = 'physical_mm'     # 'physical_mm' (Slicer LPS) or 'pixel'

    # --- 'ml' method inputs (ignored by the other methods) ---------------
    LABEL_FILE    = r"...\scan_label.mha"  # matching tooth/bone segmentation label, the file WITHOUT 0000 in its name
    HU_WINDOW     = (-500.0, 2000.0)       # HU window mapped to [0,1] for the model
    SPLINE_SMOOTH = 0.5                    # arch B-spline smoothing budget (px^2)

    # --- render knobs (shared by every method) ---------------------------
    RENDER = dict(
        trough_depth_mm=14.0,      # bucco-lingual focal-trough depth (sharp layer)
        posterior_extend_mm=22.0,  # how far the arch runs past the last molar (mm)
        sup_margin_mm=38.0,        # render extent above the arch (mm)
        inf_margin_mm=16.0,        # render extent below the arch (mm)
        render_scale=3.0,          # >1 superimposes the mid-face beyond the trough
        tone='raysum',             # 'raysum' (linear) or 'beer_lambert'
        gamma=1.0,                 # output gamma
        clahe=True,                # local-contrast (CLAHE) enhancement
        display_height=560,        # output image height in px
        show=False,                # True pops up the intermediate matplotlib figures
    )
    # The deeper geometric thresholds (bone_min / bone_max / threshold_fraction,
    # enamel HU) are NOT here -- they live inside find_coronal_roi and
    # find_arch_footprint; edit them there only if the automatic arch detection
    # misses.  Every render parameter accepted by synthesize_panoramic_from_volume
    # can also be added to the RENDER dict above.
    # =====================================================================

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # The 'ml' method reads the CBCT + label itself and returns the arch spline.
    tck = None
    if METHOD == 'ml':
        tck = ml_arch_tck(INPUT_FILE, LABEL_FILE, jaw=JAW,
                          hu_window=HU_WINDOW, spline_smooth=SPLINE_SMOOTH)

    img = sitk.ReadImage(INPUT_FILE)
    z_spacing_mm = img.GetSpacing()[2]         # axial slice thickness (mm)
    volume = sitk.GetArrayFromImage(img)
    if FLIP:
        volume = flip_volume_sagittal(volume)

    if METHOD == 'geometric':
        panoramic, roi, tck = synthesize_panoramic_from_volume(
            volume, z_spacing_mm, **RENDER)
    elif METHOD == 'manual':
        panoramic, roi, tck = synthesize_panoramic_pipeline(
            volume, z_spacing_mm, automatic=False,
            spline_csv=MANUAL_CSV, spline_coords=MANUAL_COORDS,
            image=img, flip=FLIP, **RENDER)
    elif METHOD == 'ml':
        panoramic, roi, tck = synthesize_panoramic_from_volume_manual(
            volume, z_spacing_mm, tck, **RENDER)
    else:
        raise ValueError(f"unknown METHOD {METHOD!r} (use 'geometric', 'manual' or 'ml')")

    out_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    plt.imsave(out_path, panoramic, cmap='gray', vmin=0, vmax=255)
    print("saved:", out_path)

    # =====================================================================
    #  USAGE  --  how to run each of the three arch-spline methods
    # =====================================================================
    #
    #  Edit the CONFIG block above, then run:  python alter_version.py
    #  It loads INPUT_FILE, builds the dental-arch spline with the chosen
    #  METHOD, renders the panoramic and writes it to OUTPUT_DIR/OUTPUT_NAME.
    #  All three methods share the same RENDER knobs and produce the same
    #  kind of image; they differ only in where the arch curve comes from.
    #
    #  1) GEOMETRIC   METHOD = 'geometric'   (fully automatic, no extra inputs)
    #     ---------------------------------------------------------------
    #     The arch is detected straight from the CBCT (enamel-band ROI ->
    #     footprint -> skeleton).  Just set INPUT_FILE, OUTPUT_DIR and FLIP.
    #     This is the recommended default and needs nothing beyond the scan.
    #
    #  2) MANUAL      METHOD = 'manual'      (you draw the arch)
    #     ---------------------------------------------------------------
    #     Trace the arch as markup points in 3D Slicer (or any tool) and
    #     export them to a .fcsv / .csv, then set:
    #        MANUAL_CSV     = that file
    #        MANUAL_COORDS  = 'physical_mm'  for Slicer .fcsv (LPS millimetres)
    #                         'pixel'        if the points are already in voxels
    #     A Catmull-Rom spline is fitted through the points and used as the
    #     arch; the automatic detection is skipped.  Keep FLIP consistent with
    #     the orientation the points were drawn in (see the flip note above).
    #
    #  3) PURE ML     METHOD = 'ml'          (deep-learning arch prediction)
    #     ---------------------------------------------------------------
    #     Requires PyTorch and the model shipped in
    #     'autospline/Deep_learning model' (final_model.pt), PLUS a matching
    #     tooth/bone segmentation LABEL for the scan (the model builds its
    #     input channels from it).  In ToothFairy2 the CBCT is the *_0000.mha
    #     file and the label is the same id without the _0000 suffix.  Set:
    #        INPUT_FILE  = the CBCT           LABEL_FILE = the label
    #        JAW         = 'lower' (reliable) or 'upper' (preliminary)
    #        HU_WINDOW / SPLINE_SMOOTH = model input window / arch smoothing
    #     The model is trained apex-up, so keep FLIP = False for this method.
    # =====================================================================
