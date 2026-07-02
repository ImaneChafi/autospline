import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

test_mha_link = r"C:\Users\cotyl\OneDrive\Desktop\Projet recalage 2D3D\test_mha_files\ToothFairy2F_002_0000.mha"

def show_array(image_array):
    """Visualizes 2D images given the image array"""
    plt.imshow(image_array, cmap='gray')
    plt.axis('off')
    plt.show()
def load_image(image_link):
    """Returns a numoy array with coordinates (z, y, x) given the link to an .mha or .nii file"""
    image = sitk.ReadImage(image_link)
    return sitk.GetArrayFromImage(image)
def load_MIP(image_array, axis='axial', low_axis_lim=None, up_axis_lim=None):
    """Returns the Maximum Intensity Projection (MIP) along a given axis when input with an image array object. Returns the axial MIP by default.
    If lower and upper ranges are given, the MIP will be limited to within that region of interest (both bounds inclusive)."""

    ax = {'axial':0, 'coronal':1, 'sagittal': 2}

    if axis in ax:
        axis = ax[axis]
    elif axis not in [0,1,2]:
        axis = 0

    if low_axis_lim is not None or up_axis_lim is not None:
        slicer = [slice(None)] * image_array.ndim
        slicer[axis] = slice(low_axis_lim, None if up_axis_lim is None else up_axis_lim + 1)
        image_array = image_array[tuple(slicer)]

    return np.max(image_array, axis=axis)
def load_MeIP(image_array, axis='axial', low_axis_lim=None, up_axis_lim=None):
    """Returns the Mean Intensity Projection (MeIP) along a given axis when input with an image array object. Returns the axial MeIP by default.
    If lower and upper ranges are given, the MeIP will be limited to within that region of interest (both bounds inclusive)."""

    ax = {'axial':0, 'coronal':1, 'sagittal': 2}

    if axis in ax:
        axis = ax[axis]
    elif axis not in [0,1,2]:
        axis = 0

    if low_axis_lim is not None or up_axis_lim is not None:
        slicer = [slice(None)] * image_array.ndim
        slicer[axis] = slice(low_axis_lim, None if up_axis_lim is None else up_axis_lim + 1)
        image_array = image_array[tuple(slicer)]

    return np.mean(image_array, axis=axis)
def clip_intensity(image_array, low_range=200, up_range=None):
    """Clips all voxels to fit between the input range. This function must be used BEFORE loading the MIP/MeIP
    Remove Soft Tissue: low_range = 200
    Clip Maximum Density to Bone: up_range = 800"""
    return (np.clip(image_array, low_range, up_range)-low_range).astype(np.float32)
def _gaussian(x, amp, mu, sigma):
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
def _fit_gaussian_to_peak(x, y, peak_idx):
    """Fits a 1D Gaussian to the local neighborhood of a peak in (x, y) data, bounding the fit
    window by the nearest local minima on either side of the peak. Returns (mu, sigma).
    Falls back to a FWHM-based sigma estimate if the curve fit fails to converge."""
    valley_idx, _ = find_peaks(-y)
    left = valley_idx[valley_idx < peak_idx]
    right = valley_idx[valley_idx > peak_idx]
    lo = left[-1] if len(left) else 0
    hi = right[0] if len(right) else len(y) - 1

    x_win, y_win = x[lo:hi + 1], y[lo:hi + 1]
    span = x_win[-1] - x_win[0] if len(x_win) > 1 else 1.0
    p0 = [y[peak_idx], x[peak_idx], max(span / 4, 1e-6)]
    try:
        popt, _ = curve_fit(_gaussian, x_win, y_win, p0=p0, maxfev=10000)
        return popt[1], abs(popt[2])
    except RuntimeError:
        widths, *_ = peak_widths(y, [peak_idx], rel_height=0.5)
        sigma = (widths[0] * (x[1] - x[0])) / 2.3548
        return x[peak_idx], sigma
def _fit_peak_threshold(values, peak_selector='max', bins=256, smoothing_sigma=2):
    """Builds an intensity histogram of values, smooths it slightly to suppress noise peaks, then
    picks either the peak with the largest ('max') or smallest ('min') gray value among the detected
    peaks and fits a Gaussian to it. Returns (T, mu, sigma, hist, centers) where T = mu + 1.98*sigma,
    following the threshold formula in Yun et al. (2019)."""
    hist, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    hist_smooth = gaussian_filter1d(hist.astype(float), sigma=smoothing_sigma)

    min_prominence = 0.05 * hist_smooth.max()
    peak_idx, _ = find_peaks(hist_smooth, prominence=min_prominence)
    if len(peak_idx) == 0:
        peak_idx = np.array([int(np.argmax(hist_smooth))])

    chosen = peak_idx[np.argmax(centers[peak_idx])] if peak_selector == 'max' else peak_idx[np.argmin(centers[peak_idx])]
    mu, sigma = _fit_gaussian_to_peak(centers, hist_smooth, chosen)
    T = mu + 1.98 * sigma
    return T, mu, sigma, hist, centers
def _detect_axial_range(row_counts, smoothing_sigma=2):
    """Detects the [As, Ae] axial slice index range from a row-wise pixel-count profile (counts per
    z-slice), following Yun et al. (2019)'s peak-counting logic: a single dominant peak (H==1, no
    occlusion fork) vs. two separated qualifying peaks (H>=2, occlusion fork / split jaw regions)."""
    z = np.arange(len(row_counts), dtype=float)
    counts_smooth = gaussian_filter1d(row_counts.astype(float), sigma=smoothing_sigma)

    min_prominence = 0.15 * counts_smooth.max()
    peaks, props = find_peaks(counts_smooth, height=0, prominence=min_prominence)
    if len(peaks) == 0:
        return 0, len(row_counts) - 1, {'peaks': np.array([]), 'qualifying_peaks': np.array([])}

    heights = props['peak_heights']
    highest_idx = peaks[np.argmax(heights)]
    highest_height = heights.max()

    qualifying = peaks[heights > 0.5 * highest_height]
    H = len(qualifying)

    if H == 1:
        Es = highest_idx
        _, sigma_s = _fit_gaussian_to_peak(z, counts_smooth, Es)
        As, Ae = Es - 1.5 * sigma_s, Es + 1.5 * sigma_s
    else:
        Eb, Et = qualifying.min(), qualifying.max()
        _, sigma_b = _fit_gaussian_to_peak(z, counts_smooth, Eb)
        _, sigma_t = _fit_gaussian_to_peak(z, counts_smooth, Et)
        As, Ae = Eb - 1.5 * sigma_b, Et + 1.5 * sigma_t

    As = int(np.clip(round(As), 0, len(row_counts) - 1))
    Ae = int(np.clip(round(Ae), 0, len(row_counts) - 1))
    return As, Ae, {'peaks': peaks, 'qualifying_peaks': qualifying}
def _plot_coronal_with_histogram(coronal_image, profile, As, Ae, info, title='', xlabel='Row pixel count'):
    """Plots the coronal MIP/MeIP image on the left, and its row-wise profile on the right
    (detected peaks in green, qualifying peaks in red), marking the detected axial slice range [As, Ae]
    on both."""
    _, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(10, 5))

    ax_img.imshow(coronal_image, cmap='gray', aspect='auto')
    ax_img.axhline(As, color='red', linestyle='--')
    ax_img.axhline(Ae, color='red', linestyle='--')
    ax_img.set_title(title)
    ax_img.axis('off')

    z = np.arange(len(profile))
    ax_hist.plot(profile, z)
    ax_hist.invert_yaxis()
    if len(info['peaks']):
        ax_hist.plot(profile[info['peaks']], info['peaks'], 'go', label='peaks')
    if len(info['qualifying_peaks']):
        ax_hist.plot(profile[info['qualifying_peaks']], info['qualifying_peaks'], 'r^', label='qualifying peaks')
    ax_hist.axhline(As, color='red', linestyle='--')
    ax_hist.axhline(Ae, color='red', linestyle='--')
    ax_hist.set_xlabel(xlabel)
    ax_hist.set_ylabel('Axial slice index (z)')
    ax_hist.set_title('Row-wise profile')
    ax_hist.legend()

    plt.tight_layout()
    plt.show()
def find_coronal_roi_enamel(image_array, bins=256, visualize=True):
    """Detects the [As, Ae] axial slice index range containing the teeth, by locating bright (enamel)
    peaks in the coronal MIP. Follows Yun et al. (2019): threshold the coronal MIP via a Gaussian fit
    on the histogram peak with the largest gray value (T = mu + 1.98*sigma), then apply the
    H==1/H>=2 peak-counting logic (see _detect_axial_range) on the per-row bright-pixel count.
    Returns (As, Ae), usable directly as low_axis_lim/up_axis_lim for load_MIP/load_MeIP."""
    coronal_mip = load_MIP(image_array, axis='coronal')
    T, _, _, _, _ = _fit_peak_threshold(coronal_mip.ravel(), peak_selector='max', bins=bins)
    mask = coronal_mip > T
    row_counts = mask.sum(axis=1)

    As, Ae, info = _detect_axial_range(row_counts)

    ### DELETE THIS SECTION IF IN ALREADY IMPLEMENTED INTO THE APP
    if visualize:
        _plot_coronal_with_histogram(coronal_mip, row_counts, As, Ae, info, title='Coronal MIP (enamel)')
    ###

    
    return As, Ae
def find_coronal_roi_air(image_array, visualize=True):
    """Detects the [As, Ae] axial slice index range containing the oral cavity air gap between the
    jaw bones, using the coronal MeIP. Finds the two dominant high-density peaks in the per-row
    body-masked mean intensity profile (Eb=maxilla, Et=mandible), then computes:
      v_between = deepest valley between the two peaks (center of the air cavity)
      v_below   = first valley caudal to the mandible peak (base of the mandible)
      d         = (v_below - v_between) / 2  (half the mandible span)
      As = Eb - d,  Ae = Et + d
    This pads both jaw bone peaks by d to include the alveolar bone thickness on each side.
    Returns (As, Ae), usable directly as low_axis_lim/up_axis_lim for load_MIP/load_MeIP."""
    coronal_meip = load_MeIP(image_array, axis='coronal')
    # coronal_meip > -500 excludes projection columns outside the patient's head
    body_mask = coronal_meip > -500
    counts_per_row = body_mask.sum(axis=1)
    profile = np.where(
        counts_per_row > 0,
        np.where(body_mask, coronal_meip, 0.0).sum(axis=1) / np.maximum(counts_per_row, 1),
        np.mean(coronal_meip, axis=1)
    )

    profile_smooth = gaussian_filter1d(profile.astype(float), sigma=2)
    min_prominence = 0.15 * profile_smooth.max()
    peaks, _ = find_peaks(profile_smooth, prominence=min_prominence)

    if len(peaks) < 2:
        As, Ae = 0, len(profile) - 1
        info = {'peaks': peaks, 'qualifying_peaks': np.array([])}
    else:
        # find the adjacent peak pair whose intervening valley is deepest (= the air cavity)
        best_pair, best_depth = (peaks[0], peaks[1]), -np.inf
        for i in range(len(peaks) - 1):
            valley_floor = np.min(profile_smooth[peaks[i]:peaks[i + 1] + 1])
            depth = min(profile_smooth[peaks[i]], profile_smooth[peaks[i + 1]]) - valley_floor
            if depth > best_depth:
                best_depth, best_pair = depth, (peaks[i], peaks[i + 1])
        Eb, Et = best_pair

        v_between = int(Eb + np.argmin(profile_smooth[Eb:Et + 1]))

        caudal_valleys, _ = find_peaks(-profile_smooth[Et:])
        v_below = int(Et + caudal_valleys[0]) if len(caudal_valleys) > 0 else len(profile) - 1

        d = (v_below - v_between)
        As = int(np.clip(round(Eb - d), 0, len(profile) - 1))
        Ae = int(np.clip(round(Et + d), 0, len(profile) - 1))
        info = {'peaks': peaks, 'qualifying_peaks': np.array([Eb, Et])}

    ### DELETE THIS SECTION IF IN THE APP
    if visualize:
        _plot_coronal_with_histogram(coronal_meip, profile, As, Ae, info,
                                     title='Coronal MeIP (air cavity)',
                                     xlabel='Per-row mean intensity')
    ###

    return As, Ae
def find_axial_enamel_limits(image_link, low_hu_lim=200, up_hu_lim=None, show=False):

    image = load_image(image_link)

    # no maximum value as we want the MIP
    image = clip_intensity(image, low_range=low_hu_lim, up_range=up_hu_lim)

    # determines the axial Z coordinates that limits the ROI
    lower_ROI_lim, upper_ROI_lim = find_coronal_roi_enamel(image, visualize=show)

    return lower_ROI_lim, upper_ROI_lim
def find_axial_air_limits(image_link, low_hu_lim=200, up_hu_lim=1200, show=False):
    """Returns the limits for the axial ROI using the MeIP. This method also targets the air within the oral cavity, thus appearing as a
    depression within the density histogram. We then select the beginning of both jaw bones based on the two peaks adjacent to the depression.
    Finally, we select the region of interest by estimting the approximate height of the mandible and adding half of it to both jaw bone initial peaks
    in order to target the majority of the alveolar bone."""

    image = load_image(image_link)

    # only allow values between 200 and 800 to make higher density objects similar to bone
    image = clip_intensity(image, low_range=low_hu_lim, up_range=None)

    # determines the axial Z coordinates that limits the ROI
    lower_ROI_lim, upper_ROI_lim = find_coronal_roi_air(image, visualize=show)

    return lower_ROI_lim, upper_ROI_lim


# in order to obtain the image of the ROI, run the following:

""" 1. choose whether you want to isolate the zone of interest based on:
 zones with highest density (enamel peaks) 
 or 
 if you want to localize relative to the oral cavity (low density from air, in case patient is edentulous, or has a lot of metal artefacts).

 The output will be the lower and upper limit for the axial slice"""
lower_limit, upper_limit = find_axial_air_limits(test_mha_link, show=False)
# lower_limit, upper_limit = find_axial_enamel_limits(test_mha_link, show=False)

"""2. load the 3D volume and you can select using which method the slice is generated
Maximum Intensity Projection or Mean Intensity Projection

MeIP is more useful if the palatal bone has been included in the upper limit, thus the MIP also renders
the bone all over the center. The MeIP will allow us to mitigate that effect"""
volume_array = load_image(test_mha_link)
_MIP = load_MIP(volume_array, axis='axial', low_axis_lim=lower_limit, up_axis_lim=upper_limit)
_MeIP = load_MeIP(volume_array, axis=0, low_axis_lim=lower_limit, up_axis_lim=upper_limit)

# the following simply visualizes the image
show_array(_MIP)


