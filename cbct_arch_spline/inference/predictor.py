"""
Model inference: CBCT slice → predicted arch control points.

Given a trained model and a CBCT axial slice, this module:
1. Runs the U-Net to produce a heatmap
2. Extracts keypoint peaks from the heatmap
3. Orders the peaks into a dental arch sequence
4. Returns both the raw heatmap and the ordered control points
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from scipy.ndimage import maximum_filter, gaussian_filter
from skimage.feature import peak_local_max

from config import (
    HEATMAP_THRESHOLD,
    MIN_PEAK_DISTANCE,
    MAX_KEYPOINTS,
    MIN_KEYPOINTS,
    IMAGE_SIZE,
    MODELS_DIR,
)
from data.preprocessing import resize_slice, points_resize_to_image
from spline.spline_utils import order_points_along_arch


def extract_keypoints_from_heatmap(
    heatmap: npt.NDArray[np.float32],
    threshold: float = HEATMAP_THRESHOLD,
    min_distance: int = MIN_PEAK_DISTANCE,
    max_peaks: int = MAX_KEYPOINTS,
) -> npt.NDArray[np.float64]:
    """
    Extract local maxima from a predicted heatmap.

    Returns (N, 2) array of (col, row) pixel coordinates.
    """
    # Smooth slightly to avoid noise spikes
    smoothed = gaussian_filter(heatmap, sigma=1.5)

    peaks = peak_local_max(
        smoothed,
        min_distance=min_distance,
        threshold_abs=threshold,
        num_peaks=max_peaks,
    )
    # peak_local_max returns (row, col) → convert to (col, row)
    if len(peaks) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return peaks[:, ::-1].astype(np.float64)  # (N, 2) as (col, row)


class ArchPredictor:
    """
    High-level predictor: handles model loading, preprocessing, and postprocessing.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "auto",
        image_size: int = IMAGE_SIZE,
    ):
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()
        self.image_size = image_size

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model: nn.Module,
        device: str = "auto",
    ) -> "ArchPredictor":
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model, device=device)

    def predict_from_slice(
        self,
        slice_2d: npt.NDArray[np.float32],
    ) -> dict:
        """
        Predict arch control points from a normalised 2D CBCT slice.

        Args:
            slice_2d: (H, W) float32 in [0, 1], already HU-windowed

        Returns dict with:
          - heatmap:       (H, W) predicted heatmap at model resolution
          - keypoints_px:  (N, 2) (col, row) in original slice pixel coords
          - n_keypoints:   int
        """
        original_hw = slice_2d.shape

        # Resize to model input
        slc_resized = resize_slice(slice_2d, self.image_size)

        # Prepare tensor
        tensor = torch.from_numpy(slc_resized[np.newaxis, np.newaxis]).float()
        tensor = tensor.to(self.device)

        with torch.no_grad():
            heatmap_tensor = self.model(tensor)  # (1, 1, H, W)

        heatmap = heatmap_tensor[0, 0].cpu().numpy()

        # Extract keypoints in model-resolution space
        kps_resized = extract_keypoints_from_heatmap(heatmap)

        # Map back to original slice resolution
        if len(kps_resized) > 0:
            kps_orig = points_resize_to_image(kps_resized, original_hw, self.image_size)
            kps_ordered = order_points_along_arch(kps_orig)
        else:
            kps_ordered = np.zeros((0, 2), dtype=np.float64)

        return {
            "heatmap": heatmap,
            "keypoints_px": kps_ordered,
            "n_keypoints": len(kps_ordered),
        }

    def predict_heatmap_only(
        self,
        slice_2d: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Return just the raw heatmap at model resolution (for overlaying on display)."""
        slc_resized = resize_slice(slice_2d, self.image_size)
        tensor = torch.from_numpy(slc_resized[np.newaxis, np.newaxis]).float().to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
        return out[0, 0].cpu().numpy()
