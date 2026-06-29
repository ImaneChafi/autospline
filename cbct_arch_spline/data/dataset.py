"""PyTorch dataset for CBCT arch spline detection."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter

from config import (
    ANNOTATIONS_DIR,
    DATA_DIR,
    HEATMAP_SIGMA,
    IMAGE_SIZE,
    HU_MIN,
    HU_MAX,
)
from spline.fcsv_io import load_fcsv, lps_to_voxel
from data.preprocessing import (
    load_volume,
    window_hu,
    extract_axial_slice,
    z_lps_to_voxel_index,
    resize_slice,
    slice_to_tensor_input,
    points_image_to_resize,
)


def make_gaussian_heatmap(
    points_px: npt.NDArray[np.float64],
    size: int,
    sigma: float = HEATMAP_SIGMA,
) -> npt.NDArray[np.float32]:
    """
    Create a (size x size) heatmap with a Gaussian blob at each keypoint.

    points_px: (N, 2) array of (col, row) pixel coordinates in [0, size).
    """
    heatmap = np.zeros((size, size), dtype=np.float32)
    for col, row in points_px:
        r, c = int(round(row)), int(round(col))
        if 0 <= r < size and 0 <= c < size:
            heatmap[r, c] = 1.0

    if sigma > 0:
        heatmap = gaussian_filter(heatmap, sigma=sigma)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

    return heatmap


class CBCTArchDataset(Dataset):
    """
    Dataset of (axial_slice, keypoint_heatmap) pairs.

    Expects:
      - cbct_dir: directory with NIfTI files named {case_id}.nii.gz
      - fcsv_dir: directory with .fcsv annotation files
    """

    def __init__(
        self,
        cbct_dir: str | Path,
        fcsv_dir: str | Path = ANNOTATIONS_DIR,
        augment: bool = False,
        image_size: int = IMAGE_SIZE,
    ):
        self.cbct_dir = Path(cbct_dir)
        self.fcsv_dir = Path(fcsv_dir)
        self.augment = augment
        self.image_size = image_size

        self.samples = self._find_paired_samples()
        if not self.samples:
            raise ValueError(
                f"No paired (NIfTI + fcsv) samples found.\n"
                f"  CBCT dir: {self.cbct_dir}\n"
                f"  FCSV dir: {self.fcsv_dir}\n"
                "Ensure NIfTI files follow the naming pattern {{case_id}}.nii.gz"
            )

    # Candidate volume extensions, in priority order
    VOLUME_EXTS = (".mha", ".mhd", ".nrrd", ".nii.gz", ".nii")

    def _find_volume_for_case(self, case_id: str) -> Path | None:
        """
        Locate the CBCT volume for a case id, searching the cbct_dir and a
        nested imagesTr/ subfolder (nnU-Net / ToothFairy2 layout).
        """
        search_dirs = [self.cbct_dir, self.cbct_dir / "imagesTr"]
        for d in search_dirs:
            for ext in self.VOLUME_EXTS:
                candidate = d / f"{case_id}{ext}"
                if candidate.exists():
                    return candidate
        return None

    def _find_paired_samples(self) -> list[dict]:
        samples = []
        for fcsv_path in sorted(self.fcsv_dir.glob("*.fcsv")):
            case_id = fcsv_path.stem
            nii_path = self._find_volume_for_case(case_id)
            if nii_path is not None:
                samples.append({"case_id": case_id, "nii": nii_path, "fcsv": fcsv_path})
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        annotation = load_fcsv(sample["fcsv"])
        volume, affine = load_volume(sample["nii"])
        volume = window_hu(volume)

        z_idx = z_lps_to_voxel_index(annotation["z_lps"], affine, volume.shape)
        slc = extract_axial_slice(volume, z_idx)  # (H, W)
        original_hw = slc.shape  # (H, W)

        slc_resized = resize_slice(slc, self.image_size)

        # Convert LPS points → voxel coords → pixel coords on axial slice
        pts_vox = lps_to_voxel(annotation["points_lps"], affine)  # (N, 3)
        # Axial slice: row = Y-voxel (axis 1), col = X-voxel (axis 0)
        # After .T in extract_axial_slice: row corresponds to voxel axis 1, col to axis 0
        pts_px = pts_vox[:, :2]  # (N, 2) as (col=X, row=Y) → need to check axis order

        # Scale to resized image
        pts_px_resized = points_image_to_resize(pts_px, original_hw, self.image_size)

        heatmap = make_gaussian_heatmap(pts_px_resized, self.image_size)

        if self.augment:
            slc_resized, heatmap = self._augment(slc_resized, heatmap)

        image_tensor = torch.from_numpy(slc_resized[np.newaxis]).float()  # (1, H, W)
        heatmap_tensor = torch.from_numpy(heatmap[np.newaxis]).float()   # (1, H, W)

        return {
            "image": image_tensor,
            "heatmap": heatmap_tensor,
            "case_id": sample["case_id"],
            "z_lps": annotation["z_lps"],
            "affine": torch.from_numpy(affine).float(),
            "original_hw": original_hw,
        }

    def _augment(
        self,
        image: npt.NDArray[np.float32],
        heatmap: npt.NDArray[np.float32],
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Simple augmentation: horizontal flip, brightness/contrast jitter."""
        import cv2

        # Horizontal flip (dental arches are symmetric)
        if np.random.rand() < 0.5:
            image = np.fliplr(image).copy()
            heatmap = np.fliplr(heatmap).copy()

        # Brightness jitter
        noise = np.random.uniform(-0.05, 0.05)
        image = np.clip(image + noise, 0, 1)

        # Contrast jitter
        factor = np.random.uniform(0.9, 1.1)
        image = np.clip((image - 0.5) * factor + 0.5, 0, 1)

        # Small random rotation (±10°)
        angle = np.random.uniform(-10, 10)
        h, w = image.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
        heatmap = cv2.warpAffine(heatmap, M, (w, h), flags=cv2.INTER_LINEAR)

        return image, heatmap
