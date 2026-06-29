"""CBCT volume loading and slice extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import nibabel as nib
import cv2

from config import HU_MIN, HU_MAX, IMAGE_SIZE


# File extensions handled by each backend
NIFTI_EXTS = (".nii", ".nii.gz")
ITK_EXTS = (".mha", ".mhd", ".nrrd")


def _has_ext(path: Path, exts: tuple[str, ...]) -> bool:
    name = path.name.lower()
    return any(name.endswith(e) for e in exts)


def load_volume(path: str | Path) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Load a CBCT volume from NIfTI (.nii/.nii.gz) or ITK (.mha/.mhd/.nrrd).

    Returns:
        volume: (X, Y, Z) float32 array, indexed as volume[i, j, k]
        affine: (4, 4) voxel->RAS affine matrix

    Both backends are normalised to the SAME conventions:
      - array axis order is (i, j, k) matching the affine's index columns
      - affine maps voxel index -> RAS world coordinates (mm)
    so that all downstream LPS<->voxel math in fcsv_io works identically.
    """
    path = Path(path)
    if _has_ext(path, NIFTI_EXTS):
        return load_nifti(path)
    if _has_ext(path, ITK_EXTS):
        return load_itk(path)
    raise ValueError(
        f"Unsupported file type: {path.name}\n"
        f"Supported: {NIFTI_EXTS + ITK_EXTS}"
    )


def load_nifti(path: str | Path) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Load a NIfTI file (nibabel, RAS+ affine).

    Returns:
        volume: (X, Y, Z) float32 array indexed as volume[i, j, k]
        affine: (4, 4) voxel->RAS affine matrix
    """
    img = nib.load(str(path))
    volume = np.asarray(img.dataobj, dtype=np.float32)
    affine = img.affine
    return volume, affine


def load_itk(path: str | Path) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Load an ITK volume (.mha/.mhd/.nrrd) via SimpleITK.

    SimpleITK works in LPS world coordinates and returns the array in
    (k, j, i) order. We transpose to (i, j, k) and convert the LPS geometry
    to a voxel->RAS affine, matching the NIfTI convention so the rest of the
    pipeline is backend-agnostic.

    Returns:
        volume: (X, Y, Z) float32 array indexed as volume[i, j, k]
        affine: (4, 4) voxel->RAS affine matrix
    """
    import SimpleITK as sitk

    img = sitk.ReadImage(str(path))

    # (k, j, i) -> (i, j, k)
    arr = sitk.GetArrayFromImage(img)
    volume = np.ascontiguousarray(arr.transpose(2, 1, 0)).astype(np.float32)

    spacing = np.array(img.GetSpacing(), dtype=np.float64)        # (sx, sy, sz)
    origin = np.array(img.GetOrigin(), dtype=np.float64)          # LPS mm
    direction = np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3)

    # voxel index (i,j,k) -> LPS physical point: p = origin + (D @ diag(spacing)) @ index
    affine_lps = np.eye(4, dtype=np.float64)
    affine_lps[:3, :3] = direction @ np.diag(spacing)
    affine_lps[:3, 3] = origin

    # LPS -> RAS: flip x and y signs
    ras_flip = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine = ras_flip @ affine_lps

    return volume, affine


def window_hu(volume: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Clip and normalise HU values to [0, 1]."""
    vol = np.clip(volume, HU_MIN, HU_MAX)
    vol = (vol - HU_MIN) / (HU_MAX - HU_MIN)
    return vol.astype(np.float32)


def extract_axial_slice(
    volume: npt.NDArray[np.float32],
    z_index: int,
) -> npt.NDArray[np.float32]:
    """Extract a single axial slice (H x W) at voxel z index."""
    slc = volume[:, :, z_index]
    # NIfTI X=R/L, Y=A/P in axial → transpose to (row=Y, col=X) for display
    return slc.T


def z_lps_to_voxel_index(
    z_lps: float,
    affine: npt.NDArray[np.float64],
    volume_shape: tuple[int, int, int],
) -> int:
    """
    Convert an LPS z coordinate to the corresponding voxel z index.

    NIfTI affine maps voxel→RAS. LPS z = -RAS_z_axis? No:
    LPS Superior = RAS Superior (S axis same sign). Only L↔R and P↔A flip.
    So z_ras = z_lps. We solve: affine @ [i,j,k,1] = [x_ras, y_ras, z_ras, 1]
    for k, holding i=j=0.
    """
    # z_ras == z_lps (S axis unchanged)
    z_ras = z_lps
    # affine[:,2] is the z-column (k direction in RAS)
    # affine @ [0,0,k,1] = affine[:,3] + k * affine[:,2]
    # We want the S component (index 2) to equal z_ras
    origin_ras = affine[:3, 3]
    z_col = affine[:3, 2]
    if abs(z_col[2]) < 1e-9:
        return volume_shape[2] // 2  # fallback: middle slice
    k = (z_ras - origin_ras[2]) / z_col[2]
    return int(np.clip(round(k), 0, volume_shape[2] - 1))


def resize_slice(
    slc: npt.NDArray[np.float32],
    target_size: int = IMAGE_SIZE,
) -> npt.NDArray[np.float32]:
    """Resize a 2D slice to (target_size x target_size)."""
    return cv2.resize(slc, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


def slice_to_tensor_input(slc: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Add channel dimension: (H, W) → (1, H, W)."""
    return slc[np.newaxis]  # (1, H, W)


def points_image_to_resize(
    points_px: npt.NDArray[np.float64],
    original_hw: tuple[int, int],
    target_size: int = IMAGE_SIZE,
) -> npt.NDArray[np.float64]:
    """Scale 2D pixel coordinates from original slice size to resized size."""
    scale_x = target_size / original_hw[1]
    scale_y = target_size / original_hw[0]
    scaled = points_px.copy()
    scaled[:, 0] *= scale_x
    scaled[:, 1] *= scale_y
    return scaled


def points_resize_to_image(
    points_px: npt.NDArray[np.float64],
    original_hw: tuple[int, int],
    target_size: int = IMAGE_SIZE,
) -> npt.NDArray[np.float64]:
    """Inverse of points_image_to_resize."""
    scale_x = target_size / original_hw[1]
    scale_y = target_size / original_hw[0]
    scaled = points_px.copy()
    scaled[:, 0] /= scale_x
    scaled[:, 1] /= scale_y
    return scaled
