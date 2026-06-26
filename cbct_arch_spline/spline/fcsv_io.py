"""Read and write Slicer .fcsv fiducial files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt


def load_fcsv(path: str | Path) -> dict:
    """
    Load a Slicer .fcsv file.

    Returns dict with:
      - points_lps: (N, 3) float64 array in LPS mm coordinates
      - z_lps: the common z coordinate (all points share one slice)
      - case_id: string from the label field
    """
    path = Path(path)
    points = []
    case_id = path.stem

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split(",")
            # columns: id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,...
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            points.append([x, y, z])

    points_lps = np.array(points, dtype=np.float64)
    z_lps = float(np.mean(points_lps[:, 2]))

    return {
        "points_lps": points_lps,
        "z_lps": z_lps,
        "case_id": case_id,
    }


def save_fcsv(
    path: str | Path,
    points_lps: npt.NDArray[np.float64],
    z_lps: float,
    case_id: str,
) -> None:
    """Save control points to a Slicer .fcsv file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write("# Markups fiducial file version = 5.10\n")
        f.write("# CoordinateSystem = LPS\n")
        f.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")
        for i, (x, y, _) in enumerate(points_lps, start=1):
            label = f"{case_id}-{i}"
            f.write(
                f"{i},{x:.6f},{y:.6f},{z_lps:.6f},"
                f"0,0,0,1,1,1,0,{label},,vtkMRMLScalarVolumeNode1,2,0\n"
            )


def lps_to_voxel(
    points_lps: npt.NDArray[np.float64],
    affine: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Convert LPS mm coordinates to voxel indices using NIfTI affine.

    NIfTI affine is RAS; LPS → RAS by flipping first two axes.
    """
    points_ras = points_lps.copy()
    points_ras[:, 0] = -points_lps[:, 0]  # L → R
    points_ras[:, 1] = -points_lps[:, 1]  # P → A

    inv_affine = np.linalg.inv(affine)
    # homogeneous
    ones = np.ones((len(points_ras), 1))
    pts_h = np.hstack([points_ras, ones])
    voxels_h = (inv_affine @ pts_h.T).T
    return voxels_h[:, :3]


def voxel_to_lps(
    points_voxel: npt.NDArray[np.float64],
    affine: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Convert voxel indices back to LPS mm coordinates."""
    ones = np.ones((len(points_voxel), 1))
    pts_h = np.hstack([points_voxel, ones])
    ras_h = (affine @ pts_h.T).T
    points_ras = ras_h[:, :3]

    points_lps = points_ras.copy()
    points_lps[:, 0] = -points_ras[:, 0]
    points_lps[:, 1] = -points_ras[:, 1]
    return points_lps
