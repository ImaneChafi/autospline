"""
Evaluation script: compute metrics on a test set.

Usage:
    python scripts/evaluate.py --cbct_dir /path/to/nifti --checkpoint saved_models/arch_detector_best.pth
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ANNOTATIONS_DIR
from data.dataset import CBCTArchDataset
from data.preprocessing import load_nifti, window_hu, extract_axial_slice, z_lps_to_voxel_index
from models.unet import build_model
from inference.predictor import ArchPredictor, extract_keypoints_from_heatmap
from spline.fcsv_io import load_fcsv, lps_to_voxel
from spline.spline_utils import fit_spline, mean_distance_to_spline, order_points_along_arch


def evaluate(args) -> dict:
    model = build_model(use_pretrained=False)
    predictor = ArchPredictor.from_checkpoint(args.checkpoint, model)

    fcsv_dir = Path(args.fcsv_dir)
    cbct_dir = Path(args.cbct_dir)

    fcsv_files = sorted(fcsv_dir.glob("*.fcsv"))
    if args.n_samples:
        fcsv_files = fcsv_files[:args.n_samples]

    all_distances = []
    failed = 0

    for fcsv_path in fcsv_files:
        case_id = fcsv_path.stem
        nii_path = cbct_dir / f"{case_id}.nii.gz"
        if not nii_path.exists():
            continue

        try:
            ann = load_fcsv(fcsv_path)
            volume, affine = load_nifti(nii_path)
            volume = window_hu(volume)

            z_idx = z_lps_to_voxel_index(ann["z_lps"], affine, volume.shape)
            slc = extract_axial_slice(volume, z_idx)

            result = predictor.predict_from_slice(slc)
            pred_pts = result["keypoints_px"]

            # Ground truth keypoints in pixel coords
            gt_vox = lps_to_voxel(ann["points_lps"], affine)
            gt_px = gt_vox[:, :2]

            if len(pred_pts) < 2 or len(gt_px) < 2:
                failed += 1
                continue

            # Fit dense GT curve
            gt_ordered = order_points_along_arch(gt_px)
            gt_curve, _ = fit_spline(gt_ordered, n_eval=500)

            dist = mean_distance_to_spline(pred_pts, gt_curve)
            all_distances.append(dist)
            print(f"  {case_id}: mean dist = {dist:.2f} px, {len(pred_pts)} predicted pts")

        except Exception as e:
            print(f"  {case_id}: ERROR — {e}")
            failed += 1

    metrics = {
        "n_evaluated": len(all_distances),
        "n_failed": failed,
        "mean_distance_px": float(np.mean(all_distances)) if all_distances else float("nan"),
        "median_distance_px": float(np.median(all_distances)) if all_distances else float("nan"),
        "std_distance_px": float(np.std(all_distances)) if all_distances else float("nan"),
    }
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cbct_dir", required=True)
    p.add_argument("--fcsv_dir", default=str(ANNOTATIONS_DIR))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n_samples", type=int, default=None)
    args = p.parse_args()

    metrics = evaluate(args)
    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
