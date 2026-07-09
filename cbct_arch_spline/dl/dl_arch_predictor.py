"""
Self-contained DL arch-spline predictor using HeatmapNet (Payer et al.,
MICCAI 2016) -- the best-performing architecture from our comparison
(4.0mm mean curve distance vs 16.1mm for the previous ResNet+residual
approach).

Takes FILE PATHS (not arrays) and does everything internally -- loads
the CBCT + label volumes itself (via SimpleITK, the (Z,Y,X) convention
used throughout this project), runs the model, and writes the result
as a .fcsv in the exact same format as the manual annotations.

The GUI only needs to:
  1. Call predict_arch_to_fcsv() with the CBCT/label file paths
  2. Load the resulting .fcsv with its EXISTING "Load Annotation" code

NOTE ON SLICE DISPLAY: the model works on the WHOLE volume, not on any
single slice. The .fcsv output has a fixed Z coordinate (jaw midpoint)
-- when loading it, keep the user's current slice rather than scrolling
to the stored Z. The arch curve is valid across all jaw slices.

Minimal suggested GUI wiring:

    def _run_dl_detection(self):
        out_fcsv = predict_arch_to_fcsv(
            cbct_path=self._cbct_path,
            label_path=self._label_path,
            checkpoint_path="final_model.pt",
            jaw="lower",            # or "upper"
            pipeline_path="drr_pipeline_v4.py",
            out_fcsv_path="dl_prediction.fcsv",
        )
        pts_lps, z_lps, _ = load_fcsv(out_fcsv)   # existing loader
        pts_px = lps_to_voxel(pts_lps, ...)         # existing conversion
        # DO NOT scroll to z_lps -- keep current slice
        self.canvas.set_control_points(pts_px)

Standalone test (before wiring into GUI):
  python dl_arch_predictor.py \
      --cbct case.nii.gz --label label.nii.gz \
      --checkpoint final_model.pt --jaw lower \
      --pipeline_path drr_pipeline_v4.py \
      --out_fcsv prediction.fcsv
"""
import argparse
import os

import numpy as np
import torch

from prepare_case import load_pipeline_module, prepare_for_inference, N_CONTROL
from catmull_rom_utils import catmull_rom_eval, resample_by_arclength
from architectures import HeatmapNetArch

# HU window used to normalise the MIP channel to [0, 1] before the network.
# prepare_for_inference returns the MIP in raw Hounsfield units (~[-1000, 3000]);
# feeding that raw collapses the prediction into an off-canvas loop. The training
# dataloader normalised this channel and that step is missing from the stand-alone
# inference path, so we restore it here (label_mip and geo_channel are already
# in [0, 1]).
HU_WINDOW = (-1000.0, 2000.0)


def write_fcsv(path, points_mm_xyz, label_prefix):
    """Slicer Markups Fiducial v4.11 -- identical format to the manual
    annotations, so the GUI's existing fcsv loader works unchanged."""
    with open(path, "w") as f:
        f.write("# Markups fiducial file version = 4.11\n")
        f.write("# CoordinateSystem = LPS\n")
        f.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")
        for i, (x, y, z) in enumerate(points_mm_xyz):
            f.write(f"vtkMRMLMarkupsFiducialNode_{i+1},{x:.6f},{y:.6f},{z:.6f},"
                    f"0,0,0,1,1,1,0,{label_prefix}-{i+1},,\n")


def predict_arch_to_fcsv(cbct_path, label_path, checkpoint_path, jaw,
                         out_fcsv_path,
                         pipeline_path="drr_pipeline_v4.py",
                         n_export_points=20,
                         use_pretrained=False):
    """
    Main entry point for GUI integration. Returns out_fcsv_path.

    cbct_path, label_path : paths to CBCT + matching tooth/bone label
        (.mha/.nii.gz or anything SimpleITK reads)
    checkpoint_path : path to final_model.pt  (production checkpoint
        trained on all 160 cases -- NOT a fold*_best.pt CV checkpoint)
    jaw : 'lower' or 'upper'
        Training had 155 lower vs 6 upper examples -- lower predictions
        are reliable; upper should be treated as preliminary.
    n_export_points : number of control points written to the .fcsv
        (20 is a good default -- enough for a smooth Catmull-Rom curve
        without over-populating the Slicer point list)
    """
    if jaw not in ("lower", "upper"):
        raise ValueError(f"jaw must be 'lower' or 'upper', got {jaw!r}")

    mod = load_pipeline_module(pipeline_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # HeatmapNet: the winning architecture (Payer et al., MICCAI 2016)
    # U-Net encoder-decoder -> K heatmaps -> soft-argmax -> (x,y) coords
    model = HeatmapNetArch(n_control_points=N_CONTROL,
                           use_pretrained=use_pretrained)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()

    # prepare: compute jaw-cropped MIP, label MIP, geometric-arch channel
    r = prepare_for_inference(mod, cbct_path, label_path, jaw)

    # Normalise the raw-HU MIP channel to [0, 1] to match training (see HU_WINDOW).
    mip_norm = np.clip(
        (r["mip"] - HU_WINDOW[0]) / (HU_WINDOW[1] - HU_WINDOW[0]), 0.0, 1.0
    )
    img_stack = np.stack([mip_norm, r["label_mip"], r["geo_channel"]], axis=0)
    img_t   = torch.from_numpy(img_stack).float().unsqueeze(0).to(device)
    jaw_t   = torch.tensor([0 if jaw == "lower" else 1],
                            dtype=torch.long).to(device)
    geo_cp_t = torch.from_numpy(r["geo_cp"]).float().unsqueeze(0).to(device)

    with torch.no_grad():
        pred_cp_canvas = model(img_t, jaw_t, geo_cp_t).cpu().numpy()[0]

    # canvas -> voxel space (using the correct pad/scale from prepare_for_inference)
    pad_left = r["pad_left"]
    pad_top  = r["pad_top"]
    scale    = r["scale"]
    pred_cp_voxel = pred_cp_canvas / scale - np.array([pad_left, pad_top])

    # densify then resample to a clean n_export_points Catmull-Rom curve
    dense = catmull_rom_eval(pred_cp_voxel, n_samples=1024)
    export_pts_voxel = resample_by_arclength(dense, n_export_points)

    # convert voxel indices -> LPS mm (direct multiply by spacing)
    sp = r["spacing"]
    sx, sy = sp[2], sp[1]
    z0, z1 = r["z_range"]
    z_mid_mm = (z0 + z1) / 2 * sp[0]
    points_mm = [(float(x * sx), float(y * sy), float(z_mid_mm))
                 for x, y in export_pts_voxel]

    case_label = os.path.splitext(os.path.basename(str(cbct_path)))[0]
    write_fcsv(out_fcsv_path, points_mm, case_label)
    print(f"  [{case_label}] jaw={jaw}  category={r['category']}  "
          f"geo_method={r['geo_method']}")
    print(f"  Saved: {out_fcsv_path}")
    return out_fcsv_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cbct",           required=True)
    ap.add_argument("--label",          required=True)
    ap.add_argument("--checkpoint",     required=True)
    ap.add_argument("--jaw",            required=True, choices=["lower", "upper"])
    ap.add_argument("--pipeline_path",  default="drr_pipeline_v4.py")
    ap.add_argument("--out_fcsv",       default="prediction.fcsv")
    ap.add_argument("--n_export_points", type=int, default=20)
    ap.add_argument("--use_pretrained", action="store_true", default=False)
    args = ap.parse_args()

    predict_arch_to_fcsv(
        cbct_path=args.cbct,
        label_path=args.label,
        checkpoint_path=args.checkpoint,
        jaw=args.jaw,
        out_fcsv_path=args.out_fcsv,
        pipeline_path=args.pipeline_path,
        n_export_points=args.n_export_points,
        use_pretrained=args.use_pretrained,
    )


if __name__ == "__main__":
    main()
