"""
Self-contained DL arch-spline predictor -- the single function meant
to be wired into the GUI's "Detect Arch (AI)" button as an alternative
to the existing heatmap-based predictor.

Takes FILE PATHS (not arrays) and does everything internally -- loads
the CBCT + label volumes itself (via SimpleITK, the (Z,Y,X) convention
used throughout this project), runs the model, and writes the result
as a .fcsv in the exact same format as the manual annotations. This
sidesteps any ambiguity about the GUI's internal array orientation:
the GUI only needs to hand over the same file paths it already has
from "Load CBCT", and load the resulting .fcsv with its EXISTING
"Load Annotation (.fcsv)" code -- no new array-level integration needed.

Minimal suggested GUI wiring (pseudocode, for her to adapt):

    def _run_dl_detection(self):
        out_fcsv = predict_arch_to_fcsv(
            cbct_path=self._cbct_path,        # path she loaded CBCT from
            label_path=self._label_path,       # path to matching label volume
            checkpoint_path="final_model.pt",
            jaw="lower",                        # or "upper" -- ask the user
            pipeline_path="drr_pipeline_v4.py",
            out_fcsv_path="dl_prediction.fcsv",
        )
        pts_lps, z_lps, _ = load_fcsv(out_fcsv)      # her EXISTING loader
        pts_px = lps_to_voxel(pts_lps, ...)           # her EXISTING conversion
        self.canvas.set_control_points(pts_px)

Usage (standalone CLI, for testing independent of the GUI):
  python dl_arch_predictor.py \
      --cbct case.nii.gz --label label.nii.gz \
      --checkpoint final_model.pt --jaw lower \
      --pipeline_path drr_pipeline_v4.py \
      --out_fcsv prediction.fcsv
"""
import argparse

import numpy as np
import torch

from prepare_case import load_pipeline_module, prepare_for_inference, N_CONTROL
from catmull_rom_utils import catmull_rom_eval, resample_by_arclength
from model import ArchSplineNet

# HU window used to normalise the MIP channel to [0, 1] before the network
# (see note in predict_arch_to_fcsv).
HU_WINDOW = (-1000.0, 2000.0)


def write_fcsv(path, points_mm_xyz, label_prefix):
    """Same Slicer Markups Fiducial v4.11 format as the manual
    annotations -- directly loadable by the GUI's existing fcsv loader."""
    with open(path, "w") as f:
        f.write("# Markups fiducial file version = 4.11\n")
        f.write("# CoordinateSystem = LPS\n")
        f.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")
        for i, (x, y, z) in enumerate(points_mm_xyz):
            f.write(f"vtkMRMLMarkupsFiducialNode_{i+1},{x:.6f},{y:.6f},{z:.6f},"
                   f"0,0,0,1,1,1,0,{label_prefix}-{i+1},,\n")


def predict_arch_to_fcsv(cbct_path, label_path, checkpoint_path, jaw,
                         out_fcsv_path, pipeline_path="drr_pipeline_v4.py",
                         n_export_points=20, use_pretrained=True,
                         max_correction_px=60.0):
    """
    Main entry point. Returns the path to the written .fcsv.

    cbct_path, label_path: paths to the CBCT and matching tooth/bone
        label volumes (.mha/.nii.gz -- anything SimpleITK can read).
    checkpoint_path: path to final_model.pt (the production checkpoint,
        NOT one of the fold0..fold4 cross-validation checkpoints).
    jaw: 'lower' or 'upper' -- which arch to predict. Must be supplied
        explicitly (the model was trained on 155 lower vs only 6 upper
        examples -- 'upper' predictions should be treated as
        preliminary until more upper-jaw training data exists).
    use_pretrained: set False if running somewhere without internet
        access to download ImageNet weights (must match how the
        checkpoint was actually trained, though only affects loading
        speed/availability, not correctness, since the checkpoint's
        own weights override the backbone regardless).
    """
    mod = load_pipeline_module(pipeline_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ArchSplineNet(n_control_points=N_CONTROL,
                          use_pretrained=use_pretrained,
                          max_correction_px=max_correction_px)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()

    r = prepare_for_inference(mod, cbct_path, label_path, jaw)
    # NOTE (integration fix): prepare_for_inference returns the MIP channel in
    # RAW Hounsfield units (~[-1000, 3000]). Feeding that straight into the
    # ImageNet-pretrained ResNet backbone saturates every activation, so the
    # residual head outputs a maxed-out (|delta| = max_correction_px on both
    # axes) correction on every case and the predicted arch collapses into an
    # off-canvas loop. The training dataloader normalised this channel; that
    # step was missing from the stand-alone inference path. We restore it here
    # by windowing HU to [0, 1] (label_mip and geo_channel are already [0, 1]).
    mip_norm = np.clip((r["mip"] - HU_WINDOW[0]) / (HU_WINDOW[1] - HU_WINDOW[0]), 0.0, 1.0)
    img_stack = np.stack([mip_norm, r["label_mip"], r["geo_channel"]], axis=0)
    img_t = torch.from_numpy(img_stack).float().unsqueeze(0).to(device)
    jaw_idx = 0 if jaw == "lower" else 1
    jaw_t = torch.tensor([jaw_idx], dtype=torch.long).to(device)
    geo_cp_t = torch.from_numpy(r["geo_cp"]).float().unsqueeze(0).to(device)

    with torch.no_grad():
        pred_cp_canvas = model(img_t, jaw_t, geo_cp_t).cpu().numpy()[0]

    pad_left, pad_top, scale = r["pad_left"], r["pad_top"], r["scale"]
    pred_cp_voxel = pred_cp_canvas / scale - np.array([pad_left, pad_top])

    dense = catmull_rom_eval(pred_cp_voxel, n_samples=1024)
    export_pts_voxel = resample_by_arclength(dense, n_export_points)

    sp = r["spacing"]
    sx, sy = sp[2], sp[1]
    z0, z1 = r["z_range"]
    z_mid_mm = (z0 + z1) / 2 * sp[0]
    points_mm = [(x * sx, y * sy, z_mid_mm) for x, y in export_pts_voxel]

    import os
    case_label = os.path.splitext(os.path.basename(str(cbct_path)))[0]
    write_fcsv(out_fcsv_path, points_mm, case_label)
    return out_fcsv_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cbct", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--jaw", required=True, choices=["lower", "upper"])
    ap.add_argument("--pipeline_path", default="drr_pipeline_v4.py")
    ap.add_argument("--out_fcsv", default="prediction.fcsv")
    ap.add_argument("--use_pretrained", action="store_true", default=False)
    args = ap.parse_args()

    out_path = predict_arch_to_fcsv(
        args.cbct, args.label, args.checkpoint, args.jaw,
        args.out_fcsv, args.pipeline_path, use_pretrained=args.use_pretrained)
    print(f"Saved predicted spline: {out_path}")


if __name__ == "__main__":
    main()
