"""
Launch the interactive annotation GUI.

Usage:
    python scripts/annotate.py
    python scripts/annotate.py --cbct /path/to/file.nii.gz
    python scripts/annotate.py --cbct /path/to/file.nii.gz --fcsv /path/to/ann.fcsv
    python scripts/annotate.py --cbct /path/to/file.nii.gz --model saved_models/arch_detector_best.pth
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    p = argparse.ArgumentParser(description="CBCT Dental Arch Spline Annotator")
    p.add_argument("--cbct", type=str, default=None, help="Pre-load this CBCT file")
    p.add_argument("--fcsv", type=str, default=None, help="Pre-load this .fcsv annotation")
    p.add_argument("--model", type=str, default=None, help="Pre-load model checkpoint")
    args = p.parse_args()

    from PyQt5.QtWidgets import QApplication
    import nibabel as nib
    import numpy as np

    app = QApplication(sys.argv)

    from gui.app import MainWindow
    from config import MODELS_DIR

    window = MainWindow()

    # Pre-load CBCT if provided
    if args.cbct:
        cbct_path = Path(args.cbct)
        if cbct_path.exists():
            try:
                img = nib.load(str(cbct_path))
                window._volume = np.asarray(img.dataobj, dtype=np.float32)
                window._affine = img.affine
                window._nii_path = cbct_path
                n = window._volume.shape[2]
                window._slice_slider.setMaximum(n - 1)
                mid = n // 2
                window._slice_slider.setValue(mid)
                window._on_slice_changed(mid)
                window._set_status(f"Loaded: {cbct_path.name}")
            except Exception as e:
                print(f"Failed to load CBCT: {e}")

    # Pre-load annotation if provided
    if args.fcsv and window._volume is not None:
        try:
            window._load_fcsv_from_path(Path(args.fcsv))
        except Exception as e:
            print(f"Failed to load fcsv: {e}")

    # Pre-load model if provided
    if args.model:
        model_path = Path(args.model)
        if model_path.exists():
            try:
                window._load_model_from_path(model_path)
            except Exception as e:
                print(f"Failed to load model: {e}")
        else:
            print(f"Model file not found: {model_path}")

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
