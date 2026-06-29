"""
CBCT Dental Arch Spline Annotator — main GUI application.

Features:
  - Load CBCT NIfTI volumes
  - Navigate axial slices with a slider
  - Run AI detection to get an initial spline proposal
  - Drag control points to refine
  - Add / delete control points with keyboard shortcuts
  - Export annotations as .fcsv (Slicer format)
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QFileDialog, QStatusBar,
    QGroupBox, QSizePolicy, QMessageBox, QDoubleSpinBox, QSpinBox,
    QCheckBox, QToolBar, QAction,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    WINDOW_TITLE, VIEWER_COLORMAP, SPLINE_COLOR, POINT_COLOR, POINT_RADIUS,
    HU_MIN, HU_MAX, IMAGE_SIZE,
)
from data.preprocessing import (
    load_volume, window_hu, extract_axial_slice, z_lps_to_voxel_index,
)
from spline.fcsv_io import load_fcsv, save_fcsv, lps_to_voxel, voxel_to_lps
from spline.spline_utils import fit_spline, order_points_along_arch


# ---------------------------------------------------------------------------
# Matplotlib canvas with interactive spline editing
# ---------------------------------------------------------------------------


class SplineCanvas(FigureCanvas):
    """
    Matplotlib canvas that shows a CBCT slice with draggable spline points.

    Control points are circles that can be:
      - Dragged to new positions (left-click + drag)
      - Added by Shift+click on the image
      - Deleted by right-clicking on a point
    """

    PICK_RADIUS = 12  # pixels for hit testing

    def __init__(self, parent=None, status_callback=None):
        self.fig = Figure(figsize=(8, 8), facecolor="black")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("black")
        self.fig.tight_layout(pad=0)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.status_callback = status_callback or (lambda msg: None)

        # State
        self._slice_image: Optional[npt.NDArray] = None
        self._control_points: list[list[float]] = []  # list of [col, row] in pixel coords
        self._heatmap: Optional[npt.NDArray] = None
        self._show_heatmap: bool = False

        # Drag state
        self._dragging_idx: Optional[int] = None
        self._drag_offset: tuple[float, float] = (0, 0)

        # Artists
        self._im = None
        self._heatmap_im = None
        self._spline_line: Optional[Line2D] = None
        self._point_circles: list[Circle] = []

        # Connect events
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event", self._on_motion)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_slice(self, slice_2d: npt.NDArray[np.float32]) -> None:
        self._slice_image = slice_2d
        self._redraw()

    def set_control_points(self, points: npt.NDArray[np.float64]) -> None:
        """Points as (N, 2) array of (col, row) pixel coordinates."""
        self._control_points = [list(p) for p in points]
        self._redraw()

    def get_control_points(self) -> npt.NDArray[np.float64]:
        return np.array(self._control_points, dtype=np.float64) if self._control_points else np.zeros((0, 2))

    def set_heatmap(self, heatmap: Optional[npt.NDArray]) -> None:
        self._heatmap = heatmap
        self._redraw()

    def toggle_heatmap(self, show: bool) -> None:
        self._show_heatmap = show
        self._redraw()

    def clear_points(self) -> None:
        self._control_points = []
        self._redraw()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _redraw(self) -> None:
        self.ax.clear()
        self.ax.set_facecolor("black")
        self.ax.axis("off")

        if self._slice_image is not None:
            self._im = self.ax.imshow(
                self._slice_image, cmap=VIEWER_COLORMAP,
                vmin=0, vmax=1, origin="upper", aspect="equal",
            )

        if self._show_heatmap and self._heatmap is not None:
            import cv2
            h_resized = cv2.resize(
                self._heatmap,
                (self._slice_image.shape[1], self._slice_image.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            ) if self._slice_image is not None else self._heatmap
            self.ax.imshow(
                h_resized, cmap="hot", alpha=0.4,
                vmin=0, vmax=1, origin="upper", aspect="equal",
            )

        if len(self._control_points) >= 2:
            pts = np.array(self._control_points)
            try:
                curve, _ = fit_spline(pts, n_eval=400)
                self.ax.plot(
                    curve[:, 0], curve[:, 1],
                    color=SPLINE_COLOR, linewidth=2, alpha=0.85, zorder=2,
                )
            except Exception:
                self.ax.plot(pts[:, 0], pts[:, 1], color=SPLINE_COLOR, linewidth=1.5, zorder=2)

        for i, (col, row) in enumerate(self._control_points):
            circle = Circle(
                (col, row), POINT_RADIUS,
                color=POINT_COLOR, zorder=3, linewidth=1.5,
                edgecolor="white", alpha=0.9,
            )
            self.ax.add_patch(circle)
            self.ax.text(
                col + POINT_RADIUS + 2, row, str(i + 1),
                color="white", fontsize=7, va="center", zorder=4,
            )

        self.ax.set_xlim(
            0, self._slice_image.shape[1] if self._slice_image is not None else 512
        )
        self.ax.set_ylim(
            self._slice_image.shape[0] if self._slice_image is not None else 512, 0
        )
        self.draw_idle()

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def _find_nearest_point(self, x: float, y: float) -> Optional[int]:
        """Return index of control point within PICK_RADIUS, else None."""
        if not self._control_points:
            return None
        pts = np.array(self._control_points)
        dists = np.sqrt((pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2)
        idx = int(np.argmin(dists))
        if dists[idx] <= self.PICK_RADIUS:
            return idx
        return None

    def _shift_held(self) -> bool:
        """
        Reliable Shift detection via Qt, independent of matplotlib focus.

        matplotlib's event.key is None during a click unless its canvas holds
        keyboard focus, so we ask Qt for the live modifier state instead.
        """
        return bool(QApplication.keyboardModifiers() & Qt.ShiftModifier)

    def _on_press(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None:
            return

        x, y = event.xdata, event.ydata

        if event.button == 1:  # left click
            idx = self._find_nearest_point(x, y)
            # Add a point when Shift is held, or when the click is on empty
            # space (not near an existing point). Otherwise start dragging.
            if self._shift_held() or idx is None:
                self._control_points.append([x, y])
                self.status_callback(
                    f"Added point at ({x:.1f}, {y:.1f}). "
                    f"Total: {len(self._control_points)}"
                )
                self._redraw()
            else:
                self._dragging_idx = idx
                self._drag_offset = (
                    self._control_points[idx][0] - x,
                    self._control_points[idx][1] - y,
                )

        elif event.button == 3:  # right click → delete point
            idx = self._find_nearest_point(x, y)
            if idx is not None:
                self._control_points.pop(idx)
                self.status_callback(f"Deleted point {idx + 1}. Total: {len(self._control_points)}")
                self._redraw()

    def _on_motion(self, event) -> None:
        if self._dragging_idx is None or event.inaxes != self.ax or event.xdata is None:
            return
        x, y = event.xdata, event.ydata
        dx, dy = self._drag_offset
        self._control_points[self._dragging_idx] = [x + dx, y + dy]
        self._redraw()

    def _on_release(self, event) -> None:
        if self._dragging_idx is not None:
            self.status_callback(
                f"Moved point {self._dragging_idx + 1}. "
                f"Total: {len(self._control_points)}"
            )
        self._dragging_idx = None


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1200, 850)

        # State
        self._volume: Optional[npt.NDArray] = None
        self._affine: Optional[npt.NDArray] = None
        self._nii_path: Optional[Path] = None
        self._predictor = None  # loaded on demand

        self._build_ui()
        self._build_toolbar()
        self.statusBar().showMessage("Load a CBCT NIfTI file to start.")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Right: canvas (created first — control panel wires signals to it)
        self.canvas = SplineCanvas(status_callback=self._set_status)

        # Left: controls
        ctrl_panel = self._build_control_panel()
        main_layout.addWidget(ctrl_panel, stretch=0)

        main_layout.addWidget(self.canvas, stretch=1)

    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(220)
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignTop)

        # --- File group ---
        file_group = QGroupBox("File")
        fg_layout = QVBoxLayout(file_group)

        btn_load_cbct = QPushButton("Load CBCT (.nii.gz)")
        btn_load_cbct.clicked.connect(self._load_cbct)
        fg_layout.addWidget(btn_load_cbct)

        btn_load_fcsv = QPushButton("Load Annotation (.fcsv)")
        btn_load_fcsv.clicked.connect(self._load_fcsv)
        fg_layout.addWidget(btn_load_fcsv)

        btn_save = QPushButton("Save Annotation (.fcsv)")
        btn_save.clicked.connect(self._save_annotation)
        fg_layout.addWidget(btn_save)
        layout.addWidget(file_group)

        # --- Slice navigation ---
        nav_group = QGroupBox("Axial Slice")
        ng_layout = QVBoxLayout(nav_group)

        self._slice_label = QLabel("Slice: —")
        ng_layout.addWidget(self._slice_label)

        self._slice_slider = QSlider(Qt.Horizontal)
        self._slice_slider.setMinimum(0)
        self._slice_slider.setValue(0)
        self._slice_slider.valueChanged.connect(self._on_slice_changed)
        ng_layout.addWidget(self._slice_slider)

        # HU window controls
        ng_layout.addWidget(QLabel("HU min:"))
        self._hu_min_spin = QSpinBox()
        self._hu_min_spin.setRange(-2000, 0)
        self._hu_min_spin.setValue(HU_MIN)
        self._hu_min_spin.valueChanged.connect(self._refresh_display)
        ng_layout.addWidget(self._hu_min_spin)

        ng_layout.addWidget(QLabel("HU max:"))
        self._hu_max_spin = QSpinBox()
        self._hu_max_spin.setRange(100, 4000)
        self._hu_max_spin.setValue(HU_MAX)
        self._hu_max_spin.valueChanged.connect(self._refresh_display)
        ng_layout.addWidget(self._hu_max_spin)

        layout.addWidget(nav_group)

        # --- AI Detection ---
        ai_group = QGroupBox("AI Detection")
        ag_layout = QVBoxLayout(ai_group)

        btn_load_model = QPushButton("Load Model (.pth)")
        btn_load_model.clicked.connect(self._load_model)
        ag_layout.addWidget(btn_load_model)

        btn_detect = QPushButton("Detect Arch (AI)")
        btn_detect.clicked.connect(self._run_detection)
        ag_layout.addWidget(btn_detect)

        self._chk_heatmap = QCheckBox("Show heatmap overlay")
        self._chk_heatmap.toggled.connect(self.canvas.toggle_heatmap)
        ag_layout.addWidget(self._chk_heatmap)

        layout.addWidget(ai_group)

        # --- Geometric (no AI) ---
        geo_group = QGroupBox("Geometric (no AI)")
        gg_layout = QVBoxLayout(geo_group)

        gg_layout.addWidget(QLabel("1. Click ~6 points roughly\n   along the arch"))

        gg_layout.addWidget(QLabel("Control points:"))
        self._n_control_spin = QSpinBox()
        self._n_control_spin.setRange(8, 40)
        self._n_control_spin.setValue(24)
        gg_layout.addWidget(self._n_control_spin)

        btn_fit_clicks = QPushButton("2. Fit Arch from Clicks")
        btn_fit_clicks.clicked.connect(self._fit_from_clicks)
        gg_layout.addWidget(btn_fit_clicks)

        btn_snap = QPushButton("Snap to Bright")
        btn_snap.clicked.connect(self._snap_to_bright)
        gg_layout.addWidget(btn_snap)

        btn_auto = QPushButton("Auto-detect (rough)")
        btn_auto.clicked.connect(self._auto_detect)
        gg_layout.addWidget(btn_auto)

        layout.addWidget(geo_group)

        # --- Edit group ---
        edit_group = QGroupBox("Edit")
        eg_layout = QVBoxLayout(edit_group)

        eg_layout.addWidget(QLabel("Click empty space: add point"))
        eg_layout.addWidget(QLabel("Drag a point: move it"))
        eg_layout.addWidget(QLabel("Right-click a point: delete"))

        btn_order = QPushButton("Re-order Points")
        btn_order.clicked.connect(self._reorder_points)
        eg_layout.addWidget(btn_order)

        btn_clear = QPushButton("Clear All Points")
        btn_clear.clicked.connect(self._clear_points)
        eg_layout.addWidget(btn_clear)

        layout.addWidget(edit_group)

        # --- Info ---
        self._info_label = QLabel("")
        self._info_label.setWordWrap(True)
        layout.addWidget(self._info_label)

        return panel

    def _build_toolbar(self) -> None:
        tb = self.addToolBar("Main")
        tb.setMovable(False)

        act_load = QAction("Load CBCT", self)
        act_load.triggered.connect(self._load_cbct)
        tb.addAction(act_load)

        act_save = QAction("Save .fcsv", self)
        act_save.setShortcut(QKeySequence.Save)
        act_save.triggered.connect(self._save_annotation)
        tb.addAction(act_save)

        tb.addSeparator()

        act_detect = QAction("Detect (AI)", self)
        act_detect.triggered.connect(self._run_detection)
        tb.addAction(act_detect)

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def _load_cbct(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open CBCT Volume", str(Path.home()),
            "CBCT volumes (*.mha *.mhd *.nrrd *.nii.gz *.nii);;"
            "ITK MetaImage (*.mha *.mhd *.nrrd);;"
            "NIfTI files (*.nii.gz *.nii);;"
            "All files (*)"
        )
        if not path:
            return

        self._nii_path = Path(path)
        self._set_status(f"Loading {self._nii_path.name}…")
        try:
            self._volume, self._affine = load_volume(self._nii_path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            return

        n_slices = self._volume.shape[2]
        self._slice_slider.setMaximum(n_slices - 1)
        mid = n_slices // 2
        self._slice_slider.setValue(mid)
        self._on_slice_changed(mid)
        self._set_status(f"Loaded: {self._nii_path.name} — shape {self._volume.shape}")

    def _load_fcsv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open .fcsv annotation", str(Path.home()),
            "FCSV files (*.fcsv);;All files (*)"
        )
        if not path:
            return

        try:
            ann = load_fcsv(path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            return

        if self._volume is None or self._affine is None:
            QMessageBox.warning(self, "No CBCT", "Please load a CBCT volume first.")
            return

        # Navigate to annotated z slice
        z_idx = z_lps_to_voxel_index(ann["z_lps"], self._affine, self._volume.shape)
        self._slice_slider.setValue(z_idx)

        # Convert LPS → voxel → pixel on the axial slice
        pts_vox = lps_to_voxel(ann["points_lps"], self._affine)
        slc = extract_axial_slice(self._windowed_volume(), z_idx)
        # col = voxel axis 0 scaled to slice width; row = voxel axis 1 scaled to height
        # After .T in extract_axial_slice: image row = voxel Y, image col = voxel X
        pts_px = pts_vox[:, :2]  # (col=X, row=Y)
        self.canvas.set_control_points(pts_px)
        self._set_status(f"Loaded {len(pts_px)} control points from {Path(path).name}")

    def _save_annotation(self) -> None:
        if self._nii_path is None and self._volume is None:
            QMessageBox.warning(self, "No data", "Load a CBCT first.")
            return

        pts_px = self.canvas.get_control_points()
        if len(pts_px) == 0:
            QMessageBox.warning(self, "No points", "No control points to save.")
            return

        default_name = (self._nii_path.stem if self._nii_path else "annotation") + ".fcsv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save .fcsv", str(Path.home() / default_name),
            "FCSV files (*.fcsv)"
        )
        if not path:
            return

        z_idx = self._slice_slider.value()

        if self._affine is not None:
            # Convert pixel → voxel → LPS
            pts_vox = np.column_stack([
                pts_px[:, 0],  # col = X voxel
                pts_px[:, 1],  # row = Y voxel
                np.full(len(pts_px), z_idx),
            ])
            pts_lps = voxel_to_lps(pts_vox, self._affine)
            z_lps = float(np.mean(pts_lps[:, 2]))
        else:
            # Fallback: save as pixel coords if no affine
            pts_lps = np.column_stack([pts_px, np.zeros(len(pts_px))])
            z_lps = 0.0

        case_id = Path(path).stem
        save_fcsv(path, pts_lps, z_lps, case_id)
        self._set_status(f"Saved {len(pts_lps)} points to {Path(path).name}")

    # ------------------------------------------------------------------
    # Slice navigation
    # ------------------------------------------------------------------

    def _windowed_volume(self) -> npt.NDArray[np.float32]:
        hu_min = self._hu_min_spin.value()
        hu_max = self._hu_max_spin.value()
        vol = np.clip(self._volume, hu_min, hu_max)
        return ((vol - hu_min) / (hu_max - hu_min)).astype(np.float32)

    def _on_slice_changed(self, z_idx: int) -> None:
        if self._volume is None:
            return
        self._slice_label.setText(f"Slice: {z_idx} / {self._volume.shape[2] - 1}")
        slc = extract_axial_slice(self._windowed_volume(), z_idx)
        self.canvas.set_slice(slc)

    def _refresh_display(self) -> None:
        if self._volume is not None:
            self._on_slice_changed(self._slice_slider.value())

    # ------------------------------------------------------------------
    # AI detection
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load model checkpoint", str(Path.home()),
            "PyTorch checkpoint (*.pth);;All files (*)"
        )
        if not path:
            return
        try:
            from models.unet import build_model
            from inference.predictor import ArchPredictor
            model = build_model(use_pretrained=False)
            self._predictor = ArchPredictor.from_checkpoint(path, model)
            self._set_status(f"Model loaded: {Path(path).name}")
        except Exception as e:
            QMessageBox.critical(self, "Model Load Error", str(e))

    def _run_detection(self) -> None:
        if self._volume is None:
            QMessageBox.warning(self, "No CBCT", "Load a CBCT volume first.")
            return

        if self._predictor is None:
            # Try to auto-load from default location
            default = Path(MODELS_DIR if hasattr(__import__('config'), 'MODELS_DIR') else '.') / "arch_detector_best.pth"
            if default.exists():
                self._load_model_from_path(default)
            else:
                reply = QMessageBox.question(
                    self, "No Model",
                    "No trained model loaded.\n\n"
                    "Try a rough automatic geometric detection instead?\n"
                    "(For reliable results, use 'Fit Arch from Clicks' in the "
                    "Geometric panel — Shift+click ~6 points first.)",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply == QMessageBox.Yes:
                    self._auto_detect()
                return

        z_idx = self._slice_slider.value()
        slc = extract_axial_slice(self._windowed_volume(), z_idx)

        self._set_status("Running AI detection…")
        QApplication.processEvents()

        try:
            result = self._predictor.predict_from_slice(slc)
            self.canvas.set_control_points(result["keypoints_px"])
            self.canvas.set_heatmap(result["heatmap"])
            self._set_status(
                f"Detected {result['n_keypoints']} keypoints. "
                "Shift+click to add, right-click to delete, drag to adjust."
            )
        except Exception as e:
            QMessageBox.critical(self, "Detection Error", str(e))

    def _load_model_from_path(self, path: Path) -> None:
        from models.unet import build_model
        from inference.predictor import ArchPredictor
        model = build_model(use_pretrained=False)
        self._predictor = ArchPredictor.from_checkpoint(path, model)

    # ------------------------------------------------------------------
    # Geometric (no-AI) methods
    # ------------------------------------------------------------------

    def _fit_from_clicks(self) -> None:
        """Turn the current rough clicks into a smooth, evenly-spaced arch."""
        from inference.geometric import assisted_arch_from_clicks

        clicks = self.canvas.get_control_points()
        if len(clicks) < 4:
            QMessageBox.information(
                self, "Need more points",
                "Shift+click at least 4 points (≈6 recommended) roughly along "
                "the arch first, then press 'Fit Arch from Clicks'.",
            )
            return
        slc = (
            extract_axial_slice(self._windowed_volume(), self._slice_slider.value())
            if self._volume is not None else None
        )
        try:
            control = assisted_arch_from_clicks(
                clicks, n_control=self._n_control_spin.value(), slice_2d=slc
            )
            n_clicks = len(clicks)
            self.canvas.set_control_points(control)
            self._set_status(
                f"Fitted arch: {len(control)} evenly-spaced control points "
                f"from {n_clicks} clicks. Drag any point to refine."
            )
        except Exception as e:
            QMessageBox.critical(self, "Fit Error", str(e))

    def _snap_to_bright(self) -> None:
        """Nudge each control point onto the nearest bright structure."""
        from inference.geometric import snap_points_to_bright

        pts = self.canvas.get_control_points()
        if len(pts) == 0 or self._volume is None:
            return
        z_idx = self._slice_slider.value()
        slc = extract_axial_slice(self._windowed_volume(), z_idx)
        snapped = snap_points_to_bright(slc, pts, radius=6)
        self.canvas.set_control_points(snapped)
        self._set_status("Snapped control points to nearby bright tooth/bone.")

    def _auto_detect(self) -> None:
        """Rough fully-automatic geometric guess (refine afterwards)."""
        from inference.geometric import auto_detect_arch

        if self._volume is None:
            QMessageBox.warning(self, "No CBCT", "Load a CBCT volume first.")
            return
        z_idx = self._slice_slider.value()
        slc = extract_axial_slice(self._windowed_volume(), z_idx)
        try:
            control = auto_detect_arch(slc, n_control=self._n_control_spin.value())
            self.canvas.set_control_points(control)
            self._set_status(
                f"Auto-detect: {len(control)} points (rough). Refine by dragging, "
                "or use 'Fit Arch from Clicks' for a cleaner result."
            )
        except ValueError as e:
            QMessageBox.information(self, "Auto-detect", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Auto-detect Error", str(e))

    # ------------------------------------------------------------------
    # Editing helpers
    # ------------------------------------------------------------------

    def _reorder_points(self) -> None:
        pts = self.canvas.get_control_points()
        if len(pts) < 2:
            return
        ordered = order_points_along_arch(pts)
        self.canvas.set_control_points(ordered)
        self._set_status("Points re-ordered along arch.")

    def _clear_points(self) -> None:
        reply = QMessageBox.question(
            self, "Clear", "Remove all control points?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.canvas.clear_points()
            self._set_status("Control points cleared.")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _set_status(self, msg: str) -> None:
        self.statusBar().showMessage(msg)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Left:
            self._slice_slider.setValue(self._slice_slider.value() - 1)
        elif key == Qt.Key_Right:
            self._slice_slider.setValue(self._slice_slider.value() + 1)
        else:
            super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def launch():
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    launch()
