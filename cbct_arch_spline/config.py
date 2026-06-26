"""Central configuration for all modules."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data_dir"
MODELS_DIR = PROJECT_ROOT / "saved_models"
ANNOTATIONS_DIR = Path("/Users/imanechafi/Desktop/arch_points_manual")

# Image preprocessing
HU_MIN = -500
HU_MAX = 2000
IMAGE_SIZE = 512  # model input/output size (pixels)

# Heatmap generation
HEATMAP_SIGMA = 8  # Gaussian sigma for control point heatmaps (pixels)

# Model
MODEL_BACKBONE = "efficientnet_b3"
IN_CHANNELS = 1
NUM_KEYPOINTS = 1  # single-channel heatmap (all keypoints merged)

# Training
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
VAL_SPLIT = 0.15
TEST_SPLIT = 0.10
SEED = 42

# Inference post-processing
HEATMAP_THRESHOLD = 0.3   # min heatmap value to consider as keypoint
MIN_PEAK_DISTANCE = 15    # minimum pixel distance between keypoints
MAX_KEYPOINTS = 32        # upper bound on expected control points
MIN_KEYPOINTS = 10        # lower bound on expected control points

# Spline
SPLINE_SMOOTHING = 0.0    # scipy splprep smoothing (0 = interpolating)
SPLINE_DEGREE = 3         # cubic spline

# GUI
WINDOW_TITLE = "CBCT Dental Arch Spline Annotator"
VIEWER_COLORMAP = "gray"
SPLINE_COLOR = "salmon"
POINT_COLOR = "tomato"
POINT_RADIUS = 6
