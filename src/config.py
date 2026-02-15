"""
Central configuration helpers for the railway track detection project.
Adjust paths and defaults here to keep scripts consistent.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Local dataset folder shipped with the repo (contains OBB labels).
DATA_ROOT = PROJECT_ROOT / "Dataset"
RUNS_ROOT = PROJECT_ROOT / "runs"
MODELS_ROOT = PROJECT_ROOT / "models"

# Default dataset YAML expected by Ultralytics YOLO.
DEFAULT_DATA_YAML = PROJECT_ROOT / "railway_track.yaml"
# Preferred validation image folders for the three modalities
DEFAULT_SAMPLE_DIRS = [
    DATA_ROOT / "rgb/valid/images",      # labeled RGB set
    DATA_ROOT / "thermal/valid/images",  # labeled thermal/IR set
    DATA_ROOT / "fusion/valid/images",   # optional third modality (e.g., aligned composites)
    DATA_ROOT / "valid/images",          # legacy single-modality fallback
]

# Default training hyperparameters.
DEFAULT_MODEL = "yolov8n-obb.pt"  # oriented bounding-box variant
DEFAULT_IMG_SIZE = 640
DEFAULT_EPOCHS = 5
DEFAULT_BATCH = 16
DEFAULT_DEVICE = "auto"  # "0" for GPU, "cpu" for CPU
