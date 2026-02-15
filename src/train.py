"""
Thin wrapper around Ultralytics YOLO training for the railway_track class.
Keeps CLI arguments aligned with the proposal's methodology section.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from .config import (
    DEFAULT_BATCH,
    DEFAULT_DATA_YAML,
    DEFAULT_DEVICE,
    DEFAULT_EPOCHS,
    DEFAULT_IMG_SIZE,
    DEFAULT_MODEL,
    RUNS_ROOT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO model for railway track detection")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model or checkpoint (e.g., yolov8n.pt)")
    parser.add_argument("--data", default=str(DEFAULT_DATA_YAML), help="Dataset YAML path")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMG_SIZE, help="Image size")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Batch size")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help='Training device: "0", "cpu", or "auto"')
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--project", default=str(RUNS_ROOT), help="Project directory for YOLO runs")
    parser.add_argument("--name", default="train", help="Run name under project directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data).resolve()
    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        lr0=args.lr0,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
