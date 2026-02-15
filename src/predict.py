"""
Prediction helper patterned after the structured loader/runner style in the user example.
Provides callable helpers (from bytes/path/PIL) and a small CLI.
"""

from __future__ import annotations

import io
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from ultralytics import YOLO

from .config import DEFAULT_SAMPLE_DIRS, PROJECT_ROOT

# -------------------------
# Paths & Device
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs/obb/runs/train-obb-jan313/weights/best.pt"
DEFAULT_SAVE_DIR = PROJECT_ROOT / "runs/obb/predict"

CONFIDENCE_THRESHOLD = 0.25
DEFAULT_DEVICE = "cpu"  # set to "0" for GPU if available

# Map modality -> default validation image directory
MODALITY_SAMPLE_DIRS = {
    "rgb": DEFAULT_SAMPLE_DIRS[0],
    "thermal": DEFAULT_SAMPLE_DIRS[1],
    "fusion": DEFAULT_SAMPLE_DIRS[2],
    "legacy": DEFAULT_SAMPLE_DIRS[3],
}


def _pick_sample_dir(modality: Optional[str]) -> Path:
    """
    Choose a validation image directory for fallback inference.
    - If a modality is specified, require that path to exist.
    - Otherwise, return the first existing directory from the preference order.
    - If none exist, return the first configured path (user will see a clear error).
    """
    if modality:
        if modality not in MODALITY_SAMPLE_DIRS:
            raise ValueError(f"Unknown modality '{modality}'. Choose from {list(MODALITY_SAMPLE_DIRS)}.")
        chosen = MODALITY_SAMPLE_DIRS[modality]
        if not chosen.exists():
            raise FileNotFoundError(
                f"Requested modality '{modality}' but folder not found: {chosen}. "
                "Create the folder or pick another modality."
            )
        return chosen

    for name in ("rgb", "thermal", "fusion", "legacy"):
        candidate = MODALITY_SAMPLE_DIRS[name]
        if candidate.exists():
            return candidate

    return next(iter(MODALITY_SAMPLE_DIRS.values()))

# -------------------------
# Lazy model loader
# -------------------------
_MODEL: Optional[YOLO] = None

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".flv", ".ts"}


def get_model(weights: Path = DEFAULT_WEIGHTS, device: str = DEFAULT_DEVICE) -> YOLO:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    weights = weights.expanduser().resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    _MODEL = YOLO(str(weights))
    _MODEL.to(device)
    return _MODEL


# -------------------------
# Prediction helpers
# -------------------------
def _result_to_dict(result, conf_threshold: float) -> Dict[str, Any]:
    obb = getattr(result, "obb", None)
    boxes = getattr(result, "boxes", None)

    confs = None
    labels = None
    polys = None

    if obb is not None and obb.conf is not None:
        confs = obb.conf.cpu().numpy()
        labels = obb.cls.cpu().numpy()
        polys = obb.xyxyxyxy.cpu().numpy()  # (n, 4, 2)
    elif boxes is not None and boxes.conf is not None:
        confs = boxes.conf.cpu().numpy()
        labels = boxes.cls.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        # convert axis-aligned boxes to 4-point polygons
        polys = np.stack(
            [
                xyxy[:, [0, 1]],
                xyxy[:, [2, 1]],
                xyxy[:, [2, 3]],
                xyxy[:, [0, 3]],
            ],
            axis=1,
        )
    else:
        return {"detections": [], "threshold": conf_threshold}

    detections: List[Dict[str, Any]] = []
    for cls_id, conf, poly in zip(labels, confs, polys):
        if conf < conf_threshold:
            continue
        points = [[float(x), float(y)] for x, y in poly]
        detections.append(
            {
                "label": "railway_track",
                "confidence": float(conf),
                "points": points,
            }
        )

    return {
        "detections": detections,
        "threshold": conf_threshold,
        "count": len(detections),
    }


def predict_from_pil(
    pil_img: Image.Image,
    weights: Path = DEFAULT_WEIGHTS,
    device: str = DEFAULT_DEVICE,
    imgsz: int = 640,
    conf: float = CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    model = get_model(weights, device)
    pil_img = pil_img.convert("RGB")
    results = model.predict(
        source=pil_img,
        imgsz=imgsz,
        conf=conf,
        device=device,
        max_det=300,
        save=False,
        verbose=False,
    )
    return _result_to_dict(results[0], conf)


def predict_from_bytes(
    image_bytes: bytes,
    weights: Path = DEFAULT_WEIGHTS,
    device: str = DEFAULT_DEVICE,
    imgsz: int = 640,
    conf: float = CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    pil = Image.open(io.BytesIO(image_bytes))
    return predict_from_pil(pil, weights=weights, device=device, imgsz=imgsz, conf=conf)


def predict_from_path(
    image_path: str | Path,
    weights: Path = DEFAULT_WEIGHTS,
    device: str = DEFAULT_DEVICE,
    imgsz: int = 640,
    conf: float = CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Accepts an image *or* video path.
    - Images are routed through PIL and the standard helper.
    - Videos are passed directly to YOLO; we summarize detections from the first frame.
    """
    path = Path(image_path).expanduser()
    if path.suffix.lower() in VIDEO_EXTS:
        model = get_model(weights, device)
        results = model.predict(
            source=str(path),
            imgsz=imgsz,
            conf=conf,
            device=device,
            max_det=300,
            save=True,
            verbose=False,
        )
        if not results:
            return {"detections": [], "threshold": conf, "count": 0, "note": "no frames returned"}
        return _result_to_dict(results[0], conf) | {
            "source": str(path),
            "note": "summary is from first frame of the video; see saved annotations for full run",
        }

    pil = Image.open(path)
    return predict_from_pil(pil, weights=weights, device=device, imgsz=imgsz, conf=conf)


# -------------------------
# CLI runner
# -------------------------
def parse_args():
    parser = ArgumentParser(description="Predict with YOLOv8 OBB railway-track model")
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help="Path to trained weights (.pt)",
    )
    parser.add_argument(
        "--source",
        default=None,
        help=(
            "Image/video path, directory, or stream. "
            "If omitted, uses the first available image from the selected modality's validation set."
        ),
    )
    parser.add_argument(
        "--modality",
        choices=list(MODALITY_SAMPLE_DIRS),
        default=None,
        help="Which modality's validation images to use when --source is omitted (rgb|thermal|fusion|legacy).",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help='Device: "cpu", "0", or "0,1,..."')
    parser.add_argument("--max-det", type=int, default=300, dest="max_det", help="Maximum detections per image")
    parser.add_argument("--show", action="store_true", help="Display predictions in a window")
    parser.add_argument("--half", action="store_true", help="Use FP16 (GPU only)")
    parser.add_argument(
        "--save-dir",
        default=DEFAULT_SAVE_DIR,
        help="Directory to save annotated predictions",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    weights = Path(args.weights).expanduser().resolve()
    device = args.device

    if args.source is None:
        sample_dir = _pick_sample_dir(args.modality)
        candidates = sorted(sample_dir.glob("*"))
        if not candidates:
            raise FileNotFoundError(
                f"No images found in {sample_dir}. "
                "Provide --source explicitly or populate the chosen modality folder."
            )
        source_path = candidates[0]
        print(f"No --source provided. Using sample image from {sample_dir.name} modality: {source_path}")
    else:
        source_path = Path(args.source).expanduser()

    save_dir = Path(args.save_dir).expanduser()
    project, name = save_dir.parent, save_dir.name

    model = get_model(weights, device)
    results = model.predict(
        source=str(source_path),
        imgsz=args.imgsz,
        conf=args.conf,
        device=device,
        max_det=args.max_det,
        half=args.half,
        show=args.show,
        save=True,
        project=str(project),
        name=name,
        exist_ok=True,
    )

    summary = _result_to_dict(results[0], args.conf)
    if summary["detections"]:
        best = max(d["confidence"] for d in summary["detections"])
        mean = sum(d["confidence"] for d in summary["detections"]) / summary["count"]
        print(
            f"{source_path}: detections={summary['count']}, "
            f"best_conf={best:.3f}, mean_conf={mean:.3f}"
        )
    else:
        print(f"{source_path}: no detections (conf >= {args.conf})")

    print(f"Annotated outputs saved to: {results[0].save_dir}")


if __name__ == "__main__":
    main()
