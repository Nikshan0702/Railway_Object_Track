"""
Export trained YOLO weights to ONNX and (optionally) TensorRT for edge deployment.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from .config import MODELS_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO model for edge deployment")
    parser.add_argument("--weights", required=True, help="Path to trained weights (e.g., runs/train/weights/best.pt)")
    parser.add_argument("--format", default="onnx", choices=["onnx", "engine"], help="Export format")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 quantization (where supported)")
    parser.add_argument("--half", action="store_true", help="Export in FP16 (where supported)")
    parser.add_argument("--device", default="cpu", help='Device for export: "cpu" or "0" for GPU')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    out = model.export(
        format=args.format,
        int8=args.int8,
        half=args.half,
        device=args.device,
        dynamic=False,
    )

    # Ultralytics returns the output path; move into models/ for consistency.
    exported = Path(out)
    target = MODELS_ROOT / exported.name
    if exported.resolve() != target.resolve():
        target.write_bytes(exported.read_bytes())
    print(f"Exported model to {target}")


if __name__ == "__main__":
    main()
