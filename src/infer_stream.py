"""
Lightweight real-time inference loop for onboard use.
Reads from a camera or video file, runs YOLO, overlays detections, and
optionally forwards frames to a network sink.
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
from ultralytics import YOLO

from .config import DEFAULT_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time YOLO inference on a video stream")
    parser.add_argument("--weights", default=DEFAULT_MODEL, help="Model weights path")
    parser.add_argument("--source", default=0, help="Camera index or video file/URL")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", default="auto", help='Device for inference: "0", "cpu", or "auto"')
    parser.add_argument("--view", action="store_true", help="Display annotated frames locally")
    parser.add_argument("--save", action="store_true", help="Save annotated video to disk")
    parser.add_argument("--output", default="runs/stream/output.mp4", help="Output path if --save is set")
    return parser.parse_args()


def open_writer(path: Path, fps: float, width: int, height: int) -> Optional[cv2.VideoWriter]:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(str(path), fourcc, fps, (width, height))


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = open_writer(Path(args.output), fps, width, height) if args.save else None

    last_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=args.conf, device=args.device, verbose=False)
        annotated = results[0].plot()

        # Simple FPS overlay for debugging.
        now = time.time()
        fps_est = 1.0 / max(now - last_time, 1e-5)
        last_time = now
        cv2.putText(annotated, f"FPS: {fps_est:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if args.view:
            cv2.imshow("railway_track detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if writer:
            writer.write(annotated)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
