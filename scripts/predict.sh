#!/usr/bin/env bash
set -euo pipefail

# Example prediction run on validation images with the trained OBB model.
.venv/bin/python -m src.predict \
  --weights runs/obb/runs/train-obb-jan313/weights/best.pt \
  --source Dataset/valid/images \
  --imgsz 640 \
  --conf 0.25 \
  --device cpu \
  --save-dir runs/obb/predict
