#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/predict.sh [WEIGHTS] [SOURCE]
#
# Examples:
#   bash scripts/predict.sh runs/obb/.../weights/best.pt sample1.mp4
#   bash scripts/predict.sh "" image1.png

WEIGHTS="${1:-}"
SOURCE="${2:-sample1.mp4}"

if [[ -z "${WEIGHTS}" ]]; then
  WEIGHTS="$(find runs -type f -path '*weights/best.pt' -print0 2>/dev/null | xargs -0 ls -t 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "${WEIGHTS}" ]]; then
  echo "Could not find a trained checkpoint. Train first (scripts/train.sh) or pass WEIGHTS explicitly."
  exit 1
fi

python3 -m src.predict \
  --weights "${WEIGHTS}" \
  --source "${SOURCE}" \
  --imgsz 640 \
  --conf 0.25 \
  --device auto \
  --save-dir runs/obb/predict
