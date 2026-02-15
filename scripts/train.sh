#!/usr/bin/env bash
set -euo pipefail

# Sample training command. Adjust paths and hyperparameters as needed.
python3 -m src.train \
  --model yolov8n-obb.pt \
  --data railway_track.yaml \
  --imgsz 640 \
  --epochs 200 \
  --batch 16 \
  --device auto \
  --workers 8 \
  --project runs \
  --name train
