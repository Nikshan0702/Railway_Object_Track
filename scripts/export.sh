#!/usr/bin/env bash
set -euo pipefail

# Example export command for TensorRT or ONNX.
# Update WEIGHTS to the trained checkpoint you want to export.

WEIGHTS="runs/train/weights/best.pt"

# Export to ONNX (default)
python -m src.export --weights "${WEIGHTS}" --format onnx --half

# Uncomment for TensorRT engine export on Jetson (requires TensorRT support)
# python -m src.export --weights "${WEIGHTS}" --format engine --half
