#!/usr/bin/env bash
set -euo pipefail

# Example export command for TensorRT or ONNX.
# Update WEIGHTS to the trained checkpoint you want to export.

WEIGHTS="${1:-}"
if [[ -z "${WEIGHTS}" ]]; then
  WEIGHTS="$(find runs -type f -path '*weights/best.pt' -print0 2>/dev/null | xargs -0 ls -t 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "${WEIGHTS}" ]]; then
  echo "Could not find a trained checkpoint. Train first (scripts/train.sh) or pass WEIGHTS explicitly."
  exit 1
fi

# Export to ONNX (default)
python3 -m src.export --weights "${WEIGHTS}" --format onnx --half

# Uncomment for TensorRT engine export on Jetson (requires TensorRT support)
# python3 -m src.export --weights "${WEIGHTS}" --format engine --half
