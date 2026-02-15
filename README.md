# Automated Drone-Based Railway Track Detection System

This repository contains the proposal and starter code for an undergraduate final-year project that develops a drone-based, YOLO-powered railway track detector for edge deployment (Jetson or Raspberry Pi).

- Proposal document: `docs/PROJECT_PROPOSAL.md`
- Dataset spec: `data/railway_track.yaml`
- Training wrapper: `src/train.py`
- Export helper: `src/export.py`
- Real-time inference loop: `src/infer_stream.py`
- Sample commands: `scripts/train.sh`, `scripts/export.sh`

## Quick start
1. Install dependencies (CPU example): `pip install -r requirements.txt`  
   - On Jetson or Raspberry Pi, install the appropriate PyTorch wheel first, then the rest.
2. Prepare data under `data/images/{train,val,test}` and matching `data/labels/...`; update `data/railway_track.yaml` paths if needed.
3. Train: `bash scripts/train.sh` (edit batch size/epochs to fit your GPU/CPU).
4. Export for edge: `bash scripts/export.sh` to produce ONNX or TensorRT.
5. Run real-time inference (webcam id 0): `python -m src.infer_stream --weights models/best.onnx --source 0 --view`.

## Directory layout
- `docs/` - project proposal and design narrative.
- `data/` - dataset layout and YAML config (images/labels not included).
- `src/` - training, export, and inference helpers.
- `scripts/` - example shell commands for training/export.
- `models/` - target folder for exported weights.
- `runs/` - YOLO training outputs (created automatically).
# Railway_Object_Track
