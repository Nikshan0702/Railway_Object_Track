# Railway Object Track (YOLOv8 OBB)

Drone/edge-oriented **railway track detection** using **Ultralytics YOLOv8** with **oriented bounding boxes (OBB)**. Includes helpers for training, prediction, export (ONNX / TensorRT), and a simple real-time inference loop.

- Proposal document: `docs/PROJECT_PROPOSAL.md`
- Dataset config (tracked): `railway_track.yaml`
- Train: `src/train.py`
- Predict: `src/predict.py`
- Export: `src/export.py`
- Stream inference: `src/infer_stream.py`
- Example commands: `scripts/train.sh`, `scripts/predict.sh`, `scripts/export.sh`

## Setup
Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset
This repo expects a local dataset folder at `Dataset/` (gitignored by default). The default layout used by `railway_track.yaml` is:

```text
Dataset/
  thermal/
    train/
      images/
      labels/
    valid/
      images/
      labels/
```

Update `railway_track.yaml` if your dataset is in a different location or if you use different splits/modalities.

## Train
Run the sample training script (edit hyperparameters as needed):

```bash
bash scripts/train.sh
```

Training outputs are written under `runs/` (Ultralytics will create an `obb/` subfolder for OBB runs). Your checkpoints will be in a run folder similar to:

- `runs/obb/.../weights/best.pt`
- `runs/obb/.../weights/last.pt`

## Predict (images or video)
Run inference on a file (image or video) and save annotated outputs:

```bash
python3 -m src.predict --weights runs/obb/.../weights/best.pt --source sample1.mp4
```

Outputs default to `runs/obb/predict/` (configurable via `--save-dir`).

## Export (ONNX / TensorRT)
Export a trained checkpoint to `models/`:

```bash
python3 -m src.export --weights runs/obb/.../weights/best.pt --format onnx --half
```

On Jetson (TensorRT), you can export an engine with `--format engine` (requires a TensorRT-capable Ultralytics environment).

## Real-time inference (camera)
Run a lightweight webcam/video loop:

```bash
python3 -m src.infer_stream --weights runs/obb/.../weights/best.pt --source 0 --view
```

## Notes
- `Dataset/` is ignored by Git so you can keep large images/labels locally.
- If `python` is not available on your system, use `python3` (macOS/Linux) or the activated venv’s `python`.
