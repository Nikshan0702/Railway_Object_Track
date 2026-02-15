# Automated Drone-Based Railway Track Detection System

## 1. Introduction and Problem Justification
Railway networks demand continuous inspection to mitigate derailment risks arising from obstructions, misalignments, and structural defects. Traditional ground patrols are time-consuming, limited in coverage, and costly. Leveraging drones with onboard computer vision enables rapid, flexible, and repeatable surveys of extended track segments. By executing real-time detection directly on the airborne platform and relaying results to a ground station, the proposed system shortens response time for safety interventions while reducing operational expenses.

## 2. Objectives and Scope
- Develop a single-class detector for `railway_track` using a YOLO family model (YOLOv5 or YOLOv8) optimized for edge devices.  
- Achieve real-time inference on Raspberry Pi 4 or NVIDIA Jetson Nano/Xavier NX with an RGB camera.  
- Stream annotated detections and metadata to a ground station for monitoring and logging.  
- Constrain sensing to RGB imagery only; exclude other classes (e.g., animals) and modalities (e.g., thermal).  
- Support daytime and low-light conditions through data diversity and augmentation.

## 3. System Architecture and Block Diagram Explanation
**Textual block diagram**
```
Drone (RGB Camera + Edge Compute)
      |  frames
      v
Pre-processing (resize/normalize)
      |
      v
YOLO Inference (INT8/FP16)
      |  boxes + scores
      v
Post-processing (NMS, track crop, GPS tag)
      |  annotated frames + metadata
      v
Communication Link (Wi-Fi/4G, RTSP/UDP)
      |
      v
Ground Station (overlay, logging, alerts)
```
**Subsystem roles**
- Drone subsystem: captures RGB video, runs inference, applies NMS, and packages results.  
- Communication layer: transmits compressed frames and JSON metadata; buffers locally if the link degrades.  
- Ground station: renders overlays, records detections, and raises alerts when detection confidence drops or frames are missed.  
- Offline training pipeline: prepares data, trains and quantizes the YOLO model, and produces deployment artifacts for the edge device.

## 4. Dataset Description
- **Collection:** Drone flights over straight, curved, and switch tracks at multiple altitudes and camera angles; include varied illumination (morning, noon, dusk) and backgrounds (urban, rural, vegetation).  
- **Labeling:** Single class `railway_track`; bounding boxes drawn around visible track segments; YOLO TXT or COCO JSON format.  
- **Split:** Stratified 70/20/10 (train/val/test) or 80/10/10 ensuring each subset covers lighting, weather, and viewpoint diversity.  
- **Augmentation:** Mosaic, horizontal flip, slight rotation, brightness/contrast jitter, Gaussian blur for motion, and random scaling; avoid extreme perspective distortion to preserve track geometry.

## 5. Machine Learning Model Description
- **Architecture choice:** YOLOv5s/YOLOv5n or YOLOv8n for favorable FPS-accuracy balance; CSP/ELAN backbone with PAN/FPN neck enabling multi-scale detection.  
- **Losses:** CIoU/DIoU for bounding boxes, BCE for objectness, BCE/focal for classification (single class).  
- **Training setup:** Input 640x640 (adjustable for latency), AdamW or SGD optimizer, cosine or one-cycle LR schedule, 200-300 epochs with early stopping on validation mAP.  
- **Edge optimization:** FP16 on Jetson via TensorRT; INT8 post-training quantization where accuracy permits; layer fusion and optional structured pruning.  
- **Exports:** ONNX to TensorRT (Jetson) or ONNX/TFLite (Raspberry Pi with Coral USB if available).

## 6. Methodology and Workflow (Step-by-Step Flow)
1. Mission planning: define flight path and safety envelope; upload waypoints to autopilot.  
2. Data acquisition: capture RGB frames; maintain rolling buffer for resilience to link drops.  
3. Onboard pre-processing: resize and normalize frames to model input.  
4. Real-time inference: run quantized YOLO; obtain bounding boxes and confidence scores.  
5. Post-processing: apply NMS; crop or highlight detected track regions; tag with timestamp and GPS/IMU pose.  
6. Communication: stream annotated frames plus metadata over Wi-Fi/4G (RTSP/UDP); buffer locally during outages.  
7. Ground station: display overlays, log detections, and trigger alerts on prolonged detection loss or low confidence.  
8. Offline evaluation: compute precision, recall, mAP@0.5, IoU, and FPS; analyze failure cases by lighting and motion blur.  
9. Iteration: update augmentations/hyperparameters; retrain, re-quantize, and redeploy the improved model.

## 7. Hardware and Software Requirements
- **Drone platform:** Quadcopter with sufficient payload for compute module and RGB camera; stabilized mount or gimbal recommended.  
- **Compute:** NVIDIA Jetson Nano/Xavier NX (preferred) or Raspberry Pi 4; optional Coral USB TPU on Pi for acceleration.  
- **Camera:** 1080p RGB, 30 fps or higher, low-distortion lens; ND filters for bright conditions.  
- **Ground station:** Laptop/PC with Wi-Fi/4G interface; storage for logs and replay.  
- **Software stack:** Python 3.10+, PyTorch with Ultralytics YOLOv5/YOLOv8, OpenCV, ONNX Runtime/TensorRT, GStreamer/FFmpeg for streaming, LabelImg/Roboflow for annotation, and ROS2 or lightweight Python server for telemetry handling.

## 8. Evaluation Metrics
- **Detection quality:** Precision, recall, and mAP@0.5 for `railway_track`.  
- **Localization quality:** Mean IoU between predicted and ground-truth boxes.  
- **Performance:** Real-time FPS on target hardware (goal 15-20 FPS on Jetson, 8-12 FPS on Raspberry Pi 4).  
- **Robustness:** False positive/negative rates across lighting/weather; uptime under intermittent connectivity.  
- **Resource use:** Inference latency, GPU/CPU utilization, and power draw during flight.

## 9. Expected Outcomes and Limitations
- **Outcomes:** Deployable edge model delivering real-time railway track detections; live overlays and logged metadata at the ground station; reproducible training and deployment pipeline.  
- **Limitations:** Single-class focus - does not detect obstacles, humans, or animals; performance may degrade in heavy rain/fog, severe motion blur, or night conditions without additional sensing; Raspberry Pi may require reduced input resolution to sustain FPS.

## 10. Ethical, Health, and Safety Considerations
- Comply with UAV regulations (airspace permissions, altitude limits, no-fly zones) and coordinate with railway authorities for test flights.  
- Maintain safe standoff distances from active lines and overhead power; ensure failsafe return-to-home and battery health checks.  
- Protect privacy by avoiding capture of adjacent private property; encrypt telemetry and stored data.  
- Use clear visual markings and pre-flight checklists; keep manual override available during missions.

## 11. Conclusion and Future Improvements
The proposed system offers a low-cost, rapid, and adaptable method for railway track surveillance using drone-mounted, YOLO-based detection executed on edge hardware. Future work can extend to multi-class detection of obstacles and defects, integrate IMU/LiDAR for precise geo-localization, incorporate domain adaptation for night/thermal imagery, and connect alerts directly to railway maintenance dispatch systems.
