# Mobile Deployment Architecture & Feasibility Plan

**Research Title**: A YOLOv8-Based Framework for Automated Coconut Tree Disease Detection and Plantation Health Monitoring Using Mobile Applications  
**Component**: Mobile Edge Deployment & Plantation Monitoring Pipeline  
**Phase**: Pre-Training Architectural Roadmap  
**Status**: **PROPOSED DEPLOYMENT SPECIFICATION — NO BENCHMARKS FABRICATED**

---

## 1. Overview & Objectives

The research framework requires translating trained deep-learning detection capabilities into a practical mobile application usable by agricultural extension workers, coconut farmers, and plantation agronomists in field environments.

Given rural plantation operating conditions (intermittent cellular network connectivity, high ambient sunlight, resource-constrained mobile hardware), the mobile deployment pipeline is planned across two complementary operational modalities:
1. **Offline Edge Inference (On-Device)**: Real-time or near-real-time detection running directly on mobile hardware via quantized lightweight runtimes (TFLite / ONNX Runtime Mobile / CoreML).
2. **Cloud/Server-Assisted Inference (API-based)**: Optional asynchronous batch synchronization when network connectivity is available, supporting higher-resolution inference, historical geotagged plantation mapping, and disease progression analytics.

---

## 2. End-to-End Pipeline Architecture

```text
+-----------------------+     +-----------------------+     +-----------------------+
|  Trained YOLOv8 Model  | --> | Model Quantization &  | --> | Cross-Platform Mobile |
|   (PyTorch .pt weights) |     | Format Conversion     |     | Client (Flutter/React)|
+-----------------------+     +-----------------------+     +-----------------------+
                                     |                             |
                                     v                             v
                              - ONNX Export                 - On-Device Camera View
                              - TFLite (FP16 / INT8)        - Bounding Box Overlay
                              - CoreML (iOS)                - Plantation Health Score
```

### Planned Conversion Workflow:
1. **PyTorch Weight Export**: Export trained YOLOv8 PyTorch checkpoint (`best.pt`) using `ultralytics.YOLO.export(format='onnx', dynamic=False, imgsz=640)`.
2. **Post-Training Quantization (PTQ)**:
   - **FP16 Half-Precision**: Reduces model weight storage by ~50% with negligible loss in Mean Average Precision (mAP).
   - **INT8 Integer Quantization**: Employs a representative calibration dataset (sampled from `data/processed/coconut_detection_clean/train/images`) to quantize activation dynamic ranges down to 8-bit integers, enabling hardware acceleration via mobile Neural Processing Units (NPUs) and Digital Signal Processors (DSPs).
3. **Target Runtime Packaging**:
   - Android: TensorFlow Lite (`.tflite`) with GPU / NNAPI delegates.
   - iOS: CoreML (`.mlpackage`) with Apple Neural Engine (ANE) acceleration.

---

## 3. Planned Performance Metrics to Measure in Future Benchmarking

When Phase 4 / deployment benchmarking begins, the following metrics will be measured empirically on physical test devices (e.g. standard Android mid-range smartphones and iOS devices). **No values are fabricated at this pre-deployment stage**:

| Performance Dimension | Metric | Measurement Unit | Planned Target Threshold |
| :--- | :--- | :--- | :--- |
| **Footprint** | Compressed Model Storage Size | Megabytes (MB) | $\le 15.0$ MB |
| **Speed** | Inference Latency per Frame | Milliseconds (ms) | $\le 80$ ms (interactive mobile view) |
| **Throughput** | Real-Time Frame Rate | Frames per Second (FPS) | $\ge 12$ FPS on mobile camera |
| **Resource Utilization** | Peak RAM Consumption | Megabytes (MB) | $\le 200$ MB during active inference |
| **Processor Load** | Average CPU Utilization | Percentage (%) | $\le 35\%$ to prevent thermal throttling |
| **Thermal / Battery** | Battery Drain Rate | % per hour active scanning | To be measured empirically in field |
| **Detection Quality** | Mobile mAP@0.5 vs. Server mAP | Relative mAP retention | $\ge 95\%$ retention after quantization |
| **Confidence Tuning** | Optimal Detection Confidence | Probability threshold | Tuned to maximize F1-score on validation set |

---

## 4. Mobile User Experience & Plantation Health Monitoring

The mobile interface is designed to support proactive plantation management:
1. **Live Camera Guidance**: Real-time crosshairs and guide frame indicating optimal distance and lighting for scanning coconut trunks, crown buds, and leaf fronds.
2. **On-Screen Disease Identification**: Detected lesions highlighted with bounding boxes and color-coded class badges:
   - `bud root dropping` (High Severity Alert)
   - `bud rot` (Critical Severity Alert)
   - `gray leaf spot` (Moderate Severity)
   - `leaf rot` (Moderate Severity)
   - `stembleeding` (High Severity Alert)
3. **Healthy / Negative Confirmation**: When scanning coconut foliage without detected disease, the application provides an affirmative "Healthy Palm Foliage Verified" status, leveraging background negative training to confirm absence of disease.
4. **Offline Geotagged Plantation Log**: Captures GPS coordinates, timestamp, and disease diagnosis into an on-device SQLite database, enabling plantation health heatmaps upon network reconnection.

---

## 5. Next Steps for Phase 4

Following completion of Phase 3 model training and ablation experiments:
1. Export top-performing model variants (`yolov8n` and `yolov8s`).
2. Run automated quantization scripts with representative calibration imagery.
3. Validate parity between PyTorch outputs and TFLite / ONNX outputs on identical test set images.
4. Deploy to reference mobile devices and log empirical latency and accuracy profiles.
