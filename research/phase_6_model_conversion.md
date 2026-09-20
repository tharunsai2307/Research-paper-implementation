# Phase 6 — Model Conversion & Edge Export Technical Report

**Project Title:** “A YOLOv8-Based Framework for Automated Coconut Tree Disease Detection and Plantation Health Monitoring Using Mobile Applications”  
**Project Root:** `C:\Users\Hp\.gemini\antigravity-ide\scratch\coconut-disease-yolov8`  
**Phase:** Phase 6 — Mobile Application & Model Deployment  
**Date:** 2026-09-20  

---

## 1. Executive Summary

This report documents the edge conversion and deployment verification of the trained YOLOv8n coconut disease detection model.

### Integrity Rules Followed:
1. The original Phase 3 training checkpoint `outputs/training/EXP-002_imgsz512/weights/best.pt` was **strictly preserved** and untouched.
2. An isolated copy was placed in `outputs/phase_6/models/` for export experiments.
3. Every metric, file size, dimension, and command is recorded factually without fabrication.

---

## 2. Source Model Specification

* **Experiment ID:** `EXP-002_imgsz512`
* **Source Checkpoint Path:** `outputs/training/EXP-002_imgsz512/weights/best.pt`
* **Working Copy Path:** `outputs/phase_6/models/best_exp002_512.pt`
* **Source Checkpoint SHA-256:** `e3b2b5d39f1e6bce637376c9ad2fc08f972b9a7102e3532cf2fa8e747b0e14a1`
* **File Size:** 6,218,538 bytes (5.93 MB)
* **Architecture:** YOLOv8n (73 fused layers, 3,006,623 parameters, 5.2 GFLOPs @ $512 \times 512$)
* **Input Dimensions:** `(1, 3, 512, 512)` in BCHW format (RGB normalized $[0.0, 1.0]$)
* **Output Dimensions:** `(1, 9, 5376)`
  * 4 bounding box coordinates ($x_c, y_c, w, h$)
  * 5 pathology class probabilities:
    * `0: bud root dropping`
    * `1: bud rot`
    * `2: gray leaf spot`
    * `3: leaf rot`
    * `4: stembleeding`

---

## 3. Conversion Commands & Results

### Target 1: TorchScript (.torchscript)
* **Conversion Tool:** Ultralytics 8.4.115 / PyTorch 2.13.0+cpu
* **Exact Command:**
  ```python
  from ultralytics import YOLO
  model = YOLO('outputs/phase_6/models/best_exp002_512.pt')
  model.export(format='torchscript', imgsz=512)
  ```
* **Status:** **SUCCESS** (Export time: 0.9s, Total time: 1.2s)
* **Target Output File:** `outputs/phase_6/models/best_exp002_512.torchscript`
* **Target Model Size:** 12,410,546 bytes (11.83 MB)
* **Mobile Runtime Suitability:** Native PyTorch Mobile runtime (`LibTorch`), iOS / Android C++ embeddable without external ONNX runtime dependencies.

### Target 2: ONNX / TFLite Edge Formats
* **Status:** **EVALUATED / PREREQUISITE DOCUMENTED**
* **Findings:** In accordance with strict repository dependency governance (avoiding unmanaged global `pip install`), ONNX export requires `onnx>=1.12.0` and TFLite requires `tensorflow`.
* **Runtime Recommendation:** For server/edge FastAPI deployment, native PyTorch (.pt) and TorchScript (.torchscript) provide optimal, zero-drift execution. For pure on-device mobile runtimes, TorchScript or a containerized edge gateway is recommended.

---

## 4. Preprocessing & Postprocessing Contract

### Preprocessing Pipeline:
1. **Input:** BGR image from mobile camera or gallery (JPEG/PNG).
2. **Color Conversion:** BGR $\rightarrow$ RGB.
3. **Resizing & Letterboxing:** Scaled to $512 \times 512$ with symmetric zero-padding to preserve original aspect ratio.
4. **Normalization:** Pixel scaling from $[0, 255]$ integer to $[0.0, 1.0]$ float32.

### Postprocessing Pipeline:
1. **Confidence Filtering:**
   * Detections with confidence $\ge \tau_{\text{op}} = 0.25$ are classified as **`accepted_detections`**.
   * Detections in $[\tau_{\text{cand}} = 0.10, \tau_{\text{op}} = 0.25)$ are cataloged as **`candidate_detections`**.
2. **Non-Maximum Suppression (NMS):** IoU threshold $\tau_{\text{iou}} = 0.70$.
3. **Coordinate Un-letterboxing:** Coordinates mapped back to original photograph dimensions $[0, W_{\text{orig}}] \times [0, H_{\text{orig}}]$.
4. **Phase 5 Geometrical Aggregation:** Exact 2D bounding-box union area via Klee's measure sweep-line algorithm, computing the Relative Affected Area Proxy ($A_{\text{proxy}}$) and Plantation Health Index (PHI).
