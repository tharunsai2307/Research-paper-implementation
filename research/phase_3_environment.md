# Phase 3 Computing Environment & System Diagnostics

**Project Title**: A YOLOv8-Based Framework for Automated Coconut Tree Disease Detection and Plantation Health Monitoring Using Mobile Applications  
**Phase**: Phase 3 — YOLOv8 Model Training & Controlled Experimentation  
**Timestamp**: 2026-09-20T09:30:00Z  

---

## 1. Hardware Architecture

| Component | Specification | Operational Status |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 Home / Pro (`Windows-11-10.0.26200-SP0`) | Native Windows x64 |
| **Processor (CPU)** | AMD64 Family 23 Model 160 Stepping 0, AuthenticAMD | 4 Physical Cores / 8 Logical Threads |
| **System Memory (RAM)** | 15.24 GB Total Physical RAM | ~8.0 GB Available for Dataloading & Caching |
| **GPU Acceleration** | None Detected (`CUDA Available: False`) | **CPU Execution Mode Enforced** |
| **GPU VRAM** | N/A | N/A |

### Research Integrity Hardware Statement
The host platform does not possess a CUDA-capable discrete GPU. All deep-learning computations (forward pass, backpropagation, and non-maximum suppression) will execute on multi-threaded CPU (`device='cpu'`). This operational reality is explicitly recognized. Training batch sizes, image dimensions, and worker counts are calibrated to avoid memory thrashing or CPU starvation, while maintaining complete scientific validity.

---

## 2. Core Python & Deep Learning Frameworks

| Package | Version | Provenance / Build |
| :--- | :--- | :--- |
| **Python** | `3.14.4` | CPython MSC v.1944 64-bit (AMD64) |
| **PyTorch** | `2.13.0+cpu` | Official CPU Build |
| **TorchVision** | `0.28.0` | Official Wheel |
| **Ultralytics** | `8.4.115` | Official YOLOv8 Implementation |
| **OpenCV** | `4.10.0.84` | Image I/O & BBox Visualization |
| **NumPy** | `2.5.1` | Array Mathematics |
| **Pillow (PIL)** | `12.2.0` | Image Decoding & Transforms |
| **Pandas** | `3.0.5` | Metric Aggregation & Tabulation |
| **Matplotlib** | `3.11.1` | Training Curves & Figure Generation |
| **Scikit-Learn** | `1.9.0` | Metric Evaluation & Statistical Utilities |

Complete pip dependencies are recorded in machine-readable format at:
[research/phase_3_environment.json](file:///C:/Users/Hp/.gemini/antigravity-ide/scratch/coconut-disease-yolov8/research/phase_3_environment.json).
