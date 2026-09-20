# Phase 3 Reproducibility & Replication Protocol

**Project Title**: A YOLOv8-Based Framework for Automated Coconut Tree Disease Detection and Plantation Health Monitoring Using Mobile Applications  
**Phase**: Phase 3 — YOLOv8 Model Training & Controlled Experimentation  
**Framework**: Ultralytics YOLOv8 (`v8.4.115`), PyTorch (`2.13.0+cpu`)  
**Publication Date**: 2026-09-20  

---

## 1. Overview & Protocol Integrity

This document provides complete instructions and exact configuration parameters to reproduce all Phase 3 experimental results identically. Every experiment was conducted with deterministic pseudo-random seeds (`seed=42`), standardized dataloaders, and fixed training splits.

---

## 2. Exact Experimental Parameters

| Parameter | EXP-001 (Baseline) | EXP-002 (Image Size) | EXP-003 (Capacity) |
| :--- | :--- | :--- | :--- |
| **Model Checkpoint** | `yolov8n.pt` | `yolov8n.pt` | `yolov8s.pt` |
| **Pretrained Source** | Ultralytics Official COCO | Ultralytics Official COCO | Ultralytics Official COCO |
| **Dataset Path** | `data/processed/coconut_detection_clean/data.yaml` | Same | Same |
| **Input Resolution (`imgsz`)** | 640 | 512 | 640 |
| **Batch Size** | 16 | 16 | 16 |
| **Epochs** | 15 | 15 | 15 |
| **Patience** | 10 | 10 | 10 |
| **Optimizer** | `AdamW` | `AdamW` | `AdamW` |
| **Initial Learning Rate (`lr0`)**| 0.001 | 0.001 | 0.001 |
| **Learning Rate Schedule** | Cosine Annealing (`lrf=0.01`) | Same | Same |
| **Momentum** | 0.937 | 0.937 | 0.937 |
| **Weight Decay** | 0.0005 | 0.0005 | 0.0005 |
| **Random Seed** | 42 | 42 | 42 |
| **Dataloader Workers** | 0 | 0 | 0 |
| **Execution Device** | CPU (`device='cpu'`) | CPU (`device='cpu'`) | CPU (`device='cpu'`) |
| **Execution Status** | **COMPLETED** | **COMPLETED** | **NOT RUN (Compute Bound)** |

---

## 3. Replication Commands

To replicate these experiments from the repository root:

```bash
# Ensure dependencies are installed
pip install -r requirements.txt

# Run the complete automated experiment suite
python scripts/train_experiments.py
```

### Direct Python Reproduction Code
```python
from ultralytics import YOLO
from pathlib import Path

DATA_YAML = "data/processed/coconut_detection_clean/data.yaml"

# Reproduce EXP-001 (Baseline YOLOv8n, imgsz=640)
model_exp1 = YOLO("yolov8n.pt")
model_exp1.train(
    data=DATA_YAML,
    epochs=15,
    imgsz=640,
    batch=16,
    seed=42,
    workers=0,
    device="cpu",
    optimizer="AdamW",
    lr0=0.001,
    patience=10,
    project="outputs/training",
    name="EXP-001_baseline_yolov8n",
    exist_ok=True
)

# Reproduce EXP-002 (Controlled Image-Size, imgsz=512)
model_exp2 = YOLO("yolov8n.pt")
model_exp2.train(
    data=DATA_YAML,
    epochs=15,
    imgsz=512,
    batch=16,
    seed=42,
    workers=0,
    device="cpu",
    optimizer="AdamW",
    lr0=0.001,
    patience=10,
    project="outputs/training",
    name="EXP-002_imgsz512",
    exist_ok=True
)
```

---

## 4. Checkpoint & Artifact Locations

All model weights and output curves are permanently archived in the project structure:
- **EXP-001 Weights**: `outputs/training/EXP-001_baseline_yolov8n/weights/best.pt` (5.94 MB)
- **EXP-002 Weights**: `outputs/training/EXP-002_imgsz512/weights/best.pt` (5.93 MB)
- **Master Registry**: `research/phase_3_experiment_registry.csv`
- **Machine-Readable Metrics**: `outputs/training/phase_3_results.json`
- **Validation Sample Predictions**: `outputs/predictions/EXP-001_val_samples/` and `EXP-002_val_samples/`
- **Comparative Visual Figures**: `outputs/figures/training/`
