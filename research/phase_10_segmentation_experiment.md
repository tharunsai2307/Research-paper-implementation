# Phase 10 Segmentation Experiment Protocol & Gate Audit

## 1. Experiment Architecture & Planned Matrix
When human polygon annotations are completed and validated, the segmentation experiment matrix is pre-configured as follows:

| Experiment ID | Architecture | Pretrained Weights | Image Size | Epochs | Batch | Optimizer | Seed | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEG-001** | YOLOv8n-seg | `yolov8n-seg.pt` | 640x640 | 100 | 16 | AdamW | 42 | **BLOCKED** |
| **SEG-002** | YOLOv8n-seg | `yolov8n-seg.pt` | 512x512 | 100 | 16 | AdamW | 42 | **BLOCKED** |

---

## 2. Hard Training Gate Decision
Under Section 9 of the Phase 10 Master Specification:
$$\text{Gate Condition} = (\text{Accepted Masks} > 0) \land (\text{Validation Passes}) \land (\text{All Splits Populated})$$

### Actual System State:
- Accepted Real Masks: **0**
- Completed Real Masks: **0**
- Pending Real Masks: **100**
- Test Set State: **Untouched**
- Hard Gate Verdict: **SEGMENTATION TRAINING = BLOCKED**

### Rationale:
Training an instance segmentation network on unannotated data or synthetic bounding box pseudo-masks constitutes research malpractice. Bounding boxes encompass large areas of healthy leaf fronds, petiole fibers, and sky background. Training a mask head on box rectangles would produce severe boundary artifacts, spurious high IoU on box-like features, and catastrophic failure on true disease lesions.

---

## 3. Dataset Configuration (`configs/coconut_segmentation.yaml`)
```yaml
path: ../data/processed/phase_10_segmentation
train: images/train
val: images/val
test: images/test

names:
  0: bud root dropping
  1: bud rot
  2: gray leaf spot
  3: leaf rot
  4: stembleeding

nc: 5
hard_gate_status: BLOCKED_PENDING_MANUAL_MASKS
```

---

## 4. Hardware and Environment Recording
- **Hardware Architecture**: AMD64 Family 25 Model 80 Stepping 0 (8 physical cores, 16 logical cores)
- **Host OS**: Microsoft Windows 11 Home (Build 10.0.26100)
- **Host Accelerator**: CPU (PyTorch CUDA available: False / No GPU claim made)
- **Python Version**: 3.14.3
- **Ultralytics Version**: 8.4.14
- **PyTorch Version**: 2.10.0+cpu
