# Phase 11 Segmentation Ground-Truth Annotation & Double-Annotator QC Workspace

## 1. Overview & Source Image Preservation
This workspace manages the human-in-the-loop polygon segmentation dataset for the 100 benchmark coconut disease images registered in Phase 9.

- **Source Image Location**: `data/external/phase_9_segmentation/images/` (100 images, unaugmented, unaltered).
- **Split Distribution**:
  - Train: 63 images
  - Validation: 27 images
  - Test: 10 images (Strictly `TEST_GROUND_TRUTH_ONLY`, isolated from hyperparameter/tool tuning).
- **Class Balance**: 20 images per class across 5 disease categories:
  0: `bud root dropping`
  1: `bud rot`
  2: `gray leaf spot`
  3: `leaf rot`
  4: `stembleeding`

---

## 2. Directory Structure
```
data/external/phase_11_segmentation/
├── annotations/
│   ├── annotator_A/        <- Primary human polygon annotations (<stem>.txt)
│   └── annotator_B/        <- Independent double-annotator validation polygons (<stem>.txt)
├── metadata/
│   ├── annotation_manifest.csv  <- Image-level annotation status, paths, and QC flags
│   ├── qc_manifest.csv          <- Inter-annotator IoU scores, disagreement review status
│   └── annotation_events.csv    <- Audit trail of annotation and review operations
└── README.md
```

---

## 3. Annotation Schema Contract
- Format: Canonical YOLO Segmentation Polygon
  ```text
  <class_id> x1 y1 x2 y2 ... xn yn
  ```
- Constraints:
  - Normalized coordinates in $[0.0, 1.0]$
  - Minimum vertices: $N \ge 3$
  - Non-zero polygon area
  - Anti-Bounding Box rule: 4-corner axis-aligned rectangles strictly rejected
  - Research integrity: zero synthetic masks, zero pseudo-masks.
