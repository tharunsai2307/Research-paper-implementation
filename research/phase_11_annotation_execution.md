# Phase 11 Human Segmentation Annotation Execution Report

## 1. Executive Summary & Objective
Phase 11 operationalizes the human-in-the-loop polygon segmentation data acquisition pipeline for the 100 benchmark coconut palm disease images established in Phase 9 and Phase 10. The fundamental objective is to obtain genuine, verified plant pathology ground-truth masks while strictly prohibiting any form of synthetic mask generation, bounding-box conversion, or unverified AI auto-fills.

### Current Dataset Status:
- **Total Registered Images**: **100**
- **Images Annotated by Human Annotator A**: **0 (PENDING)**
- **Images Double-Annotated by Annotator B**: **0 (PENDING)**
- **Accepted Masks**: **0**
- **Rejected Masks**: **0**
- **Pending Annotations**: **100 (100.0%)**
- **Segmentation Training Gate**: **BLOCKED**

---

## 2. Benchmark Corpus & Split Preservation
All 100 source images are preserved bitwise from `data/external/phase_9_segmentation/images/` without modification, re-compression, or alteration.

### Partitioning Scheme:
- **Train Split**: 63 images (63%)
- **Validation Split**: 27 images (27%)
- **Test Split**: 10 images (10%, designated strictly as `TEST_GROUND_TRUTH_ONLY`)

### Disease Category Breakdown (20 images / class):
1. `bud root dropping` (ID: 0): 20 images (Train: 13, Val: 4, Test: 3)
2. `bud rot` (ID: 1): 20 images (Train: 13, Val: 5, Test: 2)
3. `gray leaf spot` (ID: 2): 20 images (Train: 13, Val: 5, Test: 2)
4. `leaf rot` (ID: 3): 20 images (Train: 12, Val: 7, Test: 1)
5. `stembleeding` (ID: 4): 20 images (Train: 12, Val: 6, Test: 2)

---

## 3. Human Annotation Tooling
To ensure annotators can delineate complex pathological margins accurately, the tool suite was upgraded:
1. **Interactive Annotation Canvas (`scripts/annotation_tool.html`)**:
   - Pan and zoom controls (50% to 300% magnification).
   - Image forward/backward navigation across all 100 images.
   - Separate role selection for `annotator_A`, `annotator_B`, and `reviewer`.
   - Dynamic vertex editing, polygon closure, and vertex undo.
   - Real-time Shoelace area computation and anti-bounding-box warning.
2. **Annotation Engine & Validator (`scripts/annotate_polygons.py`)**:
   - Normalized coordinate bounds verification: $0.0 \le x, y \le 1.0$.
   - Minimum vertex threshold: $N \ge 3$.
   - Strict rejection of 4-corner axis-aligned box rectangles.
   - Image SHA-256 cryptographic logging into `annotation_manifest.csv`.
   - Event audit logging into `annotation_events.csv`.

---

## 4. Batch Ordering & Evaluation Isolation
Annotation execution follows a phased batch sequence:
- **Batch A**: 63 training images.
- **Batch B**: 27 validation images.
- **Batch C**: 10 test images (Evaluation ground truth only; strictly prohibited from use in tool tuning or hyperparameter selection).
