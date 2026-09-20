# Phase 9 Annotation Report: Corpus Selection & QC Architecture

## 1. Executive Summary & Corpus Selection

In strict accordance with Phase 9 instructions, manual polygon annotation must begin from the **genuine, unaugmented benchmark source imagery**, rather than downloading arbitrary web images or applying pre-annotation augmentations.

### Selected Annotation Image Pool:
- **Total Registered Images**: **100 images**
- **Provenance**: Patil et al. / Roboflow Universe (`phanidhar-reddy`) benchmark archive
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Class Balance**: Strictly uniform across all 5 target coconut disease categories (20 images per class):
  1. `bud root dropping`: 20 images
  2. `bud rot`: 20 images
  3. `gray leaf spot`: 20 images
  4. `leaf rot`: 20 images
  5. `stembleeding`: 20 images

---

## 2. Segmentation Workspace Layout (`data/external/phase_9_segmentation/`)

```
data/external/phase_9_segmentation/
├── images/        <- 100 benchmark disease images (verified bitwise copies)
├── annotations/   <- Canonical YOLO segmentation polygon files (<class_id> x1 y1 x2 y2 ...)
├── masks/         <- Binary raster mask PNGs (optional rasterization cache)
├── metadata/      <- annotation_manifest.csv (tracking annotator ID, status, QC flag)
├── qc/            <- Double-annotation review logs and inter-annotator IoU scores
├── splits/        <- train.txt (63), val.txt (27), test.txt (10)
└── provenance/    <- Source licenses and attribution records
```

---

## 3. Canonical Annotation Representation

All segmentation boundaries are represented in the canonical **YOLO Segmentation Polygon Format**:
```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> ... <xn> <yn>
```
where:
- Coordinates are normalized in $[0.0, 1.0]$.
- $n \ge 3$ vertices tracing visible necrotic tissue boundaries.

### Anti-Fabrication Safeguard:
Every annotation file is scanned by `scripts/validate_segmentation_annotations.py`. Any line containing exactly 5 tokens (`<class_id> <xc> <yc> <w> <h>`) is flagged as an invalid bounding box and triggers immediate failure of the segmentation validation gate.

---

## 4. Current Annotation & Training Gate Status

- **Images Registered in Manifest**: 100
- **Completed Manual Annotations**: **0 (PENDING)**
- **Pending Annotations**: **100**
- **Inter-Annotator Agreement**: `N/A` (Awaiting expert agronomist dual-annotation sessions)
- **Segmentation Model Training Gate**: **BLOCKED**
  ```text
  SEGMENTATION MODEL = BLOCKED
  REASON = REAL GROUND-TRUTH MASKS NOT YET ANNOTATED
  ```

In adherence to scientific integrity principles, **no synthetic polygon masks were generated from bounding boxes**, and no pre-trained segmentation models were claimed without real ground-truth evaluation.
