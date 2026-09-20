# Phase 2B Final Dataset Sanitization & Pre-Training Preparation Report

**Project Title**: A YOLOv8-Based Framework for Automated Coconut Tree Disease Detection and Plantation Health Monitoring Using Mobile Applications  
**Author**: Lead Research-Engineering Agent  
**Date**: 2026-09-20  
**Target Architecture**: Ultralytics YOLOv8 Object Detection  
**Status**: **DATASET SANITIZATION COMPLETE — CERTIFIED READY FOR PHASE 3**

---

## 1. Objective

The primary objective of Phase 2B was to construct a scientifically defensible, leakage-controlled, and reproducible dataset foundation for YOLOv8 object detection. Specifically, Phase 2B resolved the critical vulnerabilities identified during the Phase 2A audit:
1. Cross-partition data leakage caused by camera-burst capture sequences across train, validation, and test sets.
2. Complete absence of healthy/negative control samples, which would induce elevated false-positive alarms in mobile field deployments.
3. Lack of verified coordinate integrity, deterministic grouping splits, and provenance traceability.

---

## 2. Raw Dataset

The raw dataset evaluated in Phase 2A and ingested in Phase 2B resides at:
`data/raw/roboflow_coconut_detection/`

- **Total Ingested Images**: 100 images
- **Total Ingested Labels**: 100 label files
- **Total Ingested Bounding Boxes**: 120 annotations
- **Image Formats**: JPEG (`.jpg`)
- **Label Format**: Normalized YOLO text format (`class_id x_center y_center width height`)
- **Status of Raw Directory**: **Untouched and preserved in original state**. All sanitization was materialized into `data/processed/coconut_detection_clean/`.

---

## 3. Data Sources

Two authentic data sources constitute the sanitized dataset:
1. **Primary Disease Dataset**: Roboflow Universe *Coconut Tree Disease* by Phanidhar Reddy (CC BY 4.0). Originates from the 2021 Mendeley Data benchmark, providing 100 images across 5 pathological classes.
2. **Negative Control Dataset**: Wikimedia Commons (*Category:Cocos nucifera*) open-access imagery under CC BY-SA 4.0, CC BY 3.0, and CC0 / Public Domain licenses. Provides 50 verified authentic healthy coconut palm images.

All provenance, source URLs, author attribution, and cryptographic hashes are tracked in:
- `research/dataset_provenance.md`
- `research/healthy_image_provenance.csv`

---

## 4. Annotation Validation

A complete label integrity audit was executed via `scripts/verify_labels.py`.

- **Total Label Files Evaluated**: 100
- **Total Bounding Boxes Evaluated**: 120
- **Valid Bounding Boxes**: 120 (100.0%)
- **Invalid Coordinate Values**: 0
- **Out-of-Bounds Boxes**: 0
- **NaN / Infinite / Non-Numeric Values**: 0
- **Empty Lines / Malformed Rows**: 0
- **Class ID Mismatches**: 0 (all classes strictly $\in \{0, 1, 2, 3, 4\}$ matching `data.yaml`)
- **Orphan Labels / Missing Images**: 0

Machine-readable validation reports are stored at:
- `outputs/dataset_analysis/label_integrity_report.json`
- `outputs/dataset_analysis/label_integrity_report.csv`

---

## 5. Leakage Analysis

In Phase 2A, perceptual hash analysis revealed that camera burst captures of the same diseased palms were partitioned across `valid` and `test` in the original Roboflow distribution (e.g. `BudRootDropping018.jpg` and `BudRootDropping019.jpg` with dHash Hamming distance = 1). 

In Phase 2B, an exhaustive pairwise cross-partition leakage audit was conducted across:
- `train` $\leftrightarrow$ `val`
- `train` $\leftrightarrow$ `test`
- `val` $\leftrightarrow$ `test`

Using cryptographic SHA-256 and 64-bit difference hash (dHash) with a threshold of Hamming distance $\le 4$.

- **Exact Duplicate Collisions**: 0
- **Perceptual Near-Duplicate Collisions**: 0
- **Cross-Split Leakage Status**: **PASSED — ZERO CROSS-SPLIT LEAKAGE**

Full audit report: `outputs/dataset_analysis/final_leakage_report.json` and `research/final_leakage_audit.md`.

---

## 6. Grouping Strategy

To eliminate cross-split burst leakage deterministically, a Disjoint Set Union (DSU) clustering engine was implemented in `scripts/create_grouped_split.py`:
1. **Signal A (Perceptual Proximity)**: Any pair of images with dHash Hamming distance $\le 4$ is joined into the same equivalence cluster.
2. **Signal B (Sequential Burst Adjacency)**: Consecutive camera sequence frames ($|idx_1 - idx_2| = 1$) with dHash Hamming distance $\le 6$ are joined.

This clustered the 100 diseased images into **66 atomic leakage-controlled groups** (including 14 multi-image burst clusters, with the largest cluster containing 8 consecutive shots of a single tree). All images within any group are strictly assigned to the same partition.

---

## 7. Final Split

The 66 disease groups and 50 healthy negative groups were partitioned using a fixed random seed (`RANDOM_SEED = 42`) with class stratification:

| Split | Images | Image % | Diseased Images | Healthy Negatives | Bounding Boxes | BBox % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 100 | 66.67% | 63 | 37 | 78 | 65.0% |
| **Validation** | 35 | 23.33% | 27 | 8 | 30 | 25.0% |
| **Test** | 15 | 10.00% | 10 | 5 | 12 | 10.0% |
| **Total** | **150** | **100.0%** | **100** | **50** | **120** | **100.0%** |

---

## 8. Healthy / Negative Integration

In plantation health monitoring, object detection models must differentiate between healthy palm canopies and diseased lesions.
- **Negative Sample Representation**: 50 verified healthy palm images were assigned empty `.txt` annotation files.
- **YOLO Training Mechanics**: Ultralytics YOLOv8 treats empty label files as background negative samples, training the CNN backbone and detection head to penalize false-positive background detections without adding artificial bounding boxes.
- **Class Space Invariant**: No synthetic "healthy" bounding box was created. The class space remains strictly bounded to the 5 disease pathologies.
- **Visual Inspection**: Verified via contact sheet at `outputs/dataset_analysis/healthy_samples.png`.

---

## 9. Class Distribution

| Class ID | Disease Name | Train Imgs | Val Imgs | Test Imgs | Total Imgs | Train BBoxes | Val BBoxes | Test BBoxes | Total BBoxes | % Total BBoxes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | `bud root dropping` | 8 | 8 | 4 | 20 | 8 | 8 | 4 | 20 | 16.67% |
| **1** | `bud rot` | 14 | 5 | 1 | 20 | 14 | 5 | 1 | 20 | 16.67% |
| **2** | `gray leaf spot` | 15 | 3 | 2 | 20 | 30 | 6 | 4 | 40 | 33.33% |
| **3** | `leaf rot` | 15 | 3 | 2 | 20 | 15 | 3 | 2 | 20 | 16.67% |
| **4** | `stembleeding` | 11 | 8 | 1 | 20 | 11 | 8 | 1 | 20 | 16.67% |
| **-1** | *Healthy Negative (Background)* | 37 | 8 | 5 | 50 | 0 | 0 | 0 | 0 | 0.00% |

All 5 disease classes are represented across all three splits (`train`, `val`, `test`).

---

## 10. Image Quality

Image quality was audited across the 150 images in `outputs/dataset_analysis/final_image_quality.csv`:
- **Image Dimensions**: Widths range from 640 to 1024 px; heights range from 640 to 1024 px.
- **Aspect Ratios**: Range from 0.75 (portrait) to 1.33 (landscape); median 1.00.
- **Sharpness (Laplacian Variance)**: Range 19.3 to 3862.1; median 247.2. Zero images exhibit severe unrecoverable motion blur.
- **Luminance (Brightness)**: Range 40.2 to 216.5; median 119.5 (well-balanced natural illumination).
- **Decodability**: 100% of images decode without error via OpenCV and PIL.

Decisions documented in `research/data_quality_decisions.md`.

---

## 11. Excluded Data

- **Corrupted Files**: 0 files excluded (all 100 raw disease images and 50 negative images were valid).
- **Classification-Only Datasets**: Mendeley classification archive was excluded from object detection training because it lacks bounding boxes.
- **Synthetic Data**: Excluded completely per research integrity constraints.

---

## 12. Final Dataset Statistics

- **Total Images**: 150
- **Total Annotations**: 120
- **Average Bounding Boxes per Image**: 0.80 (1.20 among diseased images)
- **Median Bounding Boxes per Image**: 1.00
- **Minimum Boxes per Image**: 0 (healthy negatives)
- **Maximum Boxes per Image**: 4 (`gray leaf spot` multiple lesion clusters)
- **Images with Multiple Disease Classes**: 0 (all raw images display single pathology)
- **Majority Class**: `gray leaf spot` (40 bboxes, 33.33%)
- **Minority Classes**: `bud root dropping`, `bud rot`, `leaf rot`, `stembleeding` (20 bboxes each, 16.67%)
- **Class Imbalance Ratio**: $2.0:1$ (mild, well within acceptable bounds for YOLOv8 training)

---

## 13. Limitations

1. **Benchmark Scale**: The local sanitized dataset comprises 150 images (100 diseased + 50 healthy negatives). While robust for baseline proof-of-concept and pre-training pipeline validation, the full 3,229-image Roboflow archive should be ingested via user API key if large-scale production training is pursued.
2. **Single Disease per Image**: All raw images show isolated diseases; co-infection scenarios (e.g. palm with simultaneous bud rot and stem bleeding) are not represented in ground truth.
3. **Resolution Heterogeneity**: Roboflow images are standardized to $640 \times 640$, while Wikimedia images are native $768 \times 1024$ to $1024 \times 768$. YOLOv8 automatic letterbox resizing handles this during training.

---

## 14. Reproducibility

The entire Phase 2B pipeline is 100% deterministic and reproducible:
1. `scripts/verify_labels.py`: Validates raw label integrity.
2. `scripts/acquire_healthy_negatives.py`: Fetches and validates Wikimedia Commons CC-licensed healthy palm images.
3. `scripts/create_grouped_split.py`: Deterministically groups burst sequences and materializes `data/processed/coconut_detection_clean/` with seed 42.
4. `scripts/audit_post_split_leakage.py`: Confirms 0 cross-split leakage.
5. `scripts/analyze_clean_dataset.py`: Computes statistics, CSVs, and visualization figures.

---

## 15. Readiness for YOLOv8

The sanitized dataset satisfies all technical requirements for Ultralytics YOLOv8 training:
- Directory structure conforms to standard YOLO dataset schema (`train/images`, `train/labels`, `val/images`, `val/labels`, `test/images`, `test/labels`).
- Configuration file `data/processed/coconut_detection_clean/data.yaml` is prepared and verified.
- Bounding box coordinates are validated.
- Cross-split leakage is certified at 0.
- Healthy negative control samples are integrated.
- **Phase 3 YOLOv8 experimentation may proceed upon user approval.**
