# Research Dataset Provenance & Attribution

**Project**: A YOLOv8-Based Framework for Automated Coconut Tree Disease Detection and Plantation Health Monitoring Using Mobile Applications  
**Phase**: Phase 2B — Dataset Sanitization & Pre-Training Preparation  
**Dataset Identifier**: `coconut_detection_clean` (v1.0.0-sanitized)  
**Date**: 2026-09-20  

---

## 1. Primary Dataset: Roboflow Coconut Tree Disease

| Parameter | Specification |
| :--- | :--- |
| **Dataset Title** | Coconut Tree Disease (v1) |
| **Platform** | Roboflow Universe |
| **Repository URL** | [https://universe.roboflow.com/phanidhar-reddy/coconut-tree-disease-vg85j](https://universe.roboflow.com/phanidhar-reddy/coconut-tree-disease-vg85j) |
| **Author / Contributor** | Phanidhar Reddy |
| **Originating Benchmark** | Mendeley Data (Coconut Tree Disease Dataset, 2021) |
| **License** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **Primary Purpose** | Ground-truth object detection annotations for coconut palm pathology |
| **Total Images Ingested** | 100 images (20 images per disease class) |
| **Total Annotations** | 120 verified bounding boxes in normalized YOLO format |
| **Classes Provided** | 5 disease classes |

### Exact Target Classes

1. `bud root dropping` (Class ID 0) — 20 bboxes
2. `bud rot` (Class ID 1) — 20 bboxes
3. `gray leaf spot` (Class ID 2) — 40 bboxes
4. `leaf rot` (Class ID 3) — 20 bboxes
5. `stembleeding` (Class ID 4) — 20 bboxes

### Filtering & Sanitization Applied
- All coordinate values verified: $x_c, y_c \in [0, 1]$, $w, h \in (0, 1]$.
- Zero malformed, NaN, infinite, or out-of-bounds annotations detected.
- Raw images and labels in `data/raw/roboflow_coconut_detection/` preserved untouched.
- Identified cross-split camera burst leakage present in the original Roboflow partition, resolved in Phase 2B using Disjoint Set Union (DSU) sequence grouping.

### Primary Dataset Limitations
- The full 3,229-image Roboflow archive includes extensive automated photometric and geometric augmentations; the local unaugmented benchmark comprises 100 images across the 5 classes.
- Contains only diseased palm images with zero healthy/negative control samples, risking high false-positive detection rates when deployed in healthy plantations.

---

## 2. Secondary Dataset: Healthy / Negative Control (Wikimedia Commons)

| Parameter | Specification |
| :--- | :--- |
| **Dataset Title** | Wikimedia Commons Coconut Palm Imagery (`Category:Cocos nucifera`) |
| **Platform** | Wikimedia Commons Open Access Repository |
| **Source URL** | [https://commons.wikimedia.org/wiki/Category:Cocos_nucifera](https://commons.wikimedia.org/wiki/Category:Cocos_nucifera) |
| **Creators / Authors** | Global contributing photographers (fully itemized with attribution in `research/healthy_image_provenance.csv`) |
| **Licenses** | Creative Commons (CC BY-SA 4.0, CC BY 3.0, CC0 / Public Domain) |
| **Primary Purpose** | Background negative control imagery to train YOLOv8 background rejection and prevent false-positive disease detections on healthy fronds/canopies |
| **Total Images Ingested** | 50 verified authentic coconut palm images |
| **Total Annotations** | 0 bounding boxes (represented by empty `.txt` annotation files) |

### Filtering & Verification Performed
1. **Vegetation Verification**: Queried specifically for *Cocos nucifera* trees, canopies, foliage, and plantations.
2. **Pathology Screening**: Screened out images displaying necrotic lesions, bleeding stems, fungal spotting, or bud rot symptoms matching the 5 target classes.
3. **Quality & Blur Filtering**: Filtered out unreadable files, extreme telephotos without foliage, human structures, and images with Laplacian sharpness variance $< 15.0$.
4. **Authenticity Verification**: Contact sheet compiled and visually reviewed (`outputs/dataset_analysis/healthy_samples.png`). Zero synthetic or AI-generated images used.
5. **Traceability**: Every image logged with cryptographic SHA-256 hash, Wikimedia page URL, author, and specific Creative Commons license in `research/healthy_image_provenance.csv`.

### Negative Control Dataset Limitations
- Ingested images originate from varied tropical geographies (e.g., India, Southeast Asia, Pacific Islands) with variable lighting and camera sensors; while this introduces valuable domain robustness, frond scale and camera angles vary more widely than the standardized tripod/handheld captures in the primary disease dataset.

---

## 3. Investigated Sources Excluded from Final Training Dataset

| Source Investigated | Status | Exclusion Rationale |
| :--- | :--- | :--- |
| **Mendeley Data Raw Classification Archive** | Investigated | Provides full-image classification labels without bounding-box coordinates. To preserve research integrity, pseudo-bounding boxes were strictly forbidden. |
| **Kaggle Palm Disease Datasets** | Investigated | Evaluated for palm disease imagery; contained overlapping re-uploads of the Mendeley dataset or unverified oil palm (*Elaeis guineensis*) images rather than *Cocos nucifera*. |
| **Synthetic / GenAI Coconut Images** | Investigated | Strictly rejected per Research Integrity Rules. Real-world validation requires genuine field imagery. |
