# Research Decisions & Scientific Dataset Architecture

This document records the engineering, pathological, and computer vision decisions for **Phase 1** of the research project:
**"A YOLOv8-Based Framework for Automated Coconut Tree Disease Detection and Plantation Health Monitoring Using Mobile Applications"**.

---

## 1. Dataset Selected for Further Implementation

### Primary Detection Candidate: Coconut Tree Disease (Roboflow Universe - Phanidhar Reddy)
- **Name**: Coconut Tree Disease Dataset
- **Source**: Roboflow Universe (`phanidhar-reddy/coconut-tree-disease-vg85j`)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Images**: 2,122 images
- **Classes**: 5 classes (`bud_rot`, `bud_root_dropping`, `gray_leaf_spot`, `leaf_rot`, `stem_bleeding`)
- **Healthy class**: No explicit healthy palm class (background foliage is unannotated)
- **Detection annotations**: **YES** (Normalized YOLO bounding boxes in `.txt` format)
- **Why it is technically suitable**:
  1. It provides true bounding-box annotations that localize specific pathological lesions (e.g., fungal lesions on fronds, crown necrotic collapse, trunk bleeding exudates) rather than treating the entire palm as a single classification label.
  2. Classes directly align with the five high-impact economic coconut diseases identified in agronomic research (*Patil et al., 2023*).
  3. CC BY 4.0 licensing legally permits both academic research publication and commercial mobile application deployment with proper attribution.
  4. Native YOLO format (`class_id x_center y_center width height`) enables seamless ingestion into Ultralytics YOLOv8 pipelines without lossy geometric conversions.
- **Known limitations**:
  1. Absence of an explicit "healthy" class requires incorporating negative background samples to prevent false-positive alarms on healthy palms.
  2. Certain stem bleeding annotations encompass the entire lower trunk rather than fine lesion boundaries.
  3. Export requires Roboflow API authentication or manual web browser download.

### Benchmark & Annotation Source Pool: Coconut Tree Disease Dataset (Mendeley Data / Patil et al., 2023)
- **Name**: Coconut Tree Disease Dataset
- **Source**: Mendeley Data / *Data in Brief* (`DOI: 10.17632/gh56wbsnj5.1`, `DOI: 10.1016/j.dib.2023.109690`)
- **License**: CC BY 4.0
- **Images**: 5,798 images (100 verified samples audited in Phase 1)
- **Classes**: 5 classes (`Bud Root Dropping`, `Bud Rot`, `Gray Leaf Spot`, `Leaf Rot`, `Stem Bleeding`)
- **Healthy class**: No
- **Detection annotations**: **NO** (Folder-level classification only)
- **Why it is technically suitable**:
  1. Peer-reviewed academic provenance (*Data in Brief*, Vol. 51, 2023) ensures verified phytopathological credibility.
  2. Large sample size (5,798 images, ~1,100 per class) provides extensive natural field variability (lighting, tree age, canopy occlusion).
  3. Clean resolution uniformity (768 x 1024 JPEG, 0% corruption across audited samples).
- **Known limitations**:
  - **Classification-only**: Lacks bounding box annotations required for object detection. Cannot be used directly for YOLOv8 object detection without manual or AI-assisted bounding-box annotation.

---

## 2. Determination of Annotation Status

The candidate datasets were evaluated against the following classification taxonomy:
- **A. True object-detection dataset**: Roboflow Universe (`phanidhar-reddy/coconut-tree-disease-vg85j`, `driyog/coconut-tree-diseases`)
- **B. Classification-only dataset**: Mendeley Data (`gh56wbsnj5.1`), Kaggle (`jigarbaraiya`, `lakshanx`)
- **C. Segmentation dataset**: GitHub (`avishkakavindu/coconut-cinnamon-disease-detection-mask-rcnn`)

### Mandatory Scientific Rule on Ground Truth
> [!CAUTION]
> **Prohibition of Direct Label-to-Bounding-Box Conversion**:
> For classification-only datasets (e.g. Mendeley Data), converting `image → class label` directly into a naive full-image bounding box (`0.5 0.5 1.0 1.0`) is **strictly rejected**.
> Naive full-image bounding boxes force the YOLOv8 detector to correlate sky, soil, background weeds, and healthy palm fronds with disease classes. This corrupts the loss gradient and produces models that fail in the field.
> **Scientific Conclusion**: *"Manual or AI-assisted annotation is required before YOLOv8 object-detection training on classification datasets."*

---

## 3. Annotation Roadmap for Classification Datasets

If the 5,798-image Mendeley Data benchmark is expanded into the detection training pipeline, the following protocol will be executed:

### 1. Workload Estimation
- **Target volume**: Stratified subset of 2,500 images (500 images per class).
- **Lesion density**: Estimated 1 to 4 bounding boxes per image.
- **Total bounding box volume**: ~5,000 to 10,000 annotations.
- **Estimated human manual time**: ~40–50 annotator-hours at 60 images/hour.

### 2. Annotation Tooling
- **Primary Tool**: **CVAT (Computer Vision Annotation Tool)** or **Label Studio** (self-hosted open-source).
- **Format**: YOLOv8 Darknet/Ultralytics text format (`<class_id> <x_center> <y_center> <width> <height>` normalized to [0, 1]).

### 3. AI-Assisted Zero-Shot Pre-Annotation Workflow
To reduce manual labeling labor by ~70%:
1. **Zero-Shot Prompting**: Deploy a foundation vision model (Florence-2-large or Grounding DINO) with disease-specific text prompts:
   - *"brown necrotic rot on coconut bud"*
   - *"dark reddish-brown fluid exudation on coconut trunk"*
   - *"gray oval spots with brown margins on palm fronds"*
2. **Segmentation Refinement**: Feed candidate box prompts into Segment Anything Model 2 (SAM 2) to obtain exact lesion contours, then calculate bounding boxes (`min_x, min_y, max_x, max_y`).
3. **Human-in-the-Loop Quality Gate**: Domain experts inspect, adjust, and approve every generated bounding box. No unverified machine-generated box is admitted to the training set.

---

## 4. Alternative Datasets Evaluated

| Dataset Candidate | Primary Reasons for Deprioritization |
| :--- | :--- |
| **DRIYOG (Roboflow Universe)** | High frequency of synthetic augmented duplicates in export; class taxonomy merges pest damage (`caterpillar`) with disease symptoms (`lethal yellowing`). |
| **Kaggle (Jigar Baraiya)** | Classification only; small image count (~1,200 images); disease classes combine distinct nutritional and physiological stresses into a generic "Yellowing" class. Retained exclusively as an auxiliary pool for healthy negative control fronds. |
| **Kaggle (LakshanX)** | Classification only; focus is restricted to Weligama leaf wilt and coconut caterpillars (regional Sri Lankan pathogens); modest image count (~850 images). |
| **Avishka Kavindu (GitHub)** | Dataset not hosted in an archival repository; small sample size (~600 images); intermingles cinnamon and coconut crops. |

---

## 5. Critical Mitigation for Data Leakage (Burst Photos)

Perceptual hashing (dHash) conducted during Task 9 identified **49 near-duplicate image pairs** within sequential filenames (e.g. `BudRootDropping001.jpg` and `BudRootDropping002.jpg` with Hamming distance = 1).
- **Root Cause**: Field photographers captured continuous burst images of the same diseased palm.
- **Risk**: Standard random shuffling places burst frames of the same tree into both training and validation sets, yielding an over-optimistic evaluation and severe real-world performance degradation.
- **Engineering Decision**: When generating final splits in Phase 2, splitting must be **group-aware / tree-aware**, ensuring all burst images of a single tree or photographic sequence remain strictly within the same partition (train OR validation OR test).
