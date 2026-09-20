# Final Dataset Recommendation & Engineering Roadmap: Phase 2A

**Project Title**: A YOLOv8-Based Framework for Automated Coconut Tree Disease Detection and Plantation Health Monitoring Using Mobile Applications  
**Phase Completed**: Phase 2A — YOLO Detection Dataset Investigation  

---

## 1. Evidence-Based Research Decision

### PRIMARY CANDIDATE:
**Coconut Tree Disease Computer Vision Model (Roboflow Universe - Phanidhar Reddy)**  
*Project Slug: `coconut-tree-disease-vg85j` (v1)*

#### REASON (Objective Evidence):
1. **True Object Detection Ground Truth**: The dataset contains genuine, manually delineated bounding boxes and lesion boundary contours across 2,088 raw images (expanded to 3,229 in v1) rather than naive full-image labels. The mean normalized bounding box area is **0.1329** (~13.3% of the image area), confirming specific symptom localization.
2. **Pathological Domain Alignment**: The 5 annotated classes (`bud root dropping`, `bud rot`, `gray leaf spot`, `leaf rot`, `stembleeding`) represent the primary commercial disease threats documented in peer-reviewed coconut pathology literature (*Patil et al., 2023*).
3. **Native Ultralytics YOLOv8 Compatibility**: Annotations are structured in native normalized YOLO format (`.txt` files with class indices 0–4 and normalized coordinates) accompanied by a standardized `data.yaml` defining `train`, `valid`, and `test` splits.
4. **Permissive Licensing**: Verified under **Creative Commons Attribution 4.0 International (CC BY 4.0)**, which legally permits academic research publication, code release, model weights distribution, and commercial mobile application deployment with proper author attribution.
5. **Image Quality**: Auditing confirmed 0% corrupted files, clean 640x640 resolution standardization, normal exposure distributions, and minimal blurriness (mean Laplacian variance > 1,000 across surveyed field images).

#### LIMITATIONS:
1. **Absence of an Explicit Healthy Control Class**: Neither leaves nor crowns without disease symptoms are labeled. Training exclusively on this dataset risks high false-positive rates when the mobile application scans healthy trees in the field.
2. **Burst Capture Near-Duplicate Leakage**: dHash perceptual hashing identified that sequential field captures of identical infected palms leak across default validation/test splits (e.g., `BudRootDropping018.jpg` in validation vs. `BudRootDropping019.jpg` in test, Hamming distance = 1).
3. **Synthetic Expansion in v1**: The 3,229-image count in v1 includes 3x horizontal flip augmentations on the training split, which must be accounted for during training configuration.

---

### SECONDARY / BACKUP DATASET:
**Coconut Tree Disease Dataset (Mendeley Data / Patil et al., 2023)**  
*DOI: `10.17632/gh56wbsnj5.1` / Data in Brief, Vol. 51, 2023, 109690*

#### ANNOTATION REQUIRED:
**YES**.  
- **Limitation**: The Mendeley dataset provides 5,798 high-resolution images (768 x 1024), but is **strictly classification-only** (folder-level labels). Converting these images directly into full-frame bounding boxes (`0.5 0.5 1.0 1.0`) is scientifically invalid for object detection because it forces the model to treat the entire palm and background environment as diseased.
- **Role**: Mendeley Data is retained as an extensive out-of-domain validation reservoir and the source pool for Phase 2B semi-automated zero-shot annotation.

---

## 2. Mandatory Engineering Actions Before Final Model Training

To ensure research validity before executing YOLOv8 model training in Phase 2B:
1. **Sequence-Aware Split Decontamination**: Re-partition the Roboflow dataset using **group-aware sequence splitting** to ensure all burst captures of an individual palm remain strictly within a single partition (train OR validation OR test).
2. **Negative Background Frond Ingestion**: Introduce 150–200 verified healthy coconut palm images (without bounding box labels, i.e., empty `.txt` files) from open public domain collections (e.g. Kaggle CC0 healthy fronds) into the training split. This trains YOLOv8's background class suppression, critical for mobile application deployment.
3. **Class Token Preserving**: Preserve the exact class token `stembleeding` (unspaced) to ensure compatibility with the existing Roboflow label files.
