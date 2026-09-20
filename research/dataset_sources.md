# Dataset Candidate Analysis & Sources Repository

This document establishes the scientific provenance, licensing, technical specifications, and YOLOv8 object detection suitability for public datasets related to coconut tree disease detection and plantation health monitoring.

---

## 1. Candidate Datasets

### Candidate 1: Coconut Tree Disease (Roboflow Universe - Phanidhar Reddy)

1. **Dataset name**: Coconut Tree Disease Dataset
2. **Original source**: Roboflow Universe (`phanidhar-reddy/coconut-tree-disease-vg85j`)
3. **Source URL**: `https://universe.roboflow.com/phanidhar-reddy/coconut-tree-disease-vg85j`
4. **Associated research paper, if any**: Not specified (cites coconut tree pathology literature; derived from field collections in India)
5. **Publication year**: 2023
6. **Number of images**: 2,122 images (split into Train: 1,485, Valid: 424, Test: 213 in exported versions)
7. **Number of classes**: 5 classes
8. **Exact class names**:
   - `bud_rot`
   - `bud_root_dropping`
   - `gray_leaf_spot`
   - `leaf_rot`
   - `stem_bleeding`
9. **Healthy class availability**: NO (diseased symptoms only; background regions represent healthy plant tissue but are not labeled as an explicit 'healthy' class)
10. **Object detection annotations available?**: YES
11. **Annotation format**: YOLOv8 PyTorch format (`<class_id> <x_center> <y_center> <width> <height>` normalized to [0, 1] in `.txt` files), with export options for COCO JSON and Pascal VOC XML
12. **Image format**: JPG
13. **Approximate image resolution/range**: 640 x 640 (standardized export) to 1024 x 768 (native capture)
14. **Geographic/source information if documented**: Maharashtra and southern agricultural zones, India
15. **Dataset license**: Creative Commons Attribution 4.0 International (CC BY 4.0)
16. **Whether commercial/research use is permitted according to the stated license**: YES (commercial and research use permitted with attribution)
17. **Download method**: Roboflow Web UI export ("Download as ZIP" in YOLOv8 format) or Roboflow Python API using an API key
18. **Dataset version/date**: Version 1 (August 2023)
19. **Known limitations**:
   - Lacks an explicit healthy control class (essential for minimizing false alarms in uninfected palms).
   - Some symptom labels (e.g., severe stem bleeding) exhibit single full-trunk bounding boxes that span high background areas.
   - Requires API authentication or browser export.
20. **Suitability for YOLOv8 detection**: HIGH (Direct native format compatibility; bounding boxes target anatomical pathology sites: canopy, crown bud, trunk stem).

---

### Candidate 2: Coconut Tree Diseases (Roboflow Universe - DRIYOG)

1. **Dataset name**: Coconut Tree Diseases
2. **Original source**: Roboflow Universe (`driyog/coconut-tree-diseases`)
3. **Source URL**: `https://universe.roboflow.com/driyog/coconut-tree-diseases`
4. **Associated research paper, if any**: Not specified
5. **Publication year**: 2023
6. **Number of images**: 3,148 images
7. **Number of classes**: 4 classes
8. **Exact class names**:
   - `caterpillar_infested_area`
   - `leaf_rot`
   - `lethal_yellowing`
   - `stem_bleeding`
9. **Healthy class availability**: NO
10. **Object detection annotations available?**: YES
11. **Annotation format**: YOLOv8 PyTorch (`.txt` bounding boxes), COCO JSON
12. **Image format**: JPG
13. **Approximate image resolution/range**: 640 x 640 (resized), variable source dimensions
14. **Geographic/source information if documented**: Not specified
15. **Dataset license**: Creative Commons Attribution 4.0 International (CC BY 4.0)
16. **Whether commercial/research use is permitted according to the stated license**: YES (commercial and research use permitted with attribution)
17. **Download method**: Roboflow Web UI / Roboflow API
18. **Dataset version/date**: Version 1 (2023)
19. **Known limitations**:
   - Contains augmented copies within exported versions which increases dataset size without adding natural visual diversity.
   - Classes mix pest infestation (`caterpillar_infested_area`) with systemic phytoplasma symptoms (`lethal_yellowing`).
   - No explicit healthy leaf/tree control group.
20. **Suitability for YOLOv8 detection**: MODERATE TO HIGH (Native YOLO bounding boxes, but augmentation duplicates must be audited to prevent split leakage).

---

### Candidate 3: Coconut Tree Disease Dataset (Mendeley Data / Data in Brief)

1. **Dataset name**: Coconut Tree Disease Dataset
2. **Original source**: Mendeley Data (Elsevier)
3. **Source URL**: `https://data.mendeley.com/datasets/gh56wbsnj5/1` (DOI: `10.17632/gh56wbsnj5.1`)
4. **Associated research paper, if any**: Sandip Thite, Yogesh Suryawanshi, Kailas Patil, Prawit Chumchu. "Coconut (Cocos nucifera) tree disease dataset: A dataset for disease detection and classification for machine learning applications", *Data in Brief*, Volume 51, December 2023, 109690. DOI: `10.1016/j.dib.2023.109690`
5. **Publication year**: 2023
6. **Number of images**: 5,798 images
7. **Number of classes**: 5 classes
8. **Exact class names**:
   - `Bud Root Dropping` (1,154 images)
   - `Bud Rot` (1,085 images)
   - `Gray Leaf Spot` (1,102 images)
   - `Leaf Rot` (1,215 images)
   - `Stem Bleeding` (1,242 images)
9. **Healthy class availability**: NO (contains only diseased palm images)
10. **Object detection annotations available?**: NO (Folder-level image classification only)
11. **Annotation format**: Directory structure hierarchy (`dataset/<class_name>/<image_name>.jpg`)
12. **Image format**: JPEG
13. **Approximate image resolution/range**: 768 x 1024 pixels
14. **Geographic/source information if documented**: Kendur region, Shirur Taluka, Pune District, Maharashtra, India. Captured across varied farm plots, lighting angles, and seasonal stages.
15. **Dataset license**: Creative Commons Attribution 4.0 International (CC BY 4.0)
16. **Whether commercial/research use is permitted according to the stated license**: YES (commercial and research use permitted with attribution)
17. **Download method**: Mendeley Data web interface direct download (`Coconut Tree Disease Dataset.zip`, 967 MB)
18. **Dataset version/date**: Version 1 (published August 1, 2023)
19. **Known limitations**:
   - **Classification-only**: Lacks bounding box annotations required for object detection.
   - If converted naively to full-image bounding boxes, models learn entire tree silhouettes rather than lesion-specific features.
   - No healthy palm control class.
20. **Suitability for YOLOv8 detection**: LOW FOR DIRECT TRAINING; EXCELLENT AS SOURCE POOL FOR ANNOTATION (Requires manual or AI-assisted bounding-box annotation before YOLOv8 detection training).

---

### Candidate 4: Coconut Leaf Disease Dataset (Kaggle - Jigar Baraiya)

1. **Dataset name**: Coconut Leaf Disease Dataset
2. **Original source**: Kaggle
3. **Source URL**: `https://www.kaggle.com/datasets/jigarbaraiya/coconut-leaf-disease-dataset`
4. **Associated research paper, if any**: Not specified
5. **Publication year**: 2023
6. **Number of images**: ~1,200 images
7. **Number of classes**: 4 classes
8. **Exact class names**:
   - `Healthy`
   - `Leaf Spot`
   - `Yellowing`
   - `Pest Damage`
9. **Healthy class availability**: YES
10. **Object detection annotations available?**: NO (Image classification only)
11. **Annotation format**: Directory-based folder structure
12. **Image format**: JPG
13. **Approximate image resolution/range**: 512 x 512 to 1920 x 1080 pixels
14. **Geographic/source information if documented**: Western India (Gujarat / Maharashtra plantations)
15. **Dataset license**: Apache License 2.0
16. **Whether commercial/research use is permitted according to the stated license**: YES (commercial and research use permitted under Apache 2.0)
17. **Download method**: Kaggle API (`kaggle datasets download -d jigarbaraiya/coconut-leaf-disease-dataset`) or Kaggle Web UI
18. **Dataset version/date**: Version 1 (2023)
19. **Known limitations**:
   - Classification only; no localized bounding boxes.
   - Relatively small image count per class (~250-350 images).
   - "Yellowing" class combines nutritional deficiency, drought stress, and lethal yellowing symptoms without pathological distinction.
20. **Suitability for YOLOv8 detection**: LOW DIRECT (Valuable primarily as a source of verified 'Healthy' control imagery for background negative training in YOLOv8).

---

### Candidate 5: Infected and Healthy Coconut Leaves Dataset (Kaggle - LakshanX)

1. **Dataset name**: Infected and Healthy Coconut leaves dataset
2. **Original source**: Kaggle
3. **Source URL**: `https://www.kaggle.com/datasets/lakshanx/infected-and-healthy-coconut-leaves`
4. **Associated research paper, if any**: Associated with Sri Lankan Coconut Research Institute (CRI) leaf health studies
5. **Publication year**: 2022
6. **Number of images**: ~850 images
7. **Number of classes**: 4 classes
8. **Exact class names**:
   - `Healthy`
   - `CCI_Caterpillars`
   - `WCLWD_Flaccidity` (Weligama Coconut Leaf Wilt Disease)
   - `WCLWD_Yellowing`
9. **Healthy class availability**: YES
10. **Object detection annotations available?**: NO (Classification only)
11. **Annotation format**: Directory-based folder structure
12. **Image format**: JPG
13. **Approximate image resolution/range**: 600 x 600 to 1280 x 720 pixels
14. **Geographic/source information if documented**: Southern Province, Sri Lanka (Weligama region)
15. **Dataset license**: CC0: Public Domain
16. **Whether commercial/research use is permitted according to the stated license**: YES (public domain dedication)
17. **Download method**: Kaggle API or Kaggle Web UI
18. **Dataset version/date**: Version 1 (2022)
19. **Known limitations**:
   - Focused heavily on regional Sri Lankan pathology (Weligama leaf wilt).
   - Classification only; no bounding box coordinates.
   - Moderate sample count.
20. **Suitability for YOLOv8 detection**: LOW DIRECT (Valuable for cross-regional validation and healthy control samples).

---

### Candidate 6: Coconut and Cinnamon Disease Detection (GitHub - Avishka Kavindu)

1. **Dataset name**: Coconut and Cinnamon Disease Detection Dataset
2. **Original source**: GitHub (`avishkakavindu/coconut-cinnamon-disease-detection-mask-rcnn`)
3. **Source URL**: `https://github.com/avishkakavindu/coconut-cinnamon-disease-detection-mask-rcnn`
4. **Associated research paper, if any**: Undergraduate thesis project (Sri Lanka)
5. **Publication year**: 2022
6. **Number of images**: ~600 images (coconut subset)
7. **Number of classes**: 3 coconut classes
8. **Exact class names**:
   - `black_spot`
   - `brown_blight`
   - `tip_burn`
9. **Healthy class availability**: NO
10. **Object detection annotations available?**: YES (Instance Segmentation masks / bounding polygons)
11. **Annotation format**: Mask R-CNN JSON format (polygon masks with bounding coordinates)
12. **Image format**: JPG
13. **Approximate image resolution/range**: 800 x 600 pixels
14. **Geographic/source information if documented**: Sri Lanka
15. **Dataset license**: MIT License (for codebase; dataset hosted externally by author)
16. **Whether commercial/research use is permitted according to the stated license**: YES
17. **Download method**: Git clone (code) / External drive link provided in repository notes
18. **Dataset version/date**: 2022
19. **Known limitations**:
   - Small sample size (~200 images per class).
   - Images are hosted on an external drive rather than a permanent academic archive.
   - Cinnamon and coconut images are intermixed in repository structure.
20. **Suitability for YOLOv8 detection**: MODERATE (Can be converted from polygon masks to YOLOv8 bounding boxes via `min_x, min_y, max_x, max_y`, but dataset size is small).

---

## 2. Objective Comparison Matrix

Criteria evaluation scale: `YES` | `NO` | `PARTIAL` | `UNKNOWN`

| Evaluation Criteria | Candidate 1 (Reddy - Roboflow) | Candidate 2 (DRIYOG - Roboflow) | Candidate 3 (Patil - Mendeley) | Candidate 4 (Baraiya - Kaggle) | Candidate 5 (LakshanX - Kaggle) | Candidate 6 (Kavindu - GitHub) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Object detection annotations** | **YES** | **YES** | **NO** | **NO** | **NO** | **YES** (Masks) |
| **Healthy class available** | **NO** | **NO** | **NO** | **YES** | **YES** | **NO** |
| **Disease class count** | 5 | 4 | 5 | 3 | 3 | 3 |
| **Image count adequacy (>1,500)** | **YES** (2,122) | **YES** (3,148) | **YES** (5,798) | **PARTIAL** (~1,200) | **NO** (~850) | **NO** (~600) |
| **Annotation quality / documentation** | **YES** | **PARTIAL** | **N/A** (None) | **N/A** (None) | **N/A** (None) | **PARTIAL** |
| **License clarity** | **YES** (CC BY 4.0) | **YES** (CC BY 4.0) | **YES** (CC BY 4.0) | **YES** (Apache 2.0) | **YES** (CC0) | **PARTIAL** |
| **Dataset accessibility** | **PARTIAL** (Auth/Key) | **PARTIAL** (Auth/Key) | **YES** (Open Web) | **PARTIAL** (Kaggle Auth)| **PARTIAL** (Kaggle Auth)| **PARTIAL** (Ext. Drive) |
| **Natural field conditions** | **YES** | **YES** | **YES** | **YES** | **YES** | **PARTIAL** |
| **Research relevance** | **YES** | **YES** | **YES** | **PARTIAL** | **PARTIAL** | **PARTIAL** |
| **Class balance** | **PARTIAL** | **PARTIAL** | **YES** (~1.1k/class)| **PARTIAL** | **PARTIAL** | **PARTIAL** |
| **Compatibility with YOLOv8** | **YES** | **YES** | **NO** (Needs annot)| **NO** (Needs annot) | **NO** (Needs annot) | **PARTIAL** (Convert) |

---

## 3. Prioritization Analysis

1. **Primary Object Detection Candidate**: **Candidate 1 (`phanidhar-reddy/coconut-tree-disease-vg85j`)**
   - It is the most technically mature dataset with native YOLOv8 bounding-box annotations targeting the 5 major commercial coconut diseases (*Bud Rot, Bud Root Dropping, Gray Leaf Spot, Leaf Rot, Stem Bleeding*).
   - Under CC BY 4.0, it permits open academic research and commercial reproduction.
   
2. **Primary Benchmark Reference & Annotation Pool**: **Candidate 3 (`Patil et al. / Mendeley Data 10.17632/gh56wbsnj5.1`)**
   - It is the gold standard peer-reviewed academic publication (*Data in Brief*, 2023) containing 5,798 real-world plantation images.
   - Because it is classification-only, it serves as the ground-truth benchmark pool for semi-automated AI-assisted annotation in Phase 2.

3. **Auxiliary Negative / Healthy Control Pool**: **Candidate 4 (`jigarbaraiya/coconut-leaf-disease-dataset`) & Candidate 5 (`lakshanx/infected-and-healthy-coconut-leaves`)**
   - Both candidates provide real healthy coconut leaves (under Apache 2.0 and CC0). In YOLOv8 object detection, healthy images should be introduced as **background negative images** (images without label `.txt` files or empty `.txt` files) to suppress false positive background triggers.
