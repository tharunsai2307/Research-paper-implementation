# Roboflow Universe Dataset Audit: Coconut Tree Disease

This document records the first-principles investigation and technical audit of the Roboflow Universe dataset:
**`https://universe.roboflow.com/phanidhar-reddy/coconut-tree-disease-vg85j`**

Conducted via direct live browser inspection on **2026-09-20**.

---

## 1. Project & Origin Metadata

- **Project Title**: Coconut Tree Disease Computer Vision Model
- **Creator / Workspace**: Phanidhar Reddy (`phanidhar-reddy`)
- **Project URL**: [https://universe.roboflow.com/phanidhar-reddy/coconut-tree-disease-vg85j](https://universe.roboflow.com/phanidhar-reddy/coconut-tree-disease-vg85j)
- **Task Type**: Object Detection
- **Dataset Versions Available**: 1 version (`v1`, generated February 1, 2025 at 11:36 AM UTC)
- **Model Engine**: Roboflow 3.0 (Accurate)
- **Community Engagement**: 3 stars, active community model deployment endpoint

---

## 2. Image and Annotation Scale

- **Raw / Unique Uploaded Images**: 2,088 images
- **Dataset v1 Total Images (Post-Augmentation)**: 3,229 images
- **Annotation Modality**: 2D Object Detection bounding boxes and localized lesion boundary polygons.
- **Healthy Control Class**: **NONE**. The dataset strictly labels symptomatic lesions. Background tissue is unannotated.

---

## 3. Class Taxonomy & Naming Rigor

The dataset declares exactly **5 classes**:

| Class ID | Exact Class Name | Pathological Meaning |
| :---: | :--- | :--- |
| `0` | `bud root dropping` | Immature nut / button shed and crown drooping pathology |
| `1` | `bud rot` | Crown necrosis caused by *Phytophthora palmivora* |
| `2` | `gray leaf spot` | Leaf spot blight caused by *Pestalotiopsis palmarum* |
| `3` | `leaf rot` | Frond rot and flaccidity associated with fungal complex |
| `4` | `stembleeding` | Dark rust-red exudation caused by *Thielaviopsis paradoxa* |

> [!NOTE]
> **Class Naming Finding**:
> Class 4 is explicitly tokenized as **`stembleeding`** (single word without space or underscore) in the Roboflow project configuration, unlike the academic literature which writes "Stem Bleeding". This exact spelling must be preserved in model configuration to maintain label synchronization.

---

## 4. Split Distribution

The dataset v1 split configuration is structured as follows:

| Split | Percentage | Image Count | Purpose |
| :--- | :---: | :---: | :--- |
| **Train** | 82.3% | 2,658 | Core model parameter learning |
| **Validation** | 10.6% | 343 | Hyperparameter tuning & early stopping |
| **Test** | 7.1% | 228 | Unseen held-out benchmark evaluation |
| **Total** | **100%** | **3,229** | Complete dataset v1 export |

---

## 5. Preprocessing & Augmentation Pipeline

### Preprocessing Applied:
1. **Auto-Orient**: Strips EXIF orientation tags and normalizes pixel arrays upright.
2. **Resize**: Stretch to **640 x 640 pixels** (standard YOLO input dimensions).

### Augmentation Applied:
1. **Multiplication Factor**: **3x outputs per training example** (expanding raw training samples to 2,658 images).
2. **Geometric Transform**: **Horizontal Flip** (50% probability).
*Note: No photometric distortions (blur, noise, random cutout) were applied in v1.*

---

## 6. License & Legal Verification (Task 2)

- **Displayed License**: **Creative Commons Attribution 4.0 International (CC BY 4.0)**
- **License URL**: [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/)
- **Attribution Requirement**: Yes. You must give appropriate credit to Phanidhar Reddy, provide a link to the license, and indicate if changes were made.
- **Research Use Allowed**: **YES**. CC BY 4.0 explicitly permits academic and scientific research use.
- **Redistribution Allowed**: **YES**. Sharing, copying, and redistributing the material in any medium or format is permitted.
- **Commercial Use Allowed**: **YES**. Commercial adaptation and deployment in commercial mobile applications is permitted under CC BY 4.0 terms.
- **Restrictions**: You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

---

## 7. Download & Export Architecture (Task 3)

### Export Formats Available:
- Ultralytics YOLOv8 PyTorch (`.txt` labels + `data.yaml`)
- YOLOv11, YOLOv12, YOLO26, YOLOv9, YOLOv5, YOLOv7
- COCO JSON (`instances_train.json`)
- Pascal VOC XML
- TFRecord, PaliGemma, CreateML

### Authentication Access Requirement:
Direct HTTP downloading or API querying of Roboflow Universe without credentials is protected by Cloudflare and Roboflow account gates:
- A HEAD or GET request without credentials returns `HTTP 401 Unauthorized` or `HTTP 403 Forbidden`.
- Clicking the export button on the live web UI triggers the standard modal: *"Login or create a free account"*.
- **Official API Ingestion Snippet**:
  ```python
  from roboflow import Roboflow
  rf = Roboflow(api_key="<YOUR_API_KEY>")
  project = rf.workspace("phanidhar-reddy").project("coconut-tree-disease-vg85j")
  version = project.version(1)
  dataset = version.download("yolov8")
  ```
- **Manual Export Workflow**: Users can click *"Download as ZIP"* after signing in, and place the archive in `data/raw/roboflow_coconut_detection/`.
