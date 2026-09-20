# Phase 8 Segmentation Dataset Candidates & Verification Audit

## 1. Overview & Search Methodology

To determine whether genuine pixel masks or polygon annotations exist for coconut tree disease segmentation, systematic inquiries were conducted across academic repositories, open datasets, and computer vision platforms:
1. Mendeley Data
2. Roboflow Universe
3. Kaggle
4. GitHub Academic Repositories
5. Zenodo / OpenAIRE

---

## 2. Candidate Evaluation Matrix

| Dataset Candidate | Source Platform | Source URL | Claimed Task | Real Masks / Polygons | License Status | Suitable for Segmentation Training | Reason & Empirical Finding |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Coconut Tree Disease Dataset (Patil et al.)** | Mendeley Data | [gh56wbsnj5/1](https://data.mendeley.com/datasets/gh56wbsnj5/1) | Classification | **NO** | CC BY 4.0 (Verified) | **NO** | 5,798 images arranged in class subfolders; zero bounding boxes or polygon annotations included in raw release. |
| **Coconut Tree Disease (Phanidhar Reddy)** | Roboflow Universe | [coconut-tree-disease-vg85j](https://universe.roboflow.com/phanidhar-reddy/coconut-tree-disease-vg85j) | Object Detection | **NO** | CC BY 4.0 (Verified) | **NO** | Contains 120 verified bounding boxes (`xc, yc, w, h`). Exactly 0 polygon lines or binary masks exist. |
| **Coconut Tree Diseases (DRIYOG)** | Roboflow Universe | `driyog/coconut-tree-diseases` | Object Detection | **NO** | CC BY 4.0 | **NO** | Bounding box duplicate/derivative of Patil et al.; no segmentation masks. |
| **Coconut-Cinnamon Disease Detection (Avishka Kavindu)** | GitHub | [avishkakavindu/coconut-cinnamon](https://github.com/avishkakavindu/coconut-cinnamon-disease-detection-mask-rcnn) | Mask R-CNN Code | **NO (DATA UNHOSTED)** | MIT (Code only) | **NO** | Repository provides Mask R-CNN inference scripts, but the underlying annotated image/polygon dataset was not published due to hosting limits. |
| **On-tree mature coconut fruit detection** | Roboflow Universe | `college/on-tree-mature-coconut` | Instance Segmentation | **YES (FRUIT ONLY)** | CC BY 4.0 | **NO** | Labels healthy coconut fruit boundaries for harvesting robots; contains zero foliar/trunk disease pathology. |
| **Coconut Semantic Segmentation (Mohit Khubele)** | Roboflow Universe | `mohit-khubele/coconut-xo3qr` | Semantic Segmentation | **YES (PALM MASK)** | Public Domain / CC0 | **NO** | Segments palm trees vs background terrain; zero lesion or disease-region masks. |

---

## 3. License & Provenance Verification

For the primary benchmark dataset from which our object detection model derives:
- **Dataset Title**: Coconut Tree Disease Dataset
- **Primary Authors**: Patil et al. (2021/2023)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Attribution Requirements**: Mandatory academic citation of original DOI (`10.17632/gh56wbsnj5.1`) and Data in Brief publication.
- **Commercial-Use Restriction**: None (permitted with attribution).
- **Redistribution**: Permitted under CC BY 4.0 terms.
- **Segmentation Capability**: **NONE**. The dataset provides raw JPEG photos without segmentation annotations.

---

## 4. Scientific Verdict on Public Segmentation Data

1. **Definitive Conclusion**: There is currently **NO publicly available, verified dataset** providing pixel masks or polygon annotations for the 5 target coconut diseases (`bud root dropping`, `bud rot`, `gray leaf spot`, `leaf rot`, `stembleeding`).
2. **Research Posture**:
   - In strict compliance with Step 7 of the Master Prompt, **bounding boxes must NOT be converted into rectangular masks and presented as manual segmentation ground truth**.
   - Segmentation model training must remain formally:
     ```text
     SEGMENTATION MODEL TRAINING = BLOCKED (PENDING REAL MASK DATA)
     ```
   - Software interfaces and fallback area proxies established in Phase 7 are preserved.
