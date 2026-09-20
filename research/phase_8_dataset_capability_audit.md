# Phase 8 Dataset Capability & Provenance Audit

## 1. Overview & Scope

Phase 8 conducts a rigorous, comprehensive audit across all data assets within `data/` and `datasets/`. In strict accordance with the Master Prompt, any missing or unverifiable attributes are explicitly marked **`UNKNOWN`** rather than inferred or assumed.

---

## 2. Quantitative Data Asset Inventory

| Directory Location | Image Count | Label / Annotation Files | Verified Bounding Boxes | Polygons / Pixel Masks |
| :--- | :---: | :---: | :---: | :---: |
| `data/processed/coconut_detection_clean/train/` | 100 (63 diseased, 37 healthy) | 100 | 76 | 0 |
| `data/processed/coconut_detection_clean/val/` | 35 (27 diseased, 8 healthy) | 35 | 32 | 0 |
| `data/processed/coconut_detection_clean/test/` | 15 (10 diseased, 5 healthy) | 15 | 12 | 0 |
| `data/raw/roboflow_coconut_detection/train/` | 70 | 70 | 84 | 0 |
| `data/raw/roboflow_coconut_detection/valid/` | 20 | 20 | 24 | 0 |
| `data/raw/roboflow_coconut_detection/test/` | 10 | 10 | 12 | 0 |
| `data/raw/mendeley_coconut_disease/` (5 classes) | 100 (20/class) | 0 (Classification only) | 0 | 0 |
| `data/raw/healthy_coconut_negatives/` | 50 (Wikimedia Commons) | 0 (Empty controls) | 0 | 0 |
| **Total Images in Project** | **400** | **250** | **240** (Raw + Sanitized) | **0** |

---

## 3. Mandatory 20-Dimension Capability Audit Table

| Dimension | Audit Finding | Evidence & Source | Suitable for Research Training |
| :--- | :--- | :--- | :---: |
| **1. Number of images** | 150 sanitized benchmark images (100 diseased, 50 healthy); 400 total raw/archive images. | File system count across `data/processed/` and `data/raw/`. | **YES** (Object detection) |
| **2. Number of annotation files** | 250 text annotation files (150 in `processed`, 100 in `raw/roboflow`). | Scanned with `os.walk('data')`. | **YES** |
| **3. Annotation formats** | Normalized YOLO bounding box (`<class_id> <xc> <yc> <w> <h>`). | 100% of non-empty lines contain exactly 5 space-delimited floating-point tokens. | **YES** |
| **4. Bounding boxes** | 120 unique boxes across the 100 diseased benchmark images. | Parsed and verified in Phase 2B. | **YES** |
| **5. Polygon annotations** | **0** | 0 label lines with $> 5$ coordinate tokens. | **NO** |
| **6. Pixel masks** | **0** | Zero bitmap masks, RLE encodings, or `.png`/`.npy` mask assets exist. | **NO** |
| **7. Disease classes** | 5 classes: `bud root dropping`, `bud rot`, `gray leaf spot`, `leaf rot`, `stembleeding`. | `data.yaml` and Patil et al. ontology. | **YES** |
| **8. Healthy samples** | 50 verified negative control images from Wikimedia Commons. | Provenance recorded in `research/healthy_image_provenance.csv`. | **YES** (Background rejection) |
| **9. Tree identifiers** | **UNKNOWN** | Image filenames are class-sequential (e.g., `LeafRot007.jpg`). No physical tree tag, RFID, trunk number, or tree ID exists. | **NO** |
| **10. Dates** | **PARTIAL / UNKNOWN** | 47 images have `DateTime = None`. 103 images contain EXIF dates from only 2 days: June 30, 2023, and July 9, 2023. 3 images have legacy camera clock errors (2006/2014). | **NO** (Not longitudinal) |
| **11. Capture timestamps** | **PARTIAL** | Timestamps in June 30 / July 9 sessions differ by 1–5 seconds (camera burst jitter during walking sweeps). | **NO** (Burst photos represent identical pathological moments) |
| **12. Geographic/site identifiers** | **UNKNOWN** | No GPS EXIF tags, farm block coordinates, or orchard identifiers are embedded in the image files. | **NO** |
| **13. Repeated observations** | **NO** | Zero longitudinal revisit campaigns (e.g., Day 0, Day 7, Day 14, Day 28) exist for any indexed tree. | **NO** |
| **14. Existing severity labels** | **NO** | No expert phytopathological severity grading (e.g., percentage defoliation or Horsfall-Barratt scale) exists. | **NO** |
| **15. Existing segmentation labels**| **NO** | Zero pixel-level or contour annotations exist. | **NO** |
| **16. Existing progression labels** | **NO** | Zero transition labels (e.g., stable, expanding, necrotic) exist. | **NO** |
| **17. Dataset licenses** | CC BY 4.0 (Roboflow / Mendeley); CC BY-SA / CC0 (Wikimedia). | Verified in `research/dataset_provenance.md`. | **YES** |
| **18. Provenance** | Traced to Patil et al. (2021/2023) benchmark via Roboflow Universe (`phanidhar-reddy`). | Verified in `research/dataset_sources.md`. | **YES** |
| **19. Duplicate/near-duplicate risk**| High raw burst redundancy (resolved in Phase 2B via DSU sequence grouping). | Verified in `research/final_leakage_audit.md`. | **YES** (Sanitized split is leakage-free) |
| **20. Same physical tree identified across observations** | **NO** | Photos within bursts show the same tree seconds apart, but there is zero linkage between the June 30 and July 9 sessions. | **NO** |

---

## 4. Empirical Capability Summary

The current benchmark dataset (`coconut_detection_clean`) is exceptionally well-suited for **YOLOv8 Object Detection** and **2D Image-Space Bounding Box Area Proxy Modeling**. 

However, it **CANNOT** support:
1. Supervised pixel-level segmentation training (zero ground-truth masks).
2. Longitudinal disease progression modeling via recurrent neural networks (zero multi-week repeat observations of indexed palms).
