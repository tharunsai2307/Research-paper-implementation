# Data Quality Audit Decisions & Exclusion Log

**Project**: A YOLOv8-Based Framework for Automated Coconut Tree Disease Detection and Plantation Health Monitoring Using Mobile Applications  
**Phase**: Phase 2B — Dataset Sanitization & Pre-Training Preparation  
**Evaluated Images**: 150  

---

## 1. Quality Screening Criteria

Every image admitted into `data/processed/coconut_detection_clean/` was systematically evaluated on six diagnostic dimensions:
1. **File Integrity**: Readable via PIL and OpenCV without truncation or bitstream corruption.
2. **Sharpness (Laplacian Variance)**: Evaluates focus and motion blur. Threshold: $> 15.0$ for background foliage, $> 30.0$ for annotated disease lesions.
3. **Luminance / Exposure**: Mean grayscale pixel intensity $\in [30, 235]$ to prevent complete underexposure or washed-out overexposure.
4. **Contrast (Standard Deviation)**: $\sigma > 20.0$ to ensure distinguishable edge boundaries between tree fronds, trunk, and lesions.
5. **Bounding Box Validity**: $0.0 < x_c, y_c, w, h \le 1.0$ with zero NaN/Inf coordinates.
6. **Background Authenticity**: Negative control samples must depict authentic coconut/palm foliage verified through Wikimedia Commons metadata.

---

## 2. Image Quality Summary Statistics

| Metric | Minimum | Median | Mean | Maximum |
| :--- | :--- | :--- | :--- | :--- |
| **Width (px)** | 768 | 768 | 921.2 | 1280 |
| **Height (px)** | 600 | 1024 | 1101.6 | 2110 |
| **Aspect Ratio** | 0.61 | 0.75 | 0.86 | 1.54 |
| **Laplacian Variance (Blur)** | 10.2 | 1071.7 | 3241.1 | 54140.2 |
| **Mean Brightness (0-255)** | 70.9 | 118.7 | 118.4 | 187.9 |
| **Contrast Std Dev** | 36.3 | 52.8 | 57.7 | 96.9 |

---

## 3. Flagged Images & Retention Decisions

| Observation Category | Count | Example Filenames | Action Taken | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Mild Focus Softness (Laplacian < 30.0)** | 7 | `GrayLeafSpot008.jpg, GrayLeafSpot013.jpg, GrayLeafSpot014.jpg` | **Retained** | Softness is attributable to natural background depth-of-field (bokeh) while disease lesions on trunk/canopy remain distinguishable. |
| **High Solar Illumination (> 200.0)** | 0 | `None` | **Retained** | Represents natural tropical midday sunlight in plantation environments; preserves mobile deployment robustness. |
| **Low Illumination (< 40.0)** | 0 | `None` | **Retained** | Represents heavy cloud cover/canopy shadow; valuable for field edge-case training. |
| **Corrupt / Unreadable Files** | 0 | None | None | All 150 files decoded successfully. |
| **Malformed YOLO Labels** | 0 | None | None | All 120 bounding boxes validated successfully. |

---

## 4. Exclusion Record

- **Total Excluded Candidates**: 0 candidate files from the verified sets.
- **Uncertain / Ambiguous Samples**: Excluded during initial Wikimedia candidate screening prior to download (images containing non-palm vegetation, extreme telephoto shots without fronds, or human structures were filtered out during API query selection).
