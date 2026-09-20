# Phase 8 Longitudinal Dataset Candidates & Search Audit

## 1. Overview & Search Methodology

To determine whether repeated temporal observations of the same indexed coconut palms exist in the public scientific domain, comprehensive searches were conducted across:
1. Academic repositories (Google Scholar, ScienceDirect, IEEE Xplore, Mendeley Data)
2. Machine learning hubs (Kaggle, Roboflow Universe, Hugging Face)
3. Agricultural extension databases (ICAR-CPCRI, Coconut Research Institute of Sri Lanka)

Target criteria:
- **Specimen Tracking**: Persistent tree IDs or physical trunk tagging.
- **Revisit Timeline**: Multiple observation visits (Day 0, Day 7, Day 14, etc.) on the *same physical tree*.
- **Photographic Modality**: Canopy/trunk RGB imagery tracking disease expansion or remission.

---

## 2. Candidate Evaluation Matrix

| Candidate Source | Platform / Publication | Revisit Intervals | Specimen / Tree ID Available | Longitudinal Data Available | Suitable for LSTM/GRU Training | Reason & Finding |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Coconut Tree Disease Dataset (Patil et al., 2021)** | Mendeley Data | **NONE** | **NO** | **NO** | **NO** | Cross-sectional snapshot archive; photos gathered in single field sweeps; zero revisit tracking. |
| **Roboflow Coconut Tree Disease (`phanidhar-reddy`)** | Roboflow Universe | **NONE** (1-5s bursts) | **NO** | **NO** | **NO** | Camera burst photos taken within seconds of each other. Burst photos represent identical pathological moments, not longitudinal progression. |
| **Kaggle Coconut Leaf Disease Datasets** | Kaggle | **NONE** | **NO** | **NO** | **NO** | Isolated single-leaf cut specimens photographed on white/neutral backgrounds. Specimen destroyed during harvest. |
| **Canopy Remote Sensing (Landsat / Sentinel Time-Series)** | Academic Literature | 16-day orbit | Coarse Regional Pixel | **NO (LOW RESOLUTION)** | **NO** | Satellite multispectral data has $10\text{m} - 30\text{m}$ resolution; detects regional deforestation or wilt decline, but cannot resolve foliar lesions or trunk bleeding. |
| **UAV Multispectral Early Detection Studies** | ResearchGate / MDPI | Bi-weekly sweeps | Field plot centroid | **NO (DATA UNRELEASED)** | **NO** | Studies evaluate spectral reflectance indices (NDVI/NDRE), but raw longitudinal image sequences of individual trees are not publicly shared. |

---

## 3. Longitudinal Data Absence & Hard Training Gate

### Hard Decision Gate (Step 15):
```text
1. Real tree IDs available?            -> NO
2. Multiple observations/tree available? -> NO
3. Real temporal revisit dates available? -> NO
4. Valid severity target available?      -> NO
5. Tree-level train/val/test split?     -> NOT POSSIBLE
6. Sufficient sequences available?      -> NO (0 sequences)

GATE VERDICT:
LSTM TRAINING = BLOCKED
GRU TRAINING  = BLOCKED
```

### Absolute Scientific Integrity Declaration:
Under no circumstances will static photos be artificially sorted into pseudo-sequences or assigned fake dates. Fabricating longitudinal progressions violates fundamental research ethics. In accordance with the Master Prompt, **no LSTM or GRU models were trained on static data**.
