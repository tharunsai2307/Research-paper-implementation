# Post-Split Cross-Partition Leakage Audit Report

**Project**: A YOLOv8-Based Framework for Automated Coconut Tree Disease Detection and Plantation Health Monitoring Using Mobile Applications  
**Phase**: Phase 2B — Dataset Sanitization & Pre-Training Preparation  
**Evaluated Directory**: `C:\Users\Hp\.gemini\antigravity-ide\scratch\coconut-disease-yolov8\data\processed\coconut_detection_clean`  
**Status**: **PASSED - ZERO CROSS-SPLIT LEAKAGE**

---

## 1. Audit Methodology

To guarantee scientific defensibility and eliminate data leakage prior to YOLOv8 model training, every image across `train`, `val`, and `test` partitions was evaluated against every other image across partition boundaries using two orthogonal detection methods:

1. **Exact Duplicate Detection**:
   - Algorithm: Cryptographic SHA-256 hash
   - Criterion: Exact byte-level collision ($H(x) = H(y)$)
2. **Perceptual Near-Duplicate Detection**:
   - Algorithm: 64-bit Difference Hash (dHash) computed across luminance gradients ($8 \times 8$ matrix)
   - Criterion: Hamming Distance $\le 4$ bits (indicating camera burst shots, sequential video frames, or identical tree scenes under near-identical viewpoints)

---

## 2. Partition Summary

| Partition | Total Images | Verified Empty Labels (Healthy) | Annotated Disease Images |
| :--- | :--- | :--- | :--- |
| **Train** | 100 | 37 | 63 |
| **Validation** | 35 | 8 | 27 |
| **Test** | 15 | 5 | 10 |
| **Total** | **150** | **50** | **100** |

---

## 3. Cross-Split Leakage Results

| Partition Boundary | Exact Collisions (SHA-256) | Perceptual Near-Duplicates (dHash $\le 4$) | Leakage Status |
| :--- | :--- | :--- | :--- |
| **Train $\leftrightarrow$ Validation** | 0 | 0 | **CLEAN** |
| **Train $\leftrightarrow$ Test** | 0 | 0 | **CLEAN** |
| **Validation $\leftrightarrow$ Test** | 0 | 0 | **CLEAN** |
| **Total Cross-Split** | **0** | **0** | **PASSED - ZERO CROSS-SPLIT LEAKAGE** |

---

## 4. Resolving the Phase 2A Leakage Finding

In Phase 2A, the raw Roboflow Universe dataset exhibited camera-burst leakage where near-identical consecutive frames (e.g. `BudRootDropping018.jpg` in `valid` and `BudRootDropping019.jpg` in `test`, Hamming distance = 1) were distributed across partition boundaries, causing inflated validation performance.

In Phase 2B, Disjoint Set Union (DSU) clustering with perceptual dHash thresholds and sequential adjacency clustering successfully grouped burst sequences into unified atomic entities before stratified partitioning. Consequently, all images of each burst remain exclusively within their assigned partition.

- **Within-Train Burst Clusters**: 8 near-duplicate pairs preserved within `train` for intra-class variation.
- **Within-Validation Burst Clusters**: 38 pairs preserved within `val`.
- **Within-Test Burst Clusters**: 3 pairs preserved within `test`.
- **Cross-Split Boundary Violations**: **0 (Zero)**.

---

## 5. Conclusion & Verification

The sanitized dataset in `data/processed/coconut_detection_clean/` satisfies all cross-split leakage constraints and is certified clean of near-duplicate and exact-duplicate train/validation/test contamination.
