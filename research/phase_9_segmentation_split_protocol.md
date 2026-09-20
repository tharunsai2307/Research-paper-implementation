# Phase 9 Segmentation Split Protocol & Leakage Prevention

## 1. Overview & Protocol Objectives

To prepare the 100 benchmark disease images for manual polygon annotation and prospective segmentation model benchmarking, this protocol defines a strictly leakage-free partitioning scheme.

---

## 2. Partitioning Strategy & Disjoint Cluster Preservation

In Phase 2B (`research/final_leakage_audit.md`), the 100 diseased images were analyzed using 64-bit Difference Hashing (dHash) and Disjoint Set Union (DSU) sequence grouping to identify camera burst clusters. 

To maintain **100% scientific consistency with Phase 2B, Phase 3, and Phase 4**, the segmentation dataset split adopts the exact identical partition assignments:

```
Total Diseased Images: 100
├── TRAIN Split:       63 images (63%)
├── VALIDATION Split:  27 images (27%)
└── TEST Split:        10 images (10%)
```

### Class Distribution Across Splits

| Disease Class | Train Count | Validation Count | Test Count | Total Count |
| :--- | :---: | :---: | :---: | :---: |
| `bud root dropping` | 13 | 4 | 3 | 20 |
| `bud rot` | 13 | 5 | 2 | 20 |
| `gray leaf spot` | 13 | 5 | 2 | 20 |
| `leaf rot` | 12 | 7 | 1 | 20 |
| `stembleeding` | 12 | 6 | 2 | 20 |
| **Total Images** | **63** | **27** | **10** | **100** |

---

## 3. Strict Boundary Rules

1. **Test Set Inviolability**:
   - The 10 test images (`BudRootDropping017.jpg` - `019.jpg`, `BudRot019.jpg` - `020.jpg`, `GrayLeafSpot019.jpg` - `020.jpg`, `LeafRot005.jpg`, `StemBleeding019.jpg` - `020.jpg`) must remain completely untouched during any hyperparameter exploration.
2. **Zero Synthetic Masks**:
   - Polygons must represent visible tissue lesions delineated by human annotators.
   - Bounding boxes must **NEVER** be inflated, rasterized, or converted into pseudo-masks.
3. **Training Gate**:
   - Segmentation model training (e.g. `YOLOv8n-seg`) is **BLOCKED** until genuine human polygon annotations are delivered and validated by the automated QC script.
