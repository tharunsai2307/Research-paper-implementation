# Phase 8 Specimen-Level Temporal Splitting Protocol

## 1. The Core Scientific Principle: Zero Tree Leakage

In longitudinal machine learning, the single most destructive methodological error is **Random Frame-Level Splitting**.

When images from the same tree taken on different days are split randomly across training and testing sets:
```text
Day 0 (Tree 001) -> TRAIN
Day 7 (Tree 001) -> TEST  <--- CATASTROPHIC TEMPORAL LEAKAGE!
```

### Why Random Splitting Causes Scientific Invalidation:
1. **Background Identity Leakage**: Deep neural networks easily memorize unique orchard background patterns (adjacent trees, distinctive trunk scars, specific soil contours, static lighting).
2. **False Generalization**: The model predicts future disease severity not by learning true biological pathogen expansion, but by recognizing the palm tree's identity and retrieving its memorized state.
3. **Severe Overoptimism**: Evaluation metrics appear stellar on test sets ($R^2 > 0.95$), but completely collapse when deployed on unseen trees in a new plantation block.

---

## 2. Mandatory Specimen-Grouped Partitioning (GroupKFold by `tree_id`)

All future longitudinal datasets must be partitioned strictly by **SPECIMEN IDENTIFIER (`tree_id`)**:

```
+--------------------------------------------------------------------------+
|                  ALL TRACKED PALM SPECIMENS (N Trees)                    |
+--------------------------------------------------------------------------+
                                     |
         +---------------------------+---------------------------+
         |                                                       |
         v                                                       v
+-----------------------------------+   +----------------------------------+
|      TRAIN SPECIMENS (70%)        |   |      TEST SPECIMENS (15%)        |
|  - All observations for T001-T070 |   |  - All observations for T086-T100|
|  - Complete longitudinal arcs     |   |  - Unseen palm trees ONLY        |
+-----------------------------------+   +----------------------------------+
                 |
                 v
+-----------------------------------+
|    VALIDATION SPECIMENS (15%)     |
|  - All observations for T071-T085 |
|  - Unseen palm trees ONLY        |
+-----------------------------------+
```

### Mathematical Guarantee:
$$\text{Trees}_{\text{Train}} \cap \text{Trees}_{\text{Val}} \cap \text{Trees}_{\text{Test}} = \emptyset$$
$$\bigcup \text{Observations}(T \in \text{Partition}) \subseteq \text{Partition}$$

Every observation visit (Day 0, 7, 14, 21, 28) for an individual tree belongs **exclusively** to a single split partition.

---

## 3. Stratification Across Disease Classes

To ensure balanced class representation across splits:
1. Group specimens by primary diagnosed disease class.
2. Apply `StratifiedGroupKFold` using `primary_disease` as the stratifying label and `tree_id` as the grouping group.
3. Verify that zero specimen IDs cross partition boundaries via automated cryptographic audit before training initiation.
