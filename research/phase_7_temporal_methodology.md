# Phase 7 Temporal Progression & Alert Methodology

## 1. Overview & Research Posture

Temporal disease progression modeling aims to project future phytopathological severity and trigger proactive grower interventions. 

As established in the Phase 7 Data Capability Audit (`research/phase_7_data_capability_audit.md`) and Temporal Data Gap Report (`research/phase_7_temporal_data_gap.md`):
- **Real Longitudinal Data**: `NOT AVAILABLE` in existing public coconut disease archives.
- **Deep Recurrent Training (LSTM / GRU)**: `NOT TRAINED` (to avoid scientific fraud via pseudo-temporal synthesis).

To satisfy Steps 8–14 while maintaining complete research integrity, Phase 7 establishes:
1. **Mathematical Feature Engineering Specifications** for multi-visit palm inspections.
2. **Heuristic Baseline Formulation (Last-Observation-Carried-Forward & Empirical Linear Slope)**.
3. **Formal LSTM / GRU Model Architectures & PyTorch Contracts** ready for immediate training once longitudinal field campaigns conclude.
4. **Engineering Progression Alert Engine** implementing transparent decision rules for monitoring and intervention triage.

---

## 2. Temporal Feature Engineering Formulation

For an indexed coconut palm $i$ with $K$ historical inspection visits at timestamps $t_1 < t_2 < \dots < t_K$:
At each observation step $k$, the state vector $\mathbf{x}_k$ is defined by:

$$\mathbf{x}_k = \begin{bmatrix}
S_k \\
\Delta S_k \\
g_k \\
\Delta t_k \\
c_k \\
\mathbf{d}_k
\end{bmatrix}$$

Where:
- $S_k \in [0.0, 1.0]$: Image-space disease severity estimate (Phase 5 affected area proxy or future segmentation mask area ratio).
- $\Delta S_k = S_k - S_{k-1}$: Absolute change in severity since preceding observation ($\Delta S_1 = 0$).
- $\Delta t_k = t_k - t_{k-1}$: Time delta in days since previous observation ($\Delta t_1 = 0$).
- $g_k = \frac{\Delta S_k}{\max(\Delta t_k, 1.0)}$: Daily severity growth velocity.
- $c_k \in [0.0, 1.0]$: Mean detection confidence of accepted lesions.
- $\mathbf{d}_k \in \{0, 1\}^5$: One-hot encoded primary disease class among the 5 target classes.

---

## 3. Modeling Architectures: Baseline vs. Deep Recurrent Networks

### 3.1 Naive Baseline: Last Observation Carried Forward (LOCF)
Before any deep sequence model can be accepted in agricultural research, it must demonstrate statistically superior predictive accuracy over the standard persistence baseline:
$$\hat{S}_{k+h}^{\text{LOCF}} = S_k$$
Under the LOCF baseline, the future severity at horizon $h$ is assumed to equal the currently observed severity.

### 3.2 Dynamic Linear Trend Baseline (Slope Projection)
$$\hat{S}_{k+h}^{\text{Linear}} = \min\left(1.0, \max\left(0.0, S_k + g_k \times h\right)\right)$$
Projects the most recent daily growth rate forward by $h$ days, bounded within $[0.0, 1.0]$.

### 3.3 Deep Sequence Architectures: LSTM and GRU
For a sequential input tensor $\mathbf{X} \in \mathbb{R}^{B \times L \times D}$ where $B$ is batch size, $L$ is sequence length ($L \ge 3$), and $D = 10$ is feature dimension:

1. **Long Short-Term Memory (LSTM)**:
   $$\begin{aligned}
   \mathbf{f}_t &= \sigma(\mathbf{W}_f \mathbf{x}_t + \mathbf{U}_f \mathbf{h}_{t-1} + \mathbf{b}_f) \\
   \mathbf{i}_t &= \sigma(\mathbf{W}_i \mathbf{x}_t + \mathbf{U}_i \mathbf{h}_{t-1} + \mathbf{b}_i) \\
   \tilde{\mathbf{c}}_t &= \tanh(\mathbf{W}_c \mathbf{x}_t + \mathbf{U}_c \mathbf{h}_{t-1} + \mathbf{b}_c) \\
   \mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \\
   \mathbf{o}_t &= \sigma(\mathbf{W}_o \mathbf{x}_t + \mathbf{U}_o \mathbf{h}_{t-1} + \mathbf{b}_o) \\
   \mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
   \end{aligned}$$
2. **Gated Recurrent Unit (GRU)**:
   $$\begin{aligned}
   \mathbf{z}_t &= \sigma(\mathbf{W}_z \mathbf{x}_t + \mathbf{U}_z \mathbf{h}_{t-1} + \mathbf{b}_z) \\
   \mathbf{r}_t &= \sigma(\mathbf{W}_r \mathbf{x}_t + \mathbf{U}_r \mathbf{h}_{t-1} + \mathbf{b}_r) \\
   \tilde{\mathbf{h}}_t &= \tanh(\mathbf{W}_h \mathbf{x}_t + \mathbf{U}_h (\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{b}_h) \\
   \mathbf{h}_t &= (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t
   \end{aligned}$$
3. **Prediction Head**:
   $$\hat{S}_{t+h} = \text{Sigmoid}(\mathbf{W}_{\text{out}} \mathbf{h}_L + b_{\text{out}})$$

> [!NOTE]
> In `src/temporal/models.py`, both `CoconutProgressionLSTM` and `CoconutProgressionGRU` are fully implemented with unit test verification on deterministic synthetic test fixtures. **However, zero empirical weights or test metrics are claimed for real palm progression**, in strict adherence to our research-integrity mandate.

---

## 4. Engineering Progression Categories & Alert Engine

To translate numerical predictions into actionable agricultural decision support, we formulate explicit, transparent engineering threshold rules.

### 4.1 Progression Trajectory Classification
Given current severity $S_t$ and projected severity $\hat{S}_{t+h}$ over horizon $h = 7\text{ days}$:
$$\Delta S_{\text{proj}} = \hat{S}_{t+h} - S_t$$

| Trajectory Category | Mathematical Criterion | Pathological Interpretation |
| :--- | :--- | :--- |
| **STABLE** | $\Delta S_{\text{proj}} \le 0.005$ ($\le +0.5\%$ coverage change) | Lesion margins stationary; host defense or prophylactic spray holding. |
| **SLOW_EXPANSION** | $0.005 < \Delta S_{\text{proj}} \le 0.030$ ($+0.5\%$ to $+3.0\%$) | Gradual fungal colonization; localized secondary spotting. |
| **RAPID_EXPANSION** | $\Delta S_{\text{proj}} > 0.030$ ($> +3.0\%$ coverage expansion) | Aggressive pathogen proliferation; systemic crown/trunk threat. |

### 4.2 Automated Decision-Support Alert Logic

The Alert Engine (`src/alerts/engine.py`) synthesizes current health tier, projected severity, and progression trajectory into 4 discrete alert tiers:

```
[Current Palm State] + [Projected Severity & Trajectory]
                       |
                       v
         +-----------------------------+
         |  DECISION LOGIC TRUTH TABLE |
         +-----------------------------+
                       |
     +-----------------+-----------------+-----------------+
     |                                   |                 |
     v                                   v                 v
[NO_ALERT]                           [MONITOR]        [ATTENTION]
Healthy palm or                      Low stable       Expanding lesion
stationary minimal                   lesion           or moderate
lesion                                                severity
                                                           |
                                                           v
                                                  [URGENT_REVIEW]
                                                  Rapid expansion or
                                                  imminent bud/trunk loss
```

#### Truth Table:
1. **NO_ALERT**:
   - Status: `NO_DISEASE_DETECTED` (PHI $\ge 85.0$, current area $= 0.0$).
   - Action: Continue routine scheduled field scouting.
2. **MONITOR**:
   - Condition: $S_t \le 0.05$ AND Trajectory is `STABLE` or `SLOW_EXPANSION`.
   - Action: Re-inspect palm in 7 days; log lesion boundaries.
3. **ATTENTION**:
   - Condition: $0.05 < S_t \le 0.15$ OR ($S_t \le 0.05$ with `RAPID_EXPANSION`).
   - Action: Agronomic scouting inspection; prepare targeted topical fungicide application.
4. **URGENT_REVIEW**:
   - Condition: $S_t > 0.15$ OR `bud rot` with active progression OR `RAPID_EXPANSION` crossing moderate threshold.
   - Action: Immediate phytosanitary intervention; isolate specimen to prevent spore dispersal to adjacent palms.

---

## 5. Mandatory Agronomic & Engineering Disclaimer

All progression predictions and alerts generated by this framework represent **heuristic engineering monitoring estimates derived from photographic 2D bounding box projections**. They do **NOT** constitute certified plant pathology diagnoses, official agricultural extension advisories, or legally validated chemical treatment authorizations.
