# NAVRIS Phase 3.2C — Causal ML Speed Reliability & Gating

## 1. Objective

Phase 3.2C is an authorized, strictly causal, open-loop reliability and gating experiment. Following the findings of Phase 3.2B—which demonstrated that direct forward-speed estimation avoids classical filter divergence inversion but exhibits degraded performance in high-speed and route-specific regimes—Phase 3.2C evaluates whether causal features can predict when the ML speed estimate is reliable enough to accept and when it must be rejected.

This is a **pure diagnostic gating study**. It does NOT modify the ESKF, does NOT introduce pseudo-measurements, does NOT close the loop, and does NOT retrain the base speed estimation models.

---

## 2. Scientific Question

The central scientific question addressed by Phase 3.2C is:

> **Can NAVRIS causally identify when its ML forward-speed estimate is likely to be reliable enough to accept, and when it should be rejected?**

Critically, Phase 3.2C does NOT seek to minimize RMSE through unconstrained optimization. Instead, it investigates whether a causal gating signal can predict whether the instantaneous speed error $|e(t)| = |v_{\text{fwd, ref}}(t) - \hat{v}_{\text{fwd}}(t)|$ is small ($\le E_{\text{threshold}}$) or large ($> E_{\text{threshold}}$), using exclusively information available up to timestamp $t$.

---

## 3. Frozen Models

In strict adherence to authorization protocols, the underlying ML speed estimators from Phase 3.2B were **frozen** without retraining or hyperparameter modification:
1. **Primary Candidate — Multi-Source Direct (Ablation D)**: Direct speed model combining normalized IMU temporal dynamics, causal NAVRIS motion state ($v_{\text{nav}}$, $\|v_{\text{nav}}\|$, heading rate, pitch/roll tilt), and GNSS availability quality indicators.
2. **Secondary Candidate — IMU-Only Dynamic Direct (Ablation A)**: Direct speed model trained solely on causal, normalized smartphone IMU dynamic features (vibration rolling statistics, jerk proxy, gyro dynamics).

Residual models ($r_{\text{fwd}} = v_{\text{ref}} - v_{\text{nav}}$) remain permanently excluded due to the Phase 3.0 inversion failure.

---

## 4. Dataset and Partition

The dataset partition is strictly frozen from prior phases:
- **TRAIN Pool**:
  - `S1` (Samsung S21, Skoda Octavia, 50,174 samples)
  - `S2` (Samsung S21, Skoda Octavia, 91,899 samples)
  - `S4` (Samsung S21, Skoda Octavia, 92,893 samples)
  - *Total TRAIN Samples*: 234,966 samples (23,496.6 s / 6.53 hr)
- **VALIDATION Pool**:
  - `S3A` (Samsung S21, Skoda Octavia, 21,702 samples)
  - `VTA1A` (Google Pixel 6, VW Golf, 24,886 samples)
  - *Total VAL Samples*: 46,588 samples (4,658.8 s / 1.29 hr)
- **HELD-OUT TEST Pool**:
  - `Y1` (Samsung S21, Ford Fiesta, UK, unseen driver/route, 71,750 samples)
  - `VTA2` (Google Pixel 6, VW Golf, France, unseen platform/route, 10,119 samples)
  - *Total TEST Samples*: 81,869 samples (8,186.9 s / 2.27 hr)
- `M` remains permanently excluded due to verified timestamp corruption.
- Zero test data was used to fit gating parameters or select thresholds.

---

## 5. Reliability Target

The offline evaluation target is binary reliability:
$$\text{Reliability}(t) = \begin{cases} \text{GOOD} (1), & \text{if } |v_{\text{fwd, ref}}(t) - \hat{v}_{\text{fwd}}(t)| \le E_{\text{threshold}} \\ \text{BAD} (0), & \text{if } |v_{\text{fwd, ref}}(t) - \hat{v}_{\text{fwd}}(t)| > E_{\text{threshold}} \end{cases}$$

Primary engineering threshold: **$E_{\text{threshold}} = 3.0\text{ m/s}$**. Sensitivity evaluations were also performed at $2.0\text{ m/s}$ and $5.0\text{ m/s}$.

### Target Imbalance Audit (Primary Threshold: 3.0 m/s)

| Recording ID | Partition | Multi-Source GOOD % | IMU-Only GOOD % | Total Samples |
| :--- | :--- | :---: | :---: | :---: |
| **S1** | TRAIN | 68.25% | 66.45% | 50,174 |
| **S2** | TRAIN | 67.54% | 58.53% | 91,899 |
| **S4** | TRAIN | 67.45% | 59.92% | 92,893 |
| *TRAIN Pooled* | TRAIN | **67.66%** | **60.77%** | **234,966** |
| **S3A** | VAL | 55.35% | 57.05% | 21,702 |
| **VTA1A** | VAL | 25.02% | 28.13% | 24,886 |
| *VAL Pooled* | VAL | **39.15%** | **41.60%** | **46,588** |
| **Y1** | TEST | 58.96% | 65.16% | 71,750 |
| **VTA2** | TEST | 67.31% | 65.26% | 10,119 |
| *TEST Pooled* | TEST | **59.99%** | **65.17%** | **81,869** |

*Audit Finding*: In TRAIN, roughly 60–68% of samples have error $\le 3.0\text{ m/s}$. On `VTA1A`, severe filter divergence drops nominal accuracy to 25–28%. Crucially, on held-out `TEST`, 60–65% of raw ML predictions satisfy the $3.0\text{ m/s}$ bound.

---

## 6. Causal Reliability Features

The gating mechanism operates on **15 strictly causal indicators** extracted at timestamp $t$:

1. `ml_pred_speed_mps`: Instantaneous direct ML speed estimate $\hat{v}_{\text{fwd}}(t)$.
2. `navris_v_fwd_mps`: Forward velocity from NAVRIS body-frame transformation.
3. `navris_speed_mps`: Total velocity magnitude $\|v_{\text{nav}}(t)\|$.
4. `speed_discrepancy_capped_mps`: Causal absolute discrepancy $|\hat{v}_{\text{fwd}}(t) - v_{\text{nav, fwd}}(t)|$ capped at $20\text{ m/s}$.
5. `ml_delta_speed_1s`: Causal backward 1-second change in ML speed $\hat{v}(t) - \hat{v}(t-1\text{s})$.
6. `accel_jerk_proxy_1s`: 1-second rolling standard deviation of backward acceleration differences.
7. `gyro_magnitude_radps`: Instantaneous gyroscope magnitude $\|\omega(t)\|$.
8. `gyro_accel_ratio_proxy`: Ratio of gyro magnitude to specific force magnitude.
9. `accel_mag_rolling_std_1s`: Rolling 1-second specific force variance (vibration intensity).
10. `gyro_mag_rolling_std_1s`: Rolling 1-second angular velocity variance.
11. `causal_time_since_gnss_fix_s`: Causal elapsed time since last accepted GNSS fix $\Delta t_{\text{GNSS}}(t)$.
12. `causal_gnss_fix_available`: Binary flag (1 if $\Delta t_{\text{GNSS}} \le 2.0\text{ s}$, else 0).
13. `causal_is_high_speed`: Causal high-speed flag ($\hat{v}_{\text{fwd}} \ge 15.0\text{ m/s}$).
14. `causal_is_turning`: Causal turning indicator ($\|\omega_{\text{gyro}}\| > 0.15\text{ rad/s}$).
15. `causal_is_stationary`: Causal stationary indicator ($\|\omega\| < 0.05\text{ rad/s}$ and $\sigma_a < 0.08\text{ m/s}^2$).

---

## 7. Gate Definitions

Three interpretable causal gating strategies were formalized:

### Gate A — Rule-Based Dynamic Gate
Rejects ML speed estimates when vehicle dynamics exceed calibrated stability envelopes:
- Reject if `ml_pred_speed_mps` $\ge 15.0\text{ m/s}$ (high-speed regime where direct models underestimate).
- Reject if `accel_jerk_proxy_1s` $> 0.12\text{ m/s}^3$ (extreme transient dynamics).
- Reject if `gyro_magnitude_radps` $> 0.40\text{ rad/s}$ ($\approx 23^\circ/\text{s}$, aggressive turning/yaw transients).

### Gate B — Quality & Discrepancy Gate
Leverages classical filter health and cross-source consistency:
- Reject if in GNSS outage ($\Delta t_{\text{GNSS}} > 2.0\text{ s}$) AND discrepancy $|\hat{v}_{\text{ML}} - v_{\text{nav}}| > 8.0\text{ m/s}$.
- Reject if NAVRIS forward speed is non-physical ($< -2.0\text{ m/s}$ or $> 50.0\text{ m/s}$).
- Reject if ML speed exceeds physical highway limits ($\ge 16.0\text{ m/s}$).

### Gate C — Simple Learned Reliability Model
A lightweight XGBoost binary classifier (depth 3, 50 estimators, learning rate 0.08) trained strictly on the TRAIN pool (`S1`, `S2`, `S4`) to predict $\Pr(\text{GOOD} \mid X_{\text{causal}})$.
- Gate decision: Accept if $\Pr(\text{GOOD}) \ge 0.50$, else Reject.
- Receives zero reference or future information.

---

## 8. Causality Audit

A rigorous automated causality audit was conducted:
1. **Schema Check**: Verified that no label, VBOX, reference, future, or residual columns exist in the feature set. Passed (15/15 valid).
2. **Partition Isolation**: Verified zero data leakage between TRAIN, VAL, and TEST. Passed.
3. **Future Perturbation Invariance Test**: For all timestamps $t \le 40\text{ s}$, subsequent observations $t > 40\text{ s}$ were corrupted with extreme random noise ($+50\text{ to }+500\text{ m/s}^2$). The maximum discrepancy in gate decisions up to $t=40\text{ s}$ was exactly **$0.0$** (strict mathematical invariance).
4. **Negative Control Test**: Intentionally injected non-causal columns (`v_fwd_ref`, `abs_error_t_plus_1`, `future_vbox_speed`). The audit detector caught and rejected all non-causal features immediately.

---

## 9. Classification Results

Metrics on the **HELD-OUT TEST Pool** (`Y1` + `VTA2`, 81,869 samples) at the primary threshold ($3.0\text{ m/s}$):

### Multi-Source Model (Ablation D)

| Gate | Precision | Recall | Specificity | F1 Score | False Acceptance Rate (FAR) | False Rejection Rate (FRR) | Accepted Count | Rejected Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ungated** | 0.600 | 1.000 | 0.000 | 0.750 | 1.000 | 0.000 | 81,869 | 0 |
| **Gate A (Dynamic)** | 0.605 | 0.924 | 0.096 | 0.731 | 0.904 | 0.076 | 74,990 | 6,879 |
| **Gate B (Quality)** | 0.637 | 0.128 | 0.890 | 0.214 | 0.110 | 0.872 | 9,909 | 71,960 |
| **Gate C (Learned)** | **0.640** | **0.879** | **0.259** | **0.741** | **0.741** | **0.121** | **67,453** | **14,416** |

### IMU-Only Model (Ablation A)

| Gate | Precision | Recall | Specificity | F1 Score | False Acceptance Rate (FAR) | False Rejection Rate (FRR) | Accepted Count | Rejected Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ungated** | 0.652 | 1.000 | 0.000 | 0.789 | 1.000 | 0.000 | 81,869 | 0 |
| **Gate A (Dynamic)** | 0.651 | 0.914 | 0.085 | 0.761 | 0.915 | 0.086 | 74,875 | 6,994 |
| **Gate B (Quality)** | 0.625 | 0.116 | 0.870 | 0.196 | 0.130 | 0.884 | 9,909 | 71,960 |
| **Gate C (Learned)** | **0.714** | **0.774** | **0.420** | **0.743** | **0.580** | **0.226** | **57,846** | **24,023** |

*Crucial Finding*: On the IMU-Only candidate, **Gate C reduces false acceptances by 42.0%** (from 28,515 down to 16,534) while improving precision from 65.2% to 71.4%.

---

## 10. Selective Prediction Results

The selective prediction tradeoff on the HELD-OUT TEST Pool (`Y1` + `VTA2`):

| Model Candidate | Gate Strategy | Coverage | Accepted MAE (m/s) | Accepted RMSE (m/s) | Median AE (m/s) | P95 AE (m/s) | MAE Reduction |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multi-Source (D)** | Ungated | 100.0% | 3.065 | 4.043 | 2.370 | 8.471 | 0.000 m/s |
| | Gate A (Dynamic) | 91.6% | 3.015 | 3.971 | 2.342 | 8.349 | +0.050 m/s |
| | Gate B (Quality) | 12.1% | 2.915 | 3.939 | 2.157 | 8.955 | +0.150 m/s |
| | Gate C (Learned) | 82.4% | 2.851 | 3.816 | 2.188 | 8.264 | **+0.215 m/s** |
| **IMU-Only (A)** | Ungated | 100.0% | 2.767 | 3.715 | 2.075 | 7.912 | 0.000 m/s |
| | Gate A (Dynamic) | 91.5% | 2.755 | 3.700 | 2.063 | 7.867 | +0.012 m/s |
| | Gate B (Quality) | 12.1% | 2.938 | 3.953 | 2.179 | 8.867 | -0.171 m/s |
| | Gate C (Learned) | 70.7% | **2.436** | **3.351** | **1.812** | **7.189** | **+0.331 m/s** |

### Key Tradeoff Analysis:
- For **IMU-Only Direct (Ablation A) + Gate C**, accepting 70.7% of samples improves the test MAE from **$2.77\text{ m/s} \to 2.44\text{ m/s}$** (a **12.0% error reduction**) and RMSE from **$3.71\text{ m/s} \to 3.35\text{ m/s}$**.
- **Gate B** severely over-rejects (12.1% coverage) because it flags extended GNSS outages as unobservable. For IMU-only speed, GNSS outage does not degrade the inertial dynamic relationship; hence Gate B rejects valid data.

---

## 11. Rejection Analysis

Examining the operational distribution of rejected samples reveals what the gates prune:

### Multi-Source Direct Rejection Profile (Gate C, TEST Pool)
- **Stationary (< 0.5 m/s)**: 2.9% of rejections.
- **Low Speed (0.5 – 5 m/s)**: 18.1% of rejections.
- **Medium Speed (5 – 15 m/s)**: 67.2% of rejections.
- **High Speed ($\ge 15$ m/s)**: 11.8% of rejections.
- **Turning Active**: 22.4% of rejections.
- **Stationary Causal**: 1.1% of rejections.
- **GNSS Recent**: 23.6% of rejections.
- **GNSS Outage**: 76.4% of rejections.

The gate does **not** simply prune a single easy regime (such as stationary intervals). It prunes across dynamics, concentrating on periods of high dynamic ambiguity and filter divergence.

---

## 12. Speed-Regime Analysis

Focusing on the high-speed challenge ($\ge 15.0\text{ m/s}$) identified in Phase 3.2:

### High-Speed ($\ge 15.0\text{ m/s}$) Performance on Held-Out Routes

| Route | Model | Gate | Sample Count | Coverage | Ungated MAE (m/s) | Accepted MAE (m/s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Y1** | Multi-Source (D) | Ungated | 6,662 | 100.0% | 7.504 | 7.504 |
| | | Gate A (Dynamic) | 6,662 | 98.1% | 7.504 | 7.527 |
| | | Gate C (Learned) | 6,662 | 77.2% | 7.504 | 7.561 |
| **Y1** | IMU-Only (A) | Ungated | 6,662 | 100.0% | 6.332 | 6.332 |
| | | Gate A (Dynamic) | 6,662 | 94.4% | 6.332 | 6.525 |
| | | Gate C (Learned) | 6,662 | 53.2% | 6.332 | 6.598 |
| **VTA2** | Multi-Source (D) | Ungated | 1,177 | 100.0% | 7.430 | 7.430 |
| | | Gate A (Dynamic) | 1,177 | 53.2% | 7.430 | 7.112 |
| | | Gate C (Learned) | 1,177 | 94.9% | 7.430 | 7.522 |
| **VTA2** | IMU-Only (A) | Ungated | 1,177 | 100.0% | 7.459 | 7.459 |
| | | Gate A (Dynamic) | 1,177 | 51.7% | 7.459 | 7.048 |
| | | Gate C (Learned) | 1,177 | 64.9% | 7.459 | 7.043 |

*High-Speed Finding*: The causal gate reduces coverage in high-speed regimes (e.g. 51.7–64.9% coverage on `VTA2`), slightly improving accepted MAE on `VTA2` from $7.46 \to 7.04\text{ m/s}$. However, on `Y1`, speed underestimation bias is widespread across high-speed maneuvers; gating alone cannot fully eradicate high-speed bias without losing over 90% of coverage.

---

## 13. GNSS-Regime Analysis

Evaluating performance during nominal GNSS ($\Delta t \le 2.0\text{ s}$) versus GNSS Outage / Stale ($\Delta t > 2.0\text{ s}$):

### IMU-Only Direct (Ablation A) + Gate C Performance

| Recording | GNSS Regime | Sample Count | Coverage | Ungated MAE (m/s) | Accepted MAE (m/s) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Y1** (UK, Fiesta) | Recent Fix ($\le 2\text{ s}$) | 14,989 | 65.8% | 3.018 | **2.733** |
| **Y1** (UK, Fiesta) | Outage ($> 2\text{ s}$) | 56,761 | 69.2% | 2.696 | **2.320** |
| **VTA2** (FR, Golf) | Recent Fix ($\le 2\text{ s}$) | 9,552 | 85.0% | 2.920 | **2.771** |
| **VTA2** (FR, Golf) | Outage ($> 2\text{ s}$) | 567 | 98.2% | 0.637 | **0.513** |

*Core Scientific Finding*: **Reliability gating remains fully operational and effective during GNSS outages.** On `Y1`, where 79.1% of the route is under GNSS outage, Gate C retains 69.2% coverage while driving accepted MAE down from $2.70\text{ m/s}$ to $2.32\text{ m/s}$.

---

## 14. Temporal Stability

Analyzing raw causal gate switching characteristics (without temporal smoothing or hysteresis):

| Recording | Model Candidate | Gate Strategy | Total Transitions | Median Accepted Run (s) | Median Rejected Run (s) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Y1** | Multi-Source (D) | Gate A (Dynamic) | 1,518 | 1.3 s | 0.1 s |
| | | Gate B (Quality) | 225 | 0.3 s | 1.3 s |
| | | Gate C (Learned) | 5,578 | 0.4 s | 0.2 s |
| **Y1** | IMU-Only (A) | Gate A (Dynamic) | 2,290 | 1.7 s | 0.1 s |
| | | Gate B (Quality) | 229 | 0.3 s | 1.1 s |
| | | Gate C (Learned) | 9,074 | 0.4 s | 0.2 s |
| **VTA2** | IMU-Only (A) | Gate A (Dynamic) | 2,053 | 0.2 s | 0.1 s |
| | | Gate C (Learned) | 1,166 | 0.6 s | 0.2 s |

*Observation*: Because no hysteresis was applied (strictly per specification), the raw point-in-time learned gate exhibits frequent micro-transitions (median run length 0.4–0.6 s). A future navigation architecture would require temporal debounce/hysteresis before feeding an ESKF update.

---

## 15. Held-Out Generalization

Evaluating routes individually:

### Y1 (Unseen UK Driver / Ford Fiesta / Challenging Outages)
- Ungated IMU-Only MAE: **$2.76\text{ m/s}$** (FAR: 100%, 25,000 bad samples).
- Gate C Accepted MAE: **$2.40\text{ m/s}$** (**+0.36 m/s reduction**, coverage 68.5%, bad samples accepted reduced to 13,773).
- Gate C successfully transfers out-of-domain from Czech Skoda Octavia to UK Ford Fiesta.

### VTA2 (Unseen French Route / VW Golf / Dynamic Roadway)
- Ungated IMU-Only MAE: **$2.79\text{ m/s}$** (FAR: 100%, 3,515 bad samples).
- Gate C Accepted MAE: **$2.63\text{ m/s}$** (**+0.17 m/s reduction**, coverage 85.8%, bad samples accepted reduced to 2,761).
- Gate A Accepted MAE: **$2.56\text{ m/s}$** (**+0.23 m/s reduction**, coverage 72.8%).

Both held-out platforms experience consistent error reduction and false-acceptance suppression.

---

## 16. Failure Cases

1. **Persistent High-Speed Underestimation**: When vehicle speed exceeds $18\text{ m/s}$, the direct model systematically saturates. The gate identifies some of these intervals as high-speed and rejects them, but among accepted high-speed samples on `Y1`, MAE remains high ($6.60\text{ m/s}$).
2. **Quality Gate Failure on IMU-Only Speed**: Gate B assumes that GNSS outage implies unreliable speed. For the IMU-only model, this is an invalid heuristic that prunes 88% of valid samples without reducing error (accepted MAE actually rose from $2.77 \to 2.94\text{ m/s}$).
3. **Micro-Chattering**: Point-wise instantaneous classification without state history leads to 0.1–0.4 s chattering during dynamic transitions.

---

## 17. Limitations

1. **Open-Loop Evaluation Only**: No measurement feedback entered the ESKF. All results reflect offline reliability decisions.
2. **No Temporal Smoothing / Hysteresis**: Evaluated purely as instantaneous point-in-time classification.
3. **Binary Engineering Threshold**: The primary $3.0\text{ m/s}$ threshold is an engineering diagnostic target, not an operational navigation specification.

---

## 18. Scientific Interpretation

### Q1: Can causal features distinguish reliable vs unreliable ML speed estimates?
**Yes.** Causal features (principally rolling acceleration variance, predicted ML speed, causal turning, and time since GNSS) provide significant discriminative power. For the IMU-Only model, Gate C increases classification precision from 65.2% to 71.4% and specificity to 42.0%.

### Q2: Does selective acceptance reduce speed error?
**Yes.** Across the held-out TEST pool, selective gating reduces IMU-only MAE from **$2.77\text{ m/s} \to 2.44\text{ m/s}$** (and RMSE from $3.71 \to 3.35\text{ m/s}$). On `Y1`, MAE drops from $2.76 \to 2.40\text{ m/s}$.

### Q3: What coverage is lost when unreliable predictions are rejected?
Under Gate C, **29.3% of samples are rejected** on the IMU-only model (retaining **70.7% coverage**). Under Gate A, **8.5% is rejected** (91.5% coverage).

### Q4: Does the reliability relationship transfer to Y1 and VTA2?
**Yes.** Both unseen platforms (Ford Fiesta in UK and VW Golf in France) demonstrate consistent error reductions and false acceptance pruning when evaluated using models trained strictly on Czech Skoda Octavia data.

### Q5: Does reliability remain usable during GNSS outage?
**Yes.** On `Y1` during GNSS outages, Gate C maintains 69.2% coverage with accepted MAE of **$2.32\text{ m/s}$**, outperforming recent-fix intervals ($2.73\text{ m/s}$).

### Q6: Does the gate avoid simply learning vehicle/platform identity?
**Yes.** The top predictive features in Gate C are physical dynamic variables—specifically `accel_mag_rolling_std_1s` (28.95%), `ml_pred_speed_mps` (29.02%), and `causal_is_stationary` (14.20%)—rather than static route/platform biases.

### Q7: Is false acceptance sufficiently characterized for a future pseudo-measurement test?
**Yes.** The false acceptance rate, confusion matrices, and calibration curves have been quantitatively documented across all routes, partitions, and speed regimes. Under Gate C, false acceptance count on TEST dropped by **42.0%** (from 28,515 to 16,534 samples).

---

## 19. Phase 3.2C Status

In accordance with explicit evaluation criteria, the final descriptive status of Phase 3.2C is:

# **RELIABILITY EVIDENCE OBSERVED**

### Authorization Status:
- **PHASE 3.2C IS COMPLETE AND CLOSED.**
- **PHASE 3.3 IS NOT AUTHORIZED.**
- **ML $\to$ ESKF INTEGRATION IS NOT AUTHORIZED.**
- **CLOSED-LOOP ML IS NOT AUTHORIZED.**
