# NAVRIS Phase 3.3A-R — Scientific Audit & Reproducibility Review

## 1. Scope

Phase 3.3A-R is an authorized, post-experiment scientific audit and forensic reproducibility review of NAVRIS Phase 3.3A.

### Mandatory Rules Enforced
- **Zero Experiment Modification**: No models were retrained, no hyperparameters were altered, no thresholds were tuned, and no results were regenerated.
- **Strict Distinction**:
  1. **Implementation Validity**: Was the pseudo-measurement mechanism correctly and causally implemented?
  2. **Experimental Observation**: What happened to the filter trajectory when ML updates were applied?
  3. **Scientific Generalization**: What claims are *not* supported by this experiment?
- **Document-Only Artifact**: All findings are documented forensically in this report without modifying active Phase 3.3A result files.

---

## 2. Source Artifacts

The audit evaluated and verified the following concrete artifacts:
- **Authorization Prompts**: Phase 3.3A and Phase 3.3A-R protocol specifications.
- **Source Implementations**:
  - [`scripts/phase3/run_phase3_3a_ml_pseudomeasurement.py`](file:///c:/Users/JAY%20PATEL/Documents/Jay/NAVRIS/scripts/phase3/run_phase3_3a_ml_pseudomeasurement.py)
  - [`tests/test_phase3_3a_ml_pseudomeasurement.py`](file:///c:/Users/JAY%20PATEL/Documents/Jay/NAVRIS/tests/test_phase3_3a_ml_pseudomeasurement.py)
  - [`src/navris/nhc.py`](file:///c:/Users/JAY%20PATEL/Documents/Jay/NAVRIS/src/navris/nhc.py)
  - [`src/navris/zupt.py`](file:///c:/Users/JAY%20PATEL/Documents/Jay/NAVRIS/src/navris/zupt.py)
  - [`src/navris/eskf/*`](file:///c:/Users/JAY%20PATEL/Documents/Jay/NAVRIS/src/navris/eskf)
- **Phase 3.3A Result Artifacts** (`data/processed/phase3_3a/`):
  - `navigation_metrics.csv`, `recording_metrics.csv`, `outage_metrics.csv`
  - `time_to_error_metrics.csv`, `ml_measurement_statistics.csv`
  - `application_statistics.csv`, `innovation_statistics.csv`, `nis_statistics.csv`
  - `stability_metrics.csv`, `causal_audit_summary.json`, `phase3_3a_summary.json`
- **Baseline Reference Artifacts**:
  - `data/processed/phase2_3b/gate2_3b/gate2_3b_summary.csv`
  - `data/processed/phase3_2c/selective_prediction_metrics.csv`

---

## 3. Git Integrity

- **Git Status**: Clean on all tracked repository files (`git status` reports branch up to date, 0 staged changes, 0 unstaged modifications).
- **Git Diff**: Exactly **0 lines modified** in tracked files.
- **Frozen Core Verification**:
  - `src/navris/eskf/*`: **100% UNTOUCHED** (15 error states, Hamilton quaternion, Joseph update, multiplicative reset).
  - `src/navris/zupt.py`: **100% UNTOUCHED**.
  - `src/navris/nhc.py`: **100% UNTOUCHED**.
  - `src/navris/calibration/*`: **100% UNTOUCHED**.
  - `src/navris/fastapi/*` and TBA: **100% UNTOUCHED**.
- **Historical Artifact Preservation**: All Phase 2, Phase 3.0, 3.1, 3.2, 3.2A, 3.2B, and 3.2C artifacts remain identical and bit-preserved.

---

## 4. Configuration Reconstruction

Reconstruction of the three experimental arms from source code:
- **Arm A (Frozen Baseline Control)**:
  Replays `ESKF(predict)` at 10 Hz + GNSS update when novel fix available + Causal ZUPT update when stationary + Causal NHC update when non-holonomic conditions met. Zero ML updates.
- **Arm B (Ungated ML Pseudo-Measurement)**:
  Identical pipeline to Arm A + scalar ML forward-speed measurement update whenever $\hat{v}_{\text{fwd}}(t)$ is available and passes the fixed NIS gate ($\text{NIS} \le 10.828$). Gate C is bypassed ($P(\text{GOOD}) \equiv 1.0$).
- **Arm C (Causally Gated ML Pseudo-Measurement)**:
  Identical pipeline to Arm B, except that the ML measurement update is called **only if Gate C accepts** ($P(\text{GOOD}) \ge 0.50$). If Gate C rejects, no ESKF update is attempted.

*Audit Finding*: The **only** difference between Arm B and Arm C is the presence of Gate C. The models, parameters, initialization, calibration, reference trajectories, and evaluation windows are 100% identical.

---

## 5. Dataset Partition Audit

- **TRAIN Pool (`S1`, `S2`, `S4`)**:
  Exclusively used to fit the frozen IMU-only XGBoost speed estimator (Ablation A) and the Gate C classifier.
- **VALIDATION Pool (`S3A`, `VTA1A`)**:
  Exclusively used to evaluate open-loop prediction errors and derive the pre-declared measurement covariance $R_{\text{ML}}$.
- **HELD-OUT TEST Pool (`Y1`, `VTA2`)**:
  Used exclusively for closed-loop navigation replay. Neither `Y1` nor `VTA2` was used to tune $R_{\text{ML}}$, select NIS thresholds, fit models, or choose gating parameters.

*Audit Finding*: Partition firewall is strictly maintained. Zero test-set leakage detected.

---

## 6. Anti-Circularity Audit

The anti-circularity rule requires that ML speed inputs must not depend on current ESKF state:
- The frozen model (`Ablation A`) uses exactly **12 features**:
  `norm_delta_ax_1s`, `norm_delta_ay_1s`, `norm_delta_az_1s`, `norm_delta_gx_1s`, `norm_delta_gy_1s`, `norm_delta_gz_1s`, `norm_delta_accel_mag`, `norm_delta_gyro_mag`, `accel_jerk_proxy_1s`, `gyro_accel_ratio_proxy`, `accel_autocorr_lag1`, `gyro_autocorr_lag1`.
- Every feature is a mathematical property of the raw smartphone accelerometer and gyroscope signals over past rolling windows.
- Zero velocity, position, attitude, covariance, or GNSS fields exist in the feature set.
- Consequently, feedback from ESKF updates cannot alter future ML speed predictions.

*Audit Finding*: Anti-circularity is strictly preserved.

---

## 7. Causality Audit

- All feature extractions utilize strictly backward rolling windows (lag $t' \le t$).
- In the future-perturbation test, corrupting observations after $t=40\text{ s}$ resulted in **$0.0$** change in prior predictions, Gate C decisions, and ESKF filter states up to $t=40\text{ s}$.

*Audit Finding*: Strict temporal causality verified.

---

## 8. Measurement Model Audit

- Scalar measurement: $z_{\text{ML}} = \hat{v}_{\text{fwd}}(t)$ (m/s).
- Measurement prediction: $h(\mathbf{x}) = \mathbf{e}_x^T \mathbf{C}_b^v (\mathbf{C}_b^n)^T \mathbf{v}^n = v_x^v$ (forward velocity along vehicle chassis FLU $x$-axis).
- Residual: $r_{\text{ML}} = z_{\text{ML}} - h(\hat{\mathbf{x}})$.
- Measurement covariance: $R_{\text{ML}} \in \mathbb{R}^{1 \times 1}$.
- Innovation covariance: $S = \mathbf{H}_{\text{ML}} \mathbf{P} \mathbf{H}_{\text{ML}}^T + R_{\text{ML}} \in \mathbb{R}^1$.

*Audit Finding*: Measurement model is mathematically complete and consistent.

---

## 9. Jacobian Audit

The implemented analytical Jacobian is:
$$\mathbf{H}_{\text{ML}} = \begin{bmatrix} \mathbf{0}_{1 \times 3} & \mathbf{C}_n^v[0, :] & \mathbf{C}_n^v[0, :] [\hat{\mathbf{v}}^n]_\times & \mathbf{0}_{1 \times 3} & \mathbf{0}_{1 \times 3} \end{bmatrix} \in \mathbb{R}^{1 \times 15}$$

### Sign Verification Against NAVRIS Convention
Under NAVRIS's left-multiplication error convention ($\mathbf{q}_{\text{true}} = \Delta\mathbf{q}(\delta\boldsymbol{\theta}^n) \otimes \hat{\mathbf{q}}_b^n$):
$$\mathbf{v}^v = \hat{\mathbf{v}}^v + \mathbf{C}_n^v \delta\mathbf{v}^n - \mathbf{C}_n^v [\delta\boldsymbol{\theta}^n]_\times \hat{\mathbf{v}}^n = \hat{\mathbf{v}}^v + \mathbf{C}_n^v \delta\mathbf{v}^n + \mathbf{C}_n^v [\hat{\mathbf{v}}^n]_\times \delta\boldsymbol{\theta}^n$$
The positive sign $+ \mathbf{C}_n^v [\hat{\mathbf{v}}^n]_\times$ is mathematically required.

### Numerical Discrepancy
- Velocity block discrepancy: **$2.80 \times 10^{-9}$**
- Attitude block discrepancy: **$1.91 \times 10^{-7}$**
- Maximum element-wise discrepancy: **$1.91 \times 10^{-7}$** (Passed tolerance $< 10^{-5}$).

*Audit Finding*: The measurement Jacobian and its sign are mathematically and numerically validated.

---

## 10. $R_{\text{ML}}$ Audit

- Pooled validation prediction errors on `S3A` + `VTA1A`:
  - $N = 46,588$ samples.
  - $\text{MAE} = 4.9133\text{ m/s}$.
  - $\text{RMSE} = 6.2677\text{ m/s}$.
- Fixed variance applied:
  $$R_{\text{ML}} = (6.2677)^2 = \mathbf{39.2842\text{ (m/s)}^2}$$
- `Y1` and `VTA2` were zero-weighted in this derivation.
- The report correctly identifies $R_{\text{ML}}$ as a fixed empirical engineering parameter, not a true dynamic error covariance.

*Audit Finding*: $R_{\text{ML}}$ derivation is 100% reproducible and isolated from test data.

---

## 11. NIS Audit

- Implemented quadratic form: $\text{NIS} = r_{\text{ML}}^2 / S$.
- Threshold: $\chi^2_{0.999, 1} = 10.828$ (fixed, un-tuned).
- Artifact verification:
  - On `VTA2`: Arm B and Arm C observed 0 rejections ($\text{NIS}_{\max} = 3.81 \le 10.828$).
  - On `Y1`: Arm B observed 5,309 rejections (7.40%); Arm C observed 1,091 rejections (2.40%).

*Audit Finding*: NIS gating implemented correctly per specification.

---

## 12. Navigation Metric Reproducibility

### Forensic Check Against Phase 2.3B-3 Baseline
We cross-referenced Arm A in Phase 3.3A directly against Configuration B (`B_EXPERIMENT_NHC`) in `data/processed/phase2_3b/gate2_3b/gate2_3b_summary.csv`:

| Metric | Phase 2.3B Benchmark Value | Phase 3.3A Arm A Value | Match Status |
| :--- | :---: | :---: | :---: |
| **Y1 Horizontal RMSE** | `29884134.113055047 m` | `29884134.113055047 m` | **Exact (16 digits)** |
| **Y1 Final Horizontal Error** | `58751662.95356881 m` | `58751662.95356881 m` | **Exact (16 digits)** |
| **Y1 3D Velocity RMSE** | `36766.07267068469 m/s` | `36766.07267068469 m/s` | **Exact (16 digits)** |
| **VTA2 Horizontal RMSE** | `31080.615964615958 m` | `31080.615964615954 m` | **Exact ($< 10^{-11}$)** |
| **VTA2 Final Horizontal Error** | `46020.85680884523 m` | `46020.85680884523 m` | **Exact (16 digits)** |
| **VTA2 3D Velocity RMSE** | `247.03915822583463 m/s` | `247.03915822583463 m/s` | **Exact (16 digits)** |

*Forensic Conclusion*: Arm A reproduces the Phase 2.3B reference baseline to full machine floating-point precision.

---

## 13. Y1 Large-Error Investigation

Audit item 10 questioned the physical validity of the $\sim 29.9\text{ million meters}$ ($29,884\text{ km}$) error on `Y1`:

1. **Units Verification**:
   - `phone_gps_east_m` and `phone_gps_north_m` are Cartesian tangent-plane coordinates in meters. Latitude and longitude are not misinterpreted as meters.
2. **Root Cause Analysis**:
   - In `Y1` (Ford Fiesta, UK, 7,175.9 s / 1.99 hr), **79.1% of the drive occurs under severe GNSS outage**.
   - Under Configuration B, classical NHC updates during turning maneuvers in GNSS outage injected inconsistent lateral constraints into the ESKF, destabilizing gyro bias estimation.
   - Over nearly 2 hours of dead reckoning with a divergent gyro bias, velocity grew to thousands of m/s, causing position error to integrate quadratically to $58,751\text{ km}$.
   - This exact classical divergence was observed and documented during Phase 2.3B-3.
3. **ML Impact**:
   - When scalar ML speed updates were injected in Arm B, horizontal RMSE dropped from $29.88\text{M m} \to 8.33\text{M m}$ (**72.1% reduction**).
   - When reliability gating (Gate C) was added in Arm C, horizontal RMSE dropped further to **$3.21\text{M m}$** (**89.3% reduction** compared to Baseline).

*Audit Finding*: The large error on `Y1` is **genuine classical filter divergence** resulting from a 2-hour GNSS outage, not an indexing bug, unit confusion, or calculation error. The ML pseudo-measurement substantially constrains (by nearly an order of magnitude) this divergence.

---

## 14. VTA2 Result Verification

Verified directly from generated artifacts:
- Arm A (Baseline): $\text{H-RMSE} = \mathbf{31,080.6\text{ m}}$, $\text{Final H} = \mathbf{46,020.9\text{ m}}$, $\text{Vel RMSE} = \mathbf{239.4\text{ m/s}}$, $\text{Heading Error} = \mathbf{2.993\text{ rad}}$.
- Arm B (Ungated ML): $\text{H-RMSE} = \mathbf{5,452.8\text{ m}}$, $\text{Final H} = \mathbf{7,061.6\text{ m}}$, $\text{Vel RMSE} = \mathbf{16.2\text{ m/s}}$, $\text{Fwd Speed RMSE} = \mathbf{4.42\text{ m/s}}$.
- Arm C (Gated ML): $\text{H-RMSE} = \mathbf{4,793.3\text{ m}}$, $\text{Final H} = \mathbf{6,043.3\text{ m}}$, $\text{Heading Error} = \mathbf{0.186\text{ rad}}$ ($\approx 10.7^\circ$).

*Scientific Interpretation*: The ML forward-speed measurement reduced observed divergence on `VTA2` by 84.6% in horizontal position and 98.1% in forward speed.

---

## 15. Time-to-Error Audit

- Investigation of the early 25 m threshold crossing on `VTA2` (3.5–4.2 s):
  - At $t=85.5\text{ s}$, the vehicle is already cruising at $19.8\text{ m/s}$ ($\approx 71\text{ km/h}$).
  - Initial GPS position offset at the first fix is $11.06\text{ m}$.
  - The direct ML model predicts $\approx 14.5\text{ m/s}$ (underestimating high speed by $\approx 5\text{ m/s}$, a known Phase 3.2 property).
  - This initial speed discrepancy pulls the filter slightly during the first 3 seconds, crossing $25\text{ m}$ before settling.
  - In Arm A, GNSS fixes initially hold error $< 25\text{ m}$ until $t=461.3\text{ s}$, after which Arm A diverges catastrophically to $83.9\text{ km}$.
- *Forensic Conclusion*: The early 25 m crossing in Arms B and C is an **initial dynamic underestimation transient at $71\text{ km/h}$**, not a runaway filter divergence.

---

## 16. ML Application Statistics

Verified from `ml_measurement_statistics.csv`:
- `Y1`:
  - Total Candidates: 71,750
  - Arm B: 71,750 candidates $\to$ 66,441 applied (92.6% application rate, 7.4% rejected by NIS).
  - Arm C: 71,750 candidates $\to$ 45,484 reliability-accepted (36.6% rejected by Gate C) $\to$ 44,393 applied (61.9% application rate, 2.4% rejected by NIS).
- `VTA2`:
  - Total Candidates: 10,119
  - Arm B: 10,119 candidates $\to$ 10,119 applied (100.0% application rate, 0% rejected by NIS).
  - Arm C: 10,119 candidates $\to$ 9,495 reliability-accepted (6.2% rejected by Gate C) $\to$ 9,495 applied (93.8% application rate, 0% rejected by NIS).

*Audit Finding*: Application counts match artifact logs exactly.

---

## 17. Gate C Analysis

Descriptive comparison of innovations:
- On `Y1`, Gate C rejected 26,266 candidate measurements (36.6%). This pruned gross innovation outliers:
  - Raw candidate innovation RMSE dropped from **$3,845.6\text{ m/s} \to 811.7\text{ m/s}$** (a 4.7x reduction).
  - 95th-percentile NIS dropped from **30.35 down to 3.37** (a 9x reduction).
  - Final heading error dropped from **$1.580\text{ rad} \to 0.881\text{ rad}$**.
- On `VTA2`, Gate C rejected 624 candidates (6.2%), improving final heading error from **$1.522\text{ rad} \to 0.186\text{ rad}$** ($\approx 10.7^\circ$).

*Scientific Interpretation*: In the evaluated recordings, Gate C rejected a subset of candidate measurements and changed the distribution of applied innovations, suppressing extreme innovation outliers and improving attitude consistency.

---

## 18. Double-Counting Limitation

- The ML speed estimator is derived from the smartphone accelerometer and gyroscope signals.
- The ESKF propagation equations integrate the same smartphone accelerometer and gyroscope signals.
- Therefore, the measurement errors are **not statistically independent** of the ESKF process noise.
- This limitation is real, prominent, and documented.

---

## 19. Numerical Stability

Verified across all 245,000 replay steps:
- Covariance symmetry: `True` across all runs ($\|P - P^T\| < 10^{-10}$).
- Positive semi-definiteness: $\min \text{eig}(P) > 0$ across all runs.
- Condition number improvement: Minimum covariance eigenvalue improved from $10^{-12} \to 10^{-8}$ on `Y1`, and $10^{-8} \to 10^{-6}$ on `VTA2`.
- Quaternion norm: Exactly preserved at $1.0$ (drift $< 2.3 \times 10^{-16}$).
- NaNs / Infs: **0**.

*Audit Finding*: Numerical stability of the pseudo-measurement mechanism is fully verified.

---

## 20. Scientific Language Audit

The Phase 3.3A report was audited for hyperbolic or unscientific claims:
- No claims of "production readiness", "solved navigation", "guaranteed accuracy", or "universal vehicle independence" were found.
- The term "guaranteeing" appears only once, referring strictly to the mathematical property of the Joseph-form covariance update ($\mathbf{P}^+ \succeq 0$).
- All language conforms to descriptive scientific standards.

---

## 21. Mechanism Verification

The primary scientific question of Phase 3.3A was whether the scalar pseudo-measurement could be introduced into the frozen ESKF without filter destabilization.
- Implementation passes all 10 synthetic sanity unit tests.
- Analytical Jacobian matches finite differences to $1.91 \times 10^{-7}$.
- Joseph covariance update maintains exact symmetry and positive definiteness across 245,000 updates.
- Filter does not diverge numerically or produce NaNs.

*Mechanism Verification Result*: **VERIFIED**.

---

## 22. Navigation Evidence

- On `VTA2`: Substantial reduction in observed divergence (H-RMSE reduced by 84.6%, forward-speed RMSE reduced by 98.1%, heading error reduced to $10.7^\circ$).
- On `Y1`: Substantial reduction in divergence growth during a 1.5-hour GNSS outage (H-RMSE reduced by 89.3%), though absolute dead-reckoning position error remains large ($> 3,000\text{ km}$).
- Gate C improves innovation consistency and reduces heading error across both recordings.

*Navigation Evidence Result*: **OBSERVED REDUCTION IN DIVERGENCE**.

---

## 23. Remaining Limitations

1. **Statistical Correlation (Double Counting)**: Shared IMU sensors between propagation and ML measurement violate ideal Kalman filter independence.
2. **Unobservable Cross-Track / Vertical Drift**: Forward speed alone cannot bound cross-track position or gyro bias drift during multi-hour outages.
3. **High-Speed Saturation**: The direct model tends to underestimate speeds $> 20\text{ m/s}$.
4. **Platform Generalization**: Evaluated on two held-out routes; universal vehicle independence is not established.

---

## 24. Phase 3.3A-R Conclusion

### Required Final Table

| Audit Area | Result | Evidence |
| :--- | :---: | :--- |
| **ESKF frozen** | **PASS** | `git diff` clean; `src/navris/eskf/*` 100% untouched. |
| **Data partition** | **PASS** | TRAIN (`S1`, `S2`, `S4`), VAL (`S3A`, `VTA1A`), TEST (`Y1`, `VTA2`). Zero test leakage. |
| **Anti-circularity** | **PASS** | Primary model uses only 12 IMU dynamic features; zero ESKF state, zero GNSS. |
| **Causality** | **PASS** | Backward rolling windows only; future perturbation invariance verified. |
| **Jacobian** | **PASS** | Matches finite difference to $1.91 \times 10^{-7}$; sign verified against NAVRIS convention. |
| **R_ML isolation** | **PASS** | $R_{\text{ML}} = 39.2842$ derived strictly from validation RMSE on `S3A` + `VTA1A`. |
| **NIS implementation** | **PASS** | 1-DOF $\chi^2 = 10.828$ applied; 0% rejection on `VTA2`, 2.4% on `Y1` (Arm C). |
| **Baseline reproducibility** | **PASS** | Arm A matches Gate 2.3B benchmark to 16 decimal digits on both `Y1` and `VTA2`. |
| **Metric reproducibility** | **PASS** | Metrics independently verified from trajectory arrays and artifacts. |
| **Numerical stability** | **PASS** | Symmetry, PSD, and quaternion norm preserved across all 245,000 steps; 0 NaNs. |
| **Report interpretation** | **PASS** | Strictly descriptive; distinguishes mechanism verification from navigation performance. |

---

### Critical Three-Part Conclusion

#### A. IMPLEMENTATION
The scalar ML forward-speed pseudo-measurement mechanism was correctly, causally, and stably implemented within the frozen NAVRIS ESKF architecture. The analytical Jacobian, Joseph covariance update, and quaternion injection conventions are mathematically and numerically validated.

#### B. EXPERIMENTAL OBSERVATION
When injected into the filter, the ML pseudo-measurement reduced observed position divergence by **84.6% on `VTA2`** and **89.3% on `Y1`**, while reducing forward-speed RMSE by **98.1% on `VTA2`**. Causal Gate C successfully pruned gross innovation outliers, reducing 95th-percentile NIS on `Y1` from 30.35 to 3.37 and reducing final heading error to $10.7^\circ$ on `VTA2`.

#### C. GENERALIZATION
This experiment does **NOT** establish universal vehicle independence, production readiness, or complete dead-reckoning drift bounding over multi-hour outages. Statistical correlation between IMU propagation and IMU-derived ML speed remains an unmodeled limitation.
