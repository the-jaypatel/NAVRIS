# NAVRIS Phase 3.0: Preliminary Machine Learning Experiment Report

## Executive Summary

As part of **NAVRIS Phase 3.0**, an auditable supervised machine learning pipeline was established to evaluate whether an ML model can causally predict the local navigation velocity residual:
$$\Delta \mathbf{v}^n = \mathbf{v}_{\text{reference}}^n - \mathbf{v}_{\text{NAVRIS}}^n = \begin{bmatrix} \Delta v_E \\ \Delta v_N \\ \Delta v_U \end{bmatrix}$$
where $\mathbf{v}_{\text{reference}}^n$ is derived from reference VBOX Doppler measurements and $\mathbf{v}_{\text{NAVRIS}}^n$ is the classical dead-reckoning solution (Gate 2.3B Baseline A).

In strict adherence to the project mandate:
1. **No modifications** were made to existing classical navigation filters (`eskf.py`, `zupt.py`, `nhc.py`, calibration routines, or Phase 2 benchmarks).
2. The ML target is strictly a **local 3-component velocity correction** $[\Delta v_E, \Delta v_N, \Delta v_U]$; no latitude, longitude, global positions, or absolute trajectories were used as targets.
3. All inputs are strictly causal, spanning a 1.0-second history ($[k-9 \dots k]$) with zero future lookahead, zero target leakage, and zero reference sensor contamination.
4. Evaluation was conducted open-loop (Baseline C) on held-out recordings across an unseen driver (Y1) and an out-of-domain vehicle/country under nominal GNSS lock (VTA2).

**Key Scientific Verdict:**
While the ML model achieves significant velocity error reductions on severely divergent routes where the classical filter blew up due to unobserved heading/bias integration (reducing velocity error by 75% on unseen driver Y1), **the model fails catastrophically on nominally operating routes (VTA2)**. On VTA2, where the classical filter was well-conditioned (5.53 m/s horizontal velocity RMSE, 455 m final drift), the ML model injected massive spurious vertical and horizontal corrections, degrading velocity RMSE to 34.9 m/s horizontal / 212.4 m/s 3D and blowing up final position drift to 15,396 m (a 33-fold degradation). 

Feature importance forensics demonstrate that the model learned an inverted drift mapping ($\widehat{\Delta \mathbf{v}} \approx -\mathbf{v}_{\text{NAVRIS}}$) rather than high-frequency inertial error corrections. Consequently, **the preliminary ML experiment does NOT justify proceeding to closed-loop filter integration at this stage**.

---

## 1. Dataset & Observability Audit

The experiment utilizes the synchronized phone IMU observations and validated Gate 2.3B NAVRIS trajectory outputs from the IO-VNBD dataset.

| Recording | Driver | Vehicle | Country | Duration (s) | 10 Hz Samples | Classical Status (Gate 2.3B) | GNSS Valid % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **S1** | Driver A | Ford Fiesta | UK | 5018.2 | 50,183 | Succeeded (drifted DR) | 100% (obs), 1.3% acc |
| **S2** | Driver A | Ford Fiesta | UK | 9190.7 | 91,908 | Succeeded (drifted DR) | 100% (obs), 0.2% acc |
| **S3A** | Driver A | Ford Fiesta | UK | 2171.0 | 21,711 | Succeeded (drifted DR) | 100% (obs), 11.7% acc |
| **S4** | Driver A | Ford Fiesta | UK | 9290.1 | 92,902 | Succeeded (drifted DR) | 100% (obs), 1.5% acc |
| **M** | Driver B | Ford Fiesta | UK | 10596.4 | 105,965 | **UNOBSERVABLE** (Class F) | 0.0% acc (excluded) |
| **Y1** | Driver D | Ford Fiesta | UK | 7175.8 | 71,759 | Succeeded (drifted DR) | 100% (obs), 0.4% acc |
| **VTA1A** | Driver E | VW Golf | France | 2489.4 | 24,895 | Succeeded (drifted DR) | 100% (obs), 9.4% acc |
| **VTA2** | Driver E | VW Golf | France | 1012.7 | 10,128 | Succeeded (nominal locked) | 100% (obs), 96.7% acc |

- **Exclusion of Route M**: As established in Phase 2 Gate 2.3B, route M lacked initial excitation and suffered unobservable stationary calibration, resulting in 0 valid trajectory points. It was cleanly excluded from supervised ML training.
- **Usable Total**: 7 recordings, 363,486 synchronized 10 Hz rows (>10.0 hours of driving).

---

## 2. Recording-Level Partitioning (Zero Leakage)

### 2.1 Option A vs Option B Architectural Decision
The manifest `data/processed/manifest_with_splits.csv` assigned all 8 benchmark recordings to `split = train`, rendering it unsafe for direct reuse for benchmark test isolation.
- **Option A (Full IO-VNBD for Train, 8 Benchmarks for Test)** was rejected because:
  1. 43 of the 115 recordings in IO-VNBD are single-sensor/unpaired (CAN-only or phone-only) without reference truth.
  2. The remaining non-benchmark routes do not have precomputed, validated classical NAVRIS filter solutions ($\mathbf{v}_{\text{NAVRIS}}^n$).
  3. Generating classical filter trajectories on uncalibrated routes risks massive numerical divergence and unobservable calibration artifacts.
- **Option B (Strict Recording Split around Validated Benchmark Set)** was adopted:
  - Full data availability of synchronized sensor data and validated classical NAVRIS filter states.
  - Strict recording-level separation guaranteeing zero sample leakage.
  - Authentic domain transfer evaluation across unseen drivers (Driver D on Y1) and unseen vehicle/country with nominal GNSS lock (Driver E on VTA2 in France).

### 2.2 Split Assignment
| Split | Recordings | Driver(s) | Vehicle(s) | Country | Causal Samples | % Total |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TRAIN** | `S1`, `S2`, `S4` | Driver A | Ford Fiesta | UK | 234,966 | 64.6% |
| **VALIDATION** | `S3A`, `VTA1A` | Driver A, Driver E | Ford Fiesta, VW Golf | UK, France | 46,588 | 12.8% |
| **HELD-OUT TEST** | `Y1`, `VTA2` | Driver D, Driver E | Ford Fiesta, VW Golf | UK, France | 81,869 | 22.5% |

*Held-out test recordings never contributed to feature scaling, hyperparameter tuning, threshold selection, or model selection.*

---

## 3. Causal Feature Engineering & Window Definition

### 3.1 Causal Window
- **Sampling Rate**: 10.0 Hz ($\Delta t = 0.1$ s).
- **Window Length**: Exactly 10 samples ($w = 10$, covering $1.0$ s history).
- **Index Bounds**: For sample $k$, features use exclusively $[k-9, \dots, k]$. No sample $\ge k+1$ is ever accessed.
- **Boundary Handling**: The first 9 samples ($k < 9$) of each recording are discarded. No artificial padding or sensor fabrication is permitted.

### 3.2 Feature Schema (28 Causally Available Features)
1. **Instantaneous Phone IMU (9 features)**:
   - Specific force: `phone_accel_x_mps2`, `phone_accel_y_mps2`, `phone_accel_z_mps2`
   - Angular velocity: `phone_gyro_x_radps`, `phone_gyro_y_radps`, `phone_gyro_z_radps`
   - Kinematic norms: `accel_magnitude_mps2`, `gyro_magnitude_radps`
   - Gravity deviation: `gravity_deviation_mps2` ($|\|\mathbf{a}\| - 9.80665|$)
2. **Instantaneous NAVRIS Classical Estimates (6 features)**:
   - Filter velocity: `navris_vel_east_mps`, `navris_vel_north_mps`, `navris_vel_up_mps`
   - Planar speed: `navris_speed_mps` ($\sqrt{v_E^2 + v_N^2}$)
   - Orientation representation: `navris_heading_sin`, `navris_heading_cos` ($\sin\psi, \cos\psi$)
3. **Temporal Causal Window Statistics $[k-9 \dots k]$ (13 features)**:
   - Rolling dynamics: `accel_mag_rolling_mean_1s`, `accel_mag_rolling_std_1s`
   - Gyro dynamics: `gyro_mag_rolling_mean_1s`, `gyro_mag_rolling_std_1s`
   - 1-second kinematics deltas: $\Delta a_x$, $\Delta a_y$, $\Delta a_z$ ($a_k - a_{k-9}$)
   - 1-second gyro deltas: $\Delta \omega_x$, $\Delta \omega_y$, $\Delta \omega_z$ ($\omega_k - \omega_{k-9}$)
   - 1-second filter velocity deltas: $\Delta v_{\text{nav}, E}$, $\Delta v_{\text{nav}, N}$, $\Delta v_{\text{nav}, U}$ ($v_k - v_{k-9}$)

---

## 4. Target Definition & Missing-Data Policy

### 4.1 Target Definition
The target is the 3-component local navigation frame velocity residual:
$$\Delta v_E = v_{\text{ref}, E} - v_{\text{NAVRIS}, E} = v_{\text{ref\_spd}} \sin(\psi_{\text{ref}}) - v_{\text{A}, E}$$
$$\Delta v_N = v_{\text{ref}, N} - v_{\text{NAVRIS}, N} = v_{\text{ref\_spd}} \cos(\psi_{\text{ref}}) - v_{\text{A}, N}$$
$$\Delta v_U = v_{\text{ref}, U} - v_{\text{NAVRIS}, U} = v_{\text{ref\_vertical\_spd}} - v_{\text{A}, U}$$
where $v_{\text{ref\_spd}}$, $\psi_{\text{ref}}$, and $v_{\text{ref\_vertical\_spd}}$ are VBOX Doppler measurements.

### 4.2 Missing-Data Policy
- **Forward-Filling**: Zero forward-filling of reference velocity targets.
- **Sensor Imputation**: Zero fabricated sensor readings.
- **Data Quality Audit**: The 7 usable benchmark routes had **0 NaN values** across all candidate features and targets (100% complete).

---

## 5. Model Architecture & Preprocessing

- **Architecture**: 3 independent Gradient-Boosted Decision Tree regressors (`XGBRegressor`) for East, North, and Up axes.
- **Hyperparameters**:
  - `n_estimators`: 100
  - `max_depth`: 5
  - `learning_rate`: 0.05
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
  - `tree_method`: `hist`
  - `random_state`: 42
- **Preprocessing**: `StandardScaler` fitted strictly on $X_{\text{train}}$. Validation and test features were transformed using the frozen training scaling parameters.

---

## 6. Automated Leakage Audit

Prior to and following model training, automated audit assertions were executed:

| Audit Check | Status | Verification Detail |
| :--- | :--- | :--- |
| **No Forbidden / Ground Truth Fields in $X$** | **PASS** | Inspected all 28 feature names for `ref_`, `vbox`, `target`, `truth`, `oracle`. Zero violations. |
| **No Target Residuals in $X$** | **PASS** | Confirmed $\Delta v_E, \Delta v_N, \Delta v_U$ are not present in feature set. |
| **Split Disjointness** | **PASS** | Verified $S_{\text{train}} \cap S_{\text{val}} = \emptyset$, $S_{\text{train}} \cap S_{\text{test}} = \emptyset$, $S_{\text{val}} \cap S_{\text{test}} = \emptyset$. |
| **Causal Window Integrity** | **PASS** | Validated index mapping $k-9 \dots k$. Perturbations at $t+1$ produced zero change at $t$. |
| **Scaler Isolation** | **PASS** | Scaler mean and scale match recomputed $X_{\text{train}}$ statistics to within $< 10^{-15}$. |

---

## 7. Experimental Results & Baseline Comparison

### 7.1 Comprehensive Metric Summary Table

| Partition / Recording | Split Role | ML Target MAE 3D (m/s) | ML Target RMSE Horiz (m/s) | ML Target RMSE 3D (m/s) | Nav Baseline A Vel RMSE (m/s) | Nav Baseline B Vel RMSE (m/s) | Nav Baseline C (ML) Vel RMSE (m/s) | Pos Baseline A Final Drift (m) | Pos Baseline C (ML) Final Drift (m) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TRAIN ALL** (`S1, S2, S4`) | Train | 246.9 | 61.4 | 302.9 | 35,173.8 | 29,697.5 | **302.9** | — | — |
| **VAL S3A** | Val (same driver, new route) | 4,168.2 | 2,664.6 | 5,751.7 | 7,053.1 | **1,427.8** | 5,751.7 | 7,378,541 | 2,958,901 |
| **VAL VTA1A** | Val (unseen vehicle VW Golf) | 1,505.9 | 1,207.9 | 2,249.0 | 6,927.9 | 4,191.2 | **2,249.0** | 6,020,675 | 1,738,650 |
| **VAL ALL** | Val (Aggregate) | 2,746.1 | 2,021.6 | 4,255.9 | 6,986.5 | **3,214.5** | 4,255.9 | — | — |
| **TEST Y1** | Test (Unseen Driver D) | 2,847.3 | 425.9 | 3,979.8 | 29,246.1 | 36,768.4 | **3,979.8** | 3,977,973 | 1,448,192 |
| **TEST VTA2** | Test (Nominal locked GNSS) | 211.6 | 34.9 | 212.4 | **5.6** | 247.1 | 212.4 | **455.1** | 15,396.1 |
| **TEST ALL** | Test (Aggregate) | 2,521.5 | 398.9 | 3,726.5 | 27,379.1 | 34,421.3 | **3,726.5** | — | — |

---

## 8. Forensic Analysis & Root-Cause Failure Identification

### 8.1 Feature Importance Breakdown
Inspection of split gain across the 3 trained XGBoost models reveals extreme concentration:

| Feature Name | Category | Importance (East) | Importance (North) | Importance (Up) | Mean Importance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `navris_vel_east_mps` | NAVRIS State | **0.4233** | 0.3665 | 0.3619 | **38.39%** |
| `navris_vel_up_mps` | NAVRIS State | 0.1171 | 0.0298 | **0.5374** | **22.81%** |
| `navris_speed_mps` | NAVRIS State | **0.3774** | 0.1069 | 0.0582 | **18.08%** |
| `navris_vel_north_mps` | NAVRIS State | 0.0677 | **0.4398** | 0.0144 | **17.40%** |
| `navris_heading_sin` | NAVRIS State | 0.0096 | 0.0136 | 0.0093 | 1.08% |
| `navris_delta_vel_north_1s` | Temporal Delta | 0.0014 | 0.0223 | 0.0060 | 0.99% |
| `navris_delta_vel_east_1s` | Temporal Delta | 0.0024 | 0.0076 | 0.0106 | 0.69% |
| `gyro_mag_rolling_mean_1s` | IMU Dynamics | 0.0001 | 0.0093 | 0.0001 | 0.32% |
| *All Raw Accel & Gyro Signals* | Raw IMU | 0.0000 | 0.0001 | 0.0000 | **< 0.02%** |

**Forensic Finding 1: The Model Learned an Inversion Heuristic, Not Inertial Dynamics.**
The instantaneous NAVRIS velocity estimates account for **96.7% of total feature importance**. Raw accelerations, raw angular rates, and gravity deviations account for virtually 0.0% of decision tree splits.
Because vehicle physical speed is bounded ($0 \le v \le 30$ m/s), when the classical filter dead-reckoning drifts to thousands of m/s, the target residual is simply:
$$\Delta \mathbf{v}^n = \mathbf{v}_{\text{ref}}^n - \mathbf{v}_{\text{NAVRIS}}^n \approx -\mathbf{v}_{\text{NAVRIS}}^n$$
The gradient-boosted trees discovered that predicting the negative of the drifted filter velocity minimizes mean squared error across training runs S1, S2, and S4.

### 8.2 Catastrophic Failure on Nominal Tracking (VTA2)
The most critical revelation of this experiment occurs on held-out recording **VTA2**:
- In VTA2, GNSS is available and valid 96.7% of the time. The classical ESKF Baseline A is nominally conditioned, tracking true vehicle motion with a horizontal velocity RMSE of **5.53 m/s** and a final position drift of only **455 m**.
- Because the ML model was trained on unconstrained dead-reckoning runs (where vertical velocity integrated to $> 10,000$ m/s), it learned a severe bias. When applied to VTA2, the ML model injected:
  - An Up-axis bias of **$-208.66$ m/s**
  - A horizontal velocity RMSE degradation from **5.53 m/s to 34.91 m/s** (6x worse)
  - An open-loop horizontal position drift increase from **455.1 m to 15,396.1 m (33x worse!)**

---

## 9. Limitations & Scientific Findings

1. **State Dependence & Non-Stationarity**:
   A local navigation residual $\Delta \mathbf{v}^n$ conditioned on unconstrained filter velocity is non-stationary; it depends on how long the filter has been running in dead-reckoning mode rather than instantaneous vehicle dynamics.
2. **Distribution Mismatch between Nominal and Outage Modes**:
   A single regressor trained across divergent runs cannot distinguish between a filter operating in high-accuracy GNSS-locked mode vs. pure dead reckoning. Applying it unconditionally destroys well-calibrated navigation states.
3. **Classical NHC vs. ML on S3A**:
   On validation route S3A, classical non-holonomic constraints (Baseline B) reduced 3D velocity RMSE to **1,427.8 m/s**, whereas open-loop ML (Baseline C) achieved only **5,751.7 m/s**. Deterministic kinematic constraints outperformed the supervised ML model by 400% on S3A without requiring training data.

---

## 10. Final Decision: Proceed to Closed-Loop Correction?

### Verdict: **NO — DO NOT PROCEED TO CLOSED-LOOP ML CORRECTION**

**Justification**:
1. **Safety Criterion Not Met**: An ML model that degrades a nominal 455-meter tracking route to 15,396 meters poses an intolerable failure risk to the navigation pipeline.
2. **Lack of Dynamic Conditioning**: The model has not learned body-frame acceleration or gyro bias residuals; it has merely memorized an open-loop velocity clipping rule.
3. **Recommended Next Steps before any Closed-Loop Attempt**:
   - Reframe the ML target from global navigation-frame velocity residual ($\Delta \mathbf{v}^n$) to **body-frame acceleration / angular velocity bias corrections** ($\Delta \mathbf{f}^b, \Delta \boldsymbol{\omega}^b$) or **pseudo-measurement innovation gates**.
   - Condition ML inference explicitly on GNSS outage duration or covariance trace.
   - Retain existing Phase 2 classical navigation (ESKF + ZUPT + NHC) as the validated baseline.
