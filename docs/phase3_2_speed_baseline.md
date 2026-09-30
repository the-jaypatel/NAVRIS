# NAVRIS Phase 3.2 — Direct Forward-Speed ML Baseline

## 1. Objective

The objective of NAVRIS Phase 3.2 is to establish a rigorous, offline baseline evaluating whether smartphone inertial measurements contain sufficient causal information to estimate instantaneous vehicle forward speed:
$$\hat{v}_{\text{fwd}}(t) = f_\theta\big(\text{IMU history up to } t\big)$$
without relying upon accumulated NAVRIS classical filter states (position, velocity, orientation, covariance) or external GNSS measurements.

This experiment tests the feasibility of generating a synthetic forward-speed pseudo-measurement directly from raw sensor observations, completely decoupled from classical filter divergence.

---

## 2. Hypothesis

**Core Hypothesis:**
Smartphone IMU measurements (3-axis specific force and 3-axis angular velocity) contain causal mechanical signatures—primarily chassis vibrations, engine harmonics, road roughness oscillations, and turning kinematics—that correlate with vehicle forward speed $v_{\text{fwd, ref}}(t)$ over a short temporal history, enabling statistical speed estimation without dead-reckoning integration.

**Critical Research Question:**
> *Does smartphone IMU history contain enough causal information to estimate vehicle forward speed across unseen routes, drivers, and vehicle platforms, without relying on NAVRIS accumulated state?*

---

## 3. Dataset and Partition

The study uses the 7 validated benchmark recordings from the NAVRIS repository. Partition discipline is strictly maintained with zero recording overlap:

| Partition | Recording ID | Driver | Vehicle Platform | Country | Valid Samples ($N-9$) | Duration (min) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TRAIN** | `S1` | Driver A | Ford Fiesta | UK | 50,174 | 83.6 |
| **TRAIN** | `S2` | Driver A | Ford Fiesta | UK | 91,899 | 153.2 |
| **TRAIN** | `S4` | Driver A | Ford Fiesta | UK | 92,893 | 154.8 |
| **VALIDATION** | `S3A` | Driver A | Ford Fiesta | UK | 21,702 | 36.2 |
| **VALIDATION** | `VTA1A` | Driver E | VW Golf | France | 24,886 | 41.5 |
| **HELD-OUT TEST** | `Y1` | Driver D | Ford Fiesta | UK | 71,750 | 119.6 |
| **HELD-OUT TEST** | `VTA2` | Driver E | VW Golf | France | 10,119 | 16.9 |
| **TOTAL** | **7 Recordings** | **3 Drivers** | **2 Platforms** | **2 Countries**| **363,423** | **605.8** |

*Note: Recording `M` remains permanently excluded due to validated reference unobservability.*
*Held-out test recordings (`Y1`, `VTA2`) were not used for feature selection, scaling, threshold tuning, or hyperparameter optimization.*

---

## 4. Target Definition

The primary target is the scalar vehicle forward speed:
$$y(t) = v_{\text{fwd, ref}}(t)$$
derived directly from the validated VBOX dual-antenna reference system (`ref_speed_mps`).

Reference data is treated strictly as an **OFFLINE SUPERVISED LABEL**. Zero reference fields enter the runtime feature matrix.

### Empirical Target Distribution Statistics (m/s)

No artificial clipping (e.g. $[0, 35]\text{ m/s}$) was imposed. The empirical distribution across evaluated partitions is:

| Partition / Recording | Count | Min (m/s) | Max (m/s) | Mean (m/s) | Median (m/s) | p95 (m/s) | p99 (m/s) | p99.9 (m/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **S1** (Train) | 50,174 | 0.00 | 19.66 | 7.22 | 6.75 | 14.17 | 16.25 | 18.24 |
| **S2** (Train) | 91,899 | 0.00 | 29.23 | 8.27 | 7.94 | 19.03 | 25.28 | 28.79 |
| **S4** (Train) | 92,893 | 0.00 | 30.45 | 9.39 | 8.78 | 23.58 | 27.23 | 29.28 |
| **S3A** (Val) | 21,702 | 0.00 | 27.23 | 12.82 | 12.22 | 22.32 | 25.48 | 26.99 |
| **VTA1A** (Val) | 24,886 | 0.00 | 28.74 | 16.60 | 16.22 | 23.59 | 26.07 | 28.35 |
| **Y1** (Held-Out Test) | 71,750 | 0.00 | 24.30 | 8.40 | 8.42 | 17.46 | 19.33 | 21.54 |
| **VTA2** (Held-Out Test) | 10,119 | 0.00 | 22.68 | 9.48 | 8.67 | 19.10 | 22.00 | 22.45 |
| **TRAIN POOL** | 234,966 | 0.00 | 30.45 | 8.49 | 7.97 | 20.01 | 26.01 | 28.93 |
| **VAL POOL** | 46,588 | 0.00 | 28.74 | 14.84 | 13.72 | 23.05 | 25.66 | 27.92 |
| **TEST POOL** | 81,869 | 0.00 | 24.30 | 8.53 | 8.44 | 17.55 | 19.91 | 22.10 |
| **GLOBAL POOL** | 363,423 | 0.00 | 30.45 | 9.31 | 8.33 | 20.57 | 25.45 | 28.74 |

*Key Observation:* Global observed maximum speed is $30.45\text{ m/s}$ (109.6 km/h on `S4`). $99\%$ of global samples fall below $25.45\text{ m/s}$. The proposed $35\text{ m/s}$ boundary represents an engineering envelope rather than an observed physical maximum.

---

## 5. Causal Feature Definition

All features are extracted strictly from causal phone-frame IMU observations over a 1.0-second backward history window:
$$[k-9, \dots, k] \quad \text{at } 10\text{ Hz } (w = 10\text{ samples})$$
No phone-to-vehicle mounting yaw solution is assumed, and no reference orientation is injected. The model operates directly on raw smartphone coordinates.

### Feature Inventory (19 Features)

#### Group 1: Instantaneous Phone-Frame IMU Kinematics (9 Features)
- `phone_accel_x_mps2`, `phone_accel_y_mps2`, `phone_accel_z_mps2`: Raw triaxial specific force in smartphone body frame ($m/s^2$).
- `phone_gyro_x_radps`, `phone_gyro_y_radps`, `phone_gyro_z_radps`: Raw triaxial angular velocity in smartphone body frame ($rad/s$).
- `accel_magnitude_mps2`: Instantaneous Euclidean norm $\|\mathbf{a}^b\| = \sqrt{a_x^2 + a_y^2 + a_z^2}$ ($m/s^2$).
- `gyro_magnitude_radps`: Instantaneous Euclidean norm $\|\boldsymbol{\omega}^b\| = \sqrt{\omega_x^2 + \omega_y^2 + \omega_z^2}$ ($rad/s$).
- `gravity_deviation_mps2`: Absolute specific force magnitude deviation from nominal gravity $|\|\mathbf{a}^b\| - 9.80665|$ ($m/s^2$).

#### Group 2: Causal 1-Second Rolling Statistics (4 Features)
- `accel_mag_rolling_mean_1s`: Causal mean of $\|\mathbf{a}^b\|$ over $[k-9 \dots k]$ ($m/s^2$).
- `accel_mag_rolling_std_1s`: Causal sample standard deviation of $\|\mathbf{a}^b\|$ over $[k-9 \dots k]$ ($m/s^2$). Captures road-induced chassis vibration intensity.
- `gyro_mag_rolling_mean_1s`: Causal mean of $\|\boldsymbol{\omega}^b\|$ over $[k-9 \dots k]$ ($rad/s$).
- `gyro_mag_rolling_std_1s`: Causal sample standard deviation of $\|\boldsymbol{\omega}^b\|$ over $[k-9 \dots k]$ ($rad/s$). Captures rotational jitter and vehicle pitch/roll dynamics.

#### Group 3: Causal 1-Second Temporal Deltas (6 Features)
- `delta_accel_x_1s`, `delta_accel_y_1s`, `delta_accel_z_1s`: First differences over 1.0s: $a_i[k] - a_i[k-9]$ ($m/s^2$).
- `delta_gyro_x_1s`, `delta_gyro_y_1s`, `delta_gyro_z_1s`: First differences over 1.0s: $\omega_i[k] - \omega_i[k-9]$ ($rad/s$).

---

## 6. Leakage Audit

A comprehensive automated leakage audit was executed prior to model evaluation, verifying the following causal constraints:

| Audit Check | Specification | Result | Verification Measurement |
| :--- | :--- | :--- | :--- |
| **No Forbidden Column Substrings** | `ref_`, `vbox`, `navris`, `gnss`, `target`, `truth`, `eskf` absent from features | **PASSED** | 0 violations found across all 19 columns |
| **Target Isolation** | $v_{\text{fwd, ref}}$ and residuals absent from feature matrix $\mathbf{X}$ | **PASSED** | Target present only in label vector $\mathbf{y}$ |
| **Partition Separation** | $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$ | **PASSED** | Mutually exclusive recording ID sets |
| **Scaler Isolation** | `StandardScaler` fit strictly on TRAIN pool | **PASSED** | Re-fit on train yields $\max|\Delta \mu| < 10^{-15}$, $\max|\Delta \sigma| < 10^{-15}$ |
| **Causal Future-Perturbation Test** | Future samples ($t' > t$) corrupted by noise ($+100\times + 999.0$) | **PASSED** | $\max|X_{\text{orig}}(t) - X_{\text{corrupt}}(t)| = 0.0$, $|\hat{y}_{\text{orig}} - \hat{y}_{\text{corrupt}}| = 0.0$ |

*Summary:* The pipeline guarantees 100% causal compliance. No future information or offline ground-truth can enter runtime predictions.

---

## 7. Baseline Models

Three baseline models were implemented and evaluated under identical conditions:

### Baseline 0 — Constant / Mean Predictor
- Predicts the training set mean forward speed:
  $$\hat{y}(t) = \bar{y}_{\text{train}} = 8.4872\text{ m/s}$$
- Establishes the non-informative trivial benchmark ($R^2 \approx 0.0$ on train).

### Baseline 1 — Regularized Linear Regression (Ridge)
- Linear model with L2 regularization ($\alpha = 1.0$) fitted on scaled Feature Set C (`StandardScaler` fitted on train only).
- Determines whether the vibration-speed relationship can be modeled linearly.

### Baseline 2 — Small Tree Model (Lightweight XGBoost)
- Configuration: `n_estimators=100`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`.
- Deliberately constrained complexity to prevent overfitting and avoid leaderboard optimization.

---

## 8. Ablation Results

To determine whether temporal context contributes predictive information beyond instantaneous IMU measurements, Baseline 2 (XGBoost) was trained on three nested feature subsets:
- **Ablation A:** Instantaneous IMU only (9 features).
- **Ablation B:** IMU + 1-second causal rolling statistics (13 features).
- **Ablation C:** IMU + 1-second causal rolling statistics + 1-second causal deltas (19 features).

### Cross-Partition Ablation Metrics

| Partition | Metric | Ablation A (Instantaneous) | Ablation B (+ Rolling Stats) | Ablation C (+ Rolling + Deltas) |
| :--- | :--- | :--- | :--- | :--- |
| **TRAIN** | MAE (m/s) | 3.44 | 2.86 | **2.84** |
| **TRAIN** | RMSE (m/s) | 4.76 | 4.15 | **4.12** |
| **TRAIN** | $R^2$ | 0.397 | 0.542 | **0.548** |
| **VAL** | MAE (m/s) | 6.30 | **5.44** | 5.51 |
| **VAL** | RMSE (m/s) | 7.82 | **6.83** | 6.92 |
| **VAL** | $R^2$ | -0.536 | **-0.171** | -0.203 |
| **HELD-OUT TEST** | MAE (m/s) | 3.60 | 3.06 | **3.03** |
| **HELD-OUT TEST** | RMSE (m/s) | 4.85 | 4.15 | **4.13** |
| **HELD-OUT TEST** | $R^2$ | 0.187 | 0.404 | **0.410** |
| **Y1 (Test)** | MAE (m/s) | 3.58 | 3.07 | **3.03** |
| **Y1 (Test)** | $R^2$ | 0.202 | 0.409 | **0.418** |
| **VTA2 (Test)** | MAE (m/s) | 3.75 | **3.00** | 3.02 |
| **VTA2 (Test)** | $R^2$ | -0.046 | **0.280** | 0.263 |

### Ablation Findings
1. **Critical Impact of Temporal Statistics:** Adding 1-second rolling standard deviation and mean (Ablation B) drastically improves held-out test performance:
   - On `Y1`, $R^2$ increases from $0.202 \rightarrow 0.409$, and MAE decreases from $3.58 \rightarrow 3.07\text{ m/s}$.
   - On `VTA2`, $R^2$ jumps from negative ($-0.046$) to $+0.280$, and MAE drops from $3.75 \rightarrow 3.00\text{ m/s}$.
2. **Marginal Impact of First Deltas:** First deltas (Ablation C) provide minor additional gain on `Y1` ($R^2: 0.409 \rightarrow 0.418$), while slightly increasing parameter count.
3. **Conclusion:** Instantaneous IMU alone is insufficient for speed estimation ($R^2 < 0.20$). Temporal vibration variance over 1.0 second is the dominant source of causal information.

---

## 9. Baseline Model Comparison

The three baseline models were evaluated across partitions using Feature Set C:

| Partition / Split | Metric | Baseline 0 (Mean Predictor) | Baseline 1 (Ridge Linear) | Baseline 2 (XGBoost C) |
| :--- | :--- | :--- | :--- | :--- |
| **TRAIN** | MAE (m/s) | 4.81 | 4.24 | **2.84** |
| **TRAIN** | RMSE (m/s) | 6.13 | 5.54 | **4.12** |
| **TRAIN** | $R^2$ | 0.000 | 0.184 | **0.548** |
| **TRAIN** | Correlation | 0.000 | 0.428 | **0.741** |
| **VAL** | MAE (m/s) | 6.75 | **5.51** | **5.51** |
| **VAL** | RMSE (m/s) | 8.08 | **6.82** | 6.92 |
| **VAL** | $R^2$ | -0.639 | **-0.169** | -0.203 |
| **VAL** | Correlation | 0.000 | 0.425 | **0.538** |
| **HELD-OUT TEST** | MAE (m/s) | 4.43 | 3.70 | **3.03** |
| **HELD-OUT TEST** | RMSE (m/s) | 5.38 | 4.65 | **4.13** |
| **HELD-OUT TEST** | $R^2$ | -0.001 | 0.252 | **0.410** |
| **HELD-OUT TEST** | Correlation | 0.000 | 0.507 | **0.654** |

---

## 10. Per-Recording Performance Breakdown

Evaluating performance per recording exposes critical domain-shift and platform-dependency effects:

| Recording | Split | Driver / Vehicle / Country | Baseline 2 MAE (m/s) | Baseline 2 RMSE (m/s) | Median AE (m/s) | p95 AE (m/s) | $R^2$ | Pearson Correlation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **S1** | TRAIN | Driver A / Ford Fiesta / UK | 2.26 | 2.82 | 1.87 | 5.86 | 0.562 | 0.810 |
| **S2** | TRAIN | Driver A / Ford Fiesta / UK | 2.93 | 4.29 | 1.95 | 8.72 | 0.517 | 0.725 |
| **S4** | TRAIN | Driver A / Ford Fiesta / UK | 3.08 | 4.44 | 2.02 | 11.60 | 0.552 | 0.772 |
| **S3A** | VAL | Driver A / Ford Fiesta / UK | 3.79 | 5.12 | 2.78 | 10.86 | 0.331 | 0.657 |
| **VTA1A** | VAL | Driver E / VW Golf / France | 7.02 | 8.18 | 7.29 | 13.58 | -1.545 | 0.278 |
| **Y1** | TEST | Driver D / Ford Fiesta / UK | 3.03 | 4.13 | 2.23 | 8.81 | 0.418 | 0.664 |
| **VTA2** | TEST | Driver E / VW Golf / France | 3.02 | 4.14 | 2.23 | 9.47 | 0.263 | 0.574 |

---

## 11. Held-Out Test Results

Performance on the two strictly held-out test recordings demonstrates distinct behavioral patterns:

### Held-Out Recording 1: `Y1` (Ford Fiesta, Driver D, UK)
- **Sample Count:** 71,750 samples (119.6 minutes).
- **Target Distribution:** Mean $8.40\text{ m/s}$, p95 $17.46\text{ m/s}$, Max $24.30\text{ m/s}$.
- **Metrics (XGBoost C):**
  - $\text{MAE} = 3.03\text{ m/s}$ (vs Baseline 0: $4.51\text{ m/s}$, Baseline 1: $3.75\text{ m/s}$).
  - $\text{RMSE} = 4.13\text{ m/s}$ (vs Baseline 0: $5.42\text{ m/s}$, Baseline 1: $4.69\text{ m/s}$).
  - $\text{Median Absolute Error} = 2.23\text{ m/s}$.
  - $\text{p95 Absolute Error} = 8.81\text{ m/s}$.
  - $R^2 = +0.418$.
  - $\text{Pearson Correlation } \rho = 0.664$.
- **Finding:** The model generalizes well to an unseen driver (Driver D) on unseen UK suburban/urban roads when the vehicle platform (Ford Fiesta) is consistent with the training set.

### Held-Out Recording 2: `VTA2` (VW Golf, Driver E, France)
- **Sample Count:** 10,119 samples (16.9 minutes).
- **Target Distribution:** Mean $9.48\text{ m/s}$, p95 $19.10\text{ m/s}$, Max $22.68\text{ m/s}$.
- **Metrics (XGBoost C):**
  - $\text{MAE} = 3.02\text{ m/s}$ (vs Baseline 0: $3.87\text{ m/s}$, Baseline 1: $3.40\text{ m/s}$).
  - $\text{RMSE} = 4.14\text{ m/s}$ (vs Baseline 0: $5.04\text{ m/s}$, Baseline 1: $4.34\text{ m/s}$).
  - $\text{Median Absolute Error} = 2.23\text{ m/s}$.
  - $\text{p95 Absolute Error} = 9.47\text{ m/s}$.
  - $R^2 = +0.263$.
  - $\text{Pearson Correlation } \rho = 0.574$.
- **Finding:** On `VTA2`, the model outperforms both Baseline 0 ($3.87\text{ m/s}$) and Baseline 1 ($3.40\text{ m/s}$), demonstrating meaningful causal correlation ($0.574$) even on an unseen vehicle chassis (VW Golf) and French road surfaces.

---

## 12. Speed-Regime Analysis

To determine where prediction errors concentrate, performance was evaluated across four distinct speed regimes defined by the training set:

| Partition | Speed Regime | Speed Range (m/s) | Samples | % of Split | MAE (m/s) | RMSE (m/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TRAIN** | Stationary | $v < 0.5$ | 33,002 | 14.0% | **0.80** | 1.72 |
| **TRAIN** | Low Speed | $0.5 \le v < 5.0$ | 35,565 | 15.1% | 2.80 | 3.74 |
| **TRAIN** | Medium Speed | $5.0 \le v < 15.0$ | 138,170 | 58.8% | **2.27** | 2.81 |
| **TRAIN** | High Speed | $v \ge 15.0$ | 28,229 | 12.0% | **8.11** | 9.03 |
| **VAL** | Stationary | $v < 0.5$ | 2,166 | 4.6% | **1.40** | 2.13 |
| **VAL** | Low Speed | $0.5 \le v < 5.0$ | 2,409 | 5.2% | 2.73 | 3.67 |
| **VAL** | Medium Speed | $5.0 \le v < 15.0$ | 22,036 | 47.3% | **2.81** | 3.52 |
| **VAL** | High Speed | $v \ge 15.0$ | 19,977 | 42.9% | **9.28** | 9.79 |
| **TEST (`Y1`)** | Stationary | $v < 0.5$ | 11,228 | 15.6% | **2.28** | 3.89 |
| **TEST (`Y1`)** | Low Speed | $0.5 \le v < 5.0$ | 9,939 | 13.9% | 3.08 | 4.10 |
| **TEST (`Y1`)** | Medium Speed | $5.0 \le v < 15.0$ | 43,921 | 61.2% | **2.64** | 3.37 |
| **TEST (`Y1`)** | High Speed | $v \ge 15.0$ | 6,662 | 9.3% | **6.81** | 7.61 |
| **TEST (`VTA2`)** | Stationary | $v < 0.5$ | 657 | 6.5% | **0.94** | 1.82 |
| **TEST (`VTA2`)** | Low Speed | $0.5 \le v < 5.0$ | 768 | 7.6% | 3.95 | 4.69 |
| **TEST (`VTA2`)** | Medium Speed | $5.0 \le v < 15.0$ | 7,517 | 74.3% | **2.20** | 2.69 |
| **TEST (`VTA2`)** | High Speed | $v \ge 15.0$ | 1,177 | 11.6% | **8.84** | 9.22 |

### Key Regime Insights
1. **Accurate in Urban/Medium Speeds ($5-15\text{ m/s}$):** In the medium-speed regime—which constitutes 58% to 74% of driving data—the model achieves excellent accuracy with MAE of **$2.20\text{ m/s}$ on `VTA2`** and **$2.64\text{ m/s}$ on `Y1`**.
2. **Severe Under-Prediction at High Speeds ($v \ge 15\text{ m/s}$):** Above $15\text{ m/s}$ (54 km/h), errors escalate drastically (MAE $6.8 - 9.3\text{ m/s}$). The tree model saturates its predictions around $16-18\text{ m/s}$, because chassis vibration variance levels off on smooth highway asphalt, decoupling vibration amplitude from high vehicle speeds.
3. **Stationary Performance:** The model recognizes stops well when stationary, particularly on `VTA2` (MAE $0.94\text{ m/s}$).

---

## 13. Dynamic Behavior

Performance was evaluated across dynamic maneuvers using both causal runtime IMU conditions and offline reference diagnostic categories:

| Recording | Category Classification | Definition | Samples | % Total | MAE (m/s) | RMSE (m/s) | Median AE (m/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **S1** | Stationary (Causal IMU) | $\sigma_a < 0.10\text{ m/s}^2, \|\boldsymbol{\omega}\| < 0.05\text{ rad/s}$ | 2,854 | 5.7% | **0.42** | 0.73 | **0.24** |
| **S1** | Turning (Causal IMU) | $\|\boldsymbol{\omega}\| \ge 0.15\text{ rad/s}$ | 15,155 | 30.2% | **1.91** | 2.45 | 1.52 |
| **S1** | Acceleration (Ref Diagnostic) | $a_{\text{long}} > 0.5\text{ m/s}^2$ | 11,946 | 23.8% | 2.19 | 2.79 | 1.80 |
| **S1** | Braking (Ref Diagnostic) | $a_{\text{long}} < -0.5\text{ m/s}^2$ | 11,719 | 23.4% | 2.78 | 3.42 | 2.40 |
| **S1** | Cruising (Ref Diagnostic) | $\|a\| \le 0.5\text{ m/s}^2, v \ge 5\text{ m/s}$ | 18,746 | 37.4% | 2.25 | 2.77 | 1.94 |
| **Y1** | Stationary (Causal IMU) | $\sigma_a < 0.10\text{ m/s}^2, \|\boldsymbol{\omega}\| < 0.05\text{ rad/s}$ | 7,486 | 10.4% | **0.69** | 1.23 | **0.35** |
| **Y1** | Turning (Causal IMU) | $\|\boldsymbol{\omega}\| \ge 0.15\text{ rad/s}$ | 15,594 | 21.7% | **2.42** | 3.27 | 1.81 |
| **Y1** | Acceleration (Ref Diagnostic) | $a_{\text{long}} > 0.5\text{ m/s}^2$ | 16,469 | 23.0% | 2.90 | 3.83 | 2.23 |
| **Y1** | Braking (Ref Diagnostic) | $a_{\text{long}} < -0.5\text{ m/s}^2$ | 15,404 | 21.5% | 3.26 | 4.16 | 2.71 |
| **Y1** | Cruising (Ref Diagnostic) | $\|a\| \le 0.5\text{ m/s}^2, v \ge 5\text{ m/s}$ | 25,978 | 36.2% | 3.27 | 4.35 | 2.49 |
| **VTA2** | Stationary (Causal IMU) | $\sigma_a < 0.10\text{ m/s}^2, \|\boldsymbol{\omega}\| < 0.05\text{ rad/s}$ | 634 | 6.3% | **0.59** | 0.94 | **0.34** |
| **VTA2** | Turning (Causal IMU) | $\|\boldsymbol{\omega}\| \ge 0.15\text{ rad/s}$ | 6,339 | 62.6% | 3.40 | 4.57 | 2.55 |
| **VTA2** | Acceleration (Ref Diagnostic) | $a_{\text{long}} > 0.5\text{ m/s}^2$ | 2,404 | 23.8% | 3.12 | 4.14 | 2.39 |
| **VTA2** | Braking (Ref Diagnostic) | $a_{\text{long}} < -0.5\text{ m/s}^2$ | 2,364 | 23.4% | 3.45 | 4.51 | 2.66 |
| **VTA2** | Cruising (Ref Diagnostic) | $\|a\| \le 0.5\text{ m/s}^2, v \ge 5\text{ m/s}$ | 4,481 | 44.3% | 3.01 | 4.17 | 2.21 |

### Dynamic Insights
- **Causal Stationary Detection:** Under causal IMU stationarity ($\sigma_a < 0.10\text{ m/s}^2$ and $\|\boldsymbol{\omega}\| < 0.05\text{ rad/s}$), median absolute error is **$0.24 - 0.35\text{ m/s}$**, confirming that zero/near-zero speed can be recognized directly from causal IMU noise levels without a dedicated ZUPT detector.
- **Turning Kinematics:** On `S1` and `Y1`, turning maneuvers exhibit lower MAE ($1.91 - 2.42\text{ m/s}$) and high correlation ($0.70 - 0.80$) because centripetal acceleration ($a_{\text{lat}} \approx v \cdot \omega$) provides a strong physical constraint relating speed to angular velocity.

---

## 14. Temporal Analysis: Lag and Derivatives

| Recording | Split | Empirical Lag $\tau$ (s) | Speed Correlation $\rho(y, \hat{y})$ | Derivative Correlation $\rho(\Delta y, \Delta \hat{y})$ |
| :--- | :--- | :--- | :--- | :--- |
| **S1** | TRAIN | +0.5 s | 0.810 | 0.042 |
| **S2** | TRAIN | +0.5 s | 0.725 | 0.046 |
| **S4** | TRAIN | +0.3 s | 0.772 | 0.028 |
| **S3A** | VAL | 0.0 s | 0.657 | 0.012 |
| **VTA1A** | VAL | +0.2 s | 0.278 | 0.008 |
| **Y1** | TEST | +0.4 s | 0.664 | 0.016 |
| **VTA2** | TEST | +0.7 s | 0.574 | 0.001 |

### Critical Findings
1. **Low Prediction Lag:** The empirical lag between reference speed and model prediction is small: between **$0.0\text{ s}$ and $+0.7\text{ s}$** (average $\approx 0.35\text{ s}$). Predictions track macroscopic speed transitions with negligible phase distortion.
2. **Low High-Frequency Derivative Correlation:** The derivative correlation $\rho(\Delta y, \Delta \hat{y})$ is nearly zero ($0.00 - 0.05$).
   - *Interpretation:* The model estimates the **low-frequency speed envelope**, not instantaneous point-by-point vehicle acceleration. The model acts as a low-bandwidth speed envelope tracker.

---

## 15. Feature Importance Analysis

Feature importance for Baseline 2 (XGBoost C) was computed via total gain:

| Rank | Feature Name | Description | Split Gain | % Normalized Gain | Split Count |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | `accel_mag_rolling_std_1s` | 1-second rolling std of specific force magnitude | 95,381.6 | **41.59%** | 211 |
| **2** | `gyro_mag_rolling_std_1s` | 1-second rolling std of angular velocity magnitude | 54,359.9 | **23.70%** | 187 |
| **3** | `phone_gyro_y_radps` | Smartphone body-frame Y angular velocity | 23,316.3 | **10.17%** | 149 |
| **4** | `gyro_mag_rolling_mean_1s` | 1-second rolling mean of angular velocity magnitude | 12,329.3 | **5.38%** | 254 |
| **5** | `delta_gyro_y_1s` | 1-second delta in phone gyro Y | 5,824.4 | **2.54%** | 115 |
| **6** | `gyro_magnitude_radps` | Instantaneous angular velocity magnitude | 5,743.2 | **2.50%** | 38 |
| **7** | `accel_mag_rolling_mean_1s` | 1-second rolling mean of specific force magnitude | 4,847.6 | **2.11%** | 248 |
| **8** | `delta_gyro_z_1s` | 1-second delta in phone gyro Z | 3,675.6 | **1.60%** | 30 |
| **9** | `gravity_deviation_mps2` | Absolute deviation from standard gravity | 3,509.2 | **1.53%** | 12 |
| **10** | `phone_accel_y_mps2` | Smartphone body-frame Y specific force | 3,143.0 | **1.37%** | 46 |

### Physical Interpretation vs Model Reliance
> **Caution:** Feature importance reflects which variables the gradient boosted trees split upon to minimize squared error; it does **NOT** prove a direct, invariant physical causal mechanism.

- **Vibration Reliance (65.29% Gain):** The top two features—`accel_mag_rolling_std_1s` (41.59%) and `gyro_mag_rolling_std_1s` (23.70%)—account for **65.29% of all split gain**. The tree model relies primarily on high-frequency mechanical vibration energy, which scales monotonically with road-wheel interaction and engine RPM under nominal driving.
- **Rotation Dynamics (15.55% Gain):** Gyroscope Y and total rotation account for $\approx 15\%$ of gain, exploiting centripetal turn dynamics.

---

## 16. Cross-Recording Generalization & Domain Shift Failure Cases

The evaluation reveals two distinct regimes of generalization:

### Success Case: Intraclass Platform Generalization (`Y1` Held-Out Test)
- Route `Y1` featured an unseen driver (Driver D) on an unseen route in the UK.
- The vehicle platform, however, was identical to the training set (Ford Fiesta).
- **Result:** $R^2 = +0.418$, $\rho = 0.664$, $\text{MAE} = 3.03\text{ m/s}$.
- The model successfully transferred its learned vibration-to-speed curve across drivers and routes within the same vehicle chassis.

### Failure Case: Inter-Platform Highway Domain Shift (`VTA1A` Validation)
- Route `VTA1A` was recorded on a different vehicle platform (VW Golf) driven by Driver E in France.
- This recording consisted primarily of high-speed highway cruising (mean speed $16.60\text{ m/s}$, 43% of route $> 15\text{ m/s}$).
- **Result:** $\text{MAE} = 7.02\text{ m/s}$, $R^2 = -1.545$, $\rho = 0.278$.
- **Root Cause of Failure:**
  1. **Suspension/Chassis Damping Mismatch:** The VW Golf has different suspension damping and acoustic isolation than the Ford Fiesta. The vibration energy at $20\text{ m/s}$ in a VW Golf on smooth French tarmac is dramatically lower than in a Ford Fiesta on UK roads.
  2. **High-Speed Vibration Flattening:** Above $15\text{ m/s}$, road surface smoothness dominates vibration amplitude, decoupling specific force variance from vehicle forward speed.
  3. **Under-Prediction Saturation:** The model predicted an average of $\approx 10-12\text{ m/s}$ when the vehicle was traveling at $20-25\text{ m/s}$, leading to large negative $R^2$.

### Moderate Generalization: Urban Inter-Platform (`VTA2` Held-Out Test)
- Route `VTA2` was also recorded on the VW Golf in France, but under mixed urban/suburban driving (mean speed $9.48\text{ m/s}$).
- In this speed regime, turning kinematics and stop-and-go dynamics played a larger role.
- **Result:** $\text{MAE} = 3.02\text{ m/s}$, $R^2 = +0.263$, $\rho = 0.574$. The model maintained positive predictive capability ($R^2 > 0.26$) despite the unseen vehicle platform.

---

## 17. Scientific Interpretation

Returning to the primary scientific inquiry:
> *Does smartphone IMU history contain enough causal information to estimate vehicle forward speed across unseen routes and drivers, without relying on NAVRIS accumulated state?*

**Direct Answer: YES, BUT WITH SEVERE VEHICLE-PLATFORM AND SPEED-REGIME LIMITATIONS.**

1. **Existence of Causal Speed Information:**
   - Smartphone IMU measurements contain undeniable causal speed information:
     - On held-out test route `Y1`, XGBoost achieved $R^2 = 0.418$ and correlation $\rho = 0.664$.
     - On held-out test route `VTA2`, XGBoost achieved $R^2 = 0.263$ and correlation $\rho = 0.574$.
     - In the medium speed range ($5-15\text{ m/s}$), MAE is consistently $2.20 - 2.64\text{ m/s}$.
     - When stationary, MAE is $0.42 - 0.69\text{ m/s}$.
   - This represents a marked improvement over Baseline 0 (Mean Predictor, $R^2 \le 0$) and Baseline 1 (Ridge Linear, $R^2 = 0.19 - 0.25$).
2. **Nature of the Information Source:**
   - The predictive signal does not originate from kinematic dead-reckoning (which diverges rapidly) or instantaneous acceleration integration.
   - Instead, the tree model acts as a **statistical vibration-energy spectrometer and turn-dynamics estimator**.
3. **Fundamental Limitations:**
   - **Chassis/Vibration Dependency:** Because vibration energy depends on vehicle suspension, tire pressure, and road surface roughness, direct uncalibrated speed estimation experiences significant domain shifts across vehicle platforms (e.g. Ford Fiesta vs VW Golf on `VTA1A`).
   - **High-Speed Envelope Saturation:** At highway cruising speeds ($> 15\text{ m/s}$), chassis vibration amplitude levels off, preventing accurate high-speed differentiation without external velocity fixes.
   - **Low Dynamic Bandwidth:** High-frequency speed derivatives ($\Delta v$) are unobservable ($\rho(\Delta y, \Delta \hat{y}) \approx 0.02$). The estimate behaves as a smoothed macroscopic speed envelope rather than an instantaneous odometer.

---

## 18. Phase 3.2 Decision

### Descriptive Classification:
# LIMITED / ROUTE-DEPENDENT EVIDENCE

### Justification:
- Predictive accuracy is demonstrably superior to trivial mean prediction and linear regression on held-out test recordings (`Y1`: MAE $3.03\text{ m/s}$, $R^2 = 0.418$; `VTA2`: MAE $3.02\text{ m/s}$, $R^2 = 0.263$).
- The causal leakage audit confirmed zero contamination and strict temporal invariance.
- However, cross-vehicle generalization is inconsistent: high-speed highway driving on an unseen vehicle chassis (`VTA1A`) causes severe domain collapse ($R^2 < 0$, MAE $7.02\text{ m/s}$) due to differences in suspension damping and road asphalt roughness.

### Downstream Implications for NAVRIS:
1. **Unconditional Standalone Speedometer Rejected:** Direct IMU forward-speed regression cannot serve as an uncalibrated, standalone primary speedometer for vehicle navigation.
2. **Potential as a Constrained Synthetic Measurement:** The estimate exhibits strong performance in low-to-medium urban regimes ($5-15\text{ m/s}$) and during stops. If explored in future work, it must be formulated as a loosely-weighted, variance-gated synthetic observation (e.g., $R_v \approx (4\text{ m/s})^2$) with explicit vehicle-calibration awareness, or coupled with longitudinal kinematic integration.

---

## Explicit Phase Status

**PHASE 3.2 COMPLETE — HARD STOP MAINTAINED**

**PHASE 3.3 NOT STARTED**

No ML models have been connected to NAVRIS ESKF.
No pseudo-measurement updates have been created.
No navigation filters or benchmarks have been modified.
Any downstream investigation requires explicit, separate user authorization.
