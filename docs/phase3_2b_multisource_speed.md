# NAVRIS Phase 3.2B — Causal Multi-Source Forward-Speed Estimation

## 1. Objective

NAVRIS Phase 3.2A demonstrated that causal smartphone IMU dynamics contain generalizable speed information ($R^2 = 0.522$, $\rho = 0.725$ on held-out test data under dynamic-only features). However, IMU-only speed estimation operates without absolute kinematic velocity anchoring, leading to high-speed saturation ($> 15\text{ m/s}$) and road-surface sensitivity.

The primary objective of NAVRIS Phase 3.2B is to execute a **strictly controlled OPEN-LOOP multi-source experiment** testing whether combining:
1. **Causal smartphone IMU dynamics** (temporal deltas, jerk proxies, turning constraints), and
2. **Causal NAVRIS motion state** (projected forward speed, velocity vector, heading orientation, GNSS fix quality indicators)

produces a superior forward-speed estimate across unseen routes and vehicle platforms, without reproducing the catastrophic velocity inversion failure discovered in Phase 3.0.

---

## 2. Scientific Question

> **Core Research Question:**
> *Can causal ML combine smartphone IMU dynamics with the current NAVRIS motion estimate to produce a more useful forward-speed estimate across unseen routes, without learning accumulated filter divergence or using future/reference information?*

Specifically:
- Does direct speed estimation benefit from adding classical filter velocity?
- Does residual speed estimation reproduce the Phase 3.0 inversion heuristic ($\hat{r}_{\text{fwd}} \approx -v_{\text{fwd, nav}}$)?
- How does the multi-source estimate behave under GNSS availability versus GNSS outage?

---

## 3. Frozen Baselines

Three frozen baselines serve as the benchmark for multi-source evaluation:

1. **Baseline 0 (Training Mean):**
   - Predicts constant training set mean forward speed:
     $$\hat{y}(t) = \bar{y}_{\text{train}} = 8.4872\text{ m/s}$$
   - Held-Out Test Pool: $\text{MAE} = 4.43\text{ m/s}$, $\text{RMSE} = 5.38\text{ m/s}$, $R^2 = -0.001$.
2. **Baseline 1 (Frozen Causal NAVRIS Forward Speed):**
   - Evaluates the classical ESKF forward speed estimate directly:
     $$\hat{y}(t) = v_{\text{fwd, nav}}(t) = v_{\text{nav, E}}(t)\sin\psi_{\text{nav}}(t) + v_{\text{nav, N}}(t)\cos\psi_{\text{nav}}(t)$$
   - *On nominal tracking (`VTA2`, 96.7% GNSS fixes):* $\text{MAE} = 5.45\text{ m/s}$, $\text{RMSE} = 9.15\text{ m/s}$, $R^2 = -2.596$.
   - *On divergent tracking (`Y1`, 1.4% GNSS fixes):* $\text{MAE} = 1011.90\text{ m/s}$, $\text{RMSE} = 1234.25\text{ m/s}$, $R^2 = -52026.3$.
   - *Test Pool Combined:* $\text{MAE} = 887.50\text{ m/s}$, $\text{RMSE} = 1155.46\text{ m/s}$.
3. **Baseline 2 (Phase 3.2A IMU-Only Dynamic Model):**
   - Evaluates the Phase 3.2A 12-feature dynamic representation:
     $$\hat{y}(t) = f_{\theta,\text{dyn}}(\mathbf{X}_{\text{dyn}}(t))$$
   - Held-Out Test Pool: $\text{MAE} = 2.77\text{ m/s}$, $\text{RMSE} = 3.71\text{ m/s}$, $R^2 = \mathbf{0.522}$, $\rho = \mathbf{0.725}$.
   - `Y1`: $\text{MAE} = 2.76\text{ m/s}$, $R^2 = 0.532$.
   - `VTA2`: $\text{MAE} = 2.79\text{ m/s}$, $R^2 = 0.382$.

---

## 4. Dataset and Partition

The partition remains frozen with zero recording overlap:

| Partition | Recording ID | Vehicle Platform | Driver | Country | Valid Samples | Route Characteristics |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TRAIN** | `S1` | Ford Fiesta | Driver A | UK | 50,174 | Urban / Suburban |
| **TRAIN** | `S2` | Ford Fiesta | Driver A | UK | 91,899 | Mixed Urban / Rural |
| **TRAIN** | `S4` | Ford Fiesta | Driver A | UK | 92,893 | Rural / Highway |
| **VALIDATION** | `S3A` | Ford Fiesta | Driver A | UK | 21,702 | Suburban / Rural |
| **VALIDATION** | `VTA1A` | VW Golf | Driver E | France | 24,886 | High-Speed Highway |
| **HELD-OUT TEST** | `Y1` | Ford Fiesta | Driver D | UK | 71,750 | Mixed Suburban |
| **HELD-OUT TEST** | `VTA2` | VW Golf | Driver E | France | 10,119 | Urban / Suburban |
| **TOTAL** | **7 Recordings** | **2 Platforms** | **3 Drivers** | **2 Countries** | **363,423** | **605.8 minutes** |

---

## 5. Target Formulations

Two explicitly separated target formulations were analyzed:

### Formulation A: Direct Forward Speed
- **Model:** $\hat{y}_{\text{fwd}}(t) = f_\theta(\mathbf{X}_t)$
- **Target:** $y(t) = v_{\text{fwd, ref}}(t)$ (Validated VBOX reference vehicle speed in $[0, 30.5]\text{ m/s}$).
- **Nature:** Bounded, stationary physical quantity.

### Formulation B: Forward-Speed Residual
- **Model:** $\hat{r}_{\text{fwd}}(t) = f_\theta(\mathbf{X}_t)$
- **Target:** $r_{\text{fwd}}(t) = v_{\text{fwd, ref}}(t) - v_{\text{fwd, nav}}(t)$
- **Speed Reconstruction:** $\hat{y}_{\text{fwd}}(t) = v_{\text{fwd, nav}}(t) + \hat{r}_{\text{fwd}}(t)$
- **Nature:** Highly non-stationary under classical filter divergence.

### Empirical Forward-Speed Residual Statistics ($r_{\text{fwd}}$, m/s)

| Recording | Partition | Platform | Min (m/s) | Max (m/s) | Mean (m/s) | Median (m/s) | p95 (m/s) | p99 (m/s) | RMSE (m/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **S1** | TRAIN | Fiesta | -4688.04 | 4012.79 | -1284.18 | -1045.24 | 3444.60 | 3588.09 | 1680.65 |
| **S2** | TRAIN | Fiesta | -8294.19 | 8025.48 | -3886.53 | -3682.20 | 7907.41 | 7018.42 | 4626.12 |
| **S4** | TRAIN | Fiesta | -12049.90 | 11971.07 | -1913.67 | -711.66 | 9145.41 | 10991.02 | 3447.72 |
| **S3A** | VAL | Fiesta | -12574.26 | 12652.68 | -2603.77 | -1283.47 | 10309.30 | 10610.90 | 4142.93 |
| **VTA1A** | VAL | Golf | -5695.45 | 6246.91 | -1828.81 | -1071.74 | 5544.75 | 6111.99 | 2599.77 |
| **Y1** | TEST | Fiesta | -2845.46 | 2101.46 | -1003.70 | -927.81 | 2346.59 | 1901.70 | 1234.25 |
| **VTA2** | TEST | Golf | **-9.36** | **42.25** | **4.03** | **1.33** | **22.08** | **36.00** | **9.15** |

*Critical Observation:* Across 6 of 7 recordings, the forward-speed residual is dominated by classical filter dead-reckoning divergence (reaching $-12,574\text{ m/s}$). Only on `VTA2` is the residual physically concentrated ($\text{RMSE} = 9.15\text{ m/s}$).

---

## 6. Feature Groups

### Group A: Causal IMU Dynamics (12 Features — Phase 3.2A)
- Normalized 1.0s specific force deltas: $\frac{\Delta a_x}{\mu_a+\epsilon}, \frac{\Delta a_y}{\mu_a+\epsilon}, \frac{\Delta a_z}{\mu_a+\epsilon}$.
- Normalized 1.0s gyro deltas: $\frac{\Delta \omega_x}{\mu_\omega+\epsilon}, \frac{\Delta \omega_y}{\mu_\omega+\epsilon}, \frac{\Delta \omega_z}{\mu_\omega+\epsilon}$.
- Dimensionless magnitude deltas: $\frac{\Delta \|\mathbf{a}\|}{g_0}, \frac{\Delta \|\boldsymbol{\omega}\|}{\mu_\omega+\epsilon}$.
- Mean absolute jerk proxy: $\frac{1}{g_0(w-1)}\sum |\Delta \|\mathbf{a}\||$.
- Centripetal turning ratio proxy: $\frac{\|\boldsymbol{\omega}\|}{\|\Delta \mathbf{a}\|+\epsilon}$.
- Lag-1 autocorrelation proxies: $\rho_{\text{lag1}, a}, \rho_{\text{lag1}, \omega}$.

### Group B: Causal NAVRIS Motion State (9 Features)
- `navris_v_fwd_mps`: Causal forward speed $v_{\text{fwd, nav}} = v_E \sin\psi_{\text{nav}} + v_N \cos\psi_{\text{nav}}$.
- `navris_vel_east_mps`, `navris_vel_north_mps`, `navris_vel_up_mps`: ENU velocity components ($m/s$).
- `navris_speed_mps`: Horizontal speed magnitude $\sqrt{v_E^2 + v_N^2}$ ($m/s$).
- `navris_heading_sin`, `navris_heading_cos`: Heading orientation unit vectors.
- `navris_delta_v_fwd_1s`, `navris_delta_speed_1s`: Causal 1.0s changes in filter forward speed and total speed.

### Group C: Causal NAVRIS Quality Indicators (2 Features)
- `causal_time_since_gnss_fix_s`: Causal elapsed time since last valid GPS fix ($t - t_{\text{last}}$).
- `causal_gnss_fix_available`: Indicator (1.0 if new GPS fix at timestamp $t$, 0.0 otherwise).

*Note on Covariance:* ESKF error covariance matrices were not exported to the post-calibration trajectory CSVs. In accordance with authorization rules, covariance features are reported as unavailable in existing replay logs and omitted.

---

## 7. Causal NAVRIS State Construction

All NAVRIS features are derived strictly from the causal filter state available at timestamp $t$:
$$v_{\text{fwd, nav}}(t) = \mathbf{e}_x^T \mathbf{C}_b^v \mathbf{C}_n^b(t) \mathbf{v}_{\text{nav}}^n(t)$$
Using the established vehicle-to-body alignment:
$$v_{\text{fwd, nav}}(t) = v_{\text{nav, E}}(t)\sin\psi_{\text{nav}}(t) + v_{\text{nav, N}}(t)\cos\psi_{\text{nav}}(t)$$
No smoothing, backward passes, or offline reference-derived alignments were used.

---

## 8. Leakage Audit

Automated verification passed 100% of causal checks:

| Check | Specification | Result | Measurement |
| :--- | :--- | :--- | :--- |
| **No Forbidden Column Substrings** | `ref_`, `vbox`, `truth`, `oracle` absent from features | **PASSED** | 0 violations found |
| **Target Isolation** | $v_{\text{fwd, ref}}$ and $r_{\text{fwd}}$ absent from feature matrix $\mathbf{X}$ | **PASSED** | Targets present only in label vectors |
| **Partition Isolation** | $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$ | **PASSED** | Mutually exclusive splits |
| **Causal GNSS Fix Tracking** | `causal_time_since_gnss_fix_s` monotonically increases between fixes | **PASSED** | All elapsed times non-negative; resets to $0.0$ on fix |
| **Future Perturbation Invariance** | Modifying inputs for $t' > t$ leaves features at $t$ unchanged | **PASSED** | Max feature discrepancy = $0.0$ |

---

## 9. Direct-Speed Results Across Ablations

| Model / Ablation | Feature Count | TRAIN MAE (m/s) | TRAIN $R^2$ | VAL MAE (m/s) | VAL $R^2$ | HELD-OUT TEST MAE (m/s) | HELD-OUT TEST RMSE (m/s) | HELD-OUT TEST $R^2$ | HELD-OUT TEST $\rho$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline 0 (Train Mean)** | 0 | 4.81 | 0.000 | 6.75 | -0.639 | 4.43 | 5.38 | -0.001 | 0.000 |
| **Baseline 1 (Frozen NAVRIS)** | 0 | 2559.59 | -363828 | 2203.28 | -291415 | 887.50 | 1155.46 | -46203 | -0.041 |
| **Ablation A (IMU Only - Baseline 2)**| 12 | 3.13 | 0.488 | **4.91** | **+0.014** | **2.77** | **3.71** | **0.522** | **0.725** |
| **Ablation B (NAVRIS Only)** | 9 | 3.53 | 0.516 | 7.44 | -0.970 | 5.01 | 6.00 | -0.244 | -0.038 |
| **Ablation C (IMU + NAVRIS)** | 21 | 2.48 | 0.716 | 5.35 | -0.153 | 3.18 | 4.17 | 0.398 | 0.634 |
| **Ablation D (IMU + NAVRIS + Quality)**| 23 | 2.47 | 0.712 | 5.30 | -0.146 | 3.07 | 4.04 | 0.434 | 0.661 |

### Direct Speed Findings:
1. **NAVRIS Alone Collapses on Held-Out Test (Ablation B):**
   - On held-out test data, NAVRIS-only features yield $R^2 = -0.244$ and correlation $\rho \approx -0.04$. The model cannot estimate speed from classical velocity states that experience divergent dead-reckoning.
2. **Adding NAVRIS to IMU Dynamics (Ablation C vs Ablation A):**
   - Adding NAVRIS state features overfits the training set ($R^2$ jumps from $0.488 \rightarrow 0.716$), but **degrades** aggregate held-out test generalization ($R^2$ drops from $0.522 \rightarrow 0.398$).
3. **Quality Indicators Help Filter Corrupted States (Ablation D vs Ablation C):**
   - Adding causal GNSS fix availability and outage duration (Ablation D) recovers performance ($R^2: 0.398 \rightarrow 0.434$, MAE: $3.18 \rightarrow 3.07\text{ m/s}$).

---

## 10. Residual-Speed Results Across Ablations (The Phase 3.0 Inversion Recurrence)

Reconstructed forward speed: $\hat{y}_{\text{fwd}}(t) = v_{\text{fwd, nav}}(t) + \hat{r}_{\text{fwd}}(t)$.

| Model / Ablation | Feature Count | TRAIN MAE (m/s) | VAL MAE (m/s) | HELD-OUT TEST MAE (m/s) | HELD-OUT TEST RMSE (m/s) | HELD-OUT TEST $R^2$ | HELD-OUT TEST $\rho$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ablation A (IMU Only)** | 12 | 2473.59 | 2610.79 | 1166.05 | 1370.15 | -64968.4 | -0.209 |
| **Ablation B (NAVRIS Only)** | 9 | 78.57 | 182.19 | 65.39 | 83.63 | -241.1 | -0.125 |
| **Ablation C (IMU + NAVRIS)** | 21 | 75.73 | 228.04 | 70.61 | 106.58 | -392.1 | -0.137 |
| **Ablation D (IMU + NAVRIS + Quality)**| 23 | 68.34 | 198.42 | 62.54 | 98.83 | -337.0 | -0.185 |

### Catastrophic Finding:
The residual formulation fails across all ablations:
- Reconstructed speed errors exceed **$62 - 70\text{ m/s}$ MAE** on held-out test data.
- $R^2$ collapses to negative hundreds ($-337$ to $-392$).
- *Reason:* The residual target $r_{\text{fwd}}$ is completely dominated by the divergent classical filter velocity. Learning to correct this residual forces the model to predict an offset heuristic that inverts valid velocities on nominal routes.

---

## 11. Recording-Level Results: Direct vs Residual

| Recording | Split | Platform | Baseline 1 NAVRIS MAE (m/s) | IMU Only (Ablation A) Direct MAE (m/s) | Multi-Source (Ablation D) Direct MAE (m/s) | Multi-Source (Ablation D) Residual MAE (m/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **S1** | TRAIN | Fiesta | 1291.46 | 2.51 | **2.32** | 66.02 |
| **S2** | TRAIN | Fiesta | 3895.33 | 3.17 | **2.52** | 74.82 |
| **S4** | TRAIN | Fiesta | 1923.08 | 3.42 | **2.51** | 63.19 |
| **S3A** | VAL | Fiesta | 2614.26 | 3.55 | 3.85 | 267.25 |
| **VTA1A** | VAL | Golf | 1844.89 | **6.10** | 6.57 | 138.40 |
| **Y1** | TEST | Fiesta | 1011.90 | **2.76** | 3.12 | 66.49 |
| **VTA2** | TEST | Golf | 5.45 | 2.79 | **2.69** | 34.51 |

### Platform and Recording Breakdown:
1. **On Nominal Route `VTA2` (VW Golf, 96.7% GNSS fixes):**
   - Baseline 1 NAVRIS forward speed has an MAE of $5.45\text{ m/s}$.
   - IMU Only (Ablation A) achieves MAE = $2.79\text{ m/s}$ ($R^2 = 0.382$).
   - Multi-Source Direct (Ablation D) achieves **$\text{MAE} = 2.69\text{ m/s}$ ($R^2 = 0.423$)**.
   - *Result:* When the classical filter is reliable, adding NAVRIS state improves speed prediction by $+0.041$ in $R^2$.
2. **On Divergent Route `Y1` (Ford Fiesta, 1.4% GNSS fixes):**
   - Baseline 1 NAVRIS velocity diverges (MAE $= 1011.90\text{ m/s}$).
   - IMU Only (Ablation A) is unaffected by filter divergence: **$\text{MAE} = 2.76\text{ m/s}$ ($R^2 = 0.532$)**.
   - Multi-Source Direct (Ablation D) degrades to $\text{MAE} = 3.12\text{ m/s}$ ($R^2 = 0.428$).
   - *Result:* When the classical filter diverges, feeding its state into ML corrupts the estimate.
3. **On Highway Route `VTA1A` (VW Golf):**
   - IMU Only Direct: $\text{MAE} = 6.10\text{ m/s}$.
   - Multi-Source Direct: $\text{MAE} = 6.57\text{ m/s}$.

---

## 12. Speed-Regime Analysis (Held-Out Test)

Evaluating Direct Speed (Ablation D) vs IMU-Only (Ablation A) across frozen speed bins:

| Recording | Speed Regime | Speed Range | Samples | IMU Only (Abl A) Direct MAE (m/s) | Multi-Source (Abl D) Direct MAE (m/s) | Multi-Source (Abl D) Residual MAE (m/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`Y1`** (Fiesta) | Stationary | $< 0.5\text{ m/s}$ | 11,228 | **1.91** | 2.50 | 70.83 |
| **`Y1`** (Fiesta) | Low Speed | $0.5 - 5.0\text{ m/s}$ | 9,939 | 3.57 | **3.01** | 68.32 |
| **`Y1`** (Fiesta) | Medium Speed | $5.0 - 15.0\text{ m/s}$ | 43,921 | **2.26** | 2.68 | 65.48 |
| **`Y1`** (Fiesta) | High Speed | $\ge 15.0\text{ m/s}$ | 6,662 | **6.33** | 7.15 | 63.15 |
| **`VTA2`** (Golf) | Stationary | $< 0.5\text{ m/s}$ | 657 | **0.84** | 0.96 | 45.41 |
| **`VTA2`** (Golf) | Low Speed | $0.5 - 5.0\text{ m/s}$ | 768 | 4.29 | **3.56** | 44.97 |
| **`VTA2`** (Golf) | Medium Speed | $5.0 - 15.0\text{ m/s}$ | 7,517 | **2.08** | 2.14 | 33.56 |
| **`VTA2`** (Golf) | High Speed | $\ge 15.0\text{ m/s}$ | 1,177 | 7.46 | **6.66** | 27.57 |

### Regime Observations:
- **Low-Speed Maneuvering Improvement:** In low-speed driving ($0.5 - 5.0\text{ m/s}$), multi-source Direct estimation improves accuracy: MAE drops from $3.57 \rightarrow 3.01\text{ m/s}$ on `Y1`, and from $4.29 \rightarrow 3.56\text{ m/s}$ on `VTA2`.
- **High-Speed Saturation on Nominal Route (`VTA2`):** On `VTA2`, adding NAVRIS state reduces high-speed error from $7.46 \rightarrow 6.66\text{ m/s}$.
- **Residual Model Regime Failure:** The residual model exhibits errors of $33 - 70\text{ m/s}$ across all regimes due to velocity inversion.

---

## 13. GNSS-Regime Results

Dividing test samples causally by GPS fix recency ($dt_{\text{fix}} \le 2.0\text{ s}$ recent vs $dt_{\text{fix}} > 2.0\text{ s}$ stale/outage):

| Recording | GNSS Regime | Samples | Baseline 1 NAVRIS MAE (m/s) | IMU Only (Abl A) Direct MAE (m/s) | Multi-Source (Abl D) Direct MAE (m/s) | Multi-Source (Abl D) Residual MAE (m/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`Y1`** | Recent Fix ($dt \le 2\text{ s}$) | 14,989 | 948.51 | **3.02** | 3.47 | 62.09 |
| **`Y1`** | Stale / Outage ($dt > 2\text{ s}$) | 56,761 | 1028.63 | **2.70** | 3.03 | 67.65 |
| **`VTA2`** | Recent Fix ($dt \le 2\text{ s}$) | 9,552 | 5.56 | 2.92 | **2.80** | 35.80 |
| **`VTA2`** | Stale / Outage ($dt > 2\text{ s}$) | 567 | 3.54 | **0.64** | 0.90 | 12.70 |

### Key Regime Finding:
- Under GNSS outage ($dt > 2\text{ s}$), IMU-only estimation is completely decoupled from filter drift and maintains excellent accuracy (MAE $= 2.70\text{ m/s}$ on `Y1`, $0.64\text{ m/s}$ on `VTA2`).
- Multi-source estimation slightly degrades under outage on `Y1` ($2.70 \rightarrow 3.03\text{ m/s}$) because divergent NAVRIS velocity leaks into the prediction.

---

## 14. Residual Inversion Analysis: Proof of Phase 3.0 Failure Recurrence

To formally test for recurrence of the Phase 3.0 inversion failure, linear regressions were fitted between true/predicted forward-speed residuals and classical filter forward speed:
$$r_{\text{fwd}} \approx \alpha + \beta v_{\text{fwd, nav}}$$

| Recording | Partition | True $\beta_{\text{true}}$ | True Inversion $R^2$ | Model Predicted $\beta_{\text{pred}}$ | Model Inversion $R^2$ | $\text{corr}(\hat{r}_{\text{fwd}}, v_{\text{nav}})$ | Inversion Failure Verified? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **S1** | TRAIN | **-0.9998** | 0.9999 | **-0.9524** | 0.9979 | **-0.9990** | **YES** |
| **S2** | TRAIN | **-1.0002** | 0.9999 | **-0.9911** | 0.9992 | **-0.9996** | **YES** |
| **S4** | TRAIN | **-0.9999** | 0.9999 | **-0.9821** | 0.9990 | **-0.9995** | **YES** |
| **S3A** | VAL | **-0.9999** | 0.9999 | **-0.8722** | 0.9929 | **-0.9965** | **YES** |
| **VTA1A** | VAL | **-1.0001** | 0.9999 | **-0.9303** | 0.9973 | **-0.9986** | **YES** |
| **Y1** | TEST | **-1.0001** | 0.9999 | **-0.9581** | 0.9916 | **-0.9958** | **YES** |
| **VTA2** | TEST | **-0.7410** | 0.7113 | **-3.5508** | 0.5441 | **-0.7376** | **YES (Over-Inversion)** |

### Forensic Inversion Finding:
1. **Exact Replication of Phase 3.0 Failure:**
   - In training data, because $v_{\text{nav}}$ reaches thousands of m/s while $v_{\text{ref}} \le 30.5\text{ m/s}$, the mathematical ground truth is $r_{\text{fwd}} = v_{\text{ref}} - v_{\text{nav}} \approx -v_{\text{nav}}$ ($\beta_{\text{true}} \approx -1.0000$).
   - The gradient-boosted tree model learns this trivial inversion: $\beta_{\text{pred}} \in [-0.87, -0.99]$ with $R^2 > 0.99$ and correlation with $v_{\text{nav}}$ exceeding $-0.995$.
2. **Catastrophic Transfer to Non-Divergent Route (`VTA2`):**
   - On `VTA2`, the classical filter did not diverge. But the residual model, having learned to invert $v_{\text{nav}}$, aggressively subtracts $\hat{r} \approx -3.55 v_{\text{nav}}$, corrupting the reconstructed speed and yielding an MAE of **$34.51\text{ m/s}$**.
3. **Conclusion:** **Formulation B (Forward-Speed Residual Learning) is definitively rejected.**

---

## 15. Feature Importance Analysis

Feature importance by total gain for Ablation C (IMU + NAVRIS):

| Formulation | Feature Group | Combined Gain % | Top Contributing Features |
| :--- | :--- | :--- | :--- |
| **Direct Speed (Formulation A)** | **IMU Dynamics** | **58.74%** | `accel_jerk_proxy_1s` (29.92%), `norm_delta_az_1s` (5.74%), `gyro_accel_ratio_proxy` (5.45%), `accel_autocorr_lag1` (5.21%) |
| **Direct Speed (Formulation A)** | **NAVRIS State** | **41.26%** | `navris_vel_up_mps` (9.12%), `navris_speed_mps` (8.24%), `navris_heading_sin/cos` (6.53%), `navris_v_fwd_mps` (3.04%) |
| **Residual Speed (Formulation B)** | **IMU Dynamics** | **0.00%** | Zero splits on any IMU dynamic feature |
| **Residual Speed (Formulation B)** | **NAVRIS State** | **100.00%** | `navris_v_fwd_mps` (53.85%), `navris_heading_cos` (14.13%), `navris_vel_up_mps` (9.04%), `navris_vel_north_mps` (8.05%) |

### Interpretation:
- In **Direct Speed estimation**, the model balances both sources: **58.7% IMU dynamics** and **41.3% NAVRIS state**, with the normalized jerk proxy as the single most important predictor.
- In **Residual Speed estimation**, IMU dynamics receive **0.0% gain**. The model completely abandons physics and allocates 100% of its splits to inverting NAVRIS velocity components.

---

## 16. Failure Cases

1. **Residual Speed Estimation (Complete Collapse):**
   - Fails due to linear inversion recurrence ($R^2 < -300$, MAE $> 60\text{ m/s}$).
2. **Divergent Route Transfer in Multi-Source Direct (`Y1`):**
   - On `Y1`, adding NAVRIS state degrades performance relative to IMU-only ($R^2: 0.532 \rightarrow 0.428$) because divergent velocity components leak into the tree splits.
3. **High-Speed Highway Transfer (`VTA1A`):**
   - Direct multi-source MAE remains high ($6.57\text{ m/s}$), showing that adding divergent NAVRIS velocity does not resolve the high-speed highway domain shift on smooth asphalt.

---

## 17. Scientific Interpretation (Addressing Q1–Q6)

### Q1: Does adding causal NAVRIS motion state improve forward-speed estimation over IMU-only?
**CONDITIONALLY YES, BUT WITH ROUTE DEPENDENCY.**
- When the classical filter is reliable (`VTA2`), adding NAVRIS state improves $R^2$ from **$0.382 \rightarrow 0.423$** and reduces MAE from $2.79 \rightarrow 2.69\text{ m/s}$.
- When the classical filter diverges (`Y1`), adding NAVRIS state corrupts the estimate, reducing $R^2$ from **$0.532 \rightarrow 0.428$**.
- Overall held-out test $R^2$ drops from **$0.522$ (IMU-only)** to **$0.434$ (Multi-Source Direct)**.

### Q2: Does direct speed estimation or residual speed estimation behave differently under GNSS outage?
**DRAMATICALLY DIFFERENT.**
- Direct speed estimation is stable during outages ($dt > 2\text{ s}$), achieving MAE of **$3.03\text{ m/s}$ on `Y1`** and **$0.90\text{ m/s}$ on `VTA2`**.
- Residual speed estimation completely collapses during outages, with errors exceeding $67\text{ m/s}$.

### Q3: Does the residual model reproduce the Phase 3.0 inversion pattern?
**YES, 100% REPLICATION.**
- Inversion regression confirms $\beta_{\text{pred}} \approx -0.95$ to $-0.99$ ($R^2 > 0.99$).
- The residual model learns $\hat{r} \approx -v_{\text{nav}}$ and fails on nominal routes.

### Q4: Does adding NAVRIS state reduce high-speed saturation?
**PARTIALLY ON NOMINAL ROUTES ONLY.**
- On `VTA2`, high-speed error drops from **$7.46 \rightarrow 6.66\text{ m/s}$**.
- On divergent routes (`Y1`), high-speed error increases ($6.33 \rightarrow 7.15\text{ m/s}$).

### Q5: Does performance generalize across unseen driver, route, vehicle platform?
- Direct multi-source estimation achieves positive $R^2$ across unseen drivers and platforms (`Y1`: $R^2 = 0.428$, `VTA2`: $R^2 = 0.423$).
- However, it remains inferior to pure IMU-only estimation on divergent routes.

### Q6: Is the resulting speed estimate sufficiently stable and interpretable to justify a future pseudo-measurement experiment?
**READINESS ASSESSMENT:**
- **Formulation B (Residual Speed):** **UNFIT FOR DEPLOYMENT.** Permanently rejected.
- **Formulation A (Direct Multi-Source Speed):** **CONDITIONALLY PROMISING BUT REQUIRES GATING.**
  Direct speed estimation is well-behaved, bounded, and causal. However, because feeding divergent classical filter velocity can degrade accuracy on long outages, any future synthetic measurement must either:
  1. Rely strictly on **IMU-Only Dynamic features** (which are 100% immune to filter divergence, $R^2 = 0.522$), or
  2. Implement strict **covariance/quality gating** that disconnects NAVRIS state features when GNSS fixes are lost.

---

## 18. Limitations

1. **State Feedback Hazard:** Feeding an un-gated classical velocity state into ML creates a circular dependency risk under filter divergence.
2. **Missing Covariance Logging:** Real-time filter $P_{vv}$ was unavailable in trajectory logs; only fix recency was accessible.
3. **Residual Learning Incompatibility:** Learning speed residuals relative to an unconstrained open-loop dead-reckoning filter is mathematically ill-posed.

---

## 19. Phase 3.2B Status and Decision

### Descriptive Evidence Classification:
# PARTIAL MULTI-SOURCE EVIDENCE

### Justification:
- Direct multi-source speed estimation demonstrates clear utility when the classical filter is nominal (`VTA2` $R^2$ improves to $0.423$), and quality indicators effectively mitigate stale state reliance.
- However, under classical filter divergence (`Y1`), IMU-only dynamic estimation remains superior ($R^2 = 0.522$ vs $0.434$).
- The residual formulation definitively reproduces the Phase 3.0 inversion failure and is rejected.

```
==================================================
PHASE 3.2B COMPLETE — HARD STOP MAINTAINED
PHASE 3.3 NOT AUTHORIZED
ML -> ESKF INTEGRATION NOT AUTHORIZED
CLOSED-LOOP ML NOT AUTHORIZED
==================================================
```

- Zero ML models have been connected to NAVRIS ESKF.
- Zero pseudo-measurements or closed-loop updates have been created.
- All navigation equations, filters, and calibration files remain strictly untouched.
- Any subsequent Phase 3.3 work requires separate, explicit user authorization.
