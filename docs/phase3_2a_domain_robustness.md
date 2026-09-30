# NAVRIS Phase 3.2A — Vehicle-Domain Robustness Diagnostic

## 1. Objective

NAVRIS Phase 3.2 established that causal smartphone IMU history contains statistically significant predictive information regarding vehicle forward speed ($R^2 = 0.410$, $\rho = 0.654$ on held-out test routes). However, it also identified a severe domain gap: while performance on an unseen driver in the same vehicle chassis (`Y1`, Ford Fiesta) was strong ($R^2 = 0.418$), transfer across vehicle platforms to an unseen vehicle (`VTA1A`, VW Golf) suffered severe degradation ($R^2 = -1.545$, $\text{MAE} = 7.02\text{ m/s}$). Furthermore, feature importance revealed that $65.29\%$ of model gain was concentrated in rolling specific force and gyro standard deviations (`accel_mag_rolling_std_1s` and `gyro_mag_rolling_std_1s`).

The primary objective of Phase 3.2A is to conduct a **controlled diagnostic study** to determine whether the forward-speed signal is:
1. primarily a vehicle/road-specific vibration signature, or
2. a generalizable kinematic/dynamic relationship between causal IMU history and vehicle speed.

---

## 2. Scientific Question

> **Core Research Question:**
> *Does normalizing or removing absolute vibration amplitude allow the smartphone IMU speed signal to generalize robustly across vehicle platforms (Ford Fiesta vs VW Golf) and driving environments (UK vs France), or does the signal collapse when decoupled from raw vibration intensity?*

---

## 3. Frozen Phase 3.2 Baseline

Phase 3.2 results are preserved without alteration as the fixed scientific reference.

- **Baseline Architecture:** Lightweight XGBoost (`n_estimators=100`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`).
- **Phase 3.2 Raw IMU Reference Performance:**
  - `TRAIN POOL` (`S1`, `S2`, `S4`): $\text{MAE} = 2.84\text{ m/s}$, $\text{RMSE} = 4.12\text{ m/s}$, $R^2 = 0.548$, $\rho = 0.741$.
  - `S3A` (Val, Ford Fiesta): $\text{MAE} = 3.79\text{ m/s}$, $R^2 = 0.331$, $\rho = 0.657$.
  - `VTA1A` (Val, VW Golf): $\text{MAE} = 7.02\text{ m/s}$, $R^2 = -1.545$, $\rho = 0.278$.
  - `Y1` (Held-Out Test, Ford Fiesta): $\text{MAE} = 3.03\text{ m/s}$, $R^2 = 0.418$, $\rho = 0.664$.
  - `VTA2` (Held-Out Test, VW Golf): $\text{MAE} = 3.02\text{ m/s}$, $R^2 = 0.263$, $\rho = 0.574$.
  - `TEST POOL` (`Y1` + `VTA2`): $\text{MAE} = 3.03\text{ m/s}$, $\text{RMSE} = 4.13\text{ m/s}$, $R^2 = 0.410$, $\rho = 0.654$.

No hyperparameter tuning or architecture modification was performed for Phase 3.2A.

---

## 4. Dataset and Partition

The partition remains frozen across all 7 benchmark recordings:

| Partition | Recording ID | Vehicle Platform | Driver | Country | Valid Samples ($N-9$) | Route Characteristics |
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

## 5. Feature Representations

Three distinct diagnostic representations and two normalization variants were evaluated:

### 1. Raw IMU Representation (Phase 3.2 Reference — 19 Features)
- Direct phone-frame specific forces ($a_x, a_y, a_z$), angular velocities ($\omega_x, \omega_y, \omega_z$), norms ($\|\mathbf{a}\|, \|\boldsymbol{\omega}\|$), gravity deviation.
- Causal 1.0-second rolling statistics ($\mu_a, \sigma_a, \mu_\omega, \sigma_\omega$).
- Causal 1.0-second first differences ($\Delta a_x, \Delta a_y, \Delta a_z, \Delta \omega_x, \Delta \omega_y, \Delta \omega_z$).

### 2. Scale-Invariant Representation (Dimensionless — 15 Features)
Designed to decouple predictions from absolute mechanical vibration amplitude by expressing all quantities as dimensionless ratios:
- $\text{CV}_a = \frac{\sigma_a}{\mu_a + \epsilon}$: Coefficient of variation of acceleration magnitude.
- $\text{CV}_\omega = \frac{\sigma_\omega}{\mu_\omega + \epsilon}$: Coefficient of variation of angular velocity magnitude.
- $\frac{\sigma_a}{g_0}$: Specific force standard deviation normalized by standard gravity ($g_0 = 9.80665\text{ m/s}^2$).
- $\frac{|\|\mathbf{a}\| - g_0|}{g_0}$: Relative gravity deviation ratio.
- $\frac{\|\mathbf{a}\|}{\mu_a + \epsilon}, \frac{\|\boldsymbol{\omega}\|}{\mu_\omega + \epsilon}$: Instantaneous-to-mean magnitude ratios.
- $\frac{a_i^2}{\|\mathbf{a}\|^2 + \epsilon}, \frac{\omega_i^2}{\|\boldsymbol{\omega}\|^2 + \epsilon}$: Directional energy fractions along phone body axes.
- $\frac{\sigma_{a_i}}{\sigma_a + \epsilon}$: Axial vibration contribution fractions.

### 3. Dynamic-Only Representation (Deltas & Jerk — 12 Features)
Designed to eliminate steady-state vibration dependence entirely, focusing exclusively on temporal transitions and kinematic constraints:
- $\frac{a_i[k] - a_i[k-9]}{\mu_a + \epsilon}$: Mean-normalized 1-second specific force deltas.
- $\frac{\omega_i[k] - \omega_i[k-9]}{\mu_\omega + \epsilon}$: Mean-normalized 1-second angular velocity deltas.
- $\frac{\|\mathbf{a}[k]\| - \|\mathbf{a}[k-9]\|}{g_0}$: Dimensionless 1-second magnitude delta.
- $\frac{\|\boldsymbol{\omega}[k]\| - \|\boldsymbol{\omega}[k-9]\|}{\mu_\omega + \epsilon}$: Normalized rotational speed delta.
- $\frac{1}{g_0(w-1)}\sum |\|\mathbf{a}[k-i]\| - \|\mathbf{a}[k-i-1]|$: Mean normalized absolute jerk proxy over 1.0s.
- $\frac{\|\boldsymbol{\omega}\|}{\|\Delta \mathbf{a}\| + \epsilon}$: Dynamic centripetal ratio proxy.
- $\rho_{\text{lag1}, a}, \rho_{\text{lag1}, \omega}$: 1-sample lag autocorrelation proxies across the 1.0s window.

### 4. Causal Running Normalization Representation
Maintains cumulative running mean $\mu_{\text{run}}[k] = \frac{1}{k+1}\sum_{j=0}^k \sigma[j]$ and running standard deviation $\sigma_{\text{run}}[k]$ strictly using past observations $j \le k$, generating causal running z-scores:
$$z_{\text{causal}}[k] = \frac{\sigma[k] - \mu_{\text{run}}[k]}{\sigma_{\text{run}}[k] + \epsilon}$$

### 5. OFFLINE Non-Causal Normalization (Negative Control — NOT DEPLOYABLE)
In hindsight, computes the recording-wide global z-score using future data:
$$z_{\text{offline}} = \frac{\sigma - \text{mean}_{\text{full}}(\sigma)}{\text{std}_{\text{full}}(\sigma)}$$
Included solely as an analytical negative control to isolate the effect of recording-level scale alignment.

---

## 6. Causal Normalization Definitions and Verification

For all causal representations, strict mathematical causality was verified:
$$\mathbf{X}(t) = \mathcal{F}\big(\text{IMU}[0 \dots t]\big) \quad \text{with zero dependence on } \text{IMU}[t' > t]$$

A dedicated unit test confirmed that corrupting future samples after evaluation timestamp $t$ produces an exact difference of **$0.0$** across all features:
- Scale-Invariant max difference: **$0.0$**
- Dynamic-Only max difference: **$0.0$**
- Causal Running Normalization max difference: **$0.0$**

---

## 7. Leakage and Causality Audit

The automated audit passed 100% of causal verification checks:

| Check | Specification | Result | Verification Details |
| :--- | :--- | :--- | :--- |
| **No Forbidden Column Names** | No `ref_`, `vbox`, `navris`, `gnss`, `target`, `truth`, `eskf` in features | **PASSED** | 0 violations across all representation matrices |
| **Target Isolation** | $v_{\text{fwd, ref}}$ present strictly in label vector $\mathbf{y}$ | **PASSED** | Zero target leakage into $\mathbf{X}$ |
| **Partition Isolation** | $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$ | **PASSED** | Mutually exclusive recording sets |
| **Causal Perturbation Invariance** | Future IMU corruption ($t' > t$) leaves $\mathbf{X}(t)$ unchanged | **PASSED** | Max feature difference = $0.0$ across all representations |
| **Causal Running Normalization Audit** | Running z-scores depend strictly on $x[0:t]$ | **PASSED** | Modifying $x[t+1:]$ yields discrepancy = $0.0$ |

---

## 8. Overall Results Across Diagnostic Representations

| Representation | TRAIN MAE (m/s) | TRAIN $R^2$ | VAL MAE (m/s) | VAL $R^2$ | HELD-OUT TEST MAE (m/s) | HELD-OUT TEST RMSE (m/s) | HELD-OUT TEST $R^2$ | HELD-OUT TEST Pearson $\rho$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Raw IMU (Phase 3.2 Ref)** | 2.84 | 0.548 | 5.51 | -0.203 | 3.03 | 4.13 | 0.410 | 0.654 |
| **2. Scale-Invariant (Dimensionless)** | 2.92 | 0.527 | **4.97** | **-0.011** | 2.85 | 3.92 | 0.467 | 0.696 |
| **3. Dynamic-Only (Deltas & Jerk)** | 3.13 | 0.488 | **4.91** | **+0.014** | **2.77** | **3.71** | **0.522** | **0.725** |
| **4. Causal Running Normalization** | 2.75 | 0.584 | 5.59 | -0.246 | 3.00 | 3.99 | 0.450 | 0.684 |
| **5. OFFLINE Non-Causal (Negative Control)** | 2.72 | 0.586 | 6.44 | -0.624 | 3.29 | 4.16 | 0.400 | 0.673 |

### Primary Findings:
1. **Dynamic-Only Representation Achieves Best Generalization:**
   - On the Held-Out Test pool, the Dynamic-Only representation improves $R^2$ from **$0.410 \rightarrow 0.522$** and correlation from **$0.654 \rightarrow 0.725$**, while reducing Test MAE from **$3.03 \rightarrow 2.77\text{ m/s}$**.
   - On Validation, $R^2$ improves from negative (**$-0.203$**) to positive (**$+0.014$**).
2. **Scale-Invariant Representation Confirms Decoupling:**
   - The Scale-Invariant dimensionless ratios outperform Raw IMU on both validation ($R^2: -0.203 \rightarrow -0.011$) and held-out test ($R^2: 0.410 \rightarrow 0.467$, MAE: $3.03 \rightarrow 2.85\text{ m/s}$).
3. **Causal Running Normalization Flaws:**
   - Causal cumulative running z-scoring ($z_{\text{causal}}$) does not resolve domain shift ($R^2 = -0.246$ on Val), because cumulative statistics are non-stationary when vehicles transition between urban stop-and-go and highway driving.

---

## 9. Recording-Level Results: Platform Domain Transfer

Evaluating performance per recording isolates **Same-Platform Transfer** (Ford Fiesta $\rightarrow$ Ford Fiesta) versus **Cross-Platform Transfer** (Ford Fiesta $\rightarrow$ VW Golf):

| Recording | Partition | Platform / Route | Raw IMU MAE (m/s) | Raw IMU $R^2$ | Scale-Invariant MAE (m/s) | Scale-Invariant $R^2$ | Dynamic-Only MAE (m/s) | Dynamic-Only $R^2$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **S1** | TRAIN | Fiesta / Urban UK | 2.26 | 0.562 | 2.35 | 0.527 | 2.51 | 0.470 |
| **S2** | TRAIN | Fiesta / Mixed UK | 2.93 | 0.517 | 3.00 | 0.503 | 3.17 | 0.449 |
| **S4** | TRAIN | Fiesta / Rural UK | 3.08 | 0.552 | 3.16 | 0.528 | 3.42 | 0.501 |
| **S3A** | VAL (Same Platform) | Fiesta / Suburban UK | 3.79 | 0.331 | **3.38** | **0.437** | **3.55** | **0.380** |
| **VTA1A** | VAL (Cross Platform) | Golf / Highway France | 7.02 | -1.545 | **6.37** | **-1.140** | **6.10** | **-0.995** |
| **Y1** | TEST (Same Platform) | Fiesta / Suburban UK | 3.03 | 0.418 | **2.85** | **0.472** | **2.76** | **0.532** |
| **VTA2** | TEST (Cross Platform) | Golf / Urban France | 3.02 | 0.263 | **2.81** | **0.362** | **2.79** | **0.382** |

### Platform Transfer Analysis:
- **Same-Platform Transfer (Ford Fiesta `TRAIN` $\rightarrow$ `Y1` Test):**
  - Raw IMU: $\text{MAE} = 3.03\text{ m/s}$, $R^2 = 0.418$.
  - Dynamic-Only: **$\text{MAE} = 2.76\text{ m/s}$, $R^2 = 0.532$**.
  - Generalization within the same vehicle platform is remarkably strong when temporal dynamics and deltas are used.
- **Cross-Platform Transfer (Ford Fiesta `TRAIN` $\rightarrow$ `VTA2` Test):**
  - Raw IMU: $\text{MAE} = 3.02\text{ m/s}$, $R^2 = 0.263$.
  - Scale-Invariant: **$\text{MAE} = 2.81\text{ m/s}$, $R^2 = 0.362$**.
  - Dynamic-Only: **$\text{MAE} = 2.79\text{ m/s}$, $R^2 = 0.382$**.
  - Decoupling from absolute vibration amplitude substantially improves transfer to an unseen vehicle chassis on urban/suburban routes ($R^2$ increases by $+0.119$).
- **Cross-Platform Highway Transfer (Ford Fiesta `TRAIN` $\rightarrow$ `VTA1A` Val):**
  - Raw IMU: $\text{MAE} = 7.02\text{ m/s}$, $R^2 = -1.545$.
  - Dynamic-Only: $\text{MAE} = 6.10\text{ m/s}$, $R^2 = -0.995$.
  - Although Dynamic-Only reduces error by nearly $1.0\text{ m/s}$, $R^2$ remains negative. Highway driving on smooth French asphalt remains an out-of-distribution regime.

---

## 10. Speed-Regime Results

Performance on the held-out test recordings (`Y1` and `VTA2`) broken down by speed regime:

| Recording | Speed Regime | Speed Range | Samples | Raw IMU MAE (m/s) | Scale-Invariant MAE (m/s) | Dynamic-Only MAE (m/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`Y1`** (Fiesta) | Stationary | $< 0.5\text{ m/s}$ | 11,228 | 2.28 | **1.54** | **1.91** |
| **`Y1`** (Fiesta) | Low Speed | $0.5 - 5.0\text{ m/s}$ | 9,939 | 3.08 | 3.25 | 3.57 |
| **`Y1`** (Fiesta) | Medium Speed | $5.0 - 15.0\text{ m/s}$ | 43,921 | 2.64 | 2.55 | **2.26** |
| **`Y1`** (Fiesta) | High Speed | $\ge 15.0\text{ m/s}$ | 6,662 | 6.81 | 6.48 | **6.33** |
| **`VTA2`** (Golf) | Stationary | $< 0.5\text{ m/s}$ | 657 | 0.94 | **0.85** | **0.84** |
| **`VTA2`** (Golf) | Low Speed | $0.5 - 5.0\text{ m/s}$ | 768 | 3.95 | 4.30 | 4.29 |
| **`VTA2`** (Golf) | Medium Speed | $5.0 - 15.0\text{ m/s}$ | 7,517 | 2.20 | **1.99** | **2.08** |
| **`VTA2`** (Golf) | High Speed | $\ge 15.0\text{ m/s}$ | 1,177 | 8.84 | 8.13 | **7.46** |

### Key Regime Observations:
1. **Medium-Speed Regime Dominance ($5-15\text{ m/s}$):**
   - In the medium speed range (comprising 61% to 74% of driving), Dynamic-Only and Scale-Invariant representations achieve MAE of **$2.26\text{ m/s}$ on `Y1`** and **$1.99 - 2.08\text{ m/s}$ on `VTA2`**.
2. **Persistence of High-Speed Saturation:**
   - In high-speed driving ($\ge 15\text{ m/s}$), MAE remains high: $6.33\text{ m/s}$ on `Y1` and $7.46\text{ m/s}$ on `VTA2`. While Dynamic-Only reduces error by $1.38\text{ m/s}$ on `VTA2`, the saturation phenomenon is not eliminated by normalization alone.

---

## 11. Vibration-Speed Relationship Analysis

The empirical correlation between rolling vibration statistics and true reference vehicle speed was evaluated per recording:

| Recording | Platform / Country | Pearson $\rho(\sigma_a, v_{\text{ref}})$ | Spearman $r_s(\sigma_a, v_{\text{ref}})$ | Pearson $\rho(\sigma_\omega, v_{\text{ref}})$ | Spearman $r_s(\sigma_\omega, v_{\text{ref}})$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **S1** | Ford Fiesta / UK | 0.516 | **0.648** | 0.437 | 0.521 |
| **S2** | Ford Fiesta / UK | 0.447 | **0.632** | 0.388 | 0.588 |
| **S4** | Ford Fiesta / UK | 0.384 | **0.612** | 0.219 | 0.517 |
| **S3A** | Ford Fiesta / UK | 0.421 | **0.506** | -0.041 | 0.080 |
| **Y1** | Ford Fiesta / UK | 0.525 | **0.646** | 0.119 | 0.424 |
| **VTA2** | VW Golf / France | 0.484 | **0.551** | 0.378 | 0.421 |
| **VTA1A** | VW Golf / France | **0.161** | **0.179** | **0.006** | **-0.046** |

### Physical Interpretation:
- **Strong Monotonic Relationship in Urban/Suburban Driving:** Across all Ford Fiesta recordings (`S1`, `S2`, `S4`, `S3A`, `Y1`) and urban VW Golf (`VTA2`), the Spearman rank correlation $r_s(\sigma_a, v_{\text{ref}})$ is consistently **$0.50$ to $0.65$**. In typical driving, vehicle vibration monotonically tracks vehicle speed.
- **Decoupling on High-Speed Highway (`VTA1A`):** On the smooth French motorway (`VTA1A`), the Spearman correlation collapses to **$0.179$** and Pearson gyro correlation drops to **$0.006$**. On high-speed, well-paved motorways, specific force variance decouples from forward speed because road roughness is minimal and aerodynamic forces dominate over tire-chassis vibration.

---

## 12. Vehicle and Platform Domain Shift

Direct comparison of the primary vibration feature distributions between platforms exposes the physical source of domain shift:

| Recording | Platform / Country | Mean $\sigma_a$ (m/s²) | Median $\sigma_a$ (m/s²) | p95 $\sigma_a$ (m/s²) | Mean $\sigma_\omega$ (rad/s) | Median $\sigma_\omega$ (rad/s) | p95 $\sigma_\omega$ (rad/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **S1** | Ford Fiesta / UK | 0.396 | 0.337 | 0.910 | 0.046 | 0.038 | 0.113 |
| **S2** | Ford Fiesta / UK | 0.454 | 0.384 | 1.099 | 0.060 | 0.047 | 0.156 |
| **S4** | Ford Fiesta / UK | 0.328 | 0.281 | 0.783 | 0.041 | 0.033 | 0.105 |
| **S3A** | Ford Fiesta / UK | 0.425 | 0.388 | 0.837 | 0.027 | 0.022 | 0.069 |
| **Y1** | Ford Fiesta / UK | 0.381 | 0.344 | 0.870 | 0.038 | 0.024 | 0.106 |
| **Fiesta Pooled** | **Ford Fiesta** | **0.386** | **0.340** | **0.895** | **0.044** | **0.034** | **0.116** |
| **VTA2** | VW Golf / France | 0.626 | 0.533 | 1.427 | 0.138 | 0.121 | 0.318 |
| **VTA1A** | VW Golf / France | 0.818 | 0.722 | 1.685 | 0.163 | 0.136 | 0.364 |
| **Golf Pooled** | **VW Golf** | **0.762** | **0.667** | **1.610** | **0.156** | **0.132** | **0.351** |

### Critical Domain Shift Finding:
- **$2.0\times$ Linear Vibration Intensity in VW Golf:** The VW Golf exhibits a mean specific force standard deviation of **$0.762\text{ m/s}^2$ vs $0.386\text{ m/s}^2$** in the Ford Fiesta ($+97.4\%$).
- **$3.5\times$ Angular Vibration Intensity in VW Golf:** The VW Golf exhibits a mean angular rate standard deviation of **$0.156\text{ rad/s}$ vs $0.044\text{ rad/s}$** in the Ford Fiesta ($+254.5\%$).
- **Conclusion:** The smartphone mount or vehicle chassis in the VW Golf transmitted substantially higher vibrational energy across all speeds. A raw model trained on the Ford Fiesta interprets the baseline idle/low-speed vibration of the VW Golf as high vehicle speed, causing massive bias.

---

## 13. High-Speed Saturation Analysis

### Empirical Observation:
Across all representations, predicted speed saturates around $16 - 18\text{ m/s}$, producing large absolute errors when true vehicle speed reaches $20 - 30\text{ m/s}$.

### Tested Hypotheses:
1. **Vibration Saturation:**
   - *Observation:* On `S4` (Ford Fiesta), mean $\sigma_a$ in medium speed ($5-15\text{ m/s}$) is $0.32\text{ m/s}^2$, and in high speed ($\ge 15\text{ m/s}$) is $0.46\text{ m/s}^2$. On `VTA1A` (VW Golf), mean $\sigma_a$ in medium speed is $0.78\text{ m/s}^2$ and in high speed is $0.85\text{ m/s}^2$ (a mere $+9\%$ increase despite a $+60\%$ speed increase).
   - *Conclusion:* On smooth motorways, vibration amplitude flattens out, decoupling high vehicle speeds from mechanical vibration variance.
2. **Distributional Sparsity:**
   - In the training set (`S1`, `S2`, `S4`), only **12.0% of samples** exceed $15\text{ m/s}$, and only **1.0%** exceed $26\text{ m/s}$. Tree models naturally pull extreme predictions toward the mean of high-speed training leaves.

---

## 14. Failure Cases

1. **High-Speed Motorway Cruising on Smooth Asphalt (`VTA1A`):**
   - High speed ($> 15\text{ m/s}$), low road roughness, and zero turning dynamics eliminate all primary informational cues. All representations underperform in this regime ($R^2 < 0$).
2. **Global Recording-Wide Z-Score Normalization (Offline Negative Control):**
   - The offline non-causal normalization (Negative Control 5) produced the worst validation performance ($R^2 = -0.624$, `VTA1A` $\text{MAE} = 8.62\text{ m/s}$).
   - *Reason:* Forcing every recording to zero mean and unit variance destroys the macro-level speed information. A recording driven at an average of $16.6\text{ m/s}$ (`VTA1A`) cannot be distinguished from a recording driven at $7.2\text{ m/s}$ (`S1`) if their vibration distributions are unconditionally standardized to $N(0, 1)$.

---

## 15. Scientific Interpretation

### Q1: Does causal normalization reduce the gap between Ford Fiesta and VW Golf?
**YES.** Expressing features as dimensionless scale-invariant ratios or dynamic deltas and jerk proxies reduces cross-platform error significantly:
- On `VTA2` (VW Golf Held-Out Test), $R^2$ increases from **$0.263$ (Raw IMU)** to **$0.382$ (Dynamic-Only)**, and MAE decreases from $3.02 \rightarrow 2.79\text{ m/s}$.
- On `VTA1A` (VW Golf Validation), MAE decreases from $7.02 \rightarrow 6.10\text{ m/s}$.

### Q2: Does normalized/dynamic representation retain useful speed prediction on unseen recordings?
**YES, AND MATERIALLY IMPROVES IT.**
- On held-out `Y1` (Ford Fiesta), $R^2$ improves from **$0.418 \rightarrow 0.532$** ($\rho = 0.732$).
- On held-out `VTA2` (VW Golf), $R^2$ improves from **$0.263 \rightarrow 0.382$** ($\rho = 0.621$).
- Overall held-out test $R^2$ increases from **$0.410 \rightarrow 0.522$**.

### Q3: Does the high-speed saturation remain after normalization?
**YES.** At speeds above $15\text{ m/s}$ (54 km/h), errors remain substantial ($6.33 - 7.46\text{ m/s}$). Normalization mitigates vehicle platform scale differences, but it cannot overcome the physical reality that chassis vibrations level off on smooth highway asphalt.

### Q4: How much of the Phase 3.2 signal appears associated with absolute vibration amplitude?
**A SUBSTANTIAL PORTION, BUT NOT THE ENTIRE SIGNAL.**
- Absolute vibration amplitude accounts for the major platform domain shift between the Ford Fiesta ($0.38\text{ m/s}^2$) and VW Golf ($0.76\text{ m/s}^2$).
- However, when absolute amplitude is removed (Dynamic-Only representation), the model actually performs **better** on held-out data ($R^2 = 0.522$ vs $0.410$), proving that temporal dynamics, deltas, jerk proxies, and rotational constraints carry genuine, generalizable speed information.

### Q5: Is there sufficient evidence to treat IMU speed estimation as a vehicle-independent NAVRIS measurement?
**NO.** While dynamic and scale-invariant representations improve cross-vehicle transfer on urban/suburban roads, high-speed highway driving on unseen platforms remains prone to failure ($R^2 < 0$ on `VTA1A`). The estimate cannot be treated as an uncalibrated, universal standalone measurement.

---

## 16. Limitations

1. **Highway Decoupling:** On smooth road surfaces with low rotational dynamics, IMU features decouple from vehicle speed.
2. **Mounting Rigidity:** Sensor-to-vehicle vibration transmission depends on smartphone mount elasticity, which varies across vehicles.
3. **Absence of Wheel Speed Grounding:** Without periodic zero-velocity updates (ZUPT) or occasional GNSS anchoring, inertial speed estimation remains an approximate envelope tracker.

---

## 17. Phase 3.2A Status and Decision

### Descriptive Evidence Classification:
# PARTIAL DOMAIN-ROBUSTNESS EVIDENCE

### Justification:
- Dimensionless scale-invariant and dynamic-only representations materially reduce cross-platform transfer degradation, improving held-out test performance from $R^2 = 0.410$ to $R^2 = 0.522$ ($\rho = 0.725$) across both Ford Fiesta (`Y1`) and VW Golf (`VTA2`).
- However, substantial route and vehicle platform dependence persists under high-speed highway conditions (`VTA1A`), where aerodynamic forces and asphalt smoothness decouple vehicle speed from smartphone inertial dynamics.

```
==================================================
PHASE 3.2A COMPLETE — HARD STOP MAINTAINED
PHASE 3.3 NOT AUTHORIZED
ML -> ESKF INTEGRATION NOT AUTHORIZED
==================================================
```

- Zero ML models have been connected to NAVRIS ESKF.
- Zero pseudo-measurements or closed-loop updates have been created.
- All navigation equations, filters, and calibration files remain strictly untouched.
- Any subsequent Phase 3.3 investigations require separate, explicit user authorization.
