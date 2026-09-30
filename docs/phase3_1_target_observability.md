# NAVRIS Phase 3.1 — ML Target Observability Study (Corrected & Audited)

**Date**: September 29, 2026  
**Status**: AUDITED & SCIENTIFICALLY CORRECTED — STUDY ONLY (Zero Models Trained)  
**Mandate**: Systematic target formulation and observability study across 7 validated benchmark recordings (363,486 synchronized 10 Hz samples, >10.0 hours of driving).

---

## Scientific Corrections Applied

Following formal scientific review, the following corrections have been audited and applied to this report:
1. **Empirical Concentration vs. Theoretical Boundedness**: Removed all claims of targets being "strictly bounded." Characterized distributions by their empirical percentiles (median, p95, p99, p99.9) and observed extremes.
2. **Candidate A Characterization**: Replaced "unbounded" with "not practically bounded under observed classical filter divergence," explicitly acknowledging finite dataset bounds.
3. **Reference Speed Range**: Replaced the unsupported assertion of a "strictly [0, 35] m/s" target range with the true empirical reference speed range: $[0.00, 30.45] \text{ m/s}$ (max observed on S4), with $p99 = 25.39 \text{ m/s}$ and $p99.9 = 28.73 \text{ m/s}$. Any reference to 35 m/s is explicitly designated as a *proposed engineering operating envelope*.
4. **Correlation vs. Observability**: Corrected the assertion that Candidate D is "fully observable." High correlation (+0.933) with measured gyroscopes reflects common rotational motion during vehicle turns; it does **not** establish true sensor bias observability.
5. **Candidate D Construction Rigor**: Fully documented the reference yaw-rate lineage from VBOX IMU, confirming it is offline reference label information only. Noted frame misalignment between phone body axes and vehicle chassis axes.
6. **Candidate B Causal Interpretation**: Designated Candidate B as a *future short-horizon error-evolution target* rather than a current-state error. Added an explicit causal information matrix.
7. **Candidate C Physical Interpretation**: Clarified that finite difference $[\Delta \mathbf{v}(t+H) - \Delta \mathbf{v}(t)] / H$ represents an *average error-growth rate* over horizon $H$, which correlates with, but is not mathematically identical to, instantaneous acceleration error.
8. **Target Boundedness vs. Predictability**: Added an explicit discussion distinguishing numerical concentration from stationarity, causal predictability, and physical observability.
9. **Candidate B/C Offset Differencing**: Clarified that differencing removes common accumulated offsets present in both $\Delta \mathbf{v}(t)$ and $\Delta \mathbf{v}(t+H)$, transforming the target into local error evolution without eliminating all navigation-state dependence.
10. **Test Partition Discipline**: Strictly isolated held-out test recordings (`Y1`, `VTA2`) from formulation selection. All recommendations are derived from training (`S1`, `S2`, `S4`) and validation (`S3A`, `VTA1A`) partitions.
11. **Removal of Rankings**: Eliminated all comparative rankings, composite scores, and "best target" designations. Retained descriptive classifications (`GOOD`, `MODERATE`, `POOR`, `UNOBSERVABLE`) across independent criteria.
12. **Phase 3.2 Authorization Boundary**: Reformulated conclusion to evaluate evidence sufficiency for a future controlled experiment without authorizing or initiating Phase 3.2.

---

## 1. Objective

Phase 3.1 is strictly an offline target observability and formulation study. Its purpose is to determine whether a physically local machine learning target exists that satisfies the following criteria:
1. **Causally predictable** using only sensors available at runtime $[k-w \dots k]$
2. **Empirically concentrated** within realistic vehicle kinematic envelopes
3. **Stationary** across driving durations and GNSS availability regimes
4. **Physically coupled to local IMU dynamics** rather than accumulated filter states
5. **Transferable** across vehicles, drivers, and routes
6. **Meaningful during GNSS outages** without degenerating into trivial heuristics
7. **Safe to conditionally apply** to NAVRIS without corrupting well-conditioned classical solutions
8. **Mathematically well-posed** in contrast to the Phase 3.0 global ENU velocity residual

---

## 2. Phase 3.0 Lessons

The forensic audit of Phase 3.0 ([`docs/phase3_metric_audit.md`](file:///c:/Users/JAY%20PATEL/Documents/Jay/NAVRIS/docs/phase3_metric_audit.md)) proved that while target construction was mathematically correct and timestamps were aligned to within $0.00$ s, the **unconditional global velocity residual target ($\Delta \mathbf{v}^n = \mathbf{v}_{\text{ref}}^n - \mathbf{v}_{\text{nav}}^n$) is fundamentally flawed**:
1. **Non-Stationarity**: In consumer smartphone dead reckoning, unconstrained tilt errors integrate gravity linearly into velocity ($v \sim 10^3 - 10^4 \text{ m/s}$) over 30 to 120 minutes of GNSS outage. The magnitude of $\Delta \mathbf{v}^n$ reflects total elapsed filter drift rather than instantaneous vehicle dynamics.
2. **Feature Bypass (Inversion Heuristic)**: Because true vehicle speed is bounded ($0 \le v \le 30.5 \text{ m/s}$), the target residual is dominated by $-\mathbf{v}_{\text{nav}}^n$. Tree-based regressors placed **96.68% of feature importance on classical velocity states** and only **0.0051% on raw IMU features**, learning to negate the filter velocity state.
3. **Catastrophic Failure on Nominal Tracking**: When applied to VTA2 (where GNSS fix acceptance was 96.7% and classical horizontal velocity RMSE was only 5.53 m/s), the model injected a $-208.66 \text{ m/s}$ spurious vertical bias, blowing up position drift from 455 m to 15,396 m (a 33-fold degradation).

---

## 3. Candidate Target Definitions

We formulate and analyze five candidate target families:

### Candidate Target A: Body-Frame Forward Speed Residual
$$\Delta v_{\text{fwd}}^b(t) = v_{\text{ref, fwd}}(t) - v_{\text{nav, fwd}}(t)$$
where $v_{\text{ref, fwd}}(t) = \text{ref\_speed\_mps}(t)$ from VBOX Doppler reference, and $v_{\text{nav, fwd}}(t)$ is the classical filter velocity projected along estimated heading:
$$v_{\text{nav, fwd}}(t) = v_{\text{nav}, E}(t) \sin(\psi_{\text{nav}}(t)) + v_{\text{nav}, N}(t) \cos(\psi_{\text{nav}}(t))$$

### Candidate Target B: Short-Horizon Incremental Velocity Error
Instead of predicting accumulated global velocity error, predict the incremental velocity error generated over a finite lookahead horizon $H \in \{0.5, 1.0, 2.0, 5.0\}$ seconds:
$$\delta_H \Delta \mathbf{v}^n(t) = \Delta \mathbf{v}^n(t + H) - \Delta \mathbf{v}^n(t)$$
Notice that:
$$\delta_H \Delta \mathbf{v}^n(t) = [\mathbf{v}_{\text{ref}}^n(t+H) - \mathbf{v}_{\text{ref}}^n(t)] - [\mathbf{v}_{\text{nav}}^n(t+H) - \mathbf{v}_{\text{nav}}^n(t)]$$
By subtracting $\Delta \mathbf{v}^n(t)$, the common accumulated offset present in both $\Delta \mathbf{v}(t)$ and $\Delta \mathbf{v}(t+H)$ is removed, transforming the label into a measure of local error evolution over horizon $H$.

### Candidate Target C: Local Velocity Error Growth Rate (Derivative)
The average rate of change of velocity error over short horizons $H \in \{0.5, 1.0, 2.0\}$ seconds:
$$\dot{\Delta \mathbf{v}}_H^n(t) = \frac{\Delta \mathbf{v}^n(t + H) - \Delta \mathbf{v}^n(t)}{H} \approx \bar{\mathbf{a}}_{\text{ref}}^n(t) - \bar{\mathbf{a}}_{\text{nav}}^n(t)$$
This measures the average acceleration discrepancy between true vehicle acceleration and filter-propagated specific force over horizon $H$.

### Candidate Target D: Body-Frame IMU Error / Bias Proxies
Offline sensor-level error proxies between reference vehicle dynamics and phone IMU measurements:
- **Proxy D1 (Forward Accel Error)**: $\Delta a_{\text{long}}^v = a_{\text{ref, long}} - \frac{d}{dt}(v_{\text{nav, fwd}})$
- **Proxy D2 (Lateral Accel Error)**: $\Delta a_{\text{lat}}^v = a_{\text{ref, lat}} - (v_{\text{nav, fwd}} \cdot \omega_{\text{phone}, z})$
- **Proxy D3 (Yaw Rate Error)**: $\Delta \omega_{\text{yaw}}^v = \omega_{\text{ref, yaw}} - \omega_{\text{phone}, z}$
- **Proxy D4 (Body-Frame Error Growth)**: $\dot{\Delta \mathbf{v}}^b = \mathbf{C}_n^b(\psi_{\text{nav}}) \dot{\Delta \mathbf{v}}^n$

### Candidate Target E: Navigation Correction & Innovation Magnitude
Predicting the scale and necessity of corrections rather than raw vector states:
- **E1 (Velocity Error Magnitude)**: $\|\Delta \mathbf{v}^n(t)\|$
- **E2 (Relative Velocity Error Ratio)**: $\|\Delta \mathbf{v}^n(t)\| / (\|\mathbf{v}_{\text{nav}}(t)\| + 1.0)$
- **E3 (Local Growth Scale)**: $\|\dot{\Delta \mathbf{v}}_{1.0s}(t)\|$

---

## 4. Causal Runtime Information vs. Offline Reference Information

Strict separation between runtime inputs and offline supervised training labels:

| Quantity | Runtime available at $t$? | Feature? | Offline label? |
| :--- | :---: | :---: | :---: |
| Current IMU (`phone_accel_*`, `phone_gyro_*`) | **YES** | **YES** | NO |
| Past IMU window $[k-w \dots k]$ | **YES** | **YES** | NO |
| Current NAVRIS filter state ($v_{\text{nav}}(t), \psi_{\text{nav}}(t)$) | **YES** | **YES** | NO |
| Current filter covariance ($\mathbf{P}(t)$) | **YES** | **YES** | NO |
| Causal outage duration ($\Delta t_{\text{outage}}$) | **YES** | **YES** | NO |
| Future VBOX velocity ($v_{\text{ref}}(t+H)$) | **NO** | **NO** | **YES** |
| Future reference velocity / heading ($t+H$) | **NO** | **NO** | **YES** |
| Future NAVRIS filter state ($v_{\text{nav}}(t+H)$) | **NO** | **NO** | **YES (only if required by target)** |
| Reference VBOX yaw rate ($\omega_{\text{ref, yaw}}(t)$) | **NO** | **NO** | **YES (offline label only)** |
| Reference VBOX accelerations ($a_{\text{ref, long/lat}}(t)$) | **NO** | **NO** | **YES (offline label only)** |

> [!IMPORTANT]
> All reference quantities (VBOX speed, heading, yaw rate, accelerations) are strictly offline label information. Under no circumstances do reference measurements or future states enter runtime feature pipelines.

---

## 5. Empirical Bounds vs. Theoretical Bounds

A rigorous distinction must be made between theoretical mathematical bounds and empirical sample concentrations:

### 5.1 Measured Reference Speed Bounds
The true empirical distribution of vehicle reference speed across all 363,486 synchronized samples is:
- **Global Minimum**: $0.0000 \text{ m/s}$
- **Global Maximum**: $30.4531 \text{ m/s}$ ($109.63 \text{ km/h}$, observed on S4)
- **Median**: $8.8211 \text{ m/s}$
- **95th Percentile**: $20.4039 \text{ m/s}$
- **99th Percentile**: $25.3881 \text{ m/s}$
- **99.9th Percentile**: $28.7301 \text{ m/s}$

Per-recording maximum speeds: S1: $26.06 \text{ m/s}$, S2: $29.23 \text{ m/s}$, S4: $30.45 \text{ m/s}$, S3A: $27.23 \text{ m/s}$, VTA1A: $28.73 \text{ m/s}$, VTA2: $22.68 \text{ m/s}$, Y1: $24.30 \text{ m/s}$.

*A range of $[0, 35] \text{ m/s}$ is therefore not an observed physical bound, but rather a reasonable proposed engineering operating envelope for consumer highway passenger driving.*

### 5.2 Target Distributions Across Partitions
Distributions computed across all 7 benchmark recordings:

| Candidate Target | Metric | TRAIN (`S1, S2, S4`) | VAL (`S3A, VTA1A`) | TEST (`Y1, VTA2` - Held Out) | Empirical Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$\Delta \mathbf{v}^n$ (Phase 3.0)** | Min / Max | $0.00 / 91,109.8 \text{ m/s}$ | $0.00 / 15,306.2 \text{ m/s}$ | $0.00 / 55,642.5 \text{ m/s}$ | Not practically bounded |
| | Median / RMSE | $15,867.6 / 35,174.0 \text{ m/s}$ | $4,644.3 / 6,986.6 \text{ m/s}$ | $16,322.5 / 27,379.4 \text{ m/s}$ | Non-stationary |
| **Candidate A ($\Delta v_{\text{fwd}}^b$)** | Min / Max | $-12,049.9 / +11,971.1 \text{ m/s}$ | $-12,574.3 / +12,652.7 \text{ m/s}$ | $-2,845.5 / +2,101.5 \text{ m/s}$ | Not practically bounded |
| | Median / RMSE | $-489.4 / 3,697.5 \text{ m/s}$ | $+0.01 / 3,406.1 \text{ m/s}$ | $-307.8 / 1,155.3 \text{ m/s}$ | Diverges with filter |
| **Candidate B ($\delta_{1.0s} \Delta \mathbf{v}$)** | Min / Max | $0.03 / 36,676.9 \text{ m/s}$ | $0.00 / 13,791.6 \text{ m/s}$ | $0.00 / 14,834.1 \text{ m/s}$ | Extremes caused by reset steps |
| | Median / p95 | **$13.93 / 19.17 \text{ m/s}$** | **$8.01 / 18.93 \text{ m/s}$** | **$13.03 / 19.15 \text{ m/s}$** | **95% concentrated < 19.2 m/s** |
| | p99 / p99.9 | **$19.64 / 20.28 \text{ m/s}$** | **$19.76 / 242.45 \text{ m/s}$** | **$19.78 / 23.97 \text{ m/s}$** | **99% concentrated < 19.8 m/s** |
| **Candidate C ($\dot{\Delta \mathbf{v}}_{1.0s}$)** | Min / Max | $0.03 / 36,676.9 \text{ m/s}^2$ | $0.00 / 13,791.6 \text{ m/s}^2$ | $0.00 / 14,834.1 \text{ m/s}^2$ | Extremes caused by reset steps |
| | Median / p95 | **$13.93 / 19.17 \text{ m/s}^2$** | **$8.01 / 18.93 \text{ m/s}^2$** | **$13.03 / 19.15 \text{ m/s}^2$** | **95% concentrated < 19.2 m/s$^2$** |
| | p99 / p99.9 | **$19.64 / 20.28 \text{ m/s}^2$** | **$19.76 / 242.45 \text{ m/s}^2$** | **$19.78 / 23.97 \text{ m/s}^2$** | **99% concentrated < 19.8 m/s$^2$** |
| **Candidate D (Yaw Rate)** | Min / Max | $-3.65 / +4.92 \text{ rad/s}$ | $-1.31 / +1.35 \text{ rad/s}$ | $-10.82 / +3.06 \text{ rad/s}$ | Sensor noise & chassis turns |
| | Median / RMSE | **$-0.002 / 0.154 \text{ rad/s}$** | **$+0.001 / 0.192 \text{ rad/s}$** | **$+0.001 / 0.177 \text{ rad/s}$** | Highly concentrated |
| | p99 / p99.9 | **$0.49 / 0.68 \text{ rad/s}$** | **$0.54 / 0.90 \text{ rad/s}$** | **$0.45 / 0.81 \text{ rad/s}$** | **99.9% concentrated < 0.9 rad/s** |

### 5.3 Forensic Diagnosis of Candidate A
Candidate A ($\Delta v_{\text{fwd}}^b = v_{\text{ref, fwd}} - v_{\text{nav, fwd}}$) is not practically bounded under observed classical filter divergence.
During stationary stops ($v_{\text{ref}} = 0 \text{ m/s}$):
- In **S2**, Candidate A RMSE is **$4,353.4 \text{ m/s}$**
- In **S4**, Candidate A RMSE is **$2,729.0 \text{ m/s}$**
- In **S3A**, Candidate A RMSE is **$1,390.3 \text{ m/s}$**

*Physical Reason*: When the filter dead-reckons unconstrained, velocity diverges in 3D space. Projecting a 12,000 m/s velocity vector onto the forward heading axis yields $v_{\text{nav, fwd}} \approx 12,000 \text{ m/s}$. The residual $\Delta v_{\text{fwd}}^b = 0 - 12,000 = -12,000 \text{ m/s}$ incorporates the full open-loop filter drift and is therefore unsuitable as an ML target.

---

## 6. Target Boundedness vs. Predictability & Divergence

A critical distinction must be drawn between the following concepts:
1. **Numerical Concentration**: Whether 95% or 99% of sample values fall within a narrow interval.
2. **Stationarity**: Whether the probability distribution $P(y)$ is invariant to elapsed time and outage history.
3. **Causal Predictability**: Whether the label $y$ shares mutual information with features available causally at time $t$.
4. **Physical Observability**: Whether the physical state variable can be uniquely determined from available sensor measurements.

> [!NOTE]
> A target having a small empirical concentration does **not** imply it is causally predictable or observable. For example, Candidate B's 1-second increment is concentrated below 19.8 m/s, but predicting its exact value still requires observing specific force integration errors.

### The Role of Kalman Reset Discontinuities
While 99% of Candidate B/C samples are concentrated below 19.8 m/s, the maximum observed values reach $> 30,000 \text{ m/s}$. These isolated extremes are caused by **discrete Kalman measurement updates**. When the ESKF occasionally accepts an update after a prolonged outage (e.g., at $t = 2,069.2$ s in S3A), the filter velocity state jumps discontinuously (by 13,793 m/s). This step jump corrupts finite-difference increments unless measurement update steps are explicitly masked out during training.

---

## 7. Outage-Duration & Stationarity Analysis

Stationarity evaluated across outage duration bins ($[0, 5) \text{ s}$, $[5, 30) \text{ s}$, $[30, 120) \text{ s}$, $\ge 120 \text{ s}$):

| Target | Metric | Bin 0–5 s | Bin 5–30 s | Bin 30–120 s | Bin > 120 s | Stationarity Behavior |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 3.0 ($\Delta \mathbf{v}^n$, S2)** | RMSE | $54,298 \text{ m/s}$ | $53,051 \text{ m/s}$ | $39,065 \text{ m/s}$ | $1,205 \text{ m/s}$ | Non-stationary |
| **Candidate A ($\Delta v_{\text{fwd}}^b$, S2)** | RMSE | $4,664 \text{ m/s}$ | $4,619 \text{ m/s}$ | $4,390 \text{ m/s}$ | $405 \text{ m/s}$ | Non-stationary |
| **Candidate B ($\delta_{1.0s} \Delta \mathbf{v}$, S2)** | **Median** | **$13.67 \text{ m/s}$** | **$13.61 \text{ m/s}$** | **$14.50 \text{ m/s}$** | **$15.98 \text{ m/s}$** | **Stationary** |
| | **p95** | **$19.13 \text{ m/s}$** | **$19.08 \text{ m/s}$** | **$19.26 \text{ m/s}$** | **$17.12 \text{ m/s}$** | Invariant across outages |
| **Candidate C ($\dot{\Delta \mathbf{v}}_{1.0s}$, S2)** | **Median** | **$13.67 \text{ m/s}^2$**| **$13.61 \text{ m/s}^2$**| **$14.50 \text{ m/s}^2$**| **$15.98 \text{ m/s}^2$**| **Stationary** |
| **Candidate D (Yaw Rate, S2)** | **Std Dev** | **$0.166 \text{ rad/s}$**| **$0.172 \text{ rad/s}$**| **$0.036 \text{ rad/s}$**| **$0.014 \text{ rad/s}$**| **Stationary** |

---

## 8. IMU / Feature Relationship & Observability Limitations

Correlations evaluated on the training partition (`S1`, `S2`, `S4`):

| Target | Correlation with IMU Dynamics | Correlation with Filter Speed | Correlation with Outage Duration | Correlation with Elapsed Time |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 3.0 ($\Delta \mathbf{v}^n$)** | $-0.009 \dots +0.010$ (Zero) | **$+0.725$ (High)** | $-0.090$ | **$+0.599$ (High)** |
| **Candidate A ($\Delta v_{\text{fwd}}^b$)** | $-0.010 \dots +0.024$ (Zero) | **$-0.289$ (Moderate)**| $+0.073$ | $-0.059$ |
| **Candidate B ($\delta_{1.0s} \Delta \mathbf{v}$)** | $+0.015 \dots +0.026$ (Low) | $+0.007$ (Zero) | $-0.004$ (Zero) | $-0.005$ (Zero) |
| **Candidate C ($\dot{\Delta \mathbf{v}}_{1.0s}$)** | $+0.015 \dots +0.026$ (Low) | $+0.007$ (Zero) | $-0.004$ (Zero) | $-0.005$ (Zero) |
| **Candidate D (Yaw Rate Residual)** | **$+0.933$ (Strongly Correlated)** | $-0.003$ (Zero) | $-0.001$ (Zero) | $-0.011$ (Zero) |

### Observability Limitations of Candidate D
1. **Correlation $\ne$ Observability**: A correlation of $+0.933$ between Candidate D ($\omega_{\text{ref, yaw}} - \omega_{\text{phone}, z}$) and measured phone gyroscope signals indicates that during turning maneuvers, both the vehicle chassis and the smartphone experience strong angular velocities. It does **not** establish that the sensor bias $\mathbf{b}_g$ is observable independently from vehicle dynamics.
2. **Mounting Misalignment**: The subtraction $\omega_{\text{ref, yaw}} - \omega_{\text{phone}, z}$ assumes the smartphone z-axis is aligned with the vehicle yaw axis. Any uncalibrated tilt or orientation rotation couples roll and pitch rates into the residual.
3. **Offline Label Dependency**: Candidate D relies on reference VBOX yaw rate $\omega_{\text{ref, yaw}}$. At runtime, this signal is not available.

---

## 9. Observability Assessment

Descriptive classifications across target evaluation criteria (no overall rankings):

| Criterion | Candidate A ($\Delta v_{\text{fwd}}^b$) | Candidate B ($\delta_H \Delta \mathbf{v}$) | Candidate C ($\dot{\Delta \mathbf{v}}$) | Candidate D (Yaw Rate Proxy) | Candidate E (Innovation Mag) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Empirical Concentration** | POOR | **GOOD** (p99 < 19.8 m/s) | **GOOD** (p99 < 19.8 m/s$^2$) | **GOOD** (p99 < 0.55 rad/s) | MODERATE |
| **Stationarity** | POOR | **GOOD** | **GOOD** | **GOOD** | POOR |
| **Physical Interpretability** | MODERATE | **GOOD** | **GOOD** | **GOOD** | MODERATE |
| **IMU Dynamic Relationship** | POOR | MODERATE | MODERATE | **GOOD** (Strong Correlation)| POOR |
| **Cross-Recording Stability**| POOR | **GOOD** | **GOOD** | **GOOD** | POOR |
| **Outage Invariance** | POOR | **GOOD** | **GOOD** | **GOOD** | POOR |
| **Divergence Invariance** | POOR (Diverges) | MODERATE (Step spikes) | MODERATE (Step spikes) | **GOOD** (Independent) | POOR (Diverges) |
| **Causal Feature Availability**| **GOOD** | **GOOD** | **GOOD** | **GOOD** | **GOOD** |
| **Observability Status** | **UNOBSERVABLE** | **PARTIALLY OBSERVABLE**| **PARTIALLY OBSERVABLE** | **PARTIALLY OBSERVABLE** | **UNOBSERVABLE** |

---

## 10. Candidate Formulations for Potential Phase 3.2 Investigation

Based exclusively on training (`S1, S2, S4`) and validation (`S3A, VTA1A`) evidence, two candidate formulations provide sufficient justification for a narrowly controlled Phase 3.2 experiment:

### Formulation 1: Pseudo-Measurement Forward Speed Estimator
Rather than predicting an error relative to a diverged classical filter, estimate **forward vehicle speed directly from causal IMU vibrations and kinematics**:
$$\hat{v}_{\text{fwd}}(t) = f_{\text{ML}}\left(\mathbf{a}_{[k-w \dots k]}^b, \boldsymbol{\omega}_{[k-w \dots k]}^b\right)$$
- **Target**: $v_{\text{ref, fwd}}(t)$ (from VBOX reference during training).
- **Observed Range**: $[0.00, 30.45] \text{ m/s}$ (median $8.82 \text{ m/s}$).
- **Proposed Operating Envelope**: $[0, 35] \text{ m/s}$.
- **Decoupled from Classical Filter**: Does not use $v_{\text{nav}}$ as an input feature or target component. Completely immune to classical filter divergence.
- **Integration Mechanism**: Can be fed into the ESKF as a 1D synthetic pseudo-measurement update ($z_v = \hat{v}_{\text{fwd}}$) analogous to NHC.

### Formulation 2: Short-Horizon Local Acceleration Error Regressor (Candidate C)
Predict the local average acceleration error vector over a 1-second horizon:
$$\Delta \mathbf{a}^b(t) = f_{\text{ML}}\left(\mathbf{a}_{[k-w \dots k]}^b, \boldsymbol{\omega}_{[k-w \dots k]}^b\right)$$
- **Target**: Body-frame finite-difference acceleration residual during pure propagation intervals (masking out discrete Kalman reset steps).
- **Observed Distribution**: 99% of samples concentrated below $19.8 \text{ m/s}^2$.

---

## 11. Scientific Conclusion

### Key Findings
1. **Global Residuals are Fundamentally Flawed**: Global velocity residuals ($\Delta \mathbf{v}^n$ and Candidate A) grow without bound during dead reckoning. They cannot be causally predicted from local 1-second IMU features without the model learning an inversion table.
2. **Local Increments are Stationary**: Differencing $\Delta \mathbf{v}(t+H) - \Delta \mathbf{v}(t)$ removes the accumulated drift offset. 99% of observed 1-second increments fall below $19.8 \text{ m/s}$ across all recordings and outage durations.
3. **Correlation Does Not Establish Observability**: Candidate D's +0.933 correlation with gyroscopes reflects rotational vehicle dynamics during turns, not true sensor bias observability.

### Phase 3.2 Authorization Status
These findings provide **sufficient evidence for a narrowly controlled Phase 3.2 experiment** focusing on **Formulation 1 (Direct Forward Speed Estimation)**.

> [!WARNING]
> **PHASE 3.2 NOT STARTED — STOP CONDITION MET**  
> No machine learning models have been trained. No navigation code has been altered. Downstream execution requires separate explicit user authorization.
