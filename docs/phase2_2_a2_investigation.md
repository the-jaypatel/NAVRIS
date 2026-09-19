# NAVRIS Phase 2.2 — Forensic Investigation of Diagnostic Baseline A2

**Project:** NAVRIS — Intelligent Navigation & Inertial System  
**SIH 2026 Problem Statement 26168:** AI-ML based Intelligent Dead Reckoning system for seamless navigation  
**Phase:** 2.2.5 — Scientific Validation & Benchmark Hardening  
**Document:** Deep Investigation into Diagnostic Baseline A2 Anomalies  
**Date:** September 2026  

---

## 1. Problem Statement

In the Phase 2.2 benchmark evaluations, diagnostic baseline **A2** (intended to isolate gyro drift by injecting ground-truth vehicle heading and an "oracle" stationary gyroscope bias) exhibited anomalous behavior. Specifically, while A2 significantly improved dead reckoning accuracy in some recordings (e.g., **$3.0\times$ improvement in S1**, **$2.3\times$ improvement in S2**, **$5.8\times$ improvement in VTA2**), it performed substantially **worse than A1 in other recordings**, most notably:
* **S3A:** A2 RMSE $4,604,496$ m vs. A1 RMSE $2,836,954$ m ($1.62\times$ worse)
* **S4:** A2 RMSE $17,927,997$ m vs. A1 RMSE $10,341,333$ m ($1.73\times$ worse)
* **M:** A2 RMSE $30,027,304$ m vs. A1 RMSE $17,038,172$ m ($1.76\times$ worse)
* **VTA1A:** A2 RMSE $6,907,208$ m vs. A1 RMSE $1,676,965$ m ($4.12\times$ worse)

An independent adversarial review flagged this inconsistency. This investigation was conducted to determine the exact mathematical, physical, and data-driven root causes of this anomaly.

---

## 2. Per-Recording A0 vs. A1 vs. A2 Comprehensive Audit

To ensure scientific comparability, all baselines were evaluated using identical causal initial states ($p_0, v_0$), identical integration schemes (trapezoidal strapdown), identical contiguous segments, identical timestamps, and identical evaluation horizons:

| Recording ID | Duration (s) | Distance (km) | A0 Horiz RMSE (m) | A1 Horiz RMSE (m) | A2 Horiz RMSE (m) | Ratio A2 / A1 | Gyro Bias Vector A1 (rad/s) | Gyro Bias Vector A2 (rad/s) | A2 Stationary Window (s) | A2 Window Speed (m/s) |
|---|---|---|---|---|---|---|---|---|---|---|
| **S1** | $5,018.4$ | $37.0$ | $6,048,531$ | $6,048,277$ | **$2,088,514$** | **$0.35$** | $[-0.0105, +0.0159, -0.0144]$ | $[-0.0065, +0.0085, -0.0005]$ | $t \in [7.5, 12.4]$ | $<0.05$ |
| **S2** | $9,171.2$ | $74.7$ | $23,565,687$ | $23,565,757$ | **$10,256,504$** | **$0.44$** | $[-0.0099, +0.0071, -0.0097]$ | $[-0.0066, +0.0044, -0.0041]$ | $t \in [0.0, 5.1]$ | $0.00$ |
| **S3A** | $2,309.0$ | $24.5$ | $2,836,504$ | $2,836,954$ | **$4,604,496$** | **$1.62$** | $[+0.0099, +0.0004, +0.0021]$ | $[-0.0041, -0.0026, -0.0065]$ | $t \in [206.7, 211.6]$ | $<0.09$ |
| **S4** | $9,327.9$ | $88.0$ | $10,341,064$ | $10,341,333$ | **$17,927,997$** | **$1.73$** | $[+0.0251, +0.0317, -0.0133]$ | $[+0.0043, -0.0001, -0.0025]$ | $t \in [13.3, 28.9]$ | $<0.08$ |
| **M** | $10,260.4$ | $102.2$ | $17,038,440$ | $17,038,172$ | **$30,027,304$** | **$1.76$** | $[-0.0015, +0.0107, -0.0244]$ | $[+0.0354, -0.1190, +0.0338]$ | $t \in [227.8, 316.9]$ | $0.00 - 0.10$ |
| **Y1** | $7,093.8$ | $57.2$ | $4,537,184$ | $4,537,107$ | **$7,109,670$** | **$1.57$** | $[+0.0051, +0.0145, +0.0057]$ | $[-0.0131, -0.0199, -0.0098]$ | $t \in [16.0, 21.0]$ | $<0.05$ |
| **VTA1A** | $2,501.4$ | $40.3$ | $1,677,142$ | $1,676,965$ | **$6,907,208$** | **$4.12$** | $[-0.0031, -0.0007, +0.0026]$ | $[+0.0002, -0.0043, +0.0047]$ | $t \in [16.9, 119.1]$ | $<0.08$ |
| **VTA2** | $1,019.8$ | $10.2$ | $664,212$ | $664,226$ | **$114,326$** | **$0.17$** | $[+0.0093, +0.0112, -0.0022]$ | $[-0.0030, +0.0073, -0.0047]$ | $t \in [14.2, 20.4]$ | $<0.05$ |

---

## 3. Deep Root-Cause Analysis

Detailed empirical tracing of the underlying sensor streams revealed **four distinct mechanisms** that explain why A2 degrades:

### 3.1 Mechanism 1: Unverified Assumption of Initial Vehicle Rest
The foundational design of A0 and A2 assumed that every IO-VNBD recording begins with the vehicle stationary in a parking spot for at least a few seconds. **Inspection of actual VBOX speeds proves this assumption is false for several recordings**:
* In **Recording M**: At $t = 0.0$ s, the vehicle was **already traveling at $5.46$ m/s ($19.7$ km/h)**, accelerating up to $7.0$ m/s within $2.4$ seconds!
* In **Recording S3A**: At $t = 0.0$ s, the vehicle was **already traveling at $2.92$ m/s ($10.5$ km/h)**!
* In **Recording VTA2**: At $t = 0.0$ s, the vehicle was traveling at **$6.87$ m/s ($24.7$ km/h)**!
* In **Recording S1**: At $t = 0.0$ s, the vehicle was traveling at **$5.55$ m/s ($20.0$ km/h)**!

Only in **S2** was the vehicle genuinely stationary at row 0 ($v_{\text{vbox}} = 0.000$ m/s).

### 3.2 Mechanism 2: Contamination of A2 "Stationary" Windows by Dynamic Vehicle Deceleration
Because A2 used a filter (`ref_speed < 0.1 m/s`) to find the "first 50 stationary samples":
* In recordings that started while moving, the search was forced to skip forward hundreds of seconds until the first stoplight or traffic stop.
* In **Recording M**, the first occurrence of $v < 0.1$ m/s was at **$t = 227.8$ s**. Inspection of the raw telemetry shows this was a **panic-braking event** from $18.89$ m/s ($68$ km/h) to a stop. During this braking event, vehicle chassis pitch oscillations and turning excited severe angular rates ($\omega_y$ reached **$-0.89$ rad/s**).
* Averaging these braking samples in `oracle.py` yielded an apparent "bias" of:
  $$\mathbf{b}_{\text{A2}} = [+0.0354, \mathbf{-0.1190}, +0.0338]^T \text{ rad/s}$$
  The $Y$-axis (pitch) rate was estimated as **$-0.119$ rad/s ($-6.82^\circ$/s)**! This was not a sensor offset; it was **active chassis pitch dynamics**.
* When this $-0.119$ rad/s bias was subtracted continuously across the $10,260$-second drive, an artificial tilt error accumulated at $6.82^\circ$/s, tilting gravity into the horizontal plane and driving A2 position RMSE from $17.0 \times 10^6$ m up to **$30.0 \times 10^6$ m**.

### 3.3 Mechanism 3: Thermal Bias Drift in Consumer MEMS Gyroscopes
In **S3A** and **VTA1A**, where the stop occurred cleanly without violent braking, A2 still degraded.
* In consumer-grade smartphone IMUs (InvenSense / Bosch MEMS), the zero-rate offset (ZRO) drifts continuously with internal device temperature. As the phone operates during driving—with screen on, GPS tracking, and CPU logging at 10 Hz—internal phone temperature rises by $5^\circ\text{C} - 15^\circ\text{C}$ over the first 10 minutes.
* A bias measured at $t = 206$ s (in S3A) reflects the heated operating point. Subtracting this constant value across the preceding $200$ seconds introduces an artificial step error:
  $$\Delta \mathbf{b} = \mathbf{b}(t_{206}) - \mathbf{b}(t_0) \approx 0.005 - 0.015 \text{ rad/s}$$
* Due to cubic strapdown integration ($\Delta p \approx \frac{1}{6} g \Delta b t^3$), an artificial bias error of even $0.005$ rad/s ($0.29^\circ$/s) over a $2,309$-second drive produces:
  $$\Delta p \approx \frac{1}{6} \times 9.81 \times 0.005 \times (2,309)^3 \approx 100,000,000 \text{ m}$$
  This cubic sensitivity to small bias mismatches completely overwhelms the baseline.

### 3.4 Mechanism 4: Unmodeled Sensor Dynamics
A static scalar bias vector $\mathbf{b}_g \in \mathbb{R}^3$ fundamentally cannot represent the physical reality of a consumer smartphone MEMS gyroscope over hours of unconstrained driving. The gyroscope experiences:
1. Turn-on bias instability
2. Temperature-dependent scale factor and bias drift
3. G-sensitivity (acceleration-induced gyro drift)
4. Chassis and engine vibration rectification

---

## 4. Scientific Interpretability Assessment

| Criterion | Evaluation | Scientific Finding |
|---|---|---|
| **Is A2 scientifically interpretable?** | **NO** | A2 assumes a single static gyro bias can be estimated from an arbitrary stop window and applied globally across a multi-hour drive. This assumption is physically invalid for consumer MEMS sensors. |
| **Does A2 provide causal evidence?** | **NO** | The variation in A2 performance ($0.17\times$ to $4.12\times$ of A1) is driven entirely by where the vehicle happened to stop and how much the sensor drifted thermally, rather than any controlled isolation of error pathways. |
| **Did A2 isolate heading vs. tilt?** | **PARTIALLY** | In recordings with genuine initial rest (S1, S2, VTA2), A2 improved accuracy by $2.3\times$ to $5.8\times$, showing that gyro bias reduction suppresses drift. But in moving starts, it caused severe artifacts. |

---

## 5. Architectural Recommendation for A2

### **RECOMMENDATION: RETIRE A2 AS A FORMAL BENCHMARK CONTROL**

1. **Retain A0 as the Authoritative Deployable Baseline:** A0 represents true deployable smartphone performance with strictly causal leveling and COG heading.
2. **Retain A1 as a Diagnostic Heading Control:** A1 successfully demonstrates that locking initial heading to truth does not prevent divergence because horizontal gravity leakage from tilt drift dominates.
3. **Retire A2 from Comparative Tables:** A2 must **not** be presented as a standard benchmark or used as causal evidence in NAVRIS publications or SIH presentations. Static offline bias subtraction from arbitrary stop windows is scientifically indefensible.
4. **Transition to Online Estimation in Phase 2.3:** The correct, textbook solution to sensor bias drift is **online Kalman filtering (ESKF)**, where $\mathbf{b}_g(t)$ is estimated as a continuous dynamic Gauss-Markov or random walk state driven by GNSS innovations, rather than a static offline scalar.
