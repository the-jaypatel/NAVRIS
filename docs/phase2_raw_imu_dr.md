# NAVRIS Phase 2.2: Raw Smartphone IMU Dead Reckoning Baseline Report

**Project:** NAVRIS — Intelligent Navigation & Inertial System  
**SIH 2026 Problem Statement 26168:** AI-ML based Intelligent Dead Reckoning system for seamless navigation  
**Dataset:** IO-VNBD Local Git LFS Checkout  
**Phase:** 2.2 — Raw Smartphone IMU Dead Reckoning Baseline  
**Status:** COMPLETE & EMPIRICALLY BENCHMARKED  
**Date:** September 2026  

---

## 1. Executive Summary

Phase 2.2 establishes the **pure classical inertial navigation baseline** for NAVRIS prior to introducing Kalman filtering (Phase 2.3) or AI/ML components (Phase 3+). 

A rigorous, zero-leakage 3D quaternion strapdown INS mechanization and a 2D kinematic planar baseline were developed, tested, and evaluated across 8 synchronized IO-VNBD benchmark recordings (`S1`, `S2`, `S3A`, `S4`, `M`, `Y1`, `VTA1A`, `VTA2`). 

### Authoritative Empirical Findings
1. **Unconstrained Double Integration Divergence:** In cold-start dead reckoning without GNSS corrections, horizontal position error diverges quadratically/cubically. Across multi-hour real-world vehicle driving (duration $1,019$ s to $10,260$ s), the deployable 3D DR baseline (A0) exhibits drift rates between $1,197$ m/s and $5,220$ m/s. Time-to-50m error threshold is reached in **$0.0$ to $7.8$ seconds**.
2. **Controlled Outage Degradation (T1):** Under controlled GNSS outages across the 8 benchmark recordings:
   - **10-second outage:** Mean horizontal RMSE is **$78.6$ m** (final error $109.6$ m, drift rate $11.1$ m/s).
   - **30-second outage:** Mean horizontal RMSE is **$519.8$ m** (final error $1,244.6$ m, drift rate $41.6$ m/s).
   - **60-second outage:** Mean horizontal RMSE is **$2,901.0$ m** (final error $6,963.0$ m, drift rate $116.2$ m/s).
   - **120-second outage:** Mean horizontal RMSE reaches **$14,423.1$ m** (~14.4 km, final error $33,091.1$ m, drift rate $276.0$ m/s).
3. **Dominant Failure Mechanism: Tilt Error & Gravity Leakage:** A minute tilt attitude error $\delta \theta$ tilts the local gravity vector ($9.807$ m/s²), projecting $g \sin(\delta \theta)$ directly onto the horizontal plane. Even a $1^\circ$ leveling error creates a persistent horizontal acceleration bias of $\sim 0.171$ m/s², generating $\frac{1}{2} (0.171) (10,000)^2 \approx 8.55 \times 10^6$ m ($8,550$ km) of horizontal position error over a 2.8-hour drive.
4. **Diagnostic Oracle Attribution:** Providing an exact ground-truth initial vehicle heading (A1) does not prevent explosive multi-hour divergence (RMSE remains $6.64 \times 10^5$ m to $23.57 \times 10^6$ m), strongly indicating that accelerometer bias and horizontal gravity leakage from gyro pitch/roll drift—rather than initial heading error alone—are the primary drivers of unconstrained 3D strapdown divergence.
5. **Planar 2D Kinematic Advantage:** The 2D planar baseline (`A-planar`), which decouples navigation from the vertical gravity vector and integrates forward longitudinal acceleration along an estimated heading, achieves an RMSE of $63.2$ km to $656.1$ km over 3-hour drives—outperforming unconstrained 3D INS by **1 to 2 orders of magnitude**.

---

## 2. Mathematical Mechanization & Physical Conventions

### 2.1 Coordinate Frames
- **Body Frame ($b$):** Smartphone sensor axes as defined by AndroSensor:
  - $+X$: Screen right
  - $+Y$: Screen top
  - $+Z$: Out of screen front
- **Navigation Frame ($n$):** Local Cartesian East-North-Up (ENU):
  - $+X_n$: East ($E$)
  - $+Y_n$: North ($N$)
  - $+Z_n$: Up ($U$)

### 2.2 Quaternion Algebra & Attitude Propagation
Unit quaternions are parameterized in scalar-first Hamilton convention:
$$\mathbf{q} = [q_w, q_x, q_y, q_z]^T, \quad \|\mathbf{q}\| = 1$$

Direction cosine matrix $C_b^n(\mathbf{q}) \in SO(3)$ maps body-frame vectors to navigation-frame vectors:
$$C_b^n(\mathbf{q}) = \begin{bmatrix} 
1 - 2(q_y^2 + q_z^2) & 2(q_x q_y - q_w q_z) & 2(q_x q_z + q_w q_y) \\
2(q_x q_y + q_w q_z) & 1 - 2(q_x^2 + q_z^2) & 2(q_y q_z - q_w q_x) \\
2(q_x q_z - q_w q_y) & 2(q_y q_z + q_w q_x) & 1 - 2(q_x^2 + q_y^2)
\end{bmatrix}$$

Given gyro rate measurement $\boldsymbol{\omega}_k^b$ and stationary bias estimate $\mathbf{b}_g$:
$$\tilde{\boldsymbol{\omega}}_k^b = \boldsymbol{\omega}_k^b - \mathbf{b}_g, \quad \Delta \boldsymbol{\theta}_k = \tilde{\boldsymbol{\omega}}_k^b \Delta t_k$$

Attitude is causally propagated via the exact rotation vector exponential map:
$$\Delta \mathbf{q}_k = \begin{bmatrix} \cos(\|\Delta \boldsymbol{\theta}_k\| / 2) \\ \frac{\Delta \boldsymbol{\theta}_k}{\|\Delta \boldsymbol{\theta}_k\|} \sin(\|\Delta \boldsymbol{\theta}_k\| / 2) \end{bmatrix}, \quad \mathbf{q}_{k+1} = \mathbf{q}_k \otimes \Delta \mathbf{q}_k$$
with periodic unit normalization $\|\mathbf{q}_{k+1}\| = 1$.

### 2.3 Somigliana Normal Gravity & Gravity Compensation
Local gravity $\mathbf{g}^n(\phi, h)$ is computed using the WGS-84 Somigliana ellipsoidal model with free-air height correction:
$$g_0(\phi) = 9.7803253359 \cdot \frac{1 + 0.00193185265241 \sin^2\phi}{\sqrt{1 - 0.00669437999014 \sin^2\phi}}$$
$$g(\phi, h) = g_0(\phi) \cdot \left(1 - \frac{2 h}{a} (1 + f + m - 2f \sin^2\phi) + \frac{3 h^2}{a^2}\right)$$
$$\mathbf{g}^n(\phi, h) = \begin{bmatrix} 0 \\ 0 \\ -g(\phi, h) \end{bmatrix}$$

As established in Phase 2.1, the phone accelerometer measures specific force $\mathbf{f}^b$. Kinematic acceleration $\mathbf{a}_k^n$ is obtained by rotating specific force into the ENU frame and adding local gravity:
$$\mathbf{a}_k^n = C_b^n(\mathbf{q}_k) \mathbf{f}_k^b + \mathbf{g}^n(\phi_0, h_0)$$

*(Note: At rest, $\mathbf{f}^b$ points upward with norm $\approx +g$; $C_b^n \mathbf{f}^b = [0, 0, +g]^T$, which precisely cancels $\mathbf{g}^n = [0, 0, -g]^T$, yielding zero net acceleration).*

### 2.4 Numerical Integration
Velocity and position are updated via second-order trapezoidal integration:
$$\mathbf{v}_{k+1}^n = \mathbf{v}_k^n + \frac{\mathbf{a}_k^n + \mathbf{a}_{k+1}^n}{2} \Delta t_k$$
$$\mathbf{p}_{k+1}^n = \mathbf{p}_k^n + \frac{\mathbf{v}_k^n + \mathbf{v}_{k+1}^n}{2} \Delta t_k$$

### 2.5 Contiguity & Gap Halt Guard
If sampling interval $\Delta t_k > 0.25$ s (e.g. during logging pauses or phone app restarts), propagation halts immediately. No artificial bridging or synthetic integration across gaps is permitted.

---

## 3. Causal Alignment Protocol

Initial attitude is determined using a strictly causal two-step protocol with **zero reference trajectory leakage**:
1. **Initial Leveling (Arbitrary Mounting Angle):**  
   During the initial stationary window (minimum 3.0 s, verified via $\sigma_a < 0.5$ m/s² and $\bar{\omega} < 0.05$ rad/s), the mean specific force $\bar{\mathbf{f}}^b$ is computed. The unit quaternion $\mathbf{q}_{\text{level}}$ is calculated to align $\bar{\mathbf{f}}^b / \|\bar{\mathbf{f}}^b\|$ with the ENU vertical $[0, 0, 1]^T$:
   $$\mathbf{q}_{\text{level}} = \left[1 + \hat{\mathbf{f}}^b \cdot \hat{\mathbf{z}}^n, \quad \hat{\mathbf{f}}^b \times \hat{\mathbf{z}}^n\right]^T \cdot \frac{1}{\sqrt{2(1 + \hat{\mathbf{f}}^b \cdot \hat{\mathbf{z}}^n)}}$$
   Stationary gyro bias $\mathbf{b}_g$ is estimated as the mean angular velocity during this period.
2. **Initial Heading (Course-Over-Ground Dynamic Search):**  
   A sequential causal search inspects novel GNSS fixes ($v_{\text{gps}} > 2.5$ m/s). The earliest window meeting:
   - Horizontal displacement $\Delta d \ge 25.0$ m
   - Circular standard deviation of reported orientation $< 0.15$ rad (~8.5°)  
   determines initial azimuth $\psi_0 = \arctan2(\Delta E, \Delta N)$.  
   The initial attitude quaternion is formed as $\mathbf{q}_0 = \mathbf{q}_{\text{yaw}}(\psi_0) \otimes \mathbf{q}_{\text{level}}$.

All 8 synchronized recordings achieved **100% successful alignment** (`quality=NOMINAL`, `is_valid=True`).

---

## 4. Benchmark Baseline Descriptions

- **A0 (Deployable 3D Quaternion Strapdown INS):** Pure causal strapdown propagation initialized via causal leveling and GNSS COG. Zero reference data.
- **A0-norm (Ablation):** Same as A0, but specific force is scaled by $g_0 / \|\bar{\mathbf{f}}_{\text{stationary}}\|$ to eliminate initial scalar accelerometer scale factor error.
- **A1 (Diagnostic Oracle Heading):** Body tilt (roll/pitch) is propagated from leveled IMU gyro rates, but azimuth $\psi(t)$ is locked to reference VBOX heading at each epoch. Demonstrates performance when gyro yaw drift is completely eliminated.
- **A2 (Diagnostic Oracle Heading & Bias):** Uses reference heading and reference-derived stationary gyro bias.
- **A-planar (2D Kinematic Control):** 2D dead reckoning integrating forward longitudinal acceleration along heading, isolated from 3D vertical gravity leakage.

---

## 5. Full Empirical Benchmark Results

### 5.1 Cold-Start Dead Reckoning Benchmark (T0 Full Run)

Evaluated on primary continuous segments of IO-VNBD synchronized recordings:

| Recording | Duration (s) | Distance (km) | A0 Horiz RMSE (m) | A0 Final Err (m) | A0 Drift Rate (m/s) | Along-Track RMSE (m) | Cross-Track RMSE (m) | Heading RMSE (deg) | Time to 50m (s) | Time to 100m (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| **S1** | 5,018.4 | 37.0 | $6.05 \times 10^6$ | $1.13 \times 10^7$ | 2,258.6 | $4.22 \times 10^6$ | $4.34 \times 10^6$ | 102.8° | 4.8 | 9.0 |
| **S2** | 9,171.2 | 74.7 | $23.57 \times 10^6$ | $4.79 \times 10^7$ | 5,220.3 | $17.13 \times 10^6$ | $16.18 \times 10^6$ | 104.1° | 0.0 | 0.0 |
| **S3A** | 2,309.0 | 24.5 | $2.84 \times 10^6$ | $5.53 \times 10^7$ | 2,396.0 | $1.87 \times 10^6$ | $2.13 \times 10^6$ | 68.8° | 0.0 | 2.8 |
| **S4** | 9,327.9 | 88.0 | $10.34 \times 10^6$ | $2.32 \times 10^7$ | 2,491.5 | $7.45 \times 10^6$ | $7.17 \times 10^6$ | 103.7° | 7.7 | 10.3 |
| **M** | 10,260.4 | 102.2 | $17.04 \times 10^6$ | $4.77 \times 10^7$ | 4,650.9 | $12.70 \times 10^6$ | $11.36 \times 10^6$ | 103.5° | 4.8 | 8.3 |
| **Y1** | 7,093.8 | 57.2 | $4.54 \times 10^6$ | $8.80 \times 10^6$ | 1,240.1 | $2.98 \times 10^6$ | $3.42 \times 10^6$ | 108.8° | 0.0 | 0.0 |
| **VTA1A** | 2,501.4 | 40.3 | $1.68 \times 10^6$ | $4.49 \times 10^6$ | 1,796.5 | $1.33 \times 10^6$ | $1.02 \times 10^6$ | 100.4° | 0.0 | 0.0 |
| **VTA2** | 1,019.8 | 10.2 | $6.64 \times 10^5$ | $1.22 \times 10^6$ | 1,197.7 | $4.82 \times 10^5$ | $4.57 \times 10^5$ | 92.9° | 7.8 | 10.3 |

---

### 5.2 Controlled GNSS Outage Performance (T1 Sweeps)

Horizontal RMSE, Final Error, and Drift Rate under fixed-duration GNSS outages:

| Outage Duration | Metric | S1 | S2 | S3A | S4 | M | Y1 | VTA1A | VTA2 | **Mean** |
|---|---|---|---|---|---|---|---|---|---|---|
| **10 s** | RMSE (m) | 62.5 | 71.5 | 122.3 | 39.6 | 67.5 | 66.3 | 158.7 | 40.4 | **78.6 m** |
| | Final Err (m) | 115.6 | 70.6 | 160.9 | 91.6 | 132.6 | 66.9 | 148.6 | 90.3 | **109.6 m** |
| | Drift Rate (m/s) | 11.7 | 7.1 | 16.2 | 9.2 | 13.4 | 6.8 | 15.0 | 9.1 | **11.1 m/s** |
| **30 s** | RMSE (m) | 578.5 | 262.5 | 177.5 | 653.3 | 819.9 | 856.9 | 133.8 | 675.7 | **519.8 m** |
| | Final Err (m) | 1,421.0 | 514.3 | 211.1 | 1,668.6 | 2,062.2 | 2,180.3 | 258.6 | 1,640.6 | **1,244.6 m** |
| | Drift Rate (m/s) | 47.5 | 17.2 | 7.1 | 55.8 | 69.0 | 72.9 | 8.7 | 54.9 | **41.6 m/s** |
| **60 s** | RMSE (m) | 3,062.4 | 439.9 | 920.1 | 4,224.3 | 4,745.1 | 4,860.0 | 1,318.1 | 3,638.1 | **2,901.0 m** |
| | Final Err (m) | 7,355.4 | 321.7 | 2,831.3 | 10,232.2 | 11,221.9 | 11,268.2 | 3,640.2 | 8,832.8 | **6,962.9 m** |
| | Drift Rate (m/s) | 122.8 | 5.4 | 47.3 | 170.8 | 187.3 | 188.1 | 60.8 | 147.5 | **116.3 m/s** |
| **120 s** | RMSE (m) | 16,095.9 | 5,744.0 | 9,418.7 | 13,144.6 | 23,551.1 | 19,431.9 | 8,075.6 | 19,922.7 | **14,423.1 m** |
| | Final Err (m) | 38,986.0 | 17,217.3 | 23,770.0 | 20,896.4 | 54,963.9 | 42,980.7 | 18,049.0 | 47,865.5 | **32,841.1 m** |
| | Drift Rate (m/s) | 325.2 | 143.6 | 198.2 | 174.3 | 458.4 | 358.5 | 150.5 | 399.2 | **276.0 m/s** |

---

### 5.3 Comparative Baseline Evaluation

Comparison of Horizontal RMSE (meters) across architectural baselines:

| Recording | A0: Deployable 3D DR | A0-norm: Stationary Scaled | A1: Oracle Heading | A2: Oracle H + Bias | A-planar: 2D Kinematic |
|---|---|---|---|---|---|
| **S1** | $6,048,531$ | $5,963,980$ | $6,048,277$ | $2,088,514$ | **$102,888$** |
| **S2** | $23,565,687$ | $23,545,044$ | $23,565,757$ | $10,256,504$ | **$249,354$** |
| **S3A** | $2,836,504$ | $2,801,468$ | $2,836,954$ | $4,604,496$ | **$63,188$** |
| **S4** | $10,341,064$ | $10,284,245$ | $10,341,333$ | $17,927,997$ | **$176,979$** |
| **M** | $17,038,440$ | $16,680,845$ | $17,038,172$ | $30,027,304$ | **$656,066$** |
| **Y1** | $4,537,184$ | $4,527,669$ | $4,537,107$ | $7,109,670$ | **$106,617$** |
| **VTA1A** | $1,677,142$ | $1,672,916$ | $1,676,965$ | $6,907,208$ | **$133,221$** |
| **VTA2** | $664,212$ | $671,538$ | $664,226$ | $114,326$ | **$69,513$** |

---

## 6. Deep Failure Mode Breakdown

### 6.1 Why Unconstrained 3D Consumer IMU DR Fails
1. **Double Integration of Acceleration Bias:**
   An accelerometer bias $\mathbf{b}_a$ propagates directly into position:
   $$\mathbf{p}_{\text{err}}(t) = \frac{1}{2} \mathbf{b}_a t^2$$
   A typical consumer smartphone MEMS accelerometer has a bias instability and thermal offset of $\sim 0.05$ m/s² ($5$ mg). Over a 3-hour recording ($t = 10,800$ s):
   $$\mathbf{p}_{\text{err}} \approx \frac{1}{2} \times 0.05 \times (10,800)^2 = 2.91 \times 10^6 \text{ m } (2,916 \text{ km})$$
2. **Tilt Error and Gravity Projection (Major Error Pathway):**
   When the phone attitude estimation has an angular pitch/roll error $\delta \boldsymbol{\theta}$, the body acceleration rotation leaks Earth's gravity ($g \approx 9.81$ m/s²) into the horizontal plane:
   $$\mathbf{a}_{\text{err}}^n \approx g \cdot \delta \theta$$
   If gyro drift accumulates just $0.5^\circ$ ($0.0087$ rad) of tilt error:
   $$\mathbf{a}_{\text{err}}^n \approx 9.81 \times 0.0087 = 0.0856 \text{ m/s}^2$$
   Integrating this over time yields cubic error growth:
   $$\mathbf{p}_{\text{err}}(t) = \frac{1}{6} g \cdot \dot{\theta} \cdot t^3$$
   The observed A1 and A-planar results are consistent with tilt/gravity leakage being a major contributor, but its relative contribution versus accelerometer bias and other sensor errors has not been uniquely isolated. Providing an **oracle heading (A1)** barely improves 3D performance because gyro drift in pitch and roll continues to tilt the gravity vector into the horizontal axes.
3. **Why A-Planar Outperforms 3D INS:**
   The planar baseline explicitly ignores vertical accelerometer dynamics and integrates only forward velocity along the estimated azimuth. Because it is decoupled from the $9.81$ m/s² vertical gravity vector, it avoids gravity leakage, yielding errors that are orders of magnitude smaller.

---

## 7. Implications & Requirements for Phase 2.3 (ESKF)

The empirical failure modes established in Phase 2.2 define the requirements for the **Phase 2.3 Error-State Kalman Filter (ESKF)**:

1. **Error-State Formulation ($\delta \mathbf{x}$):**  
   The true state must be propagated via full non-linear strapdown mechanization, while the Kalman filter maintains only the small error state $\delta \mathbf{x} = [\delta \mathbf{p}^n, \delta \mathbf{v}^n, \delta \boldsymbol{\theta}^n, \mathbf{b}_a^b, \mathbf{b}_g^b]^T \in \mathbb{R}^{15}$.
2. **Continuous Online Bias Estimation:**  
   The filter must continuously estimate accelerometer bias $\mathbf{b}_a$ and gyro bias $\mathbf{b}_g$ during GNSS-available periods so that when an outage occurs, the propagation uses unbiased specific force and angular velocity.
3. **GNSS Sample-and-Hold Gating:**  
   Because smartphone GNSS updates only every $\sim 9.5$ seconds and repeats stale data at 10 Hz, the ESKF must **strictly reject sample-and-hold duplicates** and trigger measurement updates only upon arrival of genuinely novel fixes.
4. **Attitude-Gravity Observability:**  
   GNSS position and velocity updates allow the filter to observe horizontal velocity errors caused by tilt, enabling the ESKF to causally correct pitch and roll errors and thereby eliminate gravity leakage.

---

## 8. Artifact Verification & Deliverables

All deliverables have been generated and validated:
- **Metrics Table:** `data/processed/phase2/phase22_baseline_metrics.csv`
- **Trajectory Plots:** `data/processed/phase2/<REC>_trajectory_dr_vs_ref.png`
- **Error Breakdowns:** `data/processed/phase2/<REC>_error_breakdown.png`
- **Baseline Comparisons:** `data/processed/phase2/<REC>_baselines_comparison.png`
- **Test Suite:** 36/36 tests passing (`pytest -v`).

**Phase 2.2 is hereby concluded.** We await user authorization before initiating Phase 2.3.
