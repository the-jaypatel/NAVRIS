# NAVRIS Phase 2.4 Gate 2.1 — Causal Sensor-Frame & Extrinsic Calibration Experiment Specification

**Document ID:** `NAVRIS-PHASE2-4-GATE2-1-PLAN`  
**Date:** 2026-09-24  
**Author:** NAVRIS Core Engineering Team  
**Status:** **PLANNING ONLY — AWAITING AUTHORIZATION**  
**Target Dataset:** Primary evaluation on `S1` (`data/processed/synchronized/S1_sync.parquet`)  

---

## 1. Objective

The objective of Gate 2.1 is to design a rigorous, controlled scientific experiment to determine whether the extrinsic coordinate transformation between the smartphone IMU sensor frame ($\mathcal{S}$) and the vehicle/navigation body frame ($\mathcal{V}$) can be estimated **causally at runtime**, using exclusively sensors available on commercial consumer smartphones (accelerometer, gyroscope, sparse intermittent GNSS), **without access to external ground truth (VBOX) or non-causal offline knowledge**.

Previous forensic investigations (E2, E3, Gate 1.5A/B) demonstrated that:
1. In `S1`, an offline diagnostic mount angle of $\sim 315^\circ$ ($+X_{\mathcal{S}}$ rotated $\sim -45^\circ$ relative to forward $+X_{\mathcal{V}}$) and mapping `phone_gyro_y_radps` (`GYROSCOPE Pitch`) to vehicle yaw allowed the Phase 2.3A ESKF to track successfully.
2. In `S2`, the phone was mounted forward-facing ($\approx +2.6^\circ$).
3. In `Y1`, mounting was tilted, and in `VTA1A`/`VTA2`, track test vibrations altered signal characteristics.
4. Hardcoding extrinsic orientation or gyro channel mapping across recordings is mathematically invalid and violates core autonomous navigation principles.
5. Offline cross-correlation against VBOX yaw rate (as used in Gate 1.5B) is strictly an offline benchmark curation method and is **not deployable** on consumer devices.

Gate 2.1 establishes the mathematical formulation, observability boundaries, causal information filters, candidate calibration estimators, test matrix, and verification protocol for deployable sensor-to-vehicle alignment.

---

## 2. Scientific Question

> **“Can NAVRIS estimate the smartphone-to-vehicle 3D orientation ($R_{\mathcal{S}}^{\mathcal{V}}$) and gyro channel mapping using only information available causally at runtime, without VBOX ground truth or future information?”**

### Specific Sub-Questions:
1. **Vertical Tilt (2 DOF):** Can stationary/quasi-stationary gravity observations reliably isolate roll and pitch relative to the local gravitational vector without vehicle motion?
2. **Horizontal Heading / Mount Yaw (1 DOF):** Under what conditions is horizontal mounting yaw $\psi_{\mathcal{S}/\mathcal{V}}$ observable from early motion (linear acceleration bursts, sparse GNSS course, or centripetal turn dynamics)?
3. **Gyro Axis Identification:** Can the primary vehicle yaw channel and its sign parity ($+1$ or $-1$) be autonomously resolved from dynamic road excitation?
4. **Information Thresholds:** What minimum excitation thresholds (motion duration, acceleration magnitude, turn angular velocity, GNSS fix count) are required before a calibration can be certified as valid versus declared **UNOBSERVABLE / INSUFFICIENT EXCITATION**?

---

## 3. Causal Information Boundary & Strict Operating Rules

To ensure true real-world deployability and eliminate any leakage:

1. **Strictly Causal Timeline:**
   The calibration algorithm at time $t_k$ may evaluate only samples $t \le t_k$.
   - **NO future samples:** Accessing $t > t_k$ is strictly prohibited.
   - **NO lookahead:** Dynamic windowing must be strictly causal $[t_k - W, t_k]$.
2. **Zero Ground-Truth Access During Calibration:**
   The calibrator has **zero access** to reference VBOX channels (`ref_east_m`, `ref_north_m`, `ref_speed_mps`, `ref_yaw_rate_radps`, `ref_heading_rad`).
   - VBOX data may be loaded into memory **only after** the causal calibrator has terminated and frozen its estimate, solely for post-hoc validation and error metric computation.
3. **Sparse GNSS Arrival Fidelity:**
   Smartphone GNSS is intermittent ($\sim 1\text{ Hz}$). The calibrator cannot assume high-frequency or synchronized GNSS fixes. Course or displacement estimates may only be updated when `phone_gps_is_new_fix == True`.
4. **No Magnetometer Reliance:**
   Magnetometer readings in consumer vehicles are heavily corrupted by hard/soft iron distortions from the vehicle chassis, engine block, and cabin electronics. The calibrator must not assume a calibrated or reliable magnetometer.
5. **No Blind Straight-Line Assumption:**
   The algorithm must not assume that the vehicle immediately departs in a straight line upon motion onset (e.g., parking lot egress, curved driveways, or turnaround maneuvers).

---

## 4. Mathematical Observability Analysis

Let $\mathcal{S}$ denote the smartphone sensor frame, $\mathcal{V}$ the vehicle body frame (Forward-Left-Up: $+X_{\mathcal{V}}$ Forward, $+Y_{\mathcal{V}}$ Left, $+Z_{\mathcal{V}}$ Up), and $\mathcal{N}$ the local navigation frame (East-North-Up: ENU).

The extrinsic rotation matrix $R_{\mathcal{S}}^{\mathcal{V}} \in \mathrm{SO}(3)$ transforms vectors from sensor coordinates to vehicle coordinates:
$$\mathbf{v}^{\mathcal{V}} = R_{\mathcal{S}}^{\mathcal{V}} \mathbf{v}^{\mathcal{S}}$$

We decompose $R_{\mathcal{S}}^{\mathcal{V}}$ into a tilt alignment $R_{\text{tilt}}$ (2 DOF) and a horizontal azimuth alignment $R_{\text{yaw}}(\psi)$ (1 DOF):
$$R_{\mathcal{S}}^{\mathcal{V}} = R_z(\psi_{\mathcal{S}/\mathcal{V}}) \, R_{\text{tilt}}$$

```
   ┌──────────────────────────────────────────────────────────────┐
   │                  3D Extrinsic Rotation R_S^V                 │
   └──────────────────────────────┬───────────────────────────────┘
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│       Tilt Leveling R_tilt      │       │     Horizontal Mount Yaw R_z    │
│      (Roll & Pitch - 2 DOF)     │       │        (Azimuth - 1 DOF)        │
├─────────────────────────────────┤       ├─────────────────────────────────┤
│ • Observable from gravity       │       │ • Unobservable at rest          │
│ • Static accelerometer resting  │       │ • Requires dynamic excitation   │
│ • Independent of vehicle heading│       │ • Solved via GNSS / turns       │
└─────────────────────────────────┘       └─────────────────────────────────┘
```

### 4.1 Roll and Pitch from Gravity (Observable at Rest)
When the vehicle is stationary ($\mathbf{a}^{\mathcal{V}} = \mathbf{0}$, $\boldsymbol{\omega}^{\mathcal{V}} = \mathbf{0}$), the specific force measured by the accelerometer is solely the reaction to gravity:
$$\mathbf{f}^{\mathcal{S}} = - R_{\mathcal{N}}^{\mathcal{S}} \mathbf{g}^{\mathcal{N}} = R_{\mathcal{V}}^{\mathcal{S}} [0, 0, g]^T$$

The measured unit gravity vector in sensor coordinates is:
$$\hat{\mathbf{g}}^{\mathcal{S}} = \frac{\bar{\mathbf{f}}^{\mathcal{S}}}{\|\bar{\mathbf{f}}^{\mathcal{S}}\|}$$
where $\bar{\mathbf{f}}^{\mathcal{S}} = \frac{1}{N} \sum_{i=1}^N \mathbf{f}_i^{\mathcal{S}}$ is the mean over a stationary interval $T_{\text{static}}$.

The tilt rotation $R_{\text{tilt}}$ rotates $\hat{\mathbf{g}}^{\mathcal{S}}$ into the vehicle vertical unit vector $\hat{\mathbf{z}}_{\mathcal{V}} = [0, 0, 1]^T$:
$$\mathbf{n} = \hat{\mathbf{g}}^{\mathcal{S}} \times [0, 0, 1]^T, \quad \theta = \arcsin(\|\mathbf{n}\|)$$
$$R_{\text{tilt}} = \mathbf{I} + [\mathbf{n}]_\times + [\mathbf{n}]_\times^2 \frac{1 - \cos\theta}{\sin^2\theta}$$

**Observability Conclusion:** Roll and pitch (2 DOF) are **fully observable** from stationary gravity alone. Any rotation around $\hat{\mathbf{g}}^{\mathcal{S}}$, however, leaves $\hat{\mathbf{g}}^{\mathcal{S}}$ invariant; horizontal yaw $\psi_{\mathcal{S}/\mathcal{V}}$ is **completely unobservable** at rest.

---

### 4.2 Horizontal Yaw from Linear Acceleration (Partially Observable)
When the vehicle moves forward along $+X_{\mathcal{V}}$ with longitudinal acceleration $a_x = \dot{v}$, the specific force in the leveled sensor frame $\mathbf{f}^{\mathcal{L}} = R_{\text{tilt}} \mathbf{f}^{\mathcal{S}}$ satisfies:
$$\mathbf{f}_{\text{horiz}}^{\mathcal{L}} = [f_x^{\mathcal{L}}, f_y^{\mathcal{L}}]^T = R_z(-\psi_{\mathcal{S}/\mathcal{V}}) \begin{bmatrix} a_x \\ a_y \end{bmatrix}$$

Assuming straight motion ($a_y \approx 0$):
$$\psi_{\mathcal{S}/\mathcal{V}} = \text{atan2}(f_y^{\mathcal{L}}, f_x^{\mathcal{L}})$$

**Ambiguity:** 
If the vehicle brakes instead of accelerates ($a_x < 0$), $\mathbf{f}_{\text{horiz}}^{\mathcal{L}}$ points backward ($-X_{\mathcal{V}}$), producing a **$180^\circ$ sign error**. 
To resolve this ambiguity causally, longitudinal acceleration must be gated against a speed increase ($\Delta v_{\text{gnss}} > 0$).

---

### 4.3 Horizontal Yaw from GNSS Course / Displacement (Observable During Transit)
When the vehicle transitions to motion ($v > 3.0\text{ m/s}$), intermittent GNSS fixes produce horizontal displacement:
$$\Delta \mathbf{p}_k^{\mathcal{N}} = \begin{bmatrix} E_k - E_{k-1} \\ N_k - N_{k-1} \end{bmatrix}, \quad \chi_k = \text{atan2}(\Delta E_k, \Delta N_k)$$

Under the non-holonomic constraint (wheels do not slide sideways on dry road), vehicle heading in the navigation frame is approximately equal to ground track course: $\psi_{\mathcal{V}}^{\mathcal{N}} \approx \chi_k$.

Meanwhile, dead reckoning or horizontal inertial integration in the leveled sensor frame yields sensor displacement $\Delta \mathbf{p}_k^{\mathcal{L}}$. The horizontal mounting angle is:
$$\psi_{\mathcal{S}/\mathcal{V}} = \chi_k - \text{atan2}(\Delta p_{y, k}^{\mathcal{L}}, \Delta p_{x, k}^{\mathcal{L}})$$

**Limitations:**
- Requires vehicle motion ($v > 3\text{ m/s}$) over at least 2–3 valid GNSS epochs ($2\text{ to }5\text{ s}$).
- Highly sensitive to GNSS multi-path and low-speed bearing jitter.

---

### 4.4 Gyro Axis Mapping & Forward Direction from Vehicle Turns (Observable During Turns)
During a planar vehicle turn with yaw rate $r_{\mathcal{V}} \ne 0$ and forward speed $v$:
1. **Vertical Gyro Alignment:** The vehicle angular velocity vector $\boldsymbol{\omega}^{\mathcal{V}} = [0, 0, r_{\mathcal{V}}]^T$ is purely vertical. In the leveled frame:
   $$\boldsymbol{\omega}^{\mathcal{L}} = R_{\text{tilt}} \boldsymbol{\omega}^{\mathcal{S}} \approx [0, 0, r_{\mathcal{V}}]^T$$
   The leveled z-gyro $\omega_z^{\mathcal{L}}$ must capture the entirety of $r_{\mathcal{V}}$, confirming both the gyro channel mapping and parity sign ($\text{sign}(r_{\mathcal{V}})$).
2. **Kinematic Cross-Product for Forward Vector:**
   A turn produces lateral centripetal acceleration:
   $$\mathbf{a}_{\text{lat}}^{\mathcal{V}} = [0, a_{\text{lat}}, 0]^T = [0, v \, r_{\mathcal{V}}, 0]^T$$
   Taking the cross product of horizontal leveled acceleration and leveled angular velocity:
   $$\mathbf{v}_{\text{fwd}}^{\mathcal{L}} = \mathbf{a}_{\text{horiz}}^{\mathcal{L}} \times \boldsymbol{\omega}^{\mathcal{L}} = \begin{bmatrix} 0 \\ v \, r_{\mathcal{V}} \\ 0 \end{bmatrix} \times \begin{bmatrix} 0 \\ 0 \\ r_{\mathcal{V}} \end{bmatrix} = \begin{bmatrix} v \, r_{\mathcal{V}}^2 \\ 0 \\ 0 \end{bmatrix} \propto +X_{\mathcal{V}}$$
   Because $v > 0$ and $r_{\mathcal{V}}^2 > 0$, the vector $\mathbf{a}_{\text{horiz}}^{\mathcal{L}} \times \boldsymbol{\omega}^{\mathcal{L}}$ **always points strictly forward along the vehicle forward axis $+X_{\mathcal{V}}$, regardless of whether the vehicle turns left or right!**

**Significance:** This provides an entirely inertial, self-contained method to resolve the vehicle forward axis and horizontal mounting angle $\psi_{\mathcal{S}/\mathcal{V}}$ without requiring GNSS course!

---

### 4.5 Observability Summary Matrix

| State / Parameter | Stationary ($v = 0$) | Straight Acceleration ($a_x > 0$) | Planar Turn ($|r| > 0.05$) | Moving with GNSS ($v > 3$) |
| :--- | :---: | :---: | :---: | :---: |
| **Roll $\phi$** | **Observable** | Observable | Corrupted by roll tilt | Observable |
| **Pitch $\theta$** | **Observable** | Corrupted by $a_x$ | Observable | Observable |
| **Mount Yaw $\psi_{\mathcal{S}/\mathcal{V}}$** | **UNOBSERVABLE** | Partially ($180^\circ$ ambiguity) | **Fully Observable** ($\mathbf{a} \times \boldsymbol{\omega}$) | **Fully Observable** ($\chi_{\text{gnss}}$) |
| **Gyro Vertical Axis** | Unobservable | Unobservable | **Fully Observable** ($\omega_z^{\mathcal{L}}$) | Unobservable |
| **Gyro Parity ($\pm 1$)** | Unobservable | Unobservable | **Fully Observable** | Observable via heading change |

---

## 5. Candidate Calibration Methods

We define four candidate causal calibration algorithms to be systematically evaluated on recording `S1`:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        CANDIDATE CALIBRATION METHODS                   │
├────────────────────────────────────────────────────────────────────────┤
│ Method A: Stationary Gravity Only                                      │
│   • Inputs: Resting accelerometer (t in [0, T_static])                 │
│   • Solves: Roll & Pitch (R_tilt)                                      │
│   • Mount Yaw: Declared UNOBSERVABLE (or default 0° prior)             │
├────────────────────────────────────────────────────────────────────────┤
│ Method B: Gravity Leveling + Causal GNSS Course                        │
│   • Inputs: Static accel + first 2-3 moving GNSS fixes                 │
│   • Solves: R_tilt via gravity; Mount Yaw via ground track bearing     │
│   • Requires: v > 3.0 m/s and valid GNSS fixes                         │
├────────────────────────────────────────────────────────────────────────┤
│ Method C: Gravity Leveling + Kinematic Turn Cross-Product              │
│   • Inputs: Static accel + first turn (|r| > 0.08 rad/s, v > 3.0 m/s)   │
│   • Solves: R_tilt via gravity; Mount Yaw via a_lat x omega_yaw        │
│   • Requires: At least one turn maneuver                               │
├────────────────────────────────────────────────────────────────────────┤
│ Method D: Multi-Phase Causal State Machine                             │
│   • Phase 0: Standstill detection & gravity leveling                   │
│   • Phase 1: Forward acceleration transient check                      │
│   • Phase 2: First valid turn or GNSS bearing confirmation             │
│   • Convergence Flag: Certified when independent estimators agree      │
└────────────────────────────────────────────────────────────────────────┘
```

### 5.1 Method A: Stationary Gravity Only
- **Hypothesis:** Stationary gravity resolves roll and pitch; horizontal yaw is assumed to be $0^\circ$ (phone mounted nominally facing forward) or declared unobservable.
- **Algorithm:**
  1. Detect standstill using sample variance threshold: $\text{Var}(\|\mathbf{f}^{\mathcal{S}}\|) < 0.05\text{ (m/s}^2)^2$ and $\|\boldsymbol{\omega}^{\mathcal{S}}\| < 0.05\text{ rad/s}$ over $T_{\text{static}} = 5.0\text{ s}$.
  2. Compute average gravity vector $\bar{\mathbf{f}}^{\mathcal{S}}$.
  3. Calculate $R_{\text{tilt}}$ rotating $\bar{\mathbf{f}}^{\mathcal{S}}$ to $[0, 0, g]^T$.
  4. Set $\psi_{\mathcal{S}/\mathcal{V}} = 0.0^\circ$ (nominal prior).
  5. Gyro mapping: Assume sensor axis closest to vertical is yaw (`phone_gyro_y` or `z`).
- **Expected Outcome:** Fails horizontal tracking on S1 because S1 has a physical mount angle of $\sim 315^\circ$ (an error of $\sim 45^\circ$).

### 5.2 Method B: Gravity Leveling + Causal GNSS Course / Displacement
- **Hypothesis:** Gravity provides leveling; the horizontal heading offset is resolved as soon as the vehicle accumulates sufficient GNSS displacement.
- **Algorithm:**
  1. Level phone frame via stationary gravity ($R_{\text{tilt}}$).
  2. Wait for vehicle motion onset ($v_{\text{gnss}} > 3.0\text{ m/s}$).
  3. Accumulate consecutive new GNSS fixes ($k \in [1, K]$, with $K \ge 3$).
  4. Compute GNSS ground course $\chi = \text{atan2}(E_K - E_1, N_K - N_1)$.
  5. Compute integrated horizontal inertial displacement $\Delta \mathbf{p}^{\mathcal{L}}$ over the same interval.
  6. Estimate mount yaw: $\psi_{\mathcal{S}/\mathcal{V}} = \chi - \text{atan2}(\Delta p_y^{\mathcal{L}}, \Delta p_x^{\mathcal{L}})$.
  7. Gyro mapping: Confirmed by correlating integrated gyro heading changes against $\Delta \chi$.

### 5.3 Method C: Gravity Leveling + Kinematic Turn Cross-Product ($\mathbf{a}_{\text{lat}} \times \boldsymbol{\omega}$)
- **Hypothesis:** A planar turn provides an entirely self-contained inertial determination of the forward axis and yaw gyro channel without GNSS course.
- **Algorithm:**
  1. Level phone frame via stationary gravity ($R_{\text{tilt}}$).
  2. Monitor leveled angular rate $\boldsymbol{\omega}^{\mathcal{L}}$ and horizontal acceleration $\mathbf{a}_{\text{horiz}}^{\mathcal{L}}$.
  3. Trigger turn calibration when $|r_{\mathcal{V}}| = \|\boldsymbol{\omega}^{\mathcal{L}}\| > 0.08\text{ rad/s}$ and $v_{\text{approx}} > 3.0\text{ m/s}$ for at least $1.5\text{ s}$ ($15\text{ samples}$).
  4. Compute instantaneous forward vectors: $\mathbf{u}_k = \mathbf{a}_{\text{horiz}, k}^{\mathcal{L}} \times \boldsymbol{\omega}_k^{\mathcal{L}}$.
  5. Average $\mathbf{u}_k$ over the turn: $\bar{\mathbf{u}} = \sum_{k} \mathbf{u}_k$.
  6. Mount yaw angle: $\psi_{\mathcal{S}/\mathcal{V}} = \text{atan2}(\bar{u}_y, \bar{u}_x)$.
  7. Gyro axis mapping: The axis with maximum variance during the turn is assigned as vehicle yaw; sign is determined such that $\mathbf{a}_{\text{lat}} \times \omega > 0$ along forward speed.

### 5.4 Method D: Staged Multi-Phase Causal State Machine
- **Hypothesis:** Combining stationary gravity, acceleration forward-check, and turn/GNSS validation in a staged state machine provides robust, certified convergence while explicitly declaring **UNOBSERVABLE** when motion excitation is absent.
- **State Flow:**
  - **State 0 (UNINITIALIZED):** Await stationary detection ($t \in [0, 5\text{ s}]$). Compute $R_{\text{tilt}}$.
  - **State 1 (TILT_LOCKED):** Roll and pitch locked. Await vehicle motion ($v > 3\text{ m/s}$).
  - **State 2 (FORWARD_PENDING):** Candidate forward vector estimated from longitudinal acceleration burst $\mathbf{f}_{\text{horiz}}$ (gated by $\Delta v > 0$).
  - **State 3 (DYNAMIC_CONVERGED):** Validated by either Method B (GNSS displacement $\ge 15\text{ m}$) or Method C (turn excitation $|r| > 0.08\text{ rad/s}$).
  - **State 4 (CERTIFIED_CALIBRATION):** Both GNSS and turn estimates agree within $\pm 10^\circ$. Calibration output locked.

---

## 6. Controlled Experiment Matrix on Recording S1

The experiment will execute strictly on `S1_sync.parquet` ($51,743\text{ rows}$, $5,174.2\text{ s}$ @ 10 Hz), evaluating early calibration windows ($t \in [0, T_{\text{cal}}]$):

| Experiment ID | Candidate Method | Causal Window ($T_{\text{cal}}$) | Sensor Inputs Used | Minimum Excitation Required | Expected Observable Parameters | Validation Metric Against VBOX |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-2.1-A1** | Method A (Static Gravity) | $t \in [0, 5\text{ s}]$ | Accelerometer only | 5s stationary ($v = 0$) | Roll $\phi$, Pitch $\theta$ (Yaw unobservable) | Tilt error vs VBOX pitch/roll |
| **EXP-2.1-A2** | Method A (Static Gravity) | $t \in [0, 15\text{ s}]$ | Accelerometer only | 15s stationary ($v = 0$) | Roll $\phi$, Pitch $\theta$ (Evaluate noise reduction) | Variance of tilt estimate |
| **EXP-2.1-B1** | Method B (Gravity + GNSS Course) | $t \in [0, 30\text{ s}]$ | Accel + 3 sparse GNSS fixes | $v > 3\text{ m/s}$, $\ge 3$ fixes, disp $> 20\text{ m}$ | Roll, Pitch, Mount Yaw $\psi_{\mathcal{S}/\mathcal{V}}$ | Yaw error vs VBOX heading offset |
| **EXP-2.1-B2** | Method B (Gravity + GNSS Course) | $t \in [0, 60\text{ s}]$ | Accel + 6 sparse GNSS fixes | $v > 3\text{ m/s}$, $\ge 6$ fixes, disp $> 50\text{ m}$ | Roll, Pitch, Mount Yaw $\psi_{\mathcal{S}/\mathcal{V}}$ | Stability across fix count |
| **EXP-2.1-C1** | Method C (Gravity + Turn Cross-Prod) | $t \in [0, T_{\text{turn1}}]$ (first turn) | Accel + Gyro | First turn ($|r| > 0.08\text{ rad/s}$, duration $> 1.5\text{ s}$) | Roll, Pitch, Mount Yaw, Gyro Mapping | Yaw error vs VBOX; gyro axis parity |
| **EXP-2.1-C2** | Method C (Gravity + Turn Cross-Prod) | $t \in [0, T_{\text{turn2}}]$ (second turn) | Accel + Gyro | Second turn check | Repeatability of turn-derived yaw | Angular difference $|\psi_1 - \psi_2|$ |
| **EXP-2.1-D1** | Method D (Multi-Phase Causal Estimator) | Dynamic ($t \le 60\text{ s}$) | Accel + Gyro + Sparse GNSS | Static + Motion + Turn / Course | Full 3D $R_{\mathcal{S}}^{\mathcal{V}}$ + Gyro Mapping + Certification | Complete 3D angular error vs VBOX |

---

## 7. Metrics and Observability Classification

For each experiment, the following standardized diagnostic vector will be reported:

### Quantitative Metrics:
1. **Estimated Roll ($\hat{\phi}$):** Sensor tilt angle around forward axis (degrees).
2. **Estimated Pitch ($\hat{\theta}$):** Sensor tilt angle around lateral axis (degrees).
3. **Estimated Mount Yaw ($\hat{\psi}_{\mathcal{S}/\mathcal{V}}$):** Horizontal azimuth angle relative to vehicle forward (degrees).
4. **Gyro Axis & Parity Mapping:** Identified sensor channel (`phone_gyro_x`, `y`, `z`) and sign multiplier ($\pm 1$).
5. **Convergence Time ($T_{\text{conv}}$):** Seconds from recording start until calibration locks.
6. **VBOX Reference Error ($\Delta \Theta$):**
   - Tilt error: $\Delta \theta_{\text{tilt}} = \|\hat{\mathbf{g}}_{\text{est}}^{\mathcal{V}} - [0, 0, 1]^T\|$ (degrees).
   - Horizontal yaw error: $\Delta \psi = |\hat{\psi}_{\mathcal{S}/\mathcal{V}} - \psi_{\text{vbox}}|$ (degrees, relative to ground truth reference mount angle $\sim 315^\circ$ in S1).
7. **Sensitivity to Speed ($v$):** Calibration error as a function of speed threshold ($2.0, 3.0, 5.0\text{ m/s}$).
8. **Sensitivity to Turn Rate ($r$):** Calibration error as a function of minimum turn rate threshold ($0.04, 0.08, 0.12\text{ rad/s}$).

### Observability & Certification Status:
Each experiment run must output one of three formal status tags:
- `CERTIFIED_OBSERVABLE`: Calibration mathematically observable and supported by sufficient excitation; error within acceptance threshold.
- `CONDITIONALLY_OBSERVABLE`: Calibration observable but excitation marginal (e.g. only 1 turn or noisy GNSS); carries wide confidence bounds ($\pm 15^\circ$).
- `UNOBSERVABLE / INSUFFICIENT_EXCITATION`: Insufficient motion or zero turns; horizontal mounting yaw **must not be guessed**. Algorithm must explicitly report `is_valid = False` rather than forcing an invalid estimate.

---

## 8. Acceptance and Rejection Criteria

### Primary Acceptance Criteria (For Deployable Causal Calibration):
1. **Zero Ground-Truth Dependence:** Algorithm must execute exclusively on phone accel, gyro, and phone GNSS fixes. Accessing VBOX channels during estimation is grounds for immediate rejection.
2. **Strict Causality:** Estimator at time $t_k$ must not access any sample $t > t_k$.
3. **Horizontal Angular Accuracy:**
   - On recording `S1`, the estimated mount yaw $\hat{\psi}_{\mathcal{S}/\mathcal{V}}$ must agree with the known physical mount angle ($\sim 315^\circ \pm 10^\circ$) within:
     $$\Delta \psi \le 10.0^\circ$$
4. **Gyro Axis Identification:** Must autonomously select `phone_gyro_y_radps` with parity $+1$ as the dominant vehicle yaw channel.
5. **Tilt Accuracy:** Leveled gravity vector must align with vertical within $\le 2.0^\circ$.
6. **Explicit Unobservable Declaration:** If the vehicle remains stationary or moves strictly in a straight line without turn excitation or valid GNSS fixes, the calibrator must explicitly output `UNOBSERVABLE`, refusing to emit an arbitrary yaw prior.

### Rejection Criteria:
- Use of non-causal batch optimization or future trajectory lookahead.
- Any reliance on uncalibrated smartphone magnetometer.
- Hardcoded assumptions that the phone is mounted along the vehicle forward axis ($0^\circ$).
- Divergence or instability ($> 15^\circ$ error) across initialization segments.

---

## 9. Expected Failure Modes & Edge Cases

The experiment specification explicitly identifies anticipated failure modes:

1. **Stationary Initial Outage (Method A):**
   If the vehicle is parked, horizontal yaw is fundamentally unobservable. Method A will inevitably produce a $45^\circ$ error on S1 if it assumes a $0^\circ$ prior. This failure mode must be formally documented to demonstrate the necessity of motion-based calibration.
2. **Straight Highway Departure without Turns (Method C Failure):**
   If the driver enters a long straight highway immediately after departure, Method C (turn cross-product) will never trigger. The algorithm must remain in `FORWARD_PENDING` or fall back to Method B (GNSS displacement) rather than deadlocking.
3. **GNSS Multi-Path / Tunnel Departure (Method B Failure):**
   In urban canyons or parking garages, initial GNSS fixes may exhibit $20\text{ m}$ position jumps. Method B will produce erroneous course angles if fix accuracy (`phone_gps_accuracy_m`) is not filtered.
4. **Braking vs Acceleration Ambiguity:**
   If the vehicle starts on a downhill slope and brakes immediately, longitudinal specific force is backward ($a_x < 0$). Without gating on $\Delta v > 0$, the forward axis will be inverted by $180^\circ$.
5. **Test Track Vibration Noise (as observed in VTA1A):**
   High vibration noise floors ($0.33\text{ rad/s}$) can corrupt the $\mathbf{a}_{\text{lat}} \times \boldsymbol{\omega}$ cross-product if a turn threshold of only $0.05\text{ rad/s}$ is used. Thresholds must be dynamically scaled or set above the baseline noise floor ($> 0.10\text{ rad/s}$).

---

## 10. Implementation Plan for a Future Controlled Experiment

When authorized by the user, Gate 2.1 will be implemented in the following sequence:

1. **Step 1: Module Architecture**
   - Create `src/navris/calibration/extrinsic.py` defining:
     - `CausalExtrinsicCalibrator`: State-machine implementation of Methods A, B, C, D.
     - `CalibrationResult`: Dataclass containing rotation matrix $R_{\mathcal{S}}^{\mathcal{V}}$, Euler angles, gyro channel mapping, certification flags, and diagnostic metrics.
2. **Step 2: Standalone Controlled Runner**
   - Create `scripts/run_gate2_1_calibration_experiment.py` executing the 7 experiment scenarios in Table 6.1 on `S1`.
   - Generates detailed diagnostic CSV: `data/processed/phase2_4/experiments/gate2_1_s1_calibration_results.csv`.
3. **Step 3: Synthetic & Regression Unit Tests**
   - Create `tests/test_causal_calibration.py` with synthetic test cases:
     - Synthetic stationary phone with known tilt.
     - Synthetic linear acceleration with known $180^\circ$ braking ambiguity.
     - Synthetic planar turn verifying the $\mathbf{a}_{\text{lat}} \times \boldsymbol{\omega}$ forward vector cross-product.
     - Unobservable standstill scenario testing rejection behavior.
4. **Step 4: Comprehensive Technical Report**
   - Generate `docs/phase2_4_gate2_1_calibration_report.md` documenting results, metrics, failure modes, and recommendations.

---

## 11. What Must Remain Strictly Frozen

During Gate 2.1 design and any future implementation:
- **`src/navris/eskf/` is 100% FROZEN:** No filter equations, noise covariance matrices, or state vectors will be touched.
- **NO batch ESKF replays:** Replay on S2–S8 remains strictly unauthorized.
- **NO AI/ML, NHC, ZUPT, map matching, or Android implementation.**
- **NO modification of Phase 1 or Gate 1.5B synchronized parquets.**

---

## 12. Explicit Hard Stop

This document represents the **complete scientific design specification** for Gate 2.1.  
No code implementation, script execution, or experiment replay has been initiated.

---

**GATE 2.1 STATUS: PLANNING ONLY — AWAITING AUTHORIZATION**
