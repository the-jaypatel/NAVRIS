# NAVRIS — Phase 2.3B Gate 2.3B-1: NHC Scientific Design & Specification
## Non-Holonomic Constraints for Ground Vehicle Dead Reckoning

**Author:** NAVRIS Classical Navigation Research Baseline  
**Gate:** Phase 2.3B — Sub-gate 2.3B-1 (Design & Specification ONLY)  
**Status:** PROPOSED SCIENTIFIC SPECIFICATION — PENDING GATE 2.3B-2 AUTHORIZATION  
**Execution Boundary:** STRICT HARD STOP — Zero Source/ESKF Code Modifications, Zero Benchmarks  
**Date:** September 2026  

---

## 1. Executive Summary

This document establishes the formal mathematical specification, physical kinematics, error-state observability analysis, failure mode classification, and experimental verification plan for **Non-Holonomic Constraints (NHC)** within the NAVRIS navigation engine.

Following the completion of the Gate 2.3A causal Zero-Velocity Update (ZUPT) audit, NAVRIS identified that while ZUPT is a high-fidelity local velocity stabilizer during standstills (delivering a 99.63% RMSE reduction on VTA2), it leaves the navigation system unconstrained during active vehicle motion. During motion, unobservable heading drift continues unchecked, leading to cubic position divergence ($>10^6\text{ m}$) across multi-hour suburban and highway routes.

NHC exploits the fundamental non-holonomic kinematic property of wheeled ground vehicles: assuming no lateral skidding or vertical flight, the velocity of the vehicle body frame perpendicular to the longitudinal driving axis is approximately zero:
$$v_{\text{lateral}} \approx 0, \quad v_{\text{vertical}} \approx 0$$
This specification presents:
1. Exact mapping of the 2D pseudo-measurement into the frozen 15-state NAVRIS ESKF error vector.
2. Complete analytical Jacobian derivation demonstrating how forward speed couples lateral velocity innovation directly into vehicle yaw error.
3. Critical analysis of the phone-to-vehicle extrinsic mounting rotation ($\mathbf{C}_b^v$), identifying mounting misalignment as a primary operational blocker.
4. Lever-arm and centripetal dynamics formulation ($\boldsymbol{\omega} \times \mathbf{r}$).
5. A 20-scenario synthetic validation test suite for Gate 2.3B-2.
6. Multi-dimensional acceptance criteria pre-declared prior to real-data benchmarking.

---

## 2. Motivation from Gate 2.3A Empirical Findings

The Phase 7 benchmark and Phase 8 scientific audit of Gate 2.3A revealed four critical operational boundaries of classical ZUPT on raw consumer smartphone data:

```text
Gate 2.3A Audit Insights Driving NHC:
├── 1. The Observability Void During Motion
│   └── Finding: ZUPT operates strictly when v = 0. During minutes of continuous cruising
│       (e.g., VTA1A with 0 post-departure stops), ZUPT provides zero operational aiding.
├── 2. Heading Blindness
│   └── Finding: ZUPT innovation H_zupt = [0, I, 0, 0, 0] has attitude errors delta_theta
│       in its null space. ZUPT cannot observe or correct yaw drift.
├── 3. Filter Lockout Vulnerability (Pattern B)
│   └── Finding: On recording S2, unobservable heading caused strapdown velocity to reach
│       >34,000 m/s before the first stop. The filter gated out 99.99% of subsequent ZUPTs.
└── 4. Covariance Starvation (Pattern E)
    └── Finding: On recording S3A, 690 ZUPTs during a 327s standstill collapsed velocity covariance,
        causing subsequent GNSS fixes to fail chi-square gating at t=422s.
```

Non-Holonomic Constraints directly address **Insight 1** and **Insight 2** by providing continuous velocity and heading constraints during active vehicle cruise.

---

## 3. Existing NAVRIS Architecture Assumptions

Before specifying NHC, the existing codebase was audited to establish architectural constraints:

1. **Frozen ESKF Core:**
   - Source code under `src/navris/eskf/` is strictly frozen.
   - Nominal state: $\hat{\mathbf{x}} = [\hat{\mathbf{p}}^n, \hat{\mathbf{v}}^n, \hat{\mathbf{q}}_b^n, \hat{\mathbf{b}}_a^b, \hat{\mathbf{b}}_g^b]^T \in \mathbb{R}^{16}$.
   - Error state: $\delta \mathbf{x} = [\delta \mathbf{p}^n, \delta \mathbf{v}^n, \delta \boldsymbol{\theta}^n, \delta \mathbf{b}_a^b, \delta \mathbf{b}_g^b]^T \in \mathbb{R}^{15}$.
   - Error ordering slices: `IDX_POS = slice(0, 3)`, `IDX_VEL = slice(3, 6)`, `IDX_ATT = slice(6, 9)`, `IDX_ACC_BIAS = slice(9, 12)`, `IDX_GYR_BIAS = slice(12, 15)`.
2. **Attitude Parameterization & Error Convention:**
   - Scalar-first Hamilton unit quaternion $\mathbf{q} = [q_w, q_x, q_y, q_z]^T$ representing Body-to-Navigation rotation $\mathbf{C}_b^n$.
   - Navigation-frame attitude error: $\mathbf{q}_{\text{true}} = \Delta \mathbf{q}(\delta \boldsymbol{\theta}^n) \otimes \hat{\mathbf{q}}_b^n$ (multiplication on the left).
   - Linearized DCM: $\mathbf{C}_b^n = (\mathbf{I} - [\delta \boldsymbol{\theta}^n]_\times) \hat{\mathbf{C}}_b^n$.
3. **Causal Extrinsic Calibration (`src/navris/calibration.py`):**
   - Method A estimates gravity leveling matrix $\mathbf{R}_{\text{level}} \in SO(3)$.
   - Method B estimates horizontal mounting yaw $\mathbf{R}_{\text{mount}} \in SO(3)$.
   - Extrinsic DCM mapping phone sensor frame to vehicle frame: $\mathbf{C}_b^v = \mathbf{R}_{\text{mount}} \mathbf{R}_{\text{level}}$.
   - Associated quaternion: $\mathbf{q}_b^v = \text{dcm\_to\_quat}(\mathbf{C}_b^v)$.

---

## 4. NHC Physical Model

### Ideal NHC (Theoretical Wheeled Ground Vehicle)
In ideal wheeled vehicle kinematics:
1. Wheels roll without lateral slipping ($\beta = 0$, where $\beta$ is side-slip angle).
2. The vehicle maintains continuous planar contact with the road surface without vertical bouncing.
3. In the vehicle chassis frame $\mathcal{V}$ (defined as Forward-Left-Up, FLU):
   $$\mathbf{v}^v = \begin{bmatrix} v_x^v \\ v_y^v \\ v_z^v \end{bmatrix} = \begin{bmatrix} V_{\text{forward}} \\ 0 \\ 0 \end{bmatrix}$$
   yielding the ideal twin constraints:
   $$v_y^v = 0 \quad \text{(Lateral non-slip constraint)}$$
   $$v_z^v = 0 \quad \text{(Vertical planar contact constraint)}$$

### Practical NHC (Smartphone in Real Passenger Vehicle)
In realistic consumer driving across public roadways (IO-VNBD dataset):
1. **Pneumatic Tire Elasticity & Cornering Slip:** Vehicles develop lateral slip angles $\alpha_{\text{slip}} \approx 1^\circ - 5^\circ$ during standard turns to generate centripetal cornering force ($F_y = C_\alpha \alpha_{\text{slip}}$). High-speed turning induces non-zero lateral velocity $v_y^v = V_x \tan \beta \neq 0$.
2. **Suspension Compliance & Road Topography:** Potholes, road crowning (transverse slope $2\% - 4\%$), speed bumps, and acceleration pitch / braking dive induce transient vertical velocities $v_z^v \sim \pm 0.1 - 0.5\text{ m/s}$.
3. **Smartphone Lever-Arm Centripetal Velocity:** The phone is mounted in a dashboard cradle or windshield suction mount offset from the vehicle rear axle / center of gravity by lever-arm vector $\mathbf{r}^v = [r_x, r_y, r_z]^T$. When yawing at rate $\omega_z^v$, the phone experiences tangential velocity $\mathbf{v}_{\text{lever}}^v = \boldsymbol{\omega}^v \times \mathbf{r}^v$.

### Unvalidated NHC (Unsupported Assumptions)
- Assuming $v_y^v \equiv 0$ during aggressive highway cornering or emergency braking is physically false.
- Assuming $v_y^v = v_z^v = 0$ in the raw smartphone body frame without an explicit, validated mounting rotation $\mathbf{C}_b^v$ will inject catastrophic cross-axis corruption.

---

## 5. Coordinate-Frame Definitions

The navigation engine operates across three distinct right-handed Cartesian reference frames:

```text
Coordinate Frames:
├── Navigation Frame (n): Local East-North-Up (ENU)
│   ├── +X_n: Local Geodetic East
│   ├── +Y_n: Local Geodetic North
│   └── +Z_n: Local Geodetic Up (Opposite to gravity)
├── Sensor Body Frame (b): Smartphone IMU (Android Standard)
│   ├── +X_b: Tangent to screen, pointing Right
│   ├── +Y_b: Tangent to screen, pointing Up (toward top speaker)
│   └── +Z_b: Perpendicular to screen, pointing Outward
└── Vehicle Frame (v): Automotive Chassis (Forward-Left-Up, FLU)
    ├── +X_v: Longitudinal vehicle centerline, pointing Forward
    ├── +Y_v: Transverse axis, pointing Left (port)
    └── +Z_v: Vertical axis, pointing Up through vehicle roof
```

### Transformation Relations
- $\mathbf{C}_b^n(\mathbf{q}_b^n)$: Rotates vectors from phone body frame $\mathcal{B}$ to navigation frame $\mathcal{N}$: $\mathbf{u}^n = \mathbf{C}_b^n \mathbf{u}^b$.
- $\mathbf{C}_b^v$: Constant extrinsic DCM rotating vectors from phone body frame $\mathcal{B}$ to vehicle frame $\mathcal{V}$: $\mathbf{u}^v = \mathbf{C}_b^v \mathbf{u}^b$.
- $\mathbf{C}_v^n$: Direction cosine matrix rotating vehicle frame $\mathcal{V}$ to navigation frame $\mathcal{N}$:
  $$\mathbf{C}_v^n = \mathbf{C}_b^n (\mathbf{C}_b^v)^T$$
- $\mathbf{C}_n^v$: Direction cosine matrix rotating navigation frame $\mathcal{N}$ to vehicle frame $\mathcal{V}$:
  $$\mathbf{C}_n^v = (\mathbf{C}_v^n)^T = \mathbf{C}_b^v (\mathbf{C}_b^n)^T$$

---

## 6. Measurement Equation Derivation

### Nominal Observation Vector
The NHC measurement vector $\mathbf{z}_{\text{nhc}} \in \mathbb{R}^2$ represents the theoretical zero lateral and vertical velocity in the vehicle frame:
$$\mathbf{z}_{\text{nhc}} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

### Predicted Measurement Function
Let $\hat{\mathbf{v}}^n \in \mathbb{R}^3$ be the filter's nominal velocity estimate in the navigation frame ($\mathbf{v}_{\text{nav}}$). The velocity in the vehicle chassis frame ($\mathbf{v}_{\text{vehicle}}$) is explicitly defined as:
$$\mathbf{v}_{\text{vehicle}} = \hat{\mathbf{v}}^v = \hat{\mathbf{C}}_n^v \hat{\mathbf{v}}^n = \mathbf{C}_b^v (\hat{\mathbf{C}}_b^n)^T \hat{\mathbf{v}}^n$$
where:
- $\text{forward\_speed} = v_{\text{vehicle}}[x] = \mathbf{e}_1^T \hat{\mathbf{v}}^v$ (the signed longitudinal $X_v$ component along vehicle forward).
- $v_{\text{lateral}} = v_{\text{vehicle}}[y] = \mathbf{e}_2^T \hat{\mathbf{v}}^v$ (the transverse $Y_v$ component along vehicle left).
- $v_{\text{vertical}} = v_{\text{vehicle}}[z] = \mathbf{e}_3^T \hat{\mathbf{v}}^v$ (the normal $Z_v$ component along vehicle up).

Extracting the lateral (index 1, $y$) and vertical (index 2, $z$) components yields the measurement function $\mathbf{h}(\hat{\mathbf{x}}) \in \mathbb{R}^2$:
$$\mathbf{h}(\hat{\mathbf{x}}) = \mathbf{M}_{\text{nhc}} \hat{\mathbf{v}}^v = \mathbf{M}_{\text{nhc}} \mathbf{C}_b^v (\hat{\mathbf{C}}_b^n)^T \hat{\mathbf{v}}^n$$
where $\mathbf{M}_{\text{nhc}}$ is the selection matrix:
$$\mathbf{M}_{\text{nhc}} = \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} \in \mathbb{R}^{2 \times 3}$$

> [!IMPORTANT]
> **Forward Speed Definition:** The NHC detector's speed threshold is strictly conditioned on the **signed vehicle longitudinal component** ($\text{forward\_speed} = v_{\text{vehicle}}[x]$), NOT the scalar velocity norm $\|\hat{\mathbf{v}}^n\|_2$. The activation condition:
> $$\text{forward\_speed} = v_{\text{vehicle}}[x] > 1.5\text{ m/s}$$
> requires genuine forward motion along the vehicle chassis. Reverse-driving conditions ($v_{\text{vehicle}}[x] \le 0$) are explicitly rejected from NHC activation rather than silently treated as valid forward motion.
$$\mathbf{M}_{\text{nhc}} = \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} \in \mathbb{R}^{2 \times 3}$$

### Measurement Residual
The measurement residual $\mathbf{r}_{\text{nhc}} \in \mathbb{R}^2$ is:
$$\mathbf{r}_{\text{nhc}} = \mathbf{z}_{\text{nhc}} - \mathbf{h}(\hat{\mathbf{x}}) = - \begin{bmatrix} \hat{v}_y^v \\ \hat{v}_z^v \end{bmatrix} = - \mathbf{M}_{\text{nhc}} \mathbf{C}_b^v (\hat{\mathbf{C}}_b^n)^T \hat{\mathbf{v}}^n$$

---

## 7. Analytical Error-State Jacobian Derivation

To update the 15-state ESKF, the observation function must be linearized with respect to the error state vector:
$$\delta \mathbf{x} = \begin{bmatrix} \delta \mathbf{p}^n \\ \delta \mathbf{v}^n \\ \delta \boldsymbol{\theta}^n \\ \delta \mathbf{b}_a^b \\ \delta \mathbf{b}_g^b \end{bmatrix} \in \mathbb{R}^{15}$$

### Perturbation Expansion
The true velocity in the vehicle frame is:
$$\mathbf{v}^v = \mathbf{C}_n^v \mathbf{v}^n$$
Expressing navigation velocity as $\mathbf{v}^n = \hat{\mathbf{v}}^n + \delta \mathbf{v}^n$ and true Body-to-Navigation DCM as:
$$\mathbf{C}_b^n = (\mathbf{I} + [\delta \boldsymbol{\theta}^n]_\times) \hat{\mathbf{C}}_b^n \implies (\mathbf{C}_b^n)^T = (\hat{\mathbf{C}}_b^n)^T (\mathbf{I} - [\delta \boldsymbol{\theta}^n]_\times)$$
Substituting into the vehicle frame velocity expression:
$$\mathbf{v}^v = \mathbf{C}_b^v (\mathbf{C}_b^n)^T \mathbf{v}^n = \mathbf{C}_b^v (\hat{\mathbf{C}}_b^n)^T (\mathbf{I} - [\delta \boldsymbol{\theta}^n]_\times) (\hat{\mathbf{v}}^n + \delta \mathbf{v}^n)$$
Expanding to first order in error quantities:
$$\mathbf{v}^v \approx \mathbf{C}_b^v (\hat{\mathbf{C}}_b^n)^T \hat{\mathbf{v}}^n + \mathbf{C}_b^v (\hat{\mathbf{C}}_b^n)^T \delta \mathbf{v}^n - \mathbf{C}_b^v (\hat{\mathbf{C}}_b^n)^T [\delta \boldsymbol{\theta}^n]_\times \hat{\mathbf{v}}^n$$
Recalling $\hat{\mathbf{C}}_n^v = \mathbf{C}_b^v (\hat{\mathbf{C}}_b^n)^T$ and applying the vector cross-product identity $- [\delta \boldsymbol{\theta}^n]_\times \hat{\mathbf{v}}^n = + [\hat{\mathbf{v}}^n]_\times \delta \boldsymbol{\theta}^n$:
$$\mathbf{v}^v \approx \hat{\mathbf{v}}^v + \hat{\mathbf{C}}_n^v \delta \mathbf{v}^n + \hat{\mathbf{C}}_n^v [\hat{\mathbf{v}}^n]_\times \delta \boldsymbol{\theta}^n$$
Applying the selection matrix $\mathbf{M}_{\text{nhc}}$:
$$\delta \mathbf{h} = \mathbf{M}_{\text{nhc}} \hat{\mathbf{C}}_n^v \delta \mathbf{v}^n + \mathbf{M}_{\text{nhc}} \hat{\mathbf{C}}_n^v [\hat{\mathbf{v}}^n]_\times \delta \boldsymbol{\theta}^n$$

### Complete Measurement Matrix $\mathbf{H}_{\text{nhc}} \in \mathbb{R}^{2 \times 15}$
Mapping each block directly into the NAVRIS error-state ordering:
$$\mathbf{H}_{\text{nhc}} = \begin{bmatrix}
\mathbf{0}_{2\times 3} & \mathbf{H}_v & \mathbf{H}_\theta & \mathbf{0}_{2\times 3} & \mathbf{0}_{2\times 3}
\end{bmatrix}$$
where:
1. **Position Block (indices 0:3):**
   $$\mathbf{H}_p = \mathbf{0}_{2 \times 3}$$
2. **Velocity Block (indices 3:6):**
   $$\mathbf{H}_v = \mathbf{M}_{\text{nhc}} \hat{\mathbf{C}}_n^v = \begin{bmatrix} \mathbf{e}_2^T \hat{\mathbf{C}}_n^v \\ \mathbf{e}_3^T \hat{\mathbf{C}}_n^v \end{bmatrix} \in \mathbb{R}^{2 \times 3}$$
3. **Attitude Block (indices 6:9):**
   $$\mathbf{H}_\theta = + \mathbf{M}_{\text{nhc}} \hat{\mathbf{C}}_n^v [\hat{\mathbf{v}}^n]_\times = + \begin{bmatrix} \mathbf{e}_2^T \hat{\mathbf{C}}_n^v [\hat{\mathbf{v}}^n]_\times \\ \mathbf{e}_3^T \hat{\mathbf{C}}_n^v [\hat{\mathbf{v}}^n]_\times \end{bmatrix} \in \mathbb{R}^{2 \times 3}$$
4. **Accelerometer Bias Block (indices 9:12):**
   $$\mathbf{H}_{ba} = \mathbf{0}_{2 \times 3}$$
5. **Gyroscope Bias Block (indices 12:15):**
   $$\mathbf{H}_{bg} = \mathbf{0}_{2 \times 3}$$

---

## 8. Measurement Covariance Design

### Plausible Order of Magnitude
Unlike stationary ZUPT where velocity is bounded by sub-centimeter thermal noise, moving NHC noise accounts for suspension oscillation, road roughness, and tire cornering elasticity:
- **Lateral Noise ($\sigma_{\text{lat}}$):** Normal driving side-slip ranges from $0.5^\circ - 2.0^\circ$. At $20\text{ m/s}$ ($72\text{ km/h}$), this equates to unmodeled lateral velocities of $\approx 0.17 - 0.70\text{ m/s}$.
- **Vertical Noise ($\sigma_{\text{vert}}$):** Normal road grades, bumps, and vehicle pitch oscillations produce vertical velocities of $\approx 0.10 - 0.40\text{ m/s}$.

### Pre-Declared Engineering Hypothesis Covariance
To prevent ad-hoc parameter optimization during benchmark runs, NAVRIS defines a pre-declared conservative diagonal covariance matrix:
$$\mathbf{R}_{\text{nhc}} = \begin{bmatrix} \sigma_{\text{lat}}^2 & 0 \\ 0 & \sigma_{\text{vert}}^2 \end{bmatrix} = \begin{bmatrix} (0.25)^2 & 0 \\ 0 & (0.15)^2 \end{bmatrix} = \begin{bmatrix} 0.0625 & 0 \\ 0 & 0.0225 \end{bmatrix}\text{ m}^2/\text{s}^2$$
- Lateral standard deviation: $\sigma_{\text{lat}} = 0.25\text{ m/s}$ ($25\text{ cm/s}$).
- Vertical standard deviation: $\sigma_{\text{vert}} = 0.15\text{ m/s}$ ($15\text{ cm/s}$).
- **Status:** **PRE-DECLARED CONSERVATIVE ENGINEERING HYPOTHESIS**.
- **STRICT PROTOCOL:** These values must NOT be optimized or tuned against VTA2, S1, or any real-data benchmark recording before or during the first A/B experiment. They serve as a frozen experimental baseline.

---

## 9. NIS & Innovation Gating Strategy

### Innovation Covariance
$$\mathbf{S}_{\text{nhc}} = \mathbf{H}_{\text{nhc}} \mathbf{P}_{k|k-1} \mathbf{H}_{\text{nhc}}^T + \mathbf{R}_{\text{nhc}} \in \mathbb{R}^{2 \times 2}$$
Because $\mathbf{S}_{\text{nhc}}$ is $2 \times 2$, its inversion is computed analytically without numerical Cholesky instability:
$$\mathbf{S}_{\text{nhc}}^{-1} = \frac{1}{S_{11}S_{22} - S_{12}^2} \begin{bmatrix} S_{22} & -S_{12} \\ -S_{12} & S_{11} \end{bmatrix}$$

### Normalized Innovation Squared (NIS)
$$\text{NIS}_{\text{nhc}} = \mathbf{r}_{\text{nhc}}^T \mathbf{S}_{\text{nhc}}^{-1} \mathbf{r}_{\text{nhc}}$$

### 2-DOF Chi-Square Gating
Under nominal Gaussian noise assumptions, $\text{NIS}_{\text{nhc}} \sim \chi^2(2)$.
- $95.0\%$ confidence threshold: $\chi^2_{0.95}(2) = 5.99$
- $99.0\%$ confidence threshold: $\chi^2_{0.99}(2) = 9.21$
- $99.9\%$ confidence threshold: $\chi^2_{0.999}(2) = 13.82$

**Pre-Declared Gating Rule:**
$$\chi^2_{\text{nhc,thresh}} = 13.82 \quad (2\text{ DOF, } p = 0.001)$$
If $\text{NIS}_{\text{nhc}} > 13.82$, the measurement is rejected. This protects the filter from aggressive cornering slip, lane changes, speed bumps, and road potholes.

---

## 10. Observability Analysis: Does NHC Observe Yaw?

This analysis answers whether NHC directly resolves yaw drift.

### Attitude Block Decomposition
Let vehicle velocity be predominantly forward: $\hat{\mathbf{v}}^v \approx [V_x, 0, 0]^T$. The cross-product matrix in the vehicle frame is:
$$[\hat{\mathbf{v}}^v]_\times = \begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & -V_x \\ 0 & V_x & 0 \end{bmatrix}$$
Using the similarity transformation $\hat{\mathbf{C}}_n^v [\hat{\mathbf{v}}^n]_\times = [\hat{\mathbf{v}}^v]_\times \hat{\mathbf{C}}_n^v$, the attitude Jacobian $\mathbf{H}_\theta$ expands as:
$$\mathbf{H}_\theta = - \mathbf{M}_{\text{nhc}} [\hat{\mathbf{v}}^v]_\times \hat{\mathbf{C}}_n^v = - \begin{bmatrix} 0 & 0 & -V_x \\ 0 & V_x & 0 \end{bmatrix} \hat{\mathbf{C}}_n^v$$
Multiplying out the rows:
1. **Lateral Constraint Row ($\text{row } 0$):**
   $$\mathbf{H}_{\text{lat}, \theta} = V_x \mathbf{e}_3^T \hat{\mathbf{C}}_n^v = V_x (\mathbf{u}_{\text{up}}^v)^T$$
2. **Vertical Constraint Row ($\text{row } 1$):**
   $$\mathbf{H}_{\text{vert}, \theta} = - V_x \mathbf{e}_2^T \hat{\mathbf{C}}_n^v = - V_x (\mathbf{u}_{\text{left}}^v)^T$$

### Mathematical Observability Conclusions
1. **Speed Proportionality:** The sensitivity of NHC innovations to attitude errors is **directly proportional to forward velocity $V_x$**.
   - When the vehicle is stopped ($V_x = 0$): $\mathbf{H}_\theta = \mathbf{0}_{2\times 3}$. Stationary NHC provides **zero attitude observability** (identical to ZUPT).
   - When moving at highway speeds ($V_x = 30\text{ m/s}$): $\mathbf{H}_\theta$ provides high sensitivity ($30\text{ rad/s}$ per radian of attitude error).
2. **Yaw Error Coupling:** The lateral constraint row $\mathbf{H}_{\text{lat}, \theta} = V_x (\mathbf{u}_{\text{up}}^v)^T$ projects attitude error directly along the vehicle vertical axis $\mathbf{u}_{\text{up}}^v$. Therefore, **lateral velocity innovation directly couples to vehicle yaw error $\delta \theta_{\text{yaw}}^v$**.
3. **Pitch Error Coupling:** The vertical constraint row $\mathbf{H}_{\text{vert}, \theta} = - V_x (\mathbf{u}_{\text{left}}^v)^T$ projects attitude error along the vehicle transverse axis $\mathbf{u}_{\text{left}}^v$. Therefore, **vertical velocity innovation directly couples to vehicle pitch error $\delta \theta_{\text{pitch}}^v$**.
4. **Roll Null Space:** Rotation about the forward axis $\mathbf{u}_{\text{fwd}}^v$ (vehicle roll) is in the null space of $\mathbf{H}_\theta$:
   $$\mathbf{H}_\theta \mathbf{u}_{\text{fwd}}^n = \mathbf{0}_{2\times 1}$$
   NHC provides **zero observability into vehicle roll**. Roll must be constrained by gravity leveling during standstills.
5. **Observability Distinction (What NHC Does NOT Provide):**
   NHC does **not** provide absolute geographic heading. It provides a velocity-attitude constraint: the velocity vector must align with the vehicle longitudinal axis. If the entire navigation frame estimate rotates uniformly, NHC cannot detect that rotation without external absolute heading (e.g. from novel GNSS position displacements).

---

## 11. Phone-to-Vehicle Alignment: The Critical Blocker

NHC updates transform velocity into the vehicle frame using $\mathbf{C}_b^v$:
$$\hat{\mathbf{v}}^v = \mathbf{C}_b^v (\hat{\mathbf{C}}_b^n)^T \hat{\mathbf{v}}^n$$
If the estimated extrinsic rotation $\mathbf{C}_b^v$ contains an azimuthal mounting error $\Delta \psi_{\text{mount}}$, the true forward velocity $V_x$ projects into the perceived lateral axis:
$$v_{\text{lateral,measured}} \approx V_x \sin(\Delta \psi_{\text{mount}})$$

### Quantitative Impact of Mounting Error
For a vehicle driving at $72\text{ km/h}$ ($20\text{ m/s}$):
- If $\Delta \psi_{\text{mount}} = 1.0^\circ$: $v_{\text{lat}} = 20 \times \sin(1^\circ) = 0.35\text{ m/s}$.
  $$\text{NIS} \approx \frac{(0.35)^2}{(0.25)^2} = 1.96 < 13.82 \quad \text{(Accepted)}$$
  The filter injects a false state correction, forcing heading to drift to compensate.
- If $\Delta \psi_{\text{mount}} = 5.0^\circ$: $v_{\text{lat}} = 20 \times \sin(5^\circ) = 1.74\text{ m/s}$.
  $$\text{NIS} \approx \frac{(1.74)^2}{(0.25)^2} = 48.4 \gg 13.82 \quad \text{(Rejected / Filter Lockout)}$$
- If $\Delta \psi_{\text{mount}} = 106.0^\circ$ (as observed on recording Y1): $v_{\text{lat}} = 20 \times \sin(106^\circ) = 19.2\text{ m/s}$.
  $$\text{NIS} \approx \frac{(19.2)^2}{(0.25)^2} = 5,898 \gg 13.82 \quad \text{(Catastrophic Lockout)}$$

### Current Repository Status
In Gate 2.2, Method B mounting yaw achieved stable alignment on routes with clean initial turns (S1, S3A, VTA2), but suffered unvalidated fallback on straight-line routes (S2, Y1).
- **CRITICAL SCIENTIFIC FINDING:** **NHC cannot be applied safely on routes with unvalidated mounting alignment.**
- Applying NHC with an unvalidated $\mathbf{C}_b^v$ is mathematically equivalent to injecting systematic, correlated velocity corruption at 10 Hz.
- **GATE 2.3B DEPENDENCY:** Causal verification of mounting alignment quality must precede NHC activation.

---

## 12. Lever-Arm Analysis

When the smartphone is located at a distance $\mathbf{r}^v = [r_x, r_y, r_z]^T$ from the vehicle navigation center (rear axle midpoint), the physical velocity of the phone IMU $\mathbf{v}_{\text{phone}}^v$ differs from the vehicle chassis velocity $\mathbf{v}_{\text{vehicle}}^v$:
$$\mathbf{v}_{\text{phone}}^v = \mathbf{v}_{\text{vehicle}}^v + \boldsymbol{\omega}^v \times \mathbf{r}^v$$

### Tangential Velocity Components
Expanding the cross product with vehicle angular rates $\boldsymbol{\omega}^v = [\omega_x, \omega_y, \omega_z]^T$:
$$\boldsymbol{\omega}^v \times \mathbf{r}^v = \begin{bmatrix}
\omega_y r_z - \omega_z r_y \\
\omega_z r_x - \omega_x r_z \\
\omega_x r_y - \omega_y r_x
\end{bmatrix}$$
The perceived lateral velocity is:
$$v_{\text{phone}, y}^v = v_{\text{vehicle}, y}^v + \omega_z r_x - \omega_x r_z$$

### Order-of-Magnitude Evaluation
In passenger cars, a windshield phone mount typically has a longitudinal offset $r_x \approx +1.5 - 2.0\text{ m}$ forward of the vehicle center of rotation, and vertical height $r_z \approx +0.5\text{ m}$.
- During a standard urban cornering turn at yaw rate $\omega_z = 15^\circ/\text{s} = 0.26\text{ rad/s}$:
  $$\Delta v_{\text{lat,lever}} = \omega_z r_x = 0.26\text{ rad/s} \times 1.8\text{ m} \approx 0.47\text{ m/s}$$
- This $0.47\text{ m/s}$ tangential velocity is larger than our pre-declared $\sigma_{\text{lat}} = 0.25\text{ m/s}$!
- **Conclusion:** During sharp turns, neglecting the lever arm causes genuine vehicle turns to fail NHC innovation gating or corrupt the yaw estimate.
- **Specification:** When angular rate exceeds $\|\boldsymbol{\omega}\| > 5.0^\circ/\text{s}$ ($0.087\text{ rad/s}$), NHC updates must either incorporate the lever-arm compensation or be inhibited. Because the IO-VNBD dataset does not document exact physical mounting measurements ($r_x, r_y, r_z$), the safest causal strategy is **turning inhibition**.

---

## 13. Comprehensive Failure Mode Matrix

| Failure Mode | Physical Cause | Detectability at Runtime | Effect on ESKF State | NIS Protection? | Mitigation Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mounting Yaw Error** | Inaccurate $\mathbf{C}_b^v$ from straight-line departure | Low (correlated bias) | Massive false yaw drift or total filter lockout | No (if error is small $\sim 2^\circ$), Yes (if $>5^\circ$) | Gated activation: only apply if Method B confidence is HIGH |
| **Cornering Side-Slip** | Centripetal acceleration during high-speed turns | High ($\|\boldsymbol{\omega}\|$, lateral accel) | Distorts heading; over-constrains yaw | Partial | Turn inhibitor: suspend NHC when $\|\boldsymbol{\omega}\| > 5^\circ/\text{s}$ |
| **Lever-Arm Tangential Velocity** | Phone offset from vehicle turning center ($r_x \omega_z$) | High ($\omega_z$) | Induces artificial lateral velocity | Partial | Inhibit NHC during turns ($\|\boldsymbol{\omega}\| > 5^\circ/\text{s}$) |
| **Potholes / Road Bumps** | Transient vertical shock | High (instantaneous accel spike) | Perturbs vertical velocity and pitch | Yes ($\text{NIS} > 13.82$) | 2-DOF Chi-square gating + accel variance threshold |
| **Vehicle Standstill** | Car is stopped ($V_x = 0$) | High (ZUPT detector triggers) | Degenerate Jacobian $\mathbf{H}_\theta = \mathbf{0}$; covariance starvation | No | Inhibit NHC when speed $< 1.0\text{ m/s}$ or ZUPT is active |
| **Reverse Driving** | Backing up ($V_x < 0$) | Low without wheel encoders | Inverts sign of $\mathbf{H}_\theta$; destabilizes yaw | No | Gated by positive forward GNSS displacement |
| **Phone Dislodgement** | Phone shifts in mount while driving | High (sudden accel/gyro shock) | Catastrophic divergence across all axes | Yes | Shock detector: abort NHC if residual gravity changes |
| **Covariance Starvation** | Applying 10 Hz updates indefinitely | None (asymptotic filter property) | Covariance collapses; rejects valid GNSS | No | Update decimation (e.g. 1 Hz) + minimum covariance floor |

---

## 14. NHC Detector Architecture Options

We evaluate four structural options for applying NHC updates:

```text
Detector Architecture Trade-off:
├── Option A: Unconditional 10 Hz Execution
│   ├── Pros: Simple; maximum theoretical constraints.
│   └── Cons: CATASTROPHIC RISK. Applies false constraints during stops, turns, slips, and bumps.
├── Option B: Binary Motion & Dynamic Window Detector (RECOMMENDED)
│   ├── Description: Active strictly when vehicle is moving forward (speed > 1.5 m/s),
│   │   yaw rate is low (|omega| < 5 deg/s), and acceleration variance is nominal.
│   ├── Pros: Eliminates lever-arm effects, cornering slips, and stationary degeneracy.
│   └── Cons: Requires pre-declared detector thresholds.
├── Option C: Continuous Adaptive Noise Covariance
│   ├── Description: Scale R_nhc dynamically: R_nhc(t) = R_0 + k1 * |omega_z| + k2 * |a_lat|.
│   ├── Pros: Smooth transitions between straight driving and cornering.
│   └── Cons: Introduces heuristic hyper-parameters prone to uncalibrated over-fitting.
└── Option D: Selective Straight-Line Highway Gating
    ├── Description: Restrict NHC strictly to verified straight-line highway cruising.
    └── Cons: Highly restrictive; provides zero aiding in urban corridors.
```

### Recommendation for Gate 2.3B
**Option B (Binary Motion & Dynamic Quality Detector)** is the only scientifically defensible choice. It guarantees that NHC updates are injected only when the underlying non-holonomic physical assumptions are physically valid.

---

## 15. Gate 2.3B-2 Synthetic Validation Plan

Prior to any real-data benchmark, Gate 2.3B-2 must execute a deterministic 20-scenario synthetic test suite in `tests/test_nhc.py`:

```text
Synthetic Validation Test Matrix (20 Scenarios):
├── Kinematic Baseline Tests:
│   ├── TEST 01: Constant forward speed (10 m/s) with zero errors -> Innovations identically zero
│   ├── TEST 02: Pure forward acceleration (0 to 25 m/s) -> Zero lateral/vertical residuals
│   ├── TEST 03: Pure braking deceleration (25 to 0 m/s) -> Zero lateral residuals
│   └── TEST 04: Linear motion at 45° azimuth in ENU -> Velocity rotated correctly to vehicle frame
├── Attitude & Yaw Coupling Tests:
│   ├── TEST 05: Induced +2° yaw error at 20 m/s -> Verify H_theta couples into lateral residual
│   ├── TEST 06: Induced +2° yaw error at 0 m/s -> Verify H_theta is zero (no false correction)
│   ├── TEST 07: Induced +2° pitch error at 20 m/s -> Verify vertical constraint couples into pitch
│   └── TEST 08: Induced +5° roll error at 20 m/s -> Verify roll is in the null space of H_nhc
├── Violation & Gating Rejection Tests:
│   ├── TEST 09: Simulated lateral side-slip (v_lat = 2.0 m/s) -> NIS exceeds 13.82; update rejected
│   ├── TEST 10: Simulated vertical pothole shock (v_vert = 1.5 m/s) -> NIS rejected
│   ├── TEST 11: Constant steady-state turn (15 deg/s) -> Detector inhibits NHC execution
│   └── TEST 12: Reverse driving (v_fwd = -5 m/s) -> Correct directional handling
├── Numerical & Stability Tests:
│   ├── TEST 13: Covariance symmetry and positive definiteness across 1,000 continuous updates
│   ├── TEST 14: Minimum eigenvalue lambda_min(P) > 1e-6 (no covariance collapse)
│   ├── TEST 15: Joseph update equivalence against standard Kalman formulation
│   └── TEST 16: Quaternion normalization preserved (|q| = 1.000000)
└── Error & Bias Interaction Tests:
    ├── TEST 17: Uncorrected gyro bias -> NHC prevents unbounded velocity runaway
    ├── TEST 18: Uncorrected accel bias -> NHC bounds velocity drift in lateral plane
    ├── TEST 19: Mounting misalignment (+5°) -> Triggers chi-square gating
    └── TEST 20: 100% strict causality check (zero future-sample dependencies)
```

### Mandatory Finite-Difference Jacobian Verification
Before any real-data benchmark execution, the analytical NHC measurement Jacobian $\mathbf{H}_{\text{nhc}}$ must be numerically verified using central finite-difference perturbations under the exact existing NAVRIS error-state convention:
$$\mathbf{q}_{\text{true}} = \Delta \mathbf{q}(\delta \boldsymbol{\theta}^n) \otimes \mathbf{q}_{\text{nominal}}$$
where $\Delta \mathbf{q}(\delta \boldsymbol{\theta}^n) = \text{rotvec\_to\_quat}(\delta \boldsymbol{\theta}^n)$.

For each error state component $j \in \{0, \dots, 14\}$ with perturbation step $\epsilon = 10^{-7}$:
$$\mathbf{H}_{\text{num}, :, j} = \frac{\mathbf{h}(\hat{\mathbf{x}} \oplus \epsilon \mathbf{e}_j) - \mathbf{h}(\hat{\mathbf{x}} \ominus \epsilon \mathbf{e}_j)}{2\epsilon}$$
The verification suite must explicitly validate:
1. **Position Block (0:3):** Verifies $\mathbf{H}_p = \mathbf{0}_{2\times 3}$.
2. **Velocity Block (3:6):** Verifies $\mathbf{H}_v = \mathbf{M}_{\text{nhc}} \hat{\mathbf{C}}_n^v$.
3. **Attitude Block (6:9):** Verifies $\mathbf{H}_\theta = - \mathbf{M}_{\text{nhc}} \hat{\mathbf{C}}_n^v [\hat{\mathbf{v}}^n]_\times = - \mathbf{M}_{\text{nhc}} [\hat{\mathbf{v}}^v]_\times \hat{\mathbf{C}}_n^v$.
4. **Bias Blocks (9:15):** Verifies $\mathbf{H}_{ba} = \mathbf{0}_{2\times 3}$ and $\mathbf{H}_{bg} = \mathbf{0}_{2\times 3}$.
5. **Sign & Convention Consistency:** Verifies left-quaternion multiplication and DCM transposition consistency.

> [!CAUTION]
> **MANDATORY STOP CRITERION:** The analytical and numerical Jacobians must agree within $\|\mathbf{H}_{\text{analytical}} - \mathbf{H}_{\text{num}}\|_\infty < 1.0\times 10^{-6}$. If any discrepancy exceeds this bound, **STOP AND RESOLVE THE CONVENTION MATHEMATICALLY**. Under no circumstances may the ESKF core or perturbation conventions be altered to force a test to pass.

NHC runtime execution must strictly satisfy:
1. **Zero Reference Leakage:** No use of VBOX ground truth speed, yaw, or position at any point in the detector or measurement equations.
2. **Zero Look-Ahead:** All motion tests, variances, and speeds must be derived from trailing causal windows ($t \le t_k$).
3. **Zero Backward Smoothing:** Forward causal filtering only.
4. **Frozen Pre-Declared Thresholds:** Detector and covariance parameters declared prior to evaluation.

---

## 17. Real-Data Controlled A/B Experiment Design

When authorized for Gate 2.3B-3, the real-data experiment must evaluate:

### Experimental Configurations
- **Configuration A (Control Baseline):** Frozen Gate 2.3A pipeline (Calibrated ESKF + Causal ZUPT enabled).
- **Configuration B (Experiment):** Identical pipeline + Causal NHC module enabled.
- **Isolated Variable:** The *only* difference between A and B is the activation of the causal NHC update.

### Scope: All Eight IO-VNBD Routes
No recordings may be omitted. S1, S2, S3A, S4, M, Y1, VTA1A, and VTA2 must all be evaluated.

### Evaluation Metrics
1. **Primary Navigation Metrics:**
   - Horizontal RMSE (m)
   - Final Horizontal Error (m)
   - Maximum Horizontal Error (m)
   - Along-track and Cross-track RMSE (m)
2. **Attitude & Velocity Metrics:**
   - Heading RMSE ($^\circ$) relative to VBOX reference
   - Velocity RMSE (m/s)
3. **Filter Integrity & Innovation Metrics:**
   - NHC attempted, accepted, and rejected counts
   - NHC NIS distribution (median, 95th percentile, max)
   - GNSS fix acceptance percentage (A vs B)
   - Minimum covariance eigenvalue $\lambda_{\min}(\mathbf{P})$

---

## 18. Multi-Dimensional Acceptance Criteria

Gate 2.3B acceptance shall be judged across seven objective dimensions:

```text
Gate 2.3B Acceptance Matrix:
├── 1. Mathematical Rigor: Analytical Jacobian matches numerical perturbation (<1e-6 error).
├── 2. Synthetic Verification: 20/20 synthetic scenarios pass deterministically.
├── 3. Strict Causality: Zero future/reference data access verified by automated AST audit.
├── 4. Baseline Non-Regression: Setting enable_nhc=False reproduces Gate 2.3A bit-for-bit (diff < 1e-8 m).
├── 5. Physical Consistency: On accepted updates, median NIS_nhc <= 2.0 (nominal 2-DOF chi-square).
├── 6. Cross-Track Improvement: Measurable cross-track drift reduction on observable routes (VTA2, S1).
└── 7. Scientific Restraint: Transparent reporting of all failure modes without post-hoc tuning.
```

---

## 19. Relationship to ZUPT

- **Complementary Kinematic Regimes:**
  - ZUPT operates when $V_x = 0$ (standstill).
  - NHC operates when $V_x > 1.5\text{ m/s}$ (active cruising).
- **Mutual Exclusion:** When the vehicle is stationary, NHC must be disabled. When the vehicle is moving, ZUPT is disabled.
- **Scientific Isolation in Initial Evaluation:**
  - To isolate what NHC contributes independently of ZUPT, Gate 2.3B should also evaluate an isolated ablation: **ESKF Alone vs ESKF + NHC** on key routes, ensuring that ZUPT interaction does not mask NHC defects.

---

## 20. Relationship to Future Covariance & GNSS Robustness

The Gate 2.3A audit proved that repeated updates during extended intervals cause **Covariance Starvation**.
- **The NHC Starvation Risk:** Cruising continuously for 20 minutes on a highway corresponds to thousands of consecutive NHC updates. In the absence of GNSS fixes, repeated updates will collapse velocity and attitude covariance blocks into near-zero values. When the vehicle eventually encounters a GNSS fix, the starved filter risks gating out valid fixes.
- **Strict Isolation Protocol for Gate 2.3B:**
  - The first NHC experiment must test **ONLY:**
    $$\text{Gate 2.3A Frozen Baseline} + \text{NHC}$$
  - **Do NOT** implement covariance floors, covariance fading, adaptive GNSS gating, or other starvation mitigations as part of NHC in this gate.
  - **Do NOT** decimate NHC updates to 1 Hz. NHC shall run at its native evaluation rate without ad-hoc mitigations to evaluate pure NHC behavior.
  - The covariance-starvation issue and any fading-memory / adaptive gating mechanisms must remain a **separately documented, standalone future experiment (Gate 2.3C)**.

---

## 21. Explicit Assumptions

1. The vehicle adheres to planar Ackermann steering kinematics during nominal cruising.
2. The smartphone remains rigidly fixed relative to the vehicle cabin during each recording.
3. The causal calibration matrix $\mathbf{C}_b^v$ accurately represents the physical rotation from phone to vehicle frame.

---

## 22. Explicit Unknowns & Blockers

1. **Mounting Quality Uncertainty:** On recordings S2 and Y1, causal mounting calibration experienced straight-line fallback. Applying NHC with unverified mounting is an unvalidated risk.
2. **Exact Lever-Arm Geometry:** The IO-VNBD dataset does not provide millimeter-accurate physical coordinates of phone mounts relative to vehicle axles.

---

## 23. Scientific Risks

- **Risk 1 (Yaw Corruption):** If mounting yaw is off by even $3^\circ$, NHC will continuously drive yaw into an incorrect heading.
- **Risk 2 (GNSS Lockout):** If NHC shrinks velocity covariance too aggressively, subsequent GNSS updates will be rejected.
- **Risk 3 (Filter Instability):** Numerical asymmetry in high-rate 2D updates if Joseph form is compromised.

---

## 24. Mandatory Experimental Progression & Next Step

The NAVRIS research pipeline enforces a strictly isolated, non-overlapping experimental sequence:
```text
Experimental Progression Ladder:
Gate 2.3B-1 — Design & Specification (CURRENT GATE - COMPLETE)
      ↓
Gate 2.3B-2 — Synthetic Implementation + Unit Verification (tests/test_nhc.py)
      ↓
Gate 2.3B-3 — Real-Data A/B Benchmark across all 8 recordings
      ↓
Phase 8 Scientific Audit of Gate 2.3B
      ↓
Gate 2.3C — Separate Covariance Robustness & Standstill Anti-Starvation Experiment
      ↓
Phase 3 — Future AI/ML Residual Augmentation (TCN / GRU)
```
**DO NOT COMBINE THESE EXPERIMENTS.**

**DO NOT PROCEED TO REAL-DATA BENCHMARKING.**

The repository is now implementation-ready for **Gate 2.3B-2: Synthetic Implementation & Verification** pending explicit user authorization.

---

**STATUS: GATE 2.3B-1 SPECIFICATION COMPLETE & CLARIFIED — AWAITING EXPLICIT AUTHORIZATION FOR GATE 2.3B-2**
