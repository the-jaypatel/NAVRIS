# NAVRIS Phase 2.3B: Controlled Experiment E2 Report
## Input Frame & Unit Forensics + Controlled S1 Diagnostic Replay

**Project:** NAVRIS — Intelligent Navigation & Inertial System  
**SIH 2026 Problem Statement:** 26168  
**Scope:** Experiment E2: Input Frame, Axis, and Unit Forensics on Recording S1  
**Execution Date:** 2026-09-20  
**Status:** COMPLETE — STRICT STOP ENFORCED  
**Replay Classification:** DIAGNOSTIC ORACLE — NOT DEPLOYABLE  

---

## 1. Executive Summary & Epistemic Verdicts

Following the forensic review of Phase 2.3B Gate 1 and Experiment E1, a comprehensive raw-data and physical frame audit was conducted on IO-VNBD recording S1. The investigation confirmed that the catastrophic filter divergence observed in Baseline and E1 was primarily caused by severe **input-frame misalignments** and **unit misinterpretations in the data ingestion pipeline**, rather than any defect in the Phase 2.3A Error-State Kalman Filter (ESKF) core.

### Rigorous Epistemic Classifications

* **DEMONSTRATED (Mathematical & Empirical Certainty):**
  1. **Smartphone GNSS Speed Unit Defect:** The raw CSV column GPS SPEED (Kmh) in S-S1.csv is natively logged in **m/s**, NOT km/h. The data pipeline (src/navris/ingest.py, line 282) multiplied this column by KMH_TO_MPS (1/3.6), artificially dividing all smartphone GNSS speeds by **3.6x**. At t = 156.0 s, the raw value is 11.94 (matching reference speed 12.03 m/s to within 0.7%), but was ingested as 3.32 m/s.
  2. **Gyroscope vs Accelerometer Axis Inconsistency:** Accelerometer and gyroscope data in S-S1.csv are expressed in different coordinate reference frames:
     * In the accelerometer, gravity acts primarily along +Z (f_z = +9.847 m/s^2 at rest), confirming +Z_accel is vertical UP.
     * In the gyroscope, vehicle yaw rate (heading turn rate) is captured by column +Y_gyro (GYROSCOPE Pitch), exhibiting a Pearson correlation of **r = +0.932** and linear regression slope of **k = +0.978** against VBOX ground-truth yaw rate over 42,641 dynamic samples. Column Z_gyro (GYROSCOPE Roll) has an opposing/weak correlation (r = -0.227).
     * Passing raw [gx, gy, gz] directly into strapdown integration caused actual vehicle yaw turns to be integrated as pitch rate, tilting the filter's nominal attitude by -68 deg and projecting gravitational acceleration horizontally.
  3. **Fix 4 Divergence Resolved in Replay:** When the gyroscope is aligned with the accelerometer vertical frame and initialized with true causal speed, **Fix 4 (t = 192 s) cleanly PASSES gating**: NIS = 1.54 < 16.27, with horizontal position error of **3.56 m** (compared to Baseline: 2,376 m / NIS = 50.4, and E1-A: 452 m / NIS = 159.4). 51 consecutive fixes from t = 156 s to t = 624 s are accepted with horizontal errors < 8.5 m.

* **STRONGLY SUPPORTED (High Confidence, Empirical Multi-Segment Backing):**
  * The smartphone was mounted horizontally in the vehicle with a fixed physical yaw offset of approximately -45 deg (315 deg) relative to the vehicle forward heading, resulting in vehicle longitudinal accelerations projecting equally onto +X and -Y of the phone sensor frame.

* **PLAUSIBLE HYPOTHESIS (Requires Multi-Drive Verification):**
  * The column naming convention (GYROSCOPE Yaw = X, Pitch = Y, Roll = Z) reflects an internal sensor daemon convention of the data logger app, which may be consistent across all smartphone recordings (S1 - S8) or dependent on phone orientation during mounting.

* **NOT DEMONSTRATED / STRICTLY FORBIDDEN FROM CLAIMING:**
  * That loose position-only GNSS/INS at 0.1 Hz is solved for long-term production navigation.
  * That this diagnostic replay is deployable in real time without causal frame estimation or extrinsic calibration.

---

## 2. Raw-Data Audit: Headers, Units, and Ingestion Logic

### 2.1 File Header Inspection (S-S1.csv)
Inspection of the first lines of the raw smartphone log data/raw/IO-VNBD/S-S1.csv revealed the following column schema:
`csv
TIME (s),ACCELEROMETER X (m/s²),ACCELEROMETER Y (m/s²),ACCELEROMETER Z (m/s²),GYROSCOPE Yaw (rad/s),GYROSCOPE Pitch (rad/s),GYROSCOPE Roll (rad/s),ORIENTATION Z (rad),ORIENTATION Y (rad),ORIENTATION X (rad),GPS Latitude (deg),GPS Longitude (deg),GPS Altitude (m),GPS Bearing (deg),GPS SPEED (Kmh),GPS Quality,GPS Direction
`

### 2.2 The 3.6x Speed Unit Defect
The raw column header states GPS SPEED (Kmh). However, physical inspection against reference VBOX speed (V-S1.csv) reveals that the values logged were already in **m/s**:

| Timestamp (t) | Raw GPS SPEED (Kmh) | VBOX Speed (
ef_speed_mps) | Ratio (Raw / VBOX) | Ingested phone_gps_speed_mps |
| :---: | :---: | :---: | :---: | :---: |
| **0.0 s** | 5.570 | 5.547 m/s (19.97 km/h) | **1.004** | 1.547 m/s (0.28x) |
| **147.0 s** | 15.530 | 15.510 m/s (55.84 km/h) | **1.001** | 4.314 m/s (0.28x) |
| **156.0 s** | 11.940 | 12.030 m/s (43.31 km/h) | **0.992** | 3.317 m/s (0.28x) |

#### Root Cause:
Android's Location.getSpeed() API returns ground speed natively in **meters per second** (m/s). The logging application developer appended (Kmh) to the CSV column label without performing the unit conversion (speed_mps * 3.6).

#### Ingestion Pipeline Defect:
In src/navris/ingest.py, line 282:
`python
# DEFECT IN INGESTION PIPELINE:
out['phone_gps_speed_mps'] = (
    pd.to_numeric(df['GPS SPEED (Kmh)'], errors='coerce') * KMH_TO_MPS
)
`
Where KMH_TO_MPS = 1.0 / 3.6. This line divided the already-m/s values by 3.6, creating an artificial **3.6x speed compression** across the entire dataset.

---

## 3. Frame Relationship & Dynamic Cross-Correlation Analysis

### 3.1 Gravity & Leveling Vector at Rest (t = 50.0 - 100.0 s)
During stationary vehicle intervals, the raw specific force vector is:
f_accel = [0.051 +- 0.063, 0.059 +- 0.046, 9.847 +- 0.048]^T m/s^2
* Norm: ||f|| = 9.847 m/s^2 (~1.004 g).
* Direction: Almost purely along sensor +Z (f_z / ||f|| = 0.9999).
* **Conclusion:** The accelerometer +Z axis points vertically UPward opposite to gravity.

### 3.2 Dynamic Turn & Yaw Rate Cross-Correlation
To determine which gyroscope channel measures yaw (rotation about the local vertical), cross-correlations against the VBOX ground-truth yaw rate r_vbox were evaluated across 42,641 samples (t > 156 s, moving speed > 2 m/s):

| Sensor Channel | Header Name | Correlation (r) with VBOX Yaw Rate | Regression Slope (k) |
| :--- | :--- | :---: | :---: |
| phone_gyro_x | GYROSCOPE Yaw (rad/s) | +0.041 (Uncorrelated) | +0.047 |
| phone_gyro_y | GYROSCOPE Pitch (rad/s) | **+0.932 (Dominant Yaw)** | **+0.978** |
| phone_gyro_z | GYROSCOPE Roll (rad/s) | -0.227 (Weak / Tilt) | -0.252 |

#### Robustness Across Multiple Drive Segments:
The dominance of phone_gyro_y is consistent across the entire 86-minute drive:
* t in [0, 60 s]: r(gy, r_vbox) = **+0.985** (slope: 1.002)
* t in [140, 200 s] (includes Fix 1-4): r(gy, r_vbox) = **+0.983** (slope: 0.988)
* t in [500, 560 s]: r(gy, r_vbox) = **+0.984** (slope: 0.993)
* t in [1000, 1100 s]: r(gy, r_vbox) = **+0.974** (slope: 0.981)
* t in [2000, 2100 s]: r(gy, r_vbox) = **+0.957** (slope: 0.984)
* t in [4000, 4100 s]: r(gy, r_vbox) = **+0.975** (slope: 0.979)

### 3.3 Physical Mechanism of the Fix 4 Failure in E1
Because the gyro yaw rate was on channel Y, feeding raw [gx, gy, gz] into strapdown mechanization meant that when the vehicle turned:
1. The 52 deg azimuth turn between t = 174 s and 183 s was integrated into the **pitch angle** instead of heading.
2. The nominal attitude pitch tilted violently to -36.2 deg and then -68 deg.
3. This projected g * sin(-68 deg) ~ -9.1 m/s^2 into horizontal forward acceleration, generating massive false velocity and causing Fix 4 to be rejected (NIS = 159.4).

---

## 4. Ingest Assumptions vs Observed Reality

| Component | Current Code Assumption (src/navris/) | Observed Physical Reality (S-S1.csv) | Impact on Filter |
| :--- | :--- | :--- | :--- |
| **GNSS Speed Units** | KMH_TO_MPS (1/3.6) applied to GPS SPEED (Kmh) | Column is already in m/s | 3.6x velocity underestimation (3.32 m/s vs 12.03 m/s) |
| **Accel Vertical Axis** | +Z is UP (f_z ~ +9.81) | +Z is UP (f_z = +9.85 m/s^2) | Accelerometer leveling is physically consistent |
| **Gyro Yaw Axis** | Gyro +Z is vertical yaw rotation | Gyro +Y is vertical yaw rotation (r = 0.932, k = 0.978) | Yaw integrated as pitch; fatal attitude tilt during turns |
| **Gyro Horizontal Axes** | Gyro X = roll, Y = pitch | Gyro X = pitch/roll, Z = -pitch/roll | Horizontal body rates crossed |
| **Vehicle Forward Axis** | Body +X assumed vehicle forward in init.py | Vehicle forward is angled ~ -45 deg in X-Y plane | Initial velocity direction misaligned with phone chassis |
| **Heading Alignment** | Course-over-ground aligns body +X | Body +X is not aligned with vehicle longitudinal axis | Heading initialization has fixed extrinsic mount offset |

---

## 5. Controlled Replay Setup (Diagnostic Arm E2)

> [!WARNING]
> **CLASSIFICATION: DIAGNOSTIC ORACLE — NOT DEPLOYABLE**  
> The corrections applied in Experiment E2 use extrinsic frame knowledge discovered via offline correlation against ground truth. They serve exclusively to diagnose whether the ESKF core functions correctly when fed geometrically and physically consistent sensor inputs.

### 5.1 Replay Corrections Applied
1. **Gyroscope Frame Remapping to Accelerometer Frame:**
   The gyroscope triaxial vector was transformed into the accelerometer coordinate system:
   \boldsymbol{\omega}_{\text{body}} = \begin{bmatrix} \omega_x \\ \omega_y \\ \omega_z \end{bmatrix} = \begin{bmatrix} \text{gyro}_x \\ -\text{gyro}_z \\ \text{gyro}_y \end{bmatrix}
   * Body +Z rotation rate is now driven by gyro_y (the true vehicle yaw axis).
2. **Causal Speed Correction:**
   Initial speed at t = 156.0 s initialized using raw unscaled GPS speed (11.94 m/s) oriented along the causal displacement course-over-ground (316.48 deg).
3. **Phase 2.3A ESKF Core:**
   **100% UNCHANGED AND FROZEN.** Discretization, Van Loan Q_d, Joseph-form covariance update, nominal RK4 propagation, and reset Jacobians remained identical to Phase 2.3A.

---

## 6. Baseline vs E1 vs E2 Replay Comparison

### Master Performance Summary

| Metric | Phase 2.3B Baseline | Experiment E1-A | Experiment E1-C | Experiment E2 Replay (ORACLE) |
| :--- | :---: | :---: | :---: | :---: |
| **Initial Velocity ($)** | [-2.28, 2.40, 0.0] | [-9.41, 10.30, 0.0] | [-9.41, 10.30, 0.0] | [-8.22, 8.66, 0.0] |
| **Initial Speed ($ norm)** | 3.32 m/s | 13.95 m/s | 13.95 m/s | **11.94 m/s** |
| **Fix 1 ( = 165\text{ s}$) NIS** | 4.61 | 2.27 | **1.04** | **1.97** |
| **Fix 1 Pitch Tilt ($\Delta \theta_{\text{pitch}}$)**| +12.37 deg (Fatal) | -0.22 deg | -0.10 deg | **+3.68 deg** |
| **Fix 2 ( = 174\text{ s}$) Status** | REJECTED (NIS=31.8) | ACCEPTED (NIS=7.56) | ACCEPTED (NIS=8.06) | **ACCEPTED (NIS=1.37)** |
| **Fix 2 Horiz Error** | 316.07 m | 5.83 m | 5.93 m | **6.32 m** |
| **Fix 3 ( = 183\text{ s}$) Status** | REJECTED (NIS=60.1) | ACCEPTED (NIS=6.57) | ACCEPTED (NIS=9.21) | **ACCEPTED (NIS=3.19)** |
| **Fix 3 Horiz Error** | 1,180.03 m | 2.73 m | 2.55 m | **3.46 m** |
| **Fix 4 ( = 192\text{ s}$) Status** | REJECTED (NIS=50.4) | REJECTED (NIS=159.4) | REJECTED (NIS=153.7) | **ACCEPTED (NIS=1.54)** |
| **Fix 4 Horiz Error** | 2,376.12 m | 452.19 m | 437.81 m | **3.56 m** |
| **Fix 5 ( = 205\text{ s}$) Status** | REJECTED | REJECTED | REJECTED | **ACCEPTED (NIS=1.88)** |
| **Fixes Accepted ( \in [156, 624\text{ s}]$)**| 1 / 51 (2.0%) | 3 / 51 (5.9%) | 3 / 51 (5.9%) | **51 / 51 (100.0%)** |
| **First 60s Horiz RMSE** | 2,767.52 m | 1,699.71 m | 1,686.86 m | **48.33 m (-98.3%)** |
| **Time to 50 m Error** | 4.8 s | 8.4 s | 8.4 s | **21.5 s (+348%)** |

---

## 7. Analysis of Fix 4 and Extended Tracking

### 7.1 Detailed Fix 4 Inventory ( = 192.0\text{ s}$, Epoch 360)
* **Pre-Update Filter State:**
  * Velocity prior: v_pre = [-2.68, -13.61, -2.44] m/s (speed: 14.08 m/s)
  * Nominal Pitch: -0.94 deg (remains completely level; no unphysical tilt!)
  * Position 3sigma bound: 84.38 m
* **Measurement Innovation:**
  * Residual ENU: [-4.22, 76.83, 11.38] m
  * Innovation Covariance S diagonal: [2732.1, 2731.8, 1269.8]
  * **Normalized Innovation Squared (NIS):**
    \text{NIS} = \mathbf{r}^T \mathbf{S}^{-1} \mathbf{r} = \mathbf{1.5396} \ll 16.27 \quad (\chi^2_{3, 0.999})
* **Post-Update Correction:**
  * Pitch adjustment: Delta theta_pitch = -1.41 deg (post-pitch: -2.35 deg)
  * **Absolute Horizontal Error vs Reference:** **3.56 m**

### 7.2 Extended Tracking Behavior ( = 156.0 - 624.0\text{ s}$)
In Experiment E2, the filter tracks continuously through the entire initial urban maneuvering segment:
* **Fixes 0 through 50 (51 consecutive fixes over 7.8 minutes) are all ACCEPTED.**
* Average NIS across these 51 fixes is 5.12 (well below the 16.27 gate).
* Horizontal position error at every fix remains between **2.5 m and 8.5 m**.
* Even during complete stops and sharp turns, nominal pitch remains constrained within [-15 deg, +10 deg], completely preventing the runaway gravity misprojection that crippled Baseline.

---

## 8. Remaining Uncertainties & Scientific Caveats

While Experiment E2 conclusively demonstrates that the ESKF core is sound and capable of tracking real vehicle motion when inputs are correct, critical scientific uncertainties remain:

1. **Mounting Orientation Generality Across Files:**
   * In S1, the gyro yaw is unambiguously on channel Y. However, was the phone mounted in the same portrait/landscape orientation in recordings S2 through S8?
   * If the smartphone mount varied between runs (e.g., driver pocket vs dashboard cradle vs windshield mount), fixed axis swapping will fail on other drives.
2. **Extrinsic Yaw Offset (315 deg):**
   * The phone's chassis was angled at approximately -45 deg relative to the vehicle longitudinal axis. A production system cannot assume a hardcoded mounting angle without an automated extrinsic calibration routine.
3. **Lack of Kinematic / Non-Holonomic Constraints (NHC):**
   * Even in E2, dead reckoning drift during prolonged 9-second GNSS outages accumulates 50-80 m of open-loop error during hard braking maneuvers. While the position update pulls the state back, a vehicle-specific NHC constraint (v_lateral ~ 0) or GNSS Doppler velocity update is required for long-term production stability.

---

## 9. Recommendations for Next Scientific Gate

Before authorizing batch processing across recordings S2–S8 or advancing to Phase 2.4/Phase 3, the following steps must be formally executed:

1. **Data Ingestion Fix (src/navris/ingest.py):**
   * Correct line 282 to ingest df['GPS SPEED (Kmh)'] directly without dividing by 3.6, or implement an automated unit validation check comparing raw GNSS speed to causal position displacement Delta p / Delta t.
2. **Automated Extrinsic Calibration & Axis Alignment (src/navris/inertial/align.py):**
   * Implement an automated causal leveling routine: derive vertical axis directly from the stationary gravity vector g_hat = f_bar / ||f_bar||.
   * Implement causal turn-rate axis identification: correlate gyro channels against the course-over-ground yaw rate during initial straight/turning motion to detect which gyro axis represents vertical yaw without ground truth.
3. **Formal Multi-Recording Raw Data Audit:**
   * Audit raw CSV headers and stationary gravity vectors across S2 through S8 to confirm whether the mount frame is invariant across all recordings before running batch pipelines.

---

### Strict Phase Rule Adherence
* **ESKF Core:** src/navris/eskf/ remains 100% frozen.
* **Scope Enforced:** No AI/ML, NHC, ZUPT, map matching, or batch runs were performed.
* **Stop Enforced:** Experiment E2 is complete.
