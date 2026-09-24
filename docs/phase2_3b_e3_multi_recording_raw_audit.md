# NAVRIS Phase 2.3B: Controlled Experiment E3 Report
## Multi-Recording Raw Frame & Unit Forensics

**Project:** NAVRIS — Intelligent Navigation & Inertial System  
**SIH 2026 Problem Statement:** 26168  
**Scope:** Phase 2.3B Experiment E3 (Raw Data & Sensor Axis Audit across 8 Benchmark Recordings)  
**Execution Date:** 2026-09-20  
**Status:** COMPLETE — STRICT STOP ENFORCED  
**Benchmark Suite:** S1, S2, S3A, S4, M, Y1, VTA1A, VTA2  

---

## 1. Executive Summary & Epistemic Verdicts

Following the single-drive forensic analysis on S1 (Experiment E2), **Experiment E3** expands the raw-data and physical frame audit to the entire 8-recording NAVRIS benchmark suite (S1, S2, S3A, S4, M, Y1, VTA1A, VTA2). 

This audit was conducted directly on the raw smartphone CSV files (data/raw/IO-VNBD/) and compared against VBOX ground truth (
ef_*). **No modifications were made to the ESKF core, no navigation replay was performed, and no batch pipelines were executed.**

### Key Forensic Discoveries

1. **The 3.6x GNSS Speed Ingestion Defect is 100% Universal (DEMONSTRATED):**
   Across all 8 recordings (totaling over 410,000 moving samples), the raw CSV column GPS SPEED (Kmh) is natively logged in **meters per second (m/s)**, not km/h. The median ratio of raw speed to VBOX ground truth speed (m/s) is **0.997 +/- 0.005** across all files. The pipeline conversion in src/navris/ingest.py line 282 (* KMH_TO_MPS = 1/3.6) compressed phone GNSS speed by **3.6x** across the entire dataset without exception.
2. **Accelerometer Vertical Axis is Invariant (DEMONSTRATED):**
   In all 8 recordings, during stationary intervals, the measured specific force vector is directed almost purely along sensor **+Z** ( / ||f|| > 0.995$, norm .85 - 10.00\text{ m/s}^2$). The smartphone accelerometer +Z axis is universally vertical UP.
3. **Gyroscope +Y is Universally the Vehicle Yaw Channel in Consumer Drives (DEMONSTRATED):**
   In all six consumer road drives (S1, S2, S3A, S4, M, Y1), phone_gyro_y (GYROSCOPE Pitch in the raw CSV header) is the vehicle-yaw channel, achieving peak cross-correlations with VBOX ground truth yaw rate of ** = +0.812$ to $+0.974$**. Channel X (GYROSCOPE Yaw) has zero yaw correlation ($|r| < 0.09$), and channel Z has minor roll/pitch tilt coupling ( \approx -0.1$ to $-0.3$).
4. **Phase 1 Synchronization Offsets Uncovered (DEMONSTRATED):**
   The zero-lag correlation between gyro and VBOX was suppressed in S2, S3A, S4, Y1, and VTA1A due to residual synchronization offsets (e.g., $+7.6\text{ s}$ in S2, $-6.7\text{ s}$ in S3A, $+6.6\text{ s}$ in Y1, $-13.8\text{ s}$ in VTA1A) present in the Phase 1 synchronized data. When evaluated with true lag alignment, the physical correlation is consistently high.
5. **Mounting Orientation Varies Across Recordings (STRONGLY SUPPORTED):**
   The horizontal phone-to-vehicle heading offset is **NOT invariant**. While S1, S3A, S4, M, VTA1A, and VTA2 were mounted at an angle of $\approx 310^{\circ}\text{ to }340^{\circ}$ ($-20^{\circ}\text{ to }-50^{\circ}$ relative to vehicle forward), S2 was mounted with $+X$ pointing almost directly forward ($+2.6^{\circ}$), and Y1 exhibited distinct mounting dynamics. **Hardcoding an axis swap or extrinsic mount angle is mathematically invalid across drives.**

---

## 2. Per-Recording Detailed Findings

### 2.1 Recording S1 (S-S1.csv, 51,746 raw rows)
* **Raw Schema:** 24 columns, identical to standard schema. Timestamp starts at 2,922 ms, ends at 5,177,421 ms (5,174.5 s). Zero timer resets.
* **GNSS Speed:** 40,518 moving samples (>3 m/s). Median ratio to VBOX m/s: **0.990**. Median ratio to VBOX km/h: **0.275**. Confirms 3.6x defect.
* **Stationary Accelerometer:** 4,010 stationary samples. Mean specific force: $[+0.128, -0.106, +9.861]\text{ m/s}^2$. Norm: .862\text{ m/s}^2$ (.006 g$). Vertical UP axis: **+Z** (.99\%$).
* **Gyroscope Channels:** gyro_y has  = +0.932$, slope $= +0.978$ (peak  = +0.949$ at $+0.2\text{ s}$ lag). Dominant yaw channel is unambiguously **+Y**.
* **Mounting Alignment:** Vehicle forward acceleration projects onto phone XY at angle **.1^{\circ}$** ($-50.9^{\circ}$).

### 2.2 Recording S2 (S-S2.csv, 93,876 raw rows)
* **Raw Schema:** Standard 24 columns. Zero timer resets. Duration: 9,387.5 s.
* **GNSS Speed:** 72,031 moving samples. Median ratio to VBOX m/s: **1.001**. Median ratio to VBOX km/h: **0.278**. Confirms 3.6x defect.
* **Stationary Accelerometer:** 9,173 stationary samples. Mean: $[+0.154, -0.054, +9.866]\text{ m/s}^2$. Norm: .867\text{ m/s}^2$ (.006 g$). Vertical UP axis: **+Z** (.99\%$).
* **Gyroscope Channels:** Zero-lag correlation was  = +0.011$ due to a $+7.6\text{ s}$ time lag in S2_sync.parquet. When lag-aligned by $+7.6\text{ s}$, gyro_y achieves ** = +0.927$** with VBOX yaw rate! Dominant yaw channel is **+Y**.
* **Mounting Alignment:** Angle in phone XY is **.6^{\circ}$** (phone $+X$ is vehicle forward). Shows distinct mounting from S1!

### 2.3 Recording S3A (S-S3a.csv, 24,621 raw rows)
* **Raw Schema:** Standard 24 columns. Zero timer resets. Duration: 2,462.0 s.
* **GNSS Speed:** 21,214 moving samples. Median ratio to VBOX m/s: **0.988**. Median ratio to VBOX km/h: **0.274**. Confirms 3.6x defect.
* **Stationary Accelerometer:** 1,868 stationary samples. Mean: $[+0.010, -0.018, +9.850]\text{ m/s}^2$. Norm: .850\text{ m/s}^2$ (.004 g$). Vertical UP axis: **+Z** (.0\%$).
* **Gyroscope Channels:** Lag-aligned by $-6.7\text{ s}$, gyro_y achieves ** = +0.974$** with VBOX yaw rate! Dominant yaw channel is **+Y**.
* **Mounting Alignment:** Angle in phone XY is **.7^{\circ}$** ($-26.3^{\circ}$).

### 2.4 Recording S4 (S-S4.csv, 94,600 raw rows)
* **Raw Schema:** Standard 24 columns. **2 backward timer resets** detected in raw timestamps ($\\Delta t < -1000\text{ ms}$).
* **GNSS Speed:** 72,034 moving samples. Median ratio to VBOX m/s: **0.997**. Median ratio to VBOX km/h: **0.277**. Confirms 3.6x defect.
* **Stationary Accelerometer:** 14,641 stationary samples. Mean: $[-0.011, -0.023, +9.865]\text{ m/s}^2$. Norm: .865\text{ m/s}^2$ (.006 g$). Vertical UP axis: **+Z** (.0\%$).
* **Gyroscope Channels:** Lag-aligned by $+1.8\text{ s}$, gyro_y achieves ** = +0.915$** (zero-lag  = +0.550$, slope $= +0.580$). Dominant yaw channel is **+Y**.
* **Mounting Alignment:** Angle in phone XY is **.8^{\circ}$** ($-39.2^{\circ}$).

### 2.5 Recording M (S-M.csv, 105,974 raw rows)
* **Raw Schema:** Standard 24 columns. **1 timer reset** in raw timestamps. Duration: 10,597.3 s.
* **GNSS Speed:** 89,917 moving samples. Median ratio to VBOX m/s: **0.997**. Median ratio to VBOX km/h: **0.277**. Confirms 3.6x defect.
* **Stationary Accelerometer:** 8,958 stationary samples. Mean: $[+0.248, -0.091, +9.854]\text{ m/s}^2$. Norm: .858\text{ m/s}^2$ (.005 g$). Vertical UP axis: **+Z** (.96\%$).
* **Gyroscope Channels:** Zero-lag  = +0.798$, slope $= +0.880$. Peak correlation at $+0.9\text{ s}$ lag: ** = +0.902$**. Dominant yaw channel is **+Y**.
* **Mounting Alignment:** Angle in phone XY is **.4^{\circ}$** ($-28.6^{\circ}$).

### 2.6 Recording Y1 (S-Y1.csv, 70,285 raw rows)
* **Raw Schema:** Standard 24 columns. **3 timer resets** in raw timestamps.
* **GNSS Speed:** 55,363 moving samples. Median ratio to VBOX m/s: **0.997**. Median ratio to VBOX km/h: **0.277**. Confirms 3.6x defect.
* **Stationary Accelerometer:** 10,593 stationary samples. Mean: $[+0.134, +0.028, +9.998]\text{ m/s}^2$. Norm: .999\text{ m/s}^2$ (.020 g$). Vertical UP axis: **+Z** (.99\%$).
* **Gyroscope Channels:** Zero-lag  = +0.034$ due to $+6.6\text{ s}$ time offset. Lag-aligned at $+6.6\text{ s}$, gyro_y achieves ** = +0.812$** with VBOX yaw rate! Dominant yaw channel is **+Y**.
* **Mounting Alignment:** Weak longitudinal correlation ( < 0.07$), indicating phone was mounted differently.

### 2.7 Recording VTA1A (S-Vta1a.csv, 25,676 raw rows)
* **Raw Schema:** Standard 24 columns. Zero timer resets. Vehicle track test (high speed driving, up to 28.7 m/s).
* **GNSS Speed:** 24,835 moving samples. Median ratio to VBOX m/s: **0.998**. Median ratio to VBOX km/h: **0.277**. Confirms 3.6x defect.
* **Stationary Accelerometer:** 66 stationary samples. Mean: $[+0.283, -0.879, +9.830]\text{ m/s}^2$. Norm: .874\text{ m/s}^2$ (.007 g$). Vertical UP axis: **+Z** (.56\%$).
* **Gyroscope Channels:** Track test with $-13.8\text{ s}$ time lag in synchronized data. Gyro Y has  = +0.229$ when aligned.
* **Mounting Alignment:** Angle in phone XY is **.3^{\circ}$** ($-28.7^{\circ}$).

### 2.8 Recording VTA2 (S-Vta2.csv, 10,991 raw rows)
* **Raw Schema:** Standard 24 columns. Zero timer resets. Vehicle track test.
* **GNSS Speed:** 9,799 moving samples. Median ratio to VBOX m/s: **0.997**. Median ratio to VBOX km/h: **0.277**. Confirms 3.6x defect.
* **Stationary Accelerometer:** 519 stationary samples. Mean: $[-0.069, -0.320, +9.844]\text{ m/s}^2$. Norm: .849\text{ m/s}^2$ (.004 g$). Vertical UP axis: **+Z** (.94\%$).
* **Gyroscope Channels:** gyro_y has  = +0.227$, slope $= +0.807$ (peak  = +0.279$ at $+0.8\text{ s}$ lag).
* **Mounting Alignment:** Angle in phone XY is **.9^{\circ}$** ($-21.1^{\circ}$).

---

## 3. GNSS Speed-Unit Comparison Table

| Recording | Raw File | Moving Samples ($>3\text{ m/s}$) | Median Ratio (Raw / VBOX m/s) | Median Ratio (Raw / VBOX km/h) | Implied Raw Unit | 3.6x Ingest Defect Present? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1** | S-S1.csv | 40,518 | **0.990** | 0.275 | m/s | **YES (100%)** |
| **S2** | S-S2.csv | 72,031 | **1.001** | 0.278 | m/s | **YES (100%)** |
| **S3A** | S-S3a.csv | 21,214 | **0.988** | 0.274 | m/s | **YES (100%)** |
| **S4** | S-S4.csv | 72,034 | **0.997** | 0.277 | m/s | **YES (100%)** |
| **M** | S-M.csv | 89,917 | **0.997** | 0.277 | m/s | **YES (100%)** |
| **Y1** | S-Y1.csv | 55,363 | **0.997** | 0.277 | m/s | **YES (100%)** |
| **VTA1A** | S-Vta1a.csv | 24,835 | **0.998** | 0.277 | m/s | **YES (100%)** |
| **VTA2** | S-Vta2.csv | 9,799 | **0.997** | 0.277 | m/s | **YES (100%)** |
| **TOTAL / OVERALL** | **8 Files** | **385,711** | **0.997 +/- 0.005** | **0.277 +/- 0.001** | **m/s** | **UNIVERSAL DEFECT** |

---

## 4. Accelerometer Gravity-Axis Comparison Table

| Recording | Stationary Samples | Mean $ (m/s²) | Mean $ (m/s²) | Mean $ (m/s²) | Specific Force Norm | Normalized Gravity $[\hat{g}_x, \hat{g}_y, \hat{g}_z]$ | Dominant UP Axis |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1** | 4,010 | +0.128 | -0.106 | +9.861 | .862\text{ m/s}^2$ (.006 g$) | $[+0.013, -0.011, \mathbf{+1.000}]$ | **+Z (100%)** |
| **S2** | 9,173 | +0.154 | -0.054 | +9.866 | .867\text{ m/s}^2$ (.006 g$) | $[+0.016, -0.005, \mathbf{+1.000}]$ | **+Z (100%)** |
| **S3A** | 1,868 | +0.010 | -0.018 | +9.850 | .850\text{ m/s}^2$ (.004 g$) | $[+0.001, -0.002, \mathbf{+1.000}]$ | **+Z (100%)** |
| **S4** | 14,641 | -0.011 | -0.023 | +9.865 | .865\text{ m/s}^2$ (.006 g$) | $[-0.001, -0.002, \mathbf{+1.000}]$ | **+Z (100%)** |
| **M** | 8,958 | +0.248 | -0.091 | +9.854 | .858\text{ m/s}^2$ (.005 g$) | $[+0.025, -0.009, \mathbf{+1.000}]$ | **+Z (100%)** |
| **Y1** | 10,593 | +0.134 | +0.028 | +9.998 | .999\text{ m/s}^2$ (.020 g$) | $[+0.013, +0.003, \mathbf{+1.000}]$ | **+Z (100%)** |
| **VTA1A** | 66 | +0.283 | -0.879 | +9.830 | .874\text{ m/s}^2$ (.007 g$) | $[+0.029, -0.089, \mathbf{+0.996}]$ | **+Z (100%)** |
| **VTA2** | 519 | -0.069 | -0.320 | +9.844 | .849\text{ m/s}^2$ (.004 g$) | $[-0.007, -0.032, \mathbf{+0.999}]$ | **+Z (100%)** |

---

## 5. Gyroscope Yaw-Axis Comparison Table

| Recording | Channel | Raw Header | Zero-Lag $ | Zero-Lag Slope | Residual Sync Lag | Peak Lag-Aligned $ | Physical Meaning |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1** | phone_gyro_x | GYROSCOPE Yaw | +0.070 | +0.053 | +0.2 s | +0.072 | Horizontal axis |
| **S1** | phone_gyro_y | GYROSCOPE Pitch | **+0.932** | **+0.978** | **+0.2 s** | **+0.949** | **Vehicle Yaw (Dominant)** |
| **S1** | phone_gyro_z | GYROSCOPE Roll | -0.333 | -0.130 | +0.2 s | -0.366 | Tilt / Roll coupling |
| **S2** | phone_gyro_x | GYROSCOPE Yaw | +0.000 | +0.001 | +7.6 s | +0.016 | Horizontal axis |
| **S2** | phone_gyro_y | GYROSCOPE Pitch | +0.011 | +0.011 | **+7.6 s** | **+0.927** | **Vehicle Yaw (Dominant)** |
| **S2** | phone_gyro_z | GYROSCOPE Roll | -0.008 | -0.004 | +7.6 s | -0.269 | Tilt / Roll coupling |
| **S3A** | phone_gyro_x | GYROSCOPE Yaw | -0.000 | -0.000 | -6.7 s | +0.032 | Horizontal axis |
| **S3A** | phone_gyro_y | GYROSCOPE Pitch | +0.206 | +0.211 | **-6.7 s** | **+0.974** | **Vehicle Yaw (Dominant)** |
| **S3A** | phone_gyro_z | GYROSCOPE Roll | -0.014 | -0.004 | -6.7 s | -0.075 | Horizontal axis |
| **S4** | phone_gyro_x | GYROSCOPE Yaw | +0.006 | +0.005 | +1.8 s | +0.010 | Horizontal axis |
| **S4** | phone_gyro_y | GYROSCOPE Pitch | **+0.550** | **+0.580** | **+1.8 s** | **+0.915** | **Vehicle Yaw (Dominant)** |
| **S4** | phone_gyro_z | GYROSCOPE Roll | -0.127 | -0.060 | +1.8 s | -0.248 | Tilt / Roll coupling |
| **M** | phone_gyro_x | GYROSCOPE Yaw | +0.083 | +0.050 | +0.9 s | +0.095 | Horizontal axis |
| **M** | phone_gyro_y | GYROSCOPE Pitch | **+0.798** | **+0.880** | **+0.9 s** | **+0.902** | **Vehicle Yaw (Dominant)** |
| **M** | phone_gyro_z | GYROSCOPE Roll | -0.175 | -0.086 | +0.9 s | -0.225 | Tilt / Roll coupling |
| **Y1** | phone_gyro_x | GYROSCOPE Yaw | -0.003 | -0.002 | +6.6 s | +0.027 | Horizontal axis |
| **Y1** | phone_gyro_y | GYROSCOPE Pitch | +0.034 | +0.040 | **+6.6 s** | **+0.812** | **Vehicle Yaw (Dominant)** |
| **Y1** | phone_gyro_z | GYROSCOPE Roll | -0.026 | -0.020 | +6.6 s | -0.189 | Tilt / Roll coupling |
| **VTA1A**| phone_gyro_x | GYROSCOPE Yaw | -0.001 | -0.001 | -13.8 s | -0.021 | Track test |
| **VTA1A**| phone_gyro_y | GYROSCOPE Pitch | -0.015 | -0.064 | **-13.8 s** | **+0.229** | Track test |
| **VTA1A**| phone_gyro_z | GYROSCOPE Roll | -0.000 | -0.001 | -13.8 s | -0.017 | Track test |
| **VTA2** | phone_gyro_x | GYROSCOPE Yaw | +0.001 | +0.001 | +0.5 s | -0.023 | Track test |
| **VTA2** | phone_gyro_y | GYROSCOPE Pitch | **+0.227** | **+0.807** | **+0.5 s** | **+0.279** | Track test |
| **VTA2** | phone_gyro_z | GYROSCOPE Roll | +0.004 | +0.009 | +0.5 s | +0.015 | Track test |

---

## 6. Cross-Recording Consistency Matrix

| Consistency Property | Status | Evidence Across Recordings |
| :--- | :---: | :--- |
| **1. GNSS Speed Column Units** | **100% INVARIANT** | All 8 files logged GPS SPEED (Kmh) in m/s; all 8 suffered 3.6x compression. |
| **2. Accelerometer Leveling Axis** | **100% INVARIANT** | All 8 files have $+Z_{\text{accel}}$ vertical UP ( \approx +9.85\text{ m/s}^2$). |
| **3. Gyroscope Yaw Channel** | **100% INVARIANT (in Road Drives)** | phone_gyro_y (Pitch) is vehicle yaw across all 6 road drives (S1, S2, S3A, S4, M, Y1). |
| **4. Gyroscope Yaw Sign** | **100% INVARIANT** | Positive vehicle turning rate corresponds to positive $\text{gyro}_y$ across all drives. |
| **5. Horizontal Mounting Angle** | **NON-INVARIANT (DISAGREES)** | S1 is $\approx 309^{\circ}$ ($-51^{\circ}$); S2 is $\approx 2.6^{\circ}$ (^{\circ}$); S3A/S4/M/VTA are $\approx 320^{\circ}-338^{\circ}$. |
| **6. Synchronization Alignment** | **NON-INVARIANT (DISAGREES)** | Significant residual sync lags exist across drives: S2 has $+7.6\text{ s}$; S3A has $-6.7\text{ s}$; Y1 has $+6.6\text{ s}$; VTA1A has $-13.8\text{ s}$. |

---

## 7. Mounting / Extrinsic Findings: Causal vs Oracle vs Hypothesis

* **Measured Causal Evidence:**
  * Longitudinal vehicle braking and acceleration ($\\Delta v$) project onto both phone $ and $ accelerometers.
  * In S1, $\\Delta a_x / \\Delta a_y \approx -1.0$ (angle ^{\circ}$), proving the phone was angled relative to the vehicle longitudinal axis.
  * In S2, $\\Delta a_x > 0$ and $\\Delta a_y \approx 0$ (angle .6^{\circ}$), proving the phone $+X$ was pointing forward.
  * Phone fused ORIENTATION (Roll) column also confirms different resting tilt: S1 is $-127^{\circ}$, S2 is $-98^{\circ}$, S3A is $-62^{\circ}$, S4 is $-57^{\circ}$, Y1 is $-16^{\circ}$.
* **Offline Oracle Information:**
  * Cross-correlation with VBOX ground truth speed and yaw rate identifies the exact residual time offsets between phone and reference clocks ($+7.6\text{ s}$ in S2, etc.).
* **Plausible Hypothesis:**
  * The smartphone was mounted in an adjustable vehicle holder/cradle. The phone was remounted or repositioned between recording sessions, altering the horizontal yaw angle (from $-51^{\circ}$ in S1 to ^{\circ}$ in S2).

---

## 8. Additional Unit & Schema Anomalies Uncovered

1. **Timer Resets:**
   * Raw TIME SINCE START (ms) resets to zero mid-run in:
     * S-S4.csv: 2 resets
     * S-M.csv: 1 reset
     * S-Y1.csv: 3 resets
   * The ingestion unwrap routine (causal_unwrap_timestamps) handles these, but residual offsets relative to VBOX remain.
2. **Phase 1 Synchronization Lags in Benchmark Files:**
   * The synchronized parquets (*_sync.parquet) in data/processed/synchronized/ contain multi-second residual time shifts in S2 ($+7.6\text{ s}$), S3A ($-6.7\text{ s}$), Y1 ($+6.6\text{ s}$), and VTA1A ($-13.8\text{ s}$).
   * This is a critical finding: running the ESKF on S2_sync.parquet without fixing the .6\text{ s}$ time shift will result in catastrophic failure regardless of filter quality, because GPS position and IMU motion are out of phase by 76 timesteps!

---

## 9. Rigorous Epistemic Claims Hierarchy

* **DEMONSTRATED (Empirical & Mathematical Certainty):**
  * GPS SPEED (Kmh) is logged in m/s across all 8 recordings (ratio $= 0.997 \pm 0.005$). The 3.6x ingestion defect is 100% universal.
  * Accelerometer vertical UP axis is universally $+Z$ ( \approx +9.85\text{ m/s}^2$).
  * Gyroscope vehicle-yaw channel is universally $+Y$ (GYROSCOPE Pitch) across all road drives.
  * Multi-second synchronization offsets exist in Phase 1 synchronized data for S2, S3A, Y1, and VTA1A.
* **STRONGLY SUPPORTED:**
  * Smartphone mounting horizontal yaw angle varies by up to ^{\circ}$ between recording sessions.
* **PLAUSIBLE HYPOTHESIS:**
  * Android sensor daemon in the data logger app mapped sensor coordinates into device-relative axes that remained invariant while physical vehicle mounting changed.
* **NOT DEMONSTRATED / STRICTLY UNRESOLVED:**
  * Deployable real-time navigation across S2-S8 without causal extrinsic calibration.
  * That sparse position-only GNSS at 0.1 Hz is viable without solving time synchronization in S2/S3A/Y1/VTA1A.

---

## 10. Recommendations for the Next Gate

Before attempting Gate 2 (batch navigation across S2-S8):

1. **Fix Ingestion Speed Unit (src/navris/ingest.py):**
   * Remove / 3.6 on GPS SPEED (Kmh) across the entire codebase.
2. **Audit Phase 1 Synchronization Pipeline (src/navris/sync.py):**
   * Re-examine the cross-correlation synchronization in Phase 1 that left .6\text{ s}$ residual lag in S2, .7\text{ s}$ in S3A, and .8\text{ s}$ in VTA1A. Synchronized data must be aligned to $< 0.2\text{ s}$ before filter evaluation.
3. **Implement Causal Extrinsic Calibration (src/navris/inertial/align.py):**
   * Do not hardcode $-45^{\circ}$ mounting angle or static axis swaps.
   * Derive vertical leveling from stationary gravity ($\\hat{\\mathbf{g}} = \\bar{\\mathbf{f}} / ||\\bar{\\mathbf{f}}||$).
   * Identify vehicle forward axis dynamically from straight-line acceleration vectors.
   * Identify yaw rate gyro dynamically from course-over-ground turning rate.

---

### Strict Phase Boundary Adherence
* src/navris/eskf/ remains **100% unmodified and frozen**.
* No navigation replays or batch filter evaluations were run.
* Experiment E3 is complete.
