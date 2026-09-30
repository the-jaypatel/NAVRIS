# NAVRIS Phase 3.0: Metric & Target Construction Forensic Audit

**Date**: September 29, 2026  
**Status**: COMPLETE — ALL PHASES AUDITED  
**Mandate**: Independent verification of Phase 3.0 target construction, coordinate frames, timestamp alignment, metric computations, and physical sanity.

---

## Executive Summary

This forensic audit was commissioned to determine whether the massive velocity RMSE values reported in Phase 3.0 (e.g., 7,051 m/s on S3A, 29,246 m/s on Y1) versus the small velocity RMSE on VTA2 (5.63 m/s) reflect:
1. An implementation bug, unit mismatch, frame inconsistency, or timestamp misalignment, OR
2. Genuine physical and algorithmic behavior of the underlying classical navigation solutions.

### Core Verdict
- **Target Construction**: **100% CORRECT & MATHEMATICALLY CONSISTENT**. Units are m/s, frame is Cartesian ENU, heading is clockwise from True North, and vertical speed is positive Up.
- **Coordinate & Frame Alignment**: **100% CONSISTENT**. There is zero axis transposition or sign inversion between VBOX Doppler reference and NAVRIS ESKF states.
- **Timestamp Alignment**: **EXACT**. Across all 363,486 synchronized samples across 7 routes, the maximum timestamp discrepancy is strictly $\Delta t = 0.00 \times 10^0$ s.
- **Source of Velocity Divergence**: **A. GENUINE CLASSICAL FILTER DIVERGENCE DURING GNSS OUTAGES**. On S1, S2, S3A, S4, VTA1A, and Y1, the classical filter suffered prolonged GNSS fix rejection (acceptance rates between 0.2% and 11.7%), causing unconstrained dead-reckoning integration of consumer smartphone accelerometer tilt error and bias over 0.5 to 2.6 hours. On VTA2, GNSS fix acceptance was **96.7%**, keeping classical velocity tightly bounded (5.63 m/s 3D RMSE).
- **ML Failure Validity**: **FULLY VALID & SCIENTIFICALLY SOUND**. The 96.68% feature importance on classical velocity states and the catastrophic degradation on VTA2 (blowing up error from 5.6 m/s to 212.4 m/s) are authentic consequences of formulating $\Delta \mathbf{v}^n$ as an unconditioned global target.
- **Action for Phase 3.1**: The target formulation $\Delta \mathbf{v}^n$ must be retired. Phase 3.1 must reformulate the ML objective into locally stationary body-frame dynamics (e.g., body-frame forward speed error or body-frame acceleration bias).

---

## 1. Audit Target Construction

Target construction in [`src/navris/ml/dataset.py`](file:///c:/Users/JAY%20PATEL/Documents/Jay/NAVRIS/src/navris/ml/dataset.py) was audited against source files:

### 1.1 Source Columns and Ingestion Lineage
- **Planar Reference Speed ($v_{\text{ref\_spd}}$)**:
  - Source: Raw VBOX `Velocity [km/h]` column.
  - Ingestion: [`src/navris/ingest.py`](file:///c:/Users/JAY%20PATEL/Documents/Jay/NAVRIS/src/navris/ingest.py#L413): `df_raw[vel_col] * KMH_TO_MPS` ($1 \text{ km/h} = \frac{1}{3.6} \text{ m/s}$).
  - Units in trajectory files: Meters per second ($\text{m/s}$). Confirmed.
- **Reference Heading ($\psi_{\text{ref}}$)**:
  - Source: Raw VBOX `Heading [deg]` column.
  - Ingestion: [`src/navris/ingest.py`](file:///c:/Users/JAY%20PATEL/Documents/Jay/NAVRIS/src/navris/ingest.py#L422): `np.radians(head_deg) % (2.0 * np.pi)`.
  - Units in trajectory files: Radians ($[0, 2\pi)$). Confirmed.
- **Vertical Reference Speed ($v_{\text{ref\_vertical\_spd}}$)**:
  - Source: Raw VBOX `Vertical velocity [km/h]` column.
  - Ingestion: [`src/navris/ingest.py`](file:///c:/Users/JAY%20PATEL/Documents/Jay/NAVRIS/src/navris/ingest.py#L416): `df_raw[vvel_col] * KMH_TO_MPS`.
  - Units in sync files: Meters per second ($\text{m/s}$). Confirmed.

### 1.2 Target Mathematical Implementation
In [`src/navris/ml/dataset.py`](file:///c:/Users/JAY%20PATEL/Documents/Jay/NAVRIS/src/navris/ml/dataset.py#L140-L151):
$$\mathbf{v}_{\text{ref}}^n = \begin{bmatrix} v_{\text{ref\_spd}} \sin(\psi_{\text{ref}}) \\ v_{\text{ref\_spd}} \cos(\psi_{\text{ref}}) \\ v_{\text{ref\_vertical\_spd}} \end{bmatrix} = \begin{bmatrix} v_{\text{ref}, E} \\ v_{\text{ref}, N} \\ v_{\text{ref}, U} \end{bmatrix}$$
$$\Delta \mathbf{v}^n = \mathbf{v}_{\text{ref}}^n - \mathbf{v}_{\text{NAVRIS}}^n = \begin{bmatrix} v_{\text{ref}, E} - v_{\text{nav}, E} \\ v_{\text{ref}, N} - v_{\text{nav}, N} \\ v_{\text{ref}, U} - v_{\text{nav}, U} \end{bmatrix}$$

---

## 2. Audit NAVRIS Classical Velocity

NAVRIS velocity is loaded in [`dataset.py`](file:///c:/Users/JAY%20PATEL/Documents/Jay/NAVRIS/src/navris/ml/dataset.py#L144-L146) from `data/processed/phase2_3b/gate2_3b/{rec}_trajectory_comparison.csv`:
- Source columns: `vel_A_east_mps`, `vel_A_north_mps`, `vel_A_up_mps`.
- Baseline configuration: **Baseline A** (`A_CONTROL_BASELINE`: ESKF + ZUPT, without NHC).
- Coordinate frame: Local Cartesian ENU (East, North, Up), originating directly from nominal filter state `eskf.state.v` defined in [`src/navris/eskf/state.py`](file:///c:/Users/JAY%20PATEL/Documents/Jay/NAVRIS/src/navris/eskf/state.py#L8-L9).
- Units: Meters per second ($\text{m/s}$). Confirmed.

---

## 3. Frame & Axis Consistency Verification

To independently verify whether the heading convention $\sin(\psi)$ for East and $\cos(\psi)$ for North is consistent with the coordinate system, finite-difference velocity vectors were computed from reference geodetic positions:
$$v_{\text{fd}, E}(t) = \frac{\text{ref\_east\_m}(t+\Delta t) - \text{ref\_east\_m}(t)}{\Delta t}, \quad v_{\text{fd}, N}(t) = \frac{\text{ref\_north\_m}(t+\Delta t) - \text{ref\_north\_m}(t)}{\Delta t}$$
and compared directly against the projected reference velocities:

| Route | Axis | Mean Difference ($v_{\text{fd}} - v_{\text{proj}}$) | Std Dev | RMSE |
| :--- | :--- | :--- | :--- | :--- |
| **VTA2** | **East** | $-0.0059 \text{ m/s}$ | $0.1758 \text{ m/s}$ | **0.1759 m/s** |
| **VTA2** | **North** | $+0.0163 \text{ m/s}$ | $0.2747 \text{ m/s}$ | **0.2752 m/s** |
| **VTA2** | **Up** | $-0.0096 \text{ m/s}$ | $0.3494 \text{ m/s}$ | **0.3495 m/s** |
| **S3A** | **East** | $+0.0101 \text{ m/s}$ | $0.1879 \text{ m/s}$ | **0.1882 m/s** |
| **S3A** | **North** | $+0.0089 \text{ m/s}$ | $0.2467 \text{ m/s}$ | **0.2468 m/s** |
| **S3A** | **Up** | $-0.0019 \text{ m/s}$ | $0.4128 \text{ m/s}$ | **0.4128 m/s** |

**Verdict**: The differences are negligible ($< 0.35$ m/s, consistent with GNSS discrete position fix differentiation noise). This proves conclusively:
1. Heading is 0 = North, $\pi/2$ = East.
2. $v_E = v_{\text{spd}} \sin\psi$ and $v_N = v_{\text{spd}} \cos\psi$ are strictly consistent with the navigation frame.
3. Vertical speed is positive Up.
4. There is **zero frame mismatch or axis inversion**.

---

## 4. Time Alignment & Timestamp Verification

Verification of timestamps between `*_sync.parquet` and `*_trajectory_comparison.csv`:

| Route | Rows Sync | Rows Trajectory | Max Absolute $|\Delta t|$ | Time Offset / Systematic Shift |
| :--- | :--- | :--- | :--- | :--- |
| **S1** | 50,183 | 50,183 | **0.000000 s** | None |
| **S2** | 91,908 | 91,908 | **0.000000 s** | None |
| **S3A** | 21,711 | 21,711 | **0.000000 s** | None |
| **S4** | 92,902 | 92,902 | **0.000000 s** | None |
| **VTA1A** | 24,895 | 24,895 | **0.000000 s** | None |
| **VTA2** | 10,128 | 10,128 | **0.000000 s** | None |
| **Y1** | 71,759 | 71,759 | **0.000000 s** | None |

**Verdict**: Timestamp alignment is mathematically exact down to machine precision. There is no timestamp jitter, interpolation artifact, or lag offset.

---

## 5. Audit Evaluation Metrics Implementation

Implemented in [`src/navris/ml/evaluate.py`](file:///c:/Users/JAY%20PATEL/Documents/Jay/NAVRIS/src/navris/ml/evaluate.py):
1. **Axis Error**:
   $$e_i = y_{\text{true}, i} - \hat{y}_i$$
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |e_i|, \quad \text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N e_i^2}, \quad \text{Bias} = \frac{1}{N} \sum_{i=1}^N e_i$$
2. **Horizontal Velocity RMSE**:
   $$\text{RMSE}_{\text{horiz}} = \sqrt{\frac{1}{N} \sum_{i=1}^N (e_{E, i}^2 + e_{N, i}^2)}$$
3. **3D Velocity RMSE**:
   $$\text{RMSE}_{\text{3D}} = \sqrt{\frac{1}{N} \sum_{i=1}^N (e_{E, i}^2 + e_{N, i}^2 + e_{U, i}^2)}$$
4. **Baseline C Velocity Error Equivalence**:
   $$v_{C}^n = v_{A}^n + \widehat{\Delta \mathbf{v}}^n \implies \mathbf{v}_{\text{ref}}^n - \mathbf{v}_{C}^n = (\mathbf{v}_{\text{ref}}^n - \mathbf{v}_{A}^n) - \widehat{\Delta \mathbf{v}}^n = \Delta \mathbf{v}^n - \widehat{\Delta \mathbf{v}}^n$$
   *Hence, Baseline C velocity error is mathematically identical to ML residual prediction error.*
5. **Open-Loop Position Drift**:
   $$p_E(t) = \int_0^t v_E(\tau) d\tau, \quad p_N(t) = \int_0^t v_N(\tau) d\tau \quad (\text{via trapezoidal rule})$$
   $$\text{Drift}_{\text{final}} = \sqrt{(p_{\text{nav}, E}(T) - p_{\text{ref}, E}(T))^2 + (p_{\text{nav}, N}(T) - p_{\text{ref}, N}(T))^2}$$

---

## 6. VTA2 Sanity Check (Nominal GNSS Tracking)

### 6.1 Representative Samples (Beginning, Middle, End)

```
        time_s  v_ref_E  v_ref_N  v_ref_U   nav_E    nav_N   nav_U  delta_E  delta_N  delta_U  spd_ref  spd_nav
0        85.50    15.50    12.37    -0.04   15.46    13.17    0.00     0.04    -0.80    -0.04    19.83    20.31
1        85.60    15.75    12.06    -0.04   15.55    12.97    0.02     0.20    -0.91    -0.07    19.84    20.25
2        85.70    15.90    12.00    -0.04   15.76    12.61    0.15     0.14    -0.61    -0.18    19.92    20.18
3        85.80    15.90    11.73    -0.04   16.13    12.11    0.23    -0.23    -0.38    -0.26    19.76    20.17
4        85.90    16.34    11.75    -0.04   16.22    12.21    0.17     0.13    -0.45    -0.21    20.13    20.30
5        86.00    16.26    11.18    -0.04   16.08    12.45    0.14     0.18    -1.27    -0.18    19.73    20.33
6        86.10    16.37    11.28    -0.04   16.36    12.52    0.13     0.00    -1.24    -0.17    19.88    20.61
5061    591.60     6.40    13.09    -0.21    3.43    12.92   -1.78     2.98     0.18     1.58    14.58    13.36
5062    591.70     6.50    12.94    -0.22    3.32    12.72   -1.72     3.17     0.23     1.51    14.48    13.14
5063    591.80     6.40    12.90    -0.23    3.15    12.87   -1.74     3.25     0.02     1.52    14.40    13.25
5064    591.90     6.57    12.92    -0.23    3.12    12.82   -1.78     3.45     0.10     1.55    14.49    13.19
5065    592.00     6.46    12.86    -0.21    3.01    12.61   -1.78     3.45     0.25     1.58    14.39    12.96
5066    592.10     6.44    12.85    -0.21    3.92    12.38   -1.64     2.52     0.47     1.43    14.38    12.98
5067    592.20     6.42    12.91    -0.20    3.74    12.69   -1.66     2.68     0.22     1.46    14.42    13.23
10122  1097.70    -0.16     0.01     0.01   -0.83     0.35    0.77     0.68    -0.34    -0.76     0.16     0.90
10123  1097.80    -0.09     0.01     0.01   -0.80     0.39    0.78     0.71    -0.38    -0.77     0.09     0.89
10124  1097.90    -0.05     0.00     0.01   -0.75     0.42    0.78     0.70    -0.41    -0.77     0.05     0.86
10125  1098.00    -0.03     0.00     0.01   -0.68     0.39    0.78     0.64    -0.38    -0.77     0.03     0.78
10126  1098.10    -0.02     0.00     0.01   -0.63     0.32    0.78     0.61    -0.32    -0.77     0.02     0.70
10127  1098.20    -0.02     0.00     0.01   -0.59     0.27    0.76     0.57    -0.27    -0.75     0.02     0.65
```

### 6.2 Overall VTA2 Component Statistics (10,128 Samples)

| Component | Min (m/s) | Max (m/s) | Mean (m/s) | Std Dev (m/s) | RMSE (m/s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$v_{\text{ref}, E}$** | $-14.07$ | $+21.52$ | $+2.17$ | $7.51$ | $7.82$ |
| **$v_{\text{nav}, E}$** | $-18.48$ | $+23.48$ | $+2.09$ | $7.94$ | $8.21$ |
| **$\Delta v_E$ (Target)** | $-14.71$ | $+10.04$ | $+0.08$ | $2.52$ | **2.52** |
| **$v_{\text{ref}, N}$** | $-11.27$ | $+17.69$ | $+5.09$ | $5.94$ | $7.82$ |
| **$v_{\text{nav}, N}$** | $-26.32$ | $+35.33$ | $+5.54$ | $7.28$ | $9.15$ |
| **$\Delta v_N$ (Target)** | $-36.73$ | $+25.40$ | $-0.44$ | $4.90$ | **4.92** |
| **$v_{\text{ref}, U}$** | $-0.26$ | $+0.38$ | $0.00$ | $0.07$ | $0.07$ |
| **$v_{\text{nav}, U}$** | $-4.24$ | $+3.40$ | $-0.22$ | $1.07$ | $1.09$ |
| **$\Delta v_U$ (Target)** | $-3.40$ | $+4.26$ | $+0.22$ | $1.07$ | **1.09** |
| **Horizontal Target Residual ($\sqrt{\Delta v_E^2 + \Delta v_N^2}$)** | — | — | — | — | **5.52 m/s** |
| **3D Target Residual ($\sqrt{\Delta v_E^2 + \Delta v_N^2 + \Delta v_U^2}$)** | — | — | — | — | **5.63 m/s** |

---

## 7. S3A Sanity Check (GNSS Outage Dead Reckoning)

### 7.1 Representative Samples (Beginning, Middle, End)

```
         time_s  v_ref_E  v_ref_N  v_ref_U    nav_E     nav_N    nav_U   delta_E   delta_N  delta_U  spd_ref   spd_nav
0        284.30     0.60    -0.24    -0.02     5.61     -2.07     0.00     -5.01      1.83    -0.02     0.64      5.98
1        284.40     0.43    -0.19    -0.01     5.55     -2.04    -0.00     -5.11      1.85    -0.01     0.47      5.91
2        284.50     0.32    -0.10    -0.01     5.51     -2.04     0.01     -5.19      1.95    -0.03     0.33      5.87
3        284.60     0.45    -0.16    -0.01     5.45     -2.03     0.01     -5.00      1.87    -0.02     0.48      5.82
4        284.70     0.33    -0.16    -0.01     5.40     -2.01     0.01     -5.06      1.85    -0.02     0.37      5.76
5        284.80     0.22    -0.10    -0.01     5.37     -2.02     0.03     -5.15      1.92    -0.04     0.24      5.74
6        284.90     0.30    -0.11    -0.01     5.32     -2.02     0.02     -5.03      1.91    -0.03     0.32      5.69
10852   1369.50    -0.00     0.00     0.00  3851.69  -1579.40 -1518.03  -3851.70   1579.40  1518.03     0.00   4162.94
10853   1369.60    -0.01     0.01     0.00  3852.19  -1579.92 -1518.37  -3852.20   1579.93  1518.38     0.01   4163.59
10854   1369.70    -0.01     0.01     0.00  3852.69  -1580.47 -1518.71  -3852.70   1580.48  1518.71     0.01   4164.26
10855   1369.80    -0.01     0.01     0.00  3853.18  -1581.02 -1519.04  -3853.19   1581.03  1519.04     0.01   4164.93
10856   1369.90    -0.02     0.02     0.00  3853.67  -1581.56 -1519.36  -3853.69   1581.58  1519.36     0.02   4165.59
10857   1370.00    -0.01     0.02     0.00  3854.16  -1582.09 -1519.70  -3854.18   1582.10  1519.70     0.02   4166.24
10858   1370.10    -0.01     0.01     0.00  3854.66  -1582.61 -1520.04  -3854.67   1582.63  1520.05     0.02   4166.90
21705   2454.80    -4.31    -7.95     0.06 -5445.58 -11461.29 -8560.22   5441.27  11453.35  8560.28     9.04  12689.19
21706   2454.90    -4.36    -7.93     0.06 -5445.73 -11462.26 -8561.46   5441.37  11454.34  8561.52     9.05  12690.13
21707   2455.00    -4.30    -7.96     0.06 -5445.96 -11463.16 -8562.64   5441.65  11455.20  8562.69     9.05  12691.04
21708   2455.10    -4.45    -8.03     0.07 -5446.12 -11464.08 -8563.72   5441.67  11456.05  8563.79     9.18  12691.94
21709   2455.20    -4.39    -8.05     0.06 -5446.27 -11464.98 -8564.92   5441.88  11456.93  8564.98     9.17  12692.82
21710   2455.30    -4.19    -8.16     0.05 -5446.49 -11465.90 -8566.18   5442.30  11457.75  8566.23     9.17  12693.75
```

### 7.2 Overall S3A Component Statistics (21,711 Samples)

| Component | Min (m/s) | Max (m/s) | Mean (m/s) | Std Dev (m/s) | RMSE (m/s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$v_{\text{ref}, E}$** | $-14.18$ | $+26.09$ | $+6.47$ | $8.72$ | $10.86$ |
| **$v_{\text{nav}, E}$** | $-6484.17$ | $+6738.70$ | $+1313.46$ | $3982.83$ | $4193.73$ |
| **$\Delta v_E$ (Target)** | $-6743.35$ | $+6482.92$ | $-1306.98$ | $3981.26$ | **4190.21** |
| **$v_{\text{ref}, N}$** | $-17.40$ | $+15.12$ | $-1.71$ | $5.50$ | $5.76$ |
| **$v_{\text{nav}, N}$** | $-11465.90$ | $+7.85$ | $-3139.09$ | $3649.31$ | $4813.59$ |
| **$\Delta v_N$ (Target)** | $-16.33$ | $+11457.75$ | $+3137.37$ | $3651.04$ | **4813.79** |
| **$v_{\text{ref}, U}$** | $-0.48$ | $+0.49$ | $+0.01$ | $0.08$ | $0.08$ |
| **$v_{\text{nav}, U}$** | $-8566.18$ | $+4.20$ | $-2082.12$ | $2158.78$ | $2999.22$ |
| **$\Delta v_U$ (Target)** | $-4.11$ | $+8566.23$ | $+2082.13$ | $2158.78$ | **2999.23** |
| **Horizontal Target Residual ($\sqrt{\Delta v_E^2 + \Delta v_N^2}$)** | — | — | — | — | **6382.04 m/s** |
| **3D Target Residual ($\sqrt{\Delta v_E^2 + \Delta v_N^2 + \Delta v_U^2}$)** | — | — | — | — | **7051.65 m/s** |

### 7.3 Forensic Diagnosis for S3A
- **Verdict**: **A. GENUINE CLASSICAL NAVRIS ESKF DIVERGENCE DURING GNSS OUTAGE**.
- **Evidence**:
  1. At $t = 284.3$ s (initialization), classical velocity was normal ($v_{\text{nav}} = [5.61, -2.07, 0.00]$ m/s).
  2. Between $t = 422$ s and $t = 2455$ s (>33 minutes), GNSS was rejected 88.3% of the time by the ESKF NIS gate.
  3. Over 2,171 seconds, unconstrained dead reckoning on smartphone IMU integrated tilt errors, producing an open-loop velocity drift of $-8,566$ m/s vertically and $-11,466$ m/s north.
  4. This exact velocity RMSE ($7,051.65$ m/s) was documented in Phase 2 Gate 2.3B summary (`gate2_3b_summary.csv`).

---

## 8. Target Statistics Before Any ML Prediction

Comparing the raw target residual $\Delta \mathbf{v}^n$ against the Phase 2.3B classical Baseline A filter velocity RMSE across all 7 routes:

| Route | Mean $\Delta v_E$ | Mean $\Delta v_N$ | Mean $\Delta v_U$ | Target RMSE Horiz (m/s) | Target RMSE 3D (m/s) | Gate 2.3B Classical RMSE (m/s) | Discrepancy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **S1** | $+368.5$ | $-265.3$ | $+6229.5$ | $2430.22$ | **8726.19** | **8726.20** | $0.01 \text{ m/s}$ |
| **S2** | $-4629.9$ | $+3424.2$ | $+45437.3$ | $6285.75$ | **53085.87** | **53085.87** | $0.00 \text{ m/s}$ |
| **S4** | $+1749.6$ | $-1641.8$ | $+13096.9$ | $4816.36$ | **17320.52** | **17320.51** | $0.01 \text{ m/s}$ |
| **S3A** | $-1307.0$ | $+3137.4$ | $+2082.1$ | $6382.04$ | **7051.65** | **7051.65** | $0.00 \text{ m/s}$ |
| **VTA1A** | $+2416.3$ | $+101.2$ | $+4288.6$ | $3685.19$ | **6926.62** | **6926.62** | $0.00 \text{ m/s}$ |
| **Y1** | $+473.1$ | $-288.9$ | $+23945.7$ | $1742.13$ | **29244.25** | **29244.25** | $0.00 \text{ m/s}$ |
| **VTA2** | $+0.08$ | $-0.44$ | $+0.22$ | $5.52$ | **5.63** | **5.63** | $0.00 \text{ m/s}$ |

**Scientific Conclusion**: The target residual $\Delta \mathbf{v}^n$ is **completely dominated by classical dead-reckoning filter drift**. Before any ML prediction is made, the target magnitudes on S1, S2, S4, S3A, VTA1A, and Y1 are already in the thousands of meters per second.

---

## 9. Feature Importance Forensic Confirmation

Auditing [`data/processed/phase3/feature_importances.csv`](file:///c:/Users/JAY%20PATEL/Documents/Jay/NAVRIS/data/processed/phase3/feature_importances.csv):
- `navris_vel_east_mps`: $0.383910$ ($38.39\%$)
- `navris_vel_up_mps`: $0.228068$ ($22.81\%$)
- `navris_speed_mps`: $0.180849$ ($18.08\%$)
- `navris_vel_north_mps`: $0.173970$ ($17.40\%$)
- **Top 4 NAVRIS Velocity States Sum**: **$0.966797$ ($96.68\%$, rounded to 96.7%)**.
- **All NAVRIS States Sum (including heading & 1s deltas)**: **$0.996564$ ($99.66\%$)**.
- **All Raw IMU Sensors Sum**: **$0.000051$ ($0.0051\%$)**.

The calculation of the 96.7% figure reported in Phase 3.0 is **100% verified**. The model placed virtually all split decisions on classical filter velocity, completely bypassing raw inertial dynamics.

---

## 10. Scientific Decision & Recommendations for Phase 3.1

### 10.1 Verdict on Phase 3.0 ML Failure
The Phase 3.0 ML failure is **scientifically valid evidence**. It exposes a fundamental flaw in the **problem framing**, not an implementation bug:
- In unconstrained dead reckoning, classical filter velocity $v_{\text{nav}}(t)$ diverges toward $\pm 10^4$ m/s, while vehicle physical velocity is bounded within $[0, 30]$ m/s.
- Consequently, the residual target $\Delta \mathbf{v}^n = \mathbf{v}_{\text{ref}}^n - \mathbf{v}_{\text{nav}}^n \approx -\mathbf{v}_{\text{nav}}^n$.
- Any supervised regressor trained on this objective discovers the trivial global minimum: negate the filter velocity state.
- When applied to a nominally tracked route (VTA2), this inversion logic injects massive spurious corrections, turning a 455 m position error into 15,396 m.

### 10.2 Mandatory Target Redesign for Phase 3.1
The global navigation-frame velocity residual target $\Delta \mathbf{v}^n$ must be retired.

**Phase 3.1 Candidate Architectures**:
1. **Body-Frame Forward Speed Residual**:
   $$\Delta v_{\text{fwd}}^b = v_{\text{ref, fwd}}^b - v_{\text{nav, fwd}}^b$$
   Bounded by vehicle dynamics ($\pm 5$ m/s), invariant to global orientation drift, and stationary over time.
2. **Body-Frame Specific Force & Gyroscope Bias Residuals**:
   $$\Delta \mathbf{f}^b = \mathbf{f}_{\text{true}}^b - \mathbf{f}_{\text{meas}}^b, \quad \Delta \boldsymbol{\omega}^b = \boldsymbol{\omega}_{\text{true}}^b - \boldsymbol{\omega}_{\text{meas}}^b$$
   Correcting the IMU inputs directly at the sensor level before integration.
3. **Outage-Conditioned Innovation Gating**:
   Explicitly conditioning any ML correction on outage duration $\Delta t_{\text{outage}}$ and filter covariance trace $\text{Tr}(\mathbf{P}_{vv})$, preventing spurious corrections during nominal GNSS lock.
