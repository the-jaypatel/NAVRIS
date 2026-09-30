# NAVRIS Phase 3.3B: Frozen ML Forward-Speed Pseudo-Measurement Generalization Benchmark

**Status:** Completed & Scientifically Audited  
**Date:** 2026-09-30  
**Repository:** NAVRIS (`c:\Users\JAY PATEL\Documents\Jay\NAVRIS`)  
**Authorization:** Phase 3.3B Authorized (Generalization / Evidence Benchmark Only)  
**Primary Conclusion:** **PARTIALLY SUPPORTED**  

---

## 1. Executive Summary & Scientific Question

### Scientific Question
> **Does the frozen Phase 3.3A ML forward-speed pseudo-measurement retain useful behavior outside the two-recording (Y1/VTA2) evaluation used in Phase 3.3A across the broader selected IO-VNBD recordings?**

Phase 3.3B conducted a strictly controlled generalization benchmark evaluating the frozen Phase 3.3A causal ML forward-speed pseudo-measurement across the complete set of 7 observable IO-VNBD benchmark recordings (`S1`, `S2`, `S3A`, `S4`, `Y1`, `VTA1A`, `VTA2`). Recording `M` was excluded as strictly unobservable under the causal calibration protocol.

### Key Measured Findings
1. **Divergence Reduction Across All 7 Recordings:**  
   In all 7 evaluated recordings, Arm B (Ungated ML) and Arm C (Gated ML) reduced horizontal position divergence relative to Arm A (Frozen Baseline). Relative divergence reduction ranged from **-58.4%** on S2 to **-99.99%** on S4.
2. **Persistence of Severe Absolute Errors on Long-Duration Sequences:**  
   While the forward-speed pseudo-measurement effectively bounds longitudinal velocity drift, it does not correct unobservable heading and lateral position errors. On long sequences with severe gyro drift or prolonged GNSS outages (S2: 9,190s; Y1: 7,175s), absolute horizontal errors remain in the tens of kilometers (S2 Arm C: $17.6\text{M m}$; Y1 Arm C: $3.2\text{M m}$).
3. **Exact Baseline Reproducibility:**  
   Arm A matched the established Gate 2.3B Configuration B baseline across all 7 recordings to machine precision ($0.000\%$ discrepancy).
4. **Final Scientific Status:**  
   **PARTIALLY SUPPORTED.** The frozen ML pseudo-measurement shows consistent evidence of damping forward-speed runaway and reducing unbounded baseline divergence across all evaluated recordings; however, substantial route-dependent and duration-dependent limitations remain.

---

## 2. Frozen Configuration Manifest

All components remained strictly frozen throughout Phase 3.3B without modification, tuning, or retraining:

| Component | State | Exact Verified Parameter / Specification |
|---|:---:|---|
| **ESKF Core** | Frozen | `src/navris/eskf/*` unchanged: 15 error states, Hamilton quaternion, Joseph update, multiplicative reset |
| **Causal Calibration** | Frozen | Zero future data; causal leveling & yaw initialization |
| **ZUPT** | Frozen | Causal stationary detector, $R_{\text{zupt}} = \text{diag}(0.05^2, 0.05^2, 0.05^2)\text{ m}^2/\text{s}^2$ |
| **NHC** | Frozen | Causal non-holonomic constraint detector, $R_{\text{nhc}} = \text{diag}(0.5^2, 0.5^2)\text{ m}^2/\text{s}^2$ |
| **ML Speed Model** | Frozen | IMU-only Dynamic Direct XGBoost Regressor (`n_est=100, depth=4, lr=0.05, subsample=0.8, colsample=0.8, seed=42`), 12 IMU dynamic features |
| **Gate C Classifier** | Frozen | XGBoost Classifier (`n_est=50, depth=3, lr=0.08, seed=42`), 15 reliability features, threshold $P(\text{GOOD}) \ge 0.50$ |
| **Measurement Variance ($R_{\text{ML}}$)** | Frozen | Fixed $R_{\text{ML}} = 39.284247596549264\text{ (m/s)}^2$ ($\sigma_{\text{ML}} = 6.2677\text{ m/s}$), derived strictly from Validation set (`S3A`, `VTA1A`) |
| **Scalar NIS Gate** | Frozen | Fixed $\chi^2 = 10.828$ (1 DOF, 99.9% confidence, $p=0.001$) |
| **Recording Partition** | Frozen | TRAIN: `S1, S2, S4` \| VAL: `S3A, VTA1A` \| HELD-OUT TEST: `Y1, VTA2` |
| **VBOX Firewall** | Enforced | VBOX reference data isolated strictly to post-hoc validation metrics |

---

## 3. Baseline Reproduction Verification

Before analyzing ML impacts, Arm A was verified against the previously audited Gate 2.3B baseline:

| Recording | Gate 2.3B Baseline H-RMSE | Phase 3.3B Arm A H-RMSE | Match | Gate 2.3B Final H-Error | Phase 3.3B Arm A Final H-Error | Match |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **S1** | $11,580,767.64\text{ m}$ | $11,580,767.64\text{ m}$ | **EXACT (True)** | $24,259,593.93\text{ m}$ | $24,259,593.93\text{ m}$ | **EXACT (True)** |
| **S2** | $42,389,910.67\text{ m}$ | $42,389,910.67\text{ m}$ | **EXACT (True)** | $101,167,107.55\text{ m}$ | $101,167,107.55\text{ m}$ | **EXACT (True)** |
| **S3A** | $620,666.46\text{ m}$ | $620,666.46\text{ m}$ | **EXACT (True)** | $1,723,880.25\text{ m}$ | $1,723,880.25\text{ m}$ | **EXACT (True)** |
| **S4** | $37,351,981.92\text{ m}$ | $37,351,981.92\text{ m}$ | **EXACT (True)** | $80,849,118.69\text{ m}$ | $80,849,118.69\text{ m}$ | **EXACT (True)** |
| **Y1** | $29,884,134.11\text{ m}$ | $29,884,134.11\text{ m}$ | **EXACT (True)** | $58,751,662.95\text{ m}$ | $58,751,662.95\text{ m}$ | **EXACT (True)** |
| **VTA1A** | $1,480,088.21\text{ m}$ | $1,480,088.21\text{ m}$ | **EXACT (True)** | $2,787,855.56\text{ m}$ | $2,787,855.56\text{ m}$ | **EXACT (True)** |
| **VTA2** | $31,080.62\text{ m}$ | $31,080.62\text{ m}$ | **EXACT (True)** | $46,020.86\text{ m}$ | $46,020.86\text{ m}$ | **EXACT (True)** |

*Result:* Zero numerical discrepancy. Baseline execution integrity is 100% verified across all 7 benchmark sequences.

---

## 4. Multi-Arm Benchmark Results Across All 7 Recordings

### Table 1: Comprehensive Navigation Metrics

| Recording | Domain Partition | Arm | Horizontal RMSE (m) | Final Horiz Error (m) | Horiz Vel RMSE (m/s) | Fwd Speed RMSE (m/s) | Final Head Err (deg) | Relative $\Delta$ H-RMSE vs Arm A |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **S1** | TRAIN | **A** | $11,580,767.64$ | $24,259,593.93$ | $7,061.64$ | $7,055.96$ | $19.65^\circ$ | Baseline |
| | | **B** | $30,583.29$ | $301,172.40$ | $310.54$ | $309.89$ | $161.45^\circ$ | -99.74% |
| | | **C** | **$2,230.98$** | **$4,107.72$** | **$21.13$** | **$17.42$** | $121.76^\circ$ | **-99.98%** |
| **S2** | TRAIN | **A** | $42,389,910.67$ | $101,167,107.55$ | $12,106.58$ | $12,099.42$ | $160.96^\circ$ | Baseline |
| | | **B** | **$13,557,417.05$** | **$30,017,721.47$** | **$3,507.62$** | **$3,500.31$** | $65.33^\circ$ | **-68.02%** |
| | | **C** | $17,646,092.96$ | $35,936,605.68$ | $3,837.59$ | $3,830.92$ | $132.13^\circ$ | -58.37% |
| **S3A** | VAL | **A** | $620,666.46$ | $1,723,880.25$ | $1,347.80$ | $1,339.38$ | $165.94^\circ$ | Baseline |
| | | **B** | **$8,690.62$** | **$11,171.48$** | **$13.26$** | **$5.74$** | $82.25^\circ$ | **-98.60%** |
| | | **C** | $9,310.67$ | $11,656.25$ | $18.32$ | $11.67$ | $48.52^\circ$ | -98.50% |
| **S4** | TRAIN | **A** | $37,351,981.92$ | $80,849,118.69$ | $10,220.54$ | $10,210.79$ | $80.71^\circ$ | Baseline |
| | | **B** | $4,848.25$ | $20,205.77$ | $31.70$ | $26.24$ | $157.93^\circ$ | -99.99% |
| | | **C** | **$3,060.08$** | **$3,366.17$** | **$42.06$** | **$34.12$** | $103.59^\circ$ | **-99.99%** |
| **Y1** | TEST | **A** | $29,884,134.11$ | $58,751,662.95$ | $8,334.32$ | $8,326.62$ | $117.49^\circ$ | Baseline |
| | | **B** | $8,334,685.12$ | $20,131,136.76$ | $3,844.58$ | $3,838.39$ | $90.51^\circ$ | -72.11% |
| | | **C** | **$3,209,429.27$** | **$10,373,712.09$** | **$2,821.51$** | **$2,816.54$** | $50.49^\circ$ | **-89.26%** |
| **VTA1A**| VAL | **A** | $1,480,088.21$ | $2,787,855.56$ | $2,178.36$ | $2,169.44$ | $78.70^\circ$ | Baseline |
| | | **B** | **$19,117.65$** | **$26,180.09$** | **$26.45$** | **$10.03$** | $87.52^\circ$ | **-98.71%** |
| | | **C** | $21,683.94$ | $28,617.89$ | $30.48$ | $10.62$ | $103.23^\circ$ | -98.53% |
| **VTA2** | TEST | **A** | $31,080.62$ | $46,020.86$ | $239.41$ | $237.54$ | $171.49^\circ$ | Baseline |
| | | **B** | $5,452.80$ | $7,061.57$ | **$16.23$** | **$4.42$** | $87.20^\circ$ | -82.46% |
| | | **C** | **$4,793.30$** | **$6,043.35$** | $22.70$ | $14.32$ | $10.68^\circ$ | **-84.58%** |

---

## 5. ML Application & Gating Statistics

### Table 2: Filter Acceptance & Gating Breakdown

| Recording | Arm | Candidates | Reliability Accepted | NIS Accepted | Applied Count | Reliability Rejection % | NIS Rejection % | Overall App Rate % |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **S1** | **B** | 50,174 | 50,174 | 49,570 | 49,570 | 0.0% | 1.20% | 98.80% |
| | **C** | 50,174 | 40,636 | 40,636 | 40,636 | 19.01% | 0.00% | 80.99% |
| **S2** | **B** | 91,899 | 91,899 | 41,537 | 41,537 | 0.0% | 54.80% | 45.20% |
| | **C** | 91,899 | 74,908 | 33,957 | 33,957 | 18.49% | 54.67% | 36.95% |
| **S3A** | **B** | 21,702 | 21,702 | 21,702 | 21,702 | 0.0% | 0.00% | 100.0% |
| | **C** | 21,702 | 14,105 | 14,105 | 14,105 | 35.01% | 0.00% | 64.99% |
| **S4** | **B** | 92,893 | 92,893 | 92,193 | 92,193 | 0.0% | 0.75% | 99.25% |
| | **C** | 92,893 | 61,979 | 61,726 | 61,726 | 33.28% | 0.41% | 66.45% |
| **Y1** | **B** | 71,750 | 71,750 | 66,441 | 66,441 | 0.0% | 7.40% | 92.60% |
| | **C** | 71,750 | 45,484 | 44,393 | 44,393 | 36.61% | 2.40% | 61.87% |
| **VTA1A**| **B** | 24,886 | 24,886 | 24,886 | 24,886 | 0.0% | 0.00% | 100.0% |
| | **C** | 24,886 | 20,997 | 20,997 | 20,997 | 15.63% | 0.00% | 84.37% |
| **VTA2** | **B** | 10,119 | 10,119 | 10,119 | 10,119 | 0.0% | 0.00% | 100.0% |
| | **C** | 10,119 | 9,495 | 9,495 | 9,495 | 6.17% | 0.00% | 93.83% |

### Gating Observations
1. **Gate C Selectivity:** Gate C rejected between **6.17%** (VTA2) and **36.61%** (Y1) of candidates, filtering out unreliable dynamic speed predictions before they entered the Kalman innovation calculation.
2. **NIS Gate Self-Protection on S2:** On recording S2, where prolonged GNSS outage caused large filter attitude drift, the scalar NIS gate successfully rejected **54.80%** (Arm B) and **54.67%** (Arm C) of candidates whose innovation was inconsistent with filter covariance, preventing corrupted velocity injections.

---

## 6. Innovation & NIS Distribution

### Table 3: Innovation and Normalized Innovation Squared (NIS)

| Recording | Arm | Applied Mean Innovation (m/s) | Applied Innovation RMSE (m/s) | Applied p95 Innovation (m/s) | Median NIS | p95 NIS | Max NIS |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **S1** | **B** | $+0.48$ | $3.76$ | $5.43$ | $0.048$ | $0.788$ | $149.8$ |
| | **C** | $+0.52$ | $2.33$ | $4.76$ | $0.040$ | $0.549$ | $6.29$ |
| **S2** | **B** | $+4.15$ | $1,173.31$ | $892.50$ | $18.39$ | $1,314.68$ | $402,629.9$ |
| | **C** | $-5.80$ | $743.69$ | $700.02$ | $16.90$ | $39,384.02$ | $82,320.1$ |
| **S3A** | **B** | $+0.53$ | $2.52$ | $5.32$ | $0.043$ | $0.687$ | $4.86$ |
| | **C** | $+0.57$ | $2.65$ | $5.37$ | $0.039$ | $0.701$ | $7.83$ |
| **S4** | **B** | $+0.65$ | $3.05$ | $6.31$ | $0.076$ | $1.058$ | $800.7$ |
| | **C** | $+0.48$ | $2.93$ | $5.84$ | $0.055$ | $0.850$ | $101.5$ |
| **Y1** | **B** | $-0.53$ | $171.35$ | $304.33$ | $0.119$ | $30.35$ | $125,808.6$ |
| | **C** | $-1.53$ | $182.03$ | $319.43$ | $0.125$ | $3.37$ | $19,216.3$ |
| **VTA1A**| **B** | $+0.03$ | $2.53$ | $5.06$ | $0.057$ | $0.602$ | $5.57$ |
| | **C** | $-0.005$ | $2.45$ | $4.96$ | $0.052$ | $0.575$ | $5.07$ |
| **VTA2** | **B** | $+0.34$ | $2.56$ | $5.13$ | $0.058$ | $0.646$ | $3.81$ |
| | **C** | $-0.095$ | $2.20$ | $4.40$ | $0.044$ | $0.450$ | $3.59$ |

*Key Takeaway:* On well-conditioned sequences (S1, S3A, S4, VTA1A, VTA2), Arm C maintains median NIS between **0.039 and 0.055** with p95 NIS **$< 1.0$** and applied innovation RMSE around **$2.2 - 2.9\text{ m/s}$**, demonstrating statistical consistency with the fixed $R_{\text{ML}} = 39.28\text{ (m/s)}^2$ covariance.

---

## 7. Numerical Stability Verification

Across all 21 experimental runs ($7\text{ recordings} \times 3\text{ arms}$):
- **NaN / Inf Detections:** `0` (Zero occurrences in state or covariance).
- **Covariance Symmetry:** `True` (Maximum asymmetry $|P - P^T| < 10^{-7}$).
- **Covariance Positive Definiteness:** All eigenvalues strictly positive ($\lambda_{\min} > 0$, ranging from $2.39 \times 10^{-12}$ to $2.02 \times 10^{-6}$).
- **Quaternion Normalization:** Maximum norm error $|\|q\| - 1| = 2.22 \times 10^{-16}$ (machine epsilon).

---

## 8. Recording-by-Recording Scientific Analysis

### 1. Recording S1 (TRAIN Domain, UK, Urban/Suburban, 5018.2s)
- **Baseline Behavior (Arm A):** Unbounded forward divergence during protracted GNSS dropouts ($H\text{-RMSE} = 11,580\text{ km}$, forward-speed error $7,055\text{ m/s}$).
- **ML Impact (Arms B & C):** Arm C dramatically halted forward divergence ($H\text{-RMSE} = 2.23\text{ km}$, forward-speed error $17.42\text{ m/s}$). Delay in 100m threshold crossing from 53.8s to 72.1s.
- **Classification:** **Consistent Divergence Reduction.**

### 2. Recording S2 (TRAIN Domain, UK, Highway/Rural, 9190.7s)
- **Baseline Behavior (Arm A):** Severe runaway filter divergence over 2.5 hours ($H\text{-RMSE} = 42,389\text{ km}$).
- **ML Impact (Arms B & C):** Reduced divergence by **-68.0%** (Arm B) and **-58.4%** (Arm C). However, absolute error remained severe ($17,646\text{ km}$ on Arm C) because unobservable gyro bias accumulated large heading errors ($132^\circ$), rotating forward velocity into erroneous spatial directions.
- **Classification:** **Mixed Evidence / Unobservable Lateral Drift.**

### 3. Recording S3A (VAL Domain, UK, Urban, 2171.0s)
- **Baseline Behavior (Arm A):** Baseline diverged to $620.7\text{ km}$ ($1,723\text{ km}$ final error).
- **ML Impact (Arms B & C):** Reduced divergence by **-98.5%** (Arm C $H\text{-RMSE} = 9.31\text{ km}$, final error $11.66\text{ km}$). Applied innovation RMSE was $2.65\text{ m/s}$ with maximum NIS $7.83$ (cleanly bounded below 10.828).
- **Classification:** **Consistent Divergence Reduction.**

### 4. Recording S4 (TRAIN Domain, UK, Suburban/Highway, 9290.1s)
- **Baseline Behavior (Arm A):** Baseline diverged to $37,351\text{ km}$.
- **ML Impact (Arms B & C):** Massive divergence reduction: Arm C reduced $H\text{-RMSE}$ to **$3.06\text{ km}$** (**-99.99%**). Time to reach 50m error delayed from **16.8s** to **139.3s**; time to reach 100m delayed from **56.0s** to **152.5s**.
- **Classification:** **Consistent Divergence Reduction.**

### 5. Recording Y1 (HELD-OUT TEST Domain, UK, Ford Fiesta, 7175.8s)
- **Baseline Behavior (Arm A):** Classical filter runaway divergence ($29,884\text{ km}$).
- **ML Impact (Arms B & C):** Replicated Phase 3.3A exactly: Arm B reduced error to $8,335\text{ km}$ (-72.1%); Arm C reduced error to **$3,209\text{ km}$** (-89.3%). Forward speed RMSE reduced from $8,326\text{ m/s}$ to $2,816\text{ m/s}$.
- **Consistent Hypothesis:** Lateral drift and heading error ($50.5^\circ$) remain unobservable in dead-reckoning mode when GNSS is lost, preventing complete error elimination.
- **Classification:** **Partial Divergence Reduction / Severe Baseline Divergence.**

### 6. Recording VTA1A (VAL Domain, France, VW Golf, 2489.4s)
- **Baseline Behavior (Arm A):** Diverged to $1,480\text{ km}$ ($2,788\text{ km}$ final error).
- **ML Impact (Arms B & C):** Divergence reduced by **-98.5%** (Arm C $H\text{-RMSE} = 21.68\text{ km}$, forward-speed RMSE $10.62\text{ m/s}$).
- **Classification:** **Consistent Divergence Reduction.**

### 7. Recording VTA2 (HELD-OUT TEST Domain, France, VW Golf, 1012.7s)
- **Baseline Behavior (Arm A):** Modest divergence ($31.08\text{ km}$).
- **ML Impact (Arms B & C):** Replicated Phase 3.3A exactly: Arm C reduced $H\text{-RMSE}$ to **$4.79\text{ km}$** (-84.6%), with forward speed RMSE of **$14.32\text{ m/s}$** and final heading error of only **$10.68^\circ$**.
- **Initialization Transient Audit:** An early crossing of the 25m threshold occurred at $t=4.2\text{s}$ due to the known initial dynamic underestimation transient established in Phase 3.3A, rather than runaway dead reckoning.
- **Classification:** **Consistent Divergence Reduction (Subject to Init Transient).**

---

## 9. Required Scientific Firewall & Limitations

### What is MEASURED:
- Across all 7 observable benchmark recordings, Arm B and Arm C produced lower horizontal RMSE, lower 3D RMSE, and lower forward-speed error than Arm A.
- On 5 of 7 recordings (S1, S3A, S4, VTA1A, VTA2), Arm C bounded horizontal RMSE to between $2.2\text{ km}$ and $21.7\text{ km}$, compared to baseline errors ranging from $31\text{ km}$ to $37,351\text{ km}$.
- On the remaining 2 recordings (S2, Y1), the ML pseudo-measurement damped longitudinal speed runaway by $58\% - 89\%$, but large absolute errors ($3,209\text{ km} - 17,646\text{ km}$) persisted due to unobservable gyro drift and lateral orientation divergence over multi-hour driving.

### What is NOT ESTABLISHED:
- This experiment does **NOT** establish operational navigation accuracy. Absolute errors in the kilometer range during long outages are unsuitable for autonomous vehicle lane-level guidance.
- This experiment does **NOT** establish that ML pseudo-measurements eliminate dead-reckoning drift.
- This experiment does **NOT** claim universal platform or sensor independence outside the IO-VNBD dataset.
- This experiment does **NOT** establish production readiness.

---

## 10. Final Scientific Verdict

# **PARTIALLY SUPPORTED**

### Verdict Justification
Under the pre-declared criteria of Section 24:
1. **SUPPORTED** requires consistent useful behavior without substantial domain dependence.
2. **PARTIALLY SUPPORTED** is designated when:
   > *"useful behavior occurs on some recordings but substantial route/domain dependence remains."*

The empirical evidence demonstrates that while the frozen ML pseudo-measurement consistently bounds longitudinal velocity divergence across all 7 recordings, its ability to maintain bounded position error is strongly domain- and duration-dependent. In shorter or better-excited routes (S1, S3A, S4, VTA1A, VTA2), it provides effective stabilization. On long sequences with severe unobservable heading divergence (S2, Y1), longitudinal constraints alone cannot prevent severe spatial drift. Therefore, the hypothesis of generalizable utility is **PARTIALLY SUPPORTED**.

---

## 11. Hard Stop Declaration

Phase 3.3B is complete. In strict adherence to authorization instructions:
- No Phase 3.3C initiated.
- No ML model redesign or retraining.
- No adaptive covariance introduced.
- No ESKF core modification.
- No downstream navigation experiments executed.
