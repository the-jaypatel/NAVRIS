# NAVRIS Phase 2.2 — Formal Scientific Claim Audit

**Project:** NAVRIS — Intelligent Navigation & Inertial System  
**SIH 2026 Problem Statement 26168:** AI-ML based Intelligent Dead Reckoning system for seamless navigation  
**Phase:** 2.2.5 — Scientific Validation & Benchmark Hardening  
**Document:** Systematic Epistemic Classification of Phase 2.2 Claims  
**Date:** September 2026  

---

## 1. Claim Classification Framework

To ensure absolute scientific rigor and prevent over-claiming in technical publications and project defense, all central findings and assertions from Phase 2.2 and the forensic audit are classified into four rigorous epistemic tiers:

* **Tier A — Directly Measured:** Empirical facts directly observed, recorded, and verified in the IO-VNBD dataset.
* **Tier B — Mathematical / Theoretical:** Analytically exact derivations or proofs that hold universally, independent of specific sensor noise.
* **Tier C — Strong Inference:** Hypotheses strongly supported by multiple empirical observations and physical models, but where the exact error proportion cannot be uniquely uncoupled without bench instruments.
* **Tier D — Hypothesis / Unresolved:** Plausible mechanisms or potential artifacts that remain unverified or require physical bench experiments.

---

## 2. Systematic Classification of Key Phase 2.2 Claims

| # | Scientific Claim / Finding | Tier | Justification & Epistemic Boundary |
|---|---|:---:|---|
| **1** | **Catastrophic A0 Divergence:** Unconstrained smartphone IMU strapdown dead reckoning diverges to $10^5 - 10^7$ m over multi-hour runs. | **A** | Directly measured across all 8 synchronized recordings ($5,000$ to $10,260$ s) using pure IMU signals without GNSS updates. |
| **2** | **103.92° Long-Run Heading RMSE:** Long-run heading errors cluster near $103.9^\circ$. | **B** | Analytically proven: two independent uniform random variables on $[-\pi, +\pi)$ have an exact theoretical RMS difference of $\pi/\sqrt{3} \approx 103.92^\circ$. Measured mean across all long recordings ($103.88^\circ$) matches theory within $0.04^\circ$. Indicates that unconstrained gyro integration becomes completely decorrelated from true heading over hours. |
| **3** | **Sparse GNSS Sampling & Repetition:** Smartphone GNSS arrives at $\sim 0.10 - 0.11$ Hz with $\sim 99\%$ sample-and-hold duplicates. | **A** | Directly measured in raw `S-*.csv` files across Drivers A, B, D, and E. Verified by novel fix gating logic. |
| **4** | **Initial GNSS Offset in T1 Benchmark:** Short 10s outage errors include an initial $\sim 34$ m (median) GNSS fix error. | **A** | Directly measured by decomposed benchmark on 560 outage instances. Total error is decomposed into initial fix offset vs. INS displacement error. |
| **5** | **T1 Outage Degradation Rates:** 10s outage median RMSE is $76.3$ m; 30s is $580.3$ m; 60s is $3,060.5$ m; 120s is $14,300.0$ m. | **A** | Directly measured across 560 systematically sampled outage windows on active driving segments across all 8 recordings. |
| **6** | **Warm-State Outage Acceleration:** Outages initiated after $\ge 10$ minutes of unconstrained driving diverge $\sim 2\times$ faster than cold-start outages. | **A** | Directly measured across 126 warm-state instances (10s warm median RMSE is $187.1$ m vs. $76.3$ m cold). Demonstrates the compounding impact of pre-existing attitude tilt. |
| **7** | **Planar 2D Kinematic Advantage:** 2D dead reckoning decoupling vertical gravity outperforms 3D INS by 1–2 orders of magnitude. | **A** | Directly measured: A-planar RMSE ranges from $63.2$ km to $656.1$ km over 3 hours, compared to $2,836$ km to $23,565$ km for 3D INS. |
| **8** | **Million-Meter Plausibility Argument:** Accelerometer bias of $0.05$ m/s² and gyro tilt drift of $0.0005$ rad/s produce errors on the order of millions of meters. | **B** | Analytically proven: $\Delta p_{\text{accel}} = \frac{1}{2} b_a t^2$ and $\Delta p_{\text{gravity}} \approx \frac{1}{6} g \dot{\theta} t^3$. Evaluated at $t=9,000$ s yields $2.0 \times 10^6$ m and $5.9 \times 10^8$ m respectively. Controlled bias injection experiments on real data confirm this extreme sensitivity. |
| **9** | **Gravity Leakage as Primary Driver:** Tilt-induced gravity leakage contributes more to position divergence than longitudinal accelerometer bias. | **C** | Strong inference: Supported by the fact that A-planar (which eliminates vertical gravity) reduces error by $97\%-99\%$, and gyro injection experiments exhibit cubic error scaling. However, the exact percentage split between tilt leakage and sensor bias cannot be decoupled without rate-table ground truth. |
| **10** | **A1 Oracle Heading Diagnostic Conclusion:** Perfect heading does not eliminate multi-million-meter drift. | **A** | Directly measured: A1 (initial heading locked to VBOX truth) exhibits almost identical multi-million-meter divergence as A0 ($6.05$M m vs. $6.05$M m in S1; $23.57$M m vs. $23.57$M m in S2). |
| **11** | **A2 Static Gyro Bias Inconclusiveness:** A single offline stationary gyro bias is methodologically invalid. | **A** | Directly measured: A2 improves in rest-started recordings ($0.17\times - 0.44\times$) but explodes in moving-started recordings ($1.6\times - 4.1\times$). Proves that a single static bias cannot represent a dynamic, temperature-varying MEMS sensor. |
| **12** | **Timestamp Misalignment Sensitivity:** Millisecond timestamp lags ($\pm 200$ ms) are not the cause of million-meter drift. | **A** | Directly measured: Controlled timing shifts of $-200$ ms to $+200$ ms altered position RMSE by less than $2.5\% - 8.0\%$, leaving the divergence magnitude completely unchanged. |
| **13** | **AndroSensor Cross-Sensor Axis Alignment:** `ACCELEROMETER` and `GYROSCOPE` share an identical, standard orthogonal body frame. | **D** | Unresolved hypothesis: While individual sensors obey Android conventions, the strong correlation between vehicle yaw rate and `phone_gyro_y` despite gravity being on $Z$ indicates possible cradle tilt or non-standard column mapping. Cannot be confirmed without physical rate-table bench experiments. |

---

## 3. Mandatory Adjustments to Report Language

In response to the claim audit, the following language adjustments are made across all NAVRIS documentation:

1. **Retire "Dominant Failure Mode Proven":**  
   *Old:* "Gravity leakage is proven to be the dominant failure mode."  
   *New:* "The observed A1 and A-planar results are consistent with tilt/gravity leakage being a major contributor, but its relative contribution versus accelerometer bias and other sensor errors has not been uniquely isolated."
2. **Retire "Decorrelation Proven by 103.92°":**  
   *Old:* "103.92° heading RMSE proves gyro drift caused decorrelation."  
   *New:* "The long-run heading estimate becomes effectively uninformative, with wrapped RMSE approaching the theoretical circular-error ceiling ($\pi/\sqrt{3} \approx 103.92^\circ$); this theoretical limit reflects decorrelation but does not by itself isolate the underlying sensor mechanism."
3. **Retire "Drift Rate" Terminology for A0:**  
   *Old:* "Drift rate = Final Error / Duration (m/s)."  
   *New:* "Mean divergence speed over the evaluation horizon (m/s)." (Since position error grows super-linearly, $e(T)/T$ is not a constant rate, but a horizon-dependent average speed).
4. **Remove Unverified Geographic Claims:**  
   *Old:* "NAVRIS is validated for Indian driving conditions."  
   *New:* "NAVRIS currently has an intended Indian adaptation and validation path; Indian-condition validation remains future work following dataset collection."
