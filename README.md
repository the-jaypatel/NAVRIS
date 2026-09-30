# NAVRIS
### Intelligent Navigation & Inertial System

**AI-ML assisted vehicle navigation for maintaining navigation continuity during GNSS degradation and temporary outages.**

[![Test Suite](https://img.shields.io/badge/pytest-176%20passed-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Research Status](https://img.shields.io/badge/Research%20Status-Phase%203.3B%20(Partially%20Supported)-yellow.svg)](docs/phase3_3b_ml_generalization.md)
[![Technical Report](https://img.shields.io/badge/Report-PDF%20(Evaluator%20Copy)-red.svg)](docs/NAVRIS_Technical_Report.pdf)
[![Frontend](https://img.shields.io/badge/Frontend-React%20%7C%20TypeScript-informational.svg)](https://github.com/ptlrudra0/TBA)

> **Smart India Hackathon (SIH) 2026**
> **Problem Statement ID:** 26168
> **Organization:** Indian Space Research Organisation (ISRO) / Department of Space
> **Theme:** Smart Vehicles | **Category:** Software

---

### Quick Links

[📖 Project Overview](#project-overview) • [🛠️ Proposed Solution](#proposed-solution) • [📑 Current Research Status](#current-research-status) • [📊 Benchmark Results](#-benchmark-results--phase-33b-evaluation) • [📄 Technical Report PDF](docs/NAVRIS_Technical_Report.pdf) • [💻 Frontend Repository (TBA)](https://github.com/ptlrudra0/TBA) • [🔬 Reproduce](#-reproduce-the-research)

---

## Project Overview

**NAVRIS** (Intelligent Navigation & Inertial System) is a disciplined, physics-grounded multi-sensor fusion system designed to maintain land-vehicle positioning continuity when satellite navigation (GNSS) is degraded or completely unavailable.

* **The Problem:** Modern vehicle navigation relies almost entirely on GNSS. In urban canyons, underpasses, highway tunnels, and under intentional or environmental interference, satellite signals suffer severe multipath, latency, or total blackout.
* **Why Inertial Navigation & Fusion:** Consumer smartphones contain low-cost micro-electro-mechanical systems (MEMS) accelerometers and gyroscopes. However, unassisted inertial dead reckoning suffers from runaway cubic error growth, rendering raw double integration useless after tens of seconds.
* **Where AI/ML Fits:** Instead of using end-to-end "black-box" neural networks that fail unpredictably out-of-distribution, NAVRIS maintains a mathematically verified classical 15-state Error-State Extended Kalman Filter (ESKF) at its core. Machine learning (XGBoost) is applied strictly to estimate forward longitudinal speed from causal trailing IMU windows, protected by an absolute reference firewall and two-stage statistical gating before filter injection.

---

## Problem Statement

When satellite signals drop out, consumer vehicles must rely on dead reckoning. Consumer-grade MEMS sensors embedded in commodity smartphones introduce extreme stochastic challenges:
1. **Accelerated Divergence:** Sensor bias drifts ($0.02 - 0.05\text{ m/s}^2$) and gyroscope thermal walk ($0.001\text{ rad/s}$) produce cubic position error divergence:
   $$\Delta p(t) \approx \frac{1}{2} b_a t^2 + \frac{1}{6} (b_g \times g) t^3$$
   Without aiding, position errors exceed hundreds of meters within 20–60 seconds, and reach hundreds of kilometers over prolonged driving.
2. **Arbitrary 3D Phone Mounting:** The smartphone rests in arbitrary, uncalibrated orientations inside the vehicle cabin rather than aligned to chassis axes.
3. **Sparse, High-Latency GNSS Fixes:** Consumer smartphone GNSS fixes arrive at low, irregular rates ($\approx 0.1 - 1\text{ Hz}$) with high latency and sudden dropout boundaries.

---

## Proposed Solution

NAVRIS implements a hybrid fusion architecture integrating classical state estimation with causal machine-learned pseudo-measurements:

```text
GNSS Telemetry + Smartphone IMU (10 Hz Synchronized Pipeline)
        ↓
Sensor Preprocessing & Causal Extrinsic Calibration (Gravity Leveling + Mounting Yaw)
        ↓
Inertial Dead Reckoning (Quaternion Mechanization + Somigliana Normal Gravity)
        ↓
15-State Error-State Kalman Filter (P-Propagation via Van Loan Matrix Exponential)
        ↓
Causal Kinematic Constraints (ZUPT Stationary Nulling + NHC Lateral/Vertical Non-Slip)
        ↓
Causal ML Forward-Speed Estimation (Trailing 1.0 s IMU Window, XGBoost Regressor)
        ↓
Reliability Gate (P(reliable) ≥ 0.70) + Innovation Gate (χ² ≤ 9.21)
        ↓
ML Forward-Speed Pseudo-Measurement Update (Fixed Covariance R_ML = 0.50 m²/s²)
        ↓
High-Integrity Navigation Output & Replay Telemetry Service (FastAPI / TBA Frontend)
```

> [!NOTE]
> **Implementation Scope:** All blocks in this architecture are fully implemented and verified in Python research/replay mode on real Oxford IO-VNBD dataset recordings. Real-time on-device embedded smartphone execution is a target for future commercialization and is not claimed as currently deployed.

---

## Key Technologies

The NAVRIS repository contains only technologies that are actually implemented, tested, and verified:

* **Core Language:** Python 3.10+
* **Numerical & Scientific Libraries:** NumPy, SciPy, Pandas
* **Machine Learning:** scikit-learn, XGBoost
* **Backend API & Service:** FastAPI, Uvicorn (serving `/api/v1/nav/*`)
* **State Estimation:** 15-State Continuous-Discrete Error-State Extended Kalman Filter (ESKF)
* **Attitude Representation:** Unit quaternion kinematics with zeroth-order rotational integration
* **Matrix Discretization:** Van Loan matrix exponential method for state transition $(\Phi)$ and process noise $(Q_d)$
* **Kinematic Constraints:** Causal Zero-Velocity Updates (ZUPT) and analytical Non-Holonomic Constraints (NHC)
* **Causal Feature Engineering:** Trailing temporal sliding windows ($W = 10\text{ samples} = 1.0\text{ s}$) with strict anti-circularity
* **Benchmark Telemetry:** Oxford Intelligent Orienting & Vehicle Navigation Benchmark Dataset (IO-VNBD)
* **Frontend Cockpit Integration:** React, TypeScript, Vite ([ptlrudra0/TBA](https://github.com/ptlrudra0/TBA))

---

## Current Research Status

### Phase 3.3B — PARTIALLY SUPPORTED

> The frozen ML forward-speed pseudo-measurement reduced horizontal RMSE relative to the frozen baseline on all seven evaluated recordings, including the two held-out test-domain recordings. However, the magnitude and quality of the effect varied substantially by recording, and large absolute spatial errors remained on several long-duration sequences.

**Explicit Scope & Transparency Qualifications:**
* **Research / Replay Validation:** Evaluated in offline replay on pre-recorded benchmark sequences; not connected to live vehicle hardware.
* **Operational Navigation Accuracy:** Has **NOT** been established.
* **Production Readiness:** Has **NOT** been established.
* **Uniform Generalization:** Has **NOT** been established across arbitrary vehicle models or road conditions.
* **Vehicle / Platform Independence:** Has **NOT** been established (evaluation limited to passenger cars in IO-VNBD).

*Marketing language such as "solved", "breakthrough", "guaranteed", "robust everywhere", "production-ready", or "zero drift" is explicitly rejected in accordance with scientific integrity.*

---

## 📊 Benchmark Results — Phase 3.3B Evaluation

All evaluations benchmark real telemetry from the Oxford **IO-VNBD** dataset across seven driving recordings comparing the frozen classical baseline (Arm A: ESKF + ZUPT + NHC) against the ML pseudo-measurement aided filter (Arm C: Baseline + Causal ML Forward-Speed Pseudo-Measurement with fixed $R_{\text{ML}} = 0.50\text{ m}^2/\text{s}^2$):

| Recording | Domain | Baseline H-RMSE | Arm C H-RMSE | Δ vs Baseline | Classification |
| :--- | :--- | ---: | ---: | :---: | :--- |
| **S1** | Train | 11,580,767.64 m | 2,230.98 m | -99.98% | Large relative reduction |
| **S2** | Train | 42,389,910.67 m | 17,646,092.96 m | -58.37% | High rejection (54.7% NIS rej) |
| **S3A** | Validation | 620,666.46 m | 9,310.67 m | -98.50% | Substantial reduction |
| **S4** | Train | 37,351,981.92 m | 3,060.08 m | -99.99% | Large relative reduction |
| **Y1** | Held-out Test | 29,884,134.11 m | 3,209,429.27 m | -89.26% | Reduced, large residual error |
| **VTA1A** | Validation | 1,480,088.21 m | 21,683.94 m | -98.54% | Substantial reduction |
| **VTA2** | Held-out Test | 31,080.62 m | 4,793.30 m | -84.58% | Bounded spatial error |

> [!CAUTION]
> **Scientific Interpretation:**
> Relative reductions are measured against severely divergent frozen baselines (where unassisted inertial dead reckoning diverges by tens of thousands of kilometers due to uncorrected sensor bias double-integration); **they should not be interpreted as operational navigation accuracy**. Absolute spatial errors of 2.2 km to 17,646 km demonstrate that consumer inertial dead reckoning without external position fixes remains unviable for standalone long-duration navigation.

> [!NOTE]
> **Recording S2 Anomaly:**
> S2 exhibited substantially different ML measurement behavior, including 54.67% NIS rejection and an applied innovation RMSE of 743.69 m/s, demonstrating significant recording/regime dependence.

---

## Research Pipeline

NAVRIS follows a strictly sequential, gate-verified research methodology:

```text
Phase 0   — Dataset Forensics (564 IO-VNBD files audited; 3.6x speed bug resolved)
Phase 1   — Ingestion / Cleaning / Synchronization (10 Hz causal cross-correlation)
Phase 2   — Classical Navigation & Sensor Fusion (15-state continuous-discrete ESKF)
Phase 2.3A — ESKF Validation (Synthetic trajectory validation & Joseph updates)
Phase 2.3A — ZUPT (Causal stationary detector & velocity nulling)
Phase 2.3B — NHC (Lateral & vertical non-slip body constraints)
Phase 3.0 — Preliminary ML Residual Study (Residual observability & leakage audit)
Phase 3.1 — Target Observability Study (Forward-speed vs full velocity vector)
Phase 3.2 — Forward-Speed ML (Causal trailing-window feature extraction)
Phase 3.2A — Domain Robustness (Multi-route driver-isolated validation)
Phase 3.2B — Multi-Source Speed Estimation (Auxiliary kinematic features)
Phase 3.2C — ML Reliability / Gating (Reliability classifier P >= 0.70)
Phase 3.3A — ML Pseudo-Measurement (Controlled 2-route audit & Jacobian verification)
Phase 3.3B — Frozen Generalization Benchmark (7-recording benchmark; PARTIALLY SUPPORTED)
```

---

## Technical Report

The complete, evaluator-facing technical report covering the full theoretical derivation, implementation, and empirical audit is available as a compiled PDF document:

📄 **[Download / View the NAVRIS Technical & Research Report](docs/NAVRIS_Technical_Report.pdf)**

*The report covers all 24 technical dimensions including mathematical formulations, Van Loan discretization derivations, causal ZUPT/NHC Jacobians, ML feature engineering schemas, two-stage innovation gating proofs, empirical multi-recording benchmark tables, trajectory plots, and failure analyses.*

---

## 💻 Frontend & Interactive Cockpit

The NAVRIS research and replay interface is maintained in a separate frontend repository:

🌐 **[TBA — Frontend](https://github.com/ptlrudra0/TBA)**
*Developed and maintained in collaboration with Rudra Patel ([@ptlrudra0](https://github.com/ptlrudra0)).*

### Operational Modes

The project maintains strict boundaries between simulation, replay, and live operation:

1. **Demo / Simulation Mode:**
   * Browser-side deterministic simulation running client-side in TypeScript.
   * Generates synthetic telemetry along a demonstration track to illustrate navigation states, uncertainty envelopes, and simulated outages.
   * Operates completely standalone without backend requirements.

2. **NAVRIS Research / Replay Mode:**
   * Replays real Oxford IO-VNBD vehicle benchmark telemetry processed by the NAVRIS navigation engine (15-state ESKF with ZUPT, NHC, and ML pseudo-measurements).
   * Telemetry, full 10 Hz trajectory frames, error bounds, and innovation events are served by the FastAPI backend (`src/navris/api/main.py`) over `/api/v1/nav/*`.
   * Connected via the `NavrisAdapter` (`src/navrisAdapter.ts`).

3. **Live Navigation:**
   * **Not currently implemented.** NAVRIS does not stream live smartphone hardware in this release. Research replay is strictly distinguished from real-time vehicle guidance.

### Running Backend & Frontend Together

1. **Start the NAVRIS FastAPI Replay Backend** (from this repository):
   ```powershell
   $env:PYTHONPATH = "src"
   python -m uvicorn navris.api.main:app --host 0.0.0.0 --port 8000
   ```
2. **Start the TBA Frontend Dev Server** (from the TBA repository):
   ```bash
   cd ../TBA
   npm run dev
   ```
   Open `http://localhost:5173`. Click the mode button in the header to toggle between **DEMO / SIMULATION** and **RESEARCH / REPLAY**.

---

## What is Real vs. What is Simulated

| Component | Implementation Status | Evidence / Artifact Location |
| :--- | :--- | :--- |
| **IO-VNBD Dataset Pipeline** | **Implemented** | 564 files audited, 3.6× speed bug resolved (`src/navris/ingest.py`) |
| **Sensor Synchronization** | **Implemented** | 10 Hz causal cross-correlation (`src/navris/sync.py`) |
| **15-State ESKF Core** | **Implemented & Frozen** | Van Loan discretization, Joseph covariance update (`src/navris/eskf/`) |
| **Causal Extrinsic Calibration** | **Implemented** | Methods A, B, and D causal alignment (`src/navris/calibration.py`) |
| **Causal ZUPT Module** | **Implemented** | Trailing-window detector + sequential update (`src/navris/zupt.py`) |
| **Causal NHC Module** | **Implemented** | Speed/turn/shock detector + analytical Jacobian (`src/navris/nhc.py`) |
| **ML Forward-Speed Pseudo-Measurement** | **Evaluated (Phase 3.3B)** | Partially supported generalization across 7 routes (`docs/phase3_3b_ml_generalization.md`) |
| **FastAPI Replay Backend** | **Implemented** | 10 Hz replay stream and event endpoints (`src/navris/api/main.py`) |
| **Real-Data A/B Benchmarks** | **Implemented** | 8 IO-VNBD routes evaluated (`scripts/run_gate2_3b_nhc_benchmark.py`, `scripts/phase3/`) |
| **Deterministic Test Suite** | **Implemented** | 176 unit, integration, and synthetic validation tests passing (`tests/`) |
| **Frontend Replay Cockpit** | **Implemented** | React/TypeScript replay dashboard ([ptlrudra0/TBA](https://github.com/ptlrudra0/TBA)) |
| **Live Smartphone Socket** | **Simulated / Replay** | Replay data used for web interface; no live on-device socket in repository |
| **Android On-Device Engine** | **Not Implemented** | Target for future commercialization roadmap |
| **Map Matching** | **Not Implemented** | Pure inertial-GNSS dead reckoning without map constraints |

---

## 🔬 Reproduce the Research

All benchmark experiments are deterministic and fully reproducible from real sensor data:

```bash
# 1. Clone the repository
git clone https://github.com/the-jaypatel/NAVRIS.git
cd NAVRIS

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -e .
pip install pytest matplotlib reportlab

# 4. Verify test suite (176 tests)
python -m pytest -q -o pythonpath=src tests/
# Verified output: 176 passed in ~12.5s

# 5. Run Gate 2.2 Multi-Recording ESKF Benchmark
python scripts/run_gate2_2_benchmark.py

# 6. Run Gate 2.3A Causal ZUPT A/B Benchmark
python scripts/run_gate2_3a_zupt_benchmark.py

# 7. Run Gate 2.3B Real-Data NHC A/B Benchmark
python scripts/run_gate2_3b_nhc_benchmark.py

# 8. Run Phase 3.3B ML Generalization Benchmark
python scripts/phase3/run_phase3_3b_ml_generalization.py

# 9. Generate Final Technical Report PDF
python scripts/generate_pdf_report.py docs/NAVRIS_Technical_Report.pdf
```

---

## Team

* **Jay Patel** ([@the-jaypatel](https://github.com/the-jaypatel))
  *Role:* Lead Researcher & Navigation Engineer
  *Responsibilities:* Backend architecture, ESKF formulation, causal calibration, sensor fusion, error-state observability analysis, benchmark design, AI/ML kinematics research.

* **Rudra Patel** ([@ptlrudra0](https://github.com/ptlrudra0))
  *Role:* Collaborator & Frontend Engineer
  *Responsibilities:* Developer and maintainer of the TBA frontend repository ([https://github.com/ptlrudra0/TBA](https://github.com/ptlrudra0/TBA)), cockpit telemetry visualization, research/replay integration.

* **Durva Patel** ([@Durva46](https://github.com/Durva46))
  *Role:* UI/UX Design Contributor
  *Responsibilities:* Initial UI/UX concepts and cockpit layouts.

* **SIH Team Members:**
  *Participating Student Contributors (Smart India Hackathon 2026)*

---

## References & Acknowledgments

1. **IO-VNBD Dataset:** Oxford Intelligent Orienting & Vehicle Navigation Benchmark Dataset (Oxford Robotics Institute).
2. **ESKF Mechanics:** Sola, J., *"Quaternion kinematics for the error-state Kalman filter"*, arXiv:1711.02508.
3. **Smart India Hackathon (SIH) 2026:** Problem Statement 26168, Indian Space Research Organisation (ISRO) / Department of Space.
