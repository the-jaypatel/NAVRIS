# NAVRIS — Intelligent Navigation & Inertial System

**AI-ML based Intelligent Dead Reckoning system for seamless navigation**  
*SIH 2026 Problem Statement:* 26168  
*License:* Apache License 2.0  

---

## 1. Overview

NAVRIS is an ongoing applied research effort to develop a robust, smartphone-grade inertial navigation and dead reckoning system capable of providing continuous vehicle positioning during GNSS degradation and outages.

Modern consumer smartphones are equipped with low-cost MEMS inertial measurement units (accelerometers and gyroscopes) that suffer from significant sensor bias, thermal drift, axis cross-coupling, and unpredictable mounting orientations. NAVRIS investigates how classical strapdown inertial navigation, Error-State Kalman Filtering (ESKF), and future machine-learned sensor corrections can be combined to bridge multi-second and multi-minute GNSS gaps.

---

## 2. The Navigation Problem

Unassisted consumer smartphone inertial dead reckoning suffers from rapid, runaway cubic error divergence:
1. **Low-Cost MEMS IMU Drift:** Integrating uncorrected accelerometer biases ($0.02 - 0.05\text{ m/s}^2$) and gyroscope drift ($0.0005\text{ rad/s}$) causes horizontal position errors to diverge into hundreds of meters within $10 - 20$ seconds, and millions of meters over multi-hour drives.
2. **Sparse Consumer GNSS:** Android and smartphone GNSS fixes arrive at low sampling rates (typically 0.1 Hz / once every $9 - 10$ seconds) with frequent signal outages in urban canyons, tunnels, and underpasses.
3. **Arbitrary Sensor Orientation:** Unlike automotive-grade IMUs bolted rigidly to the chassis, smartphones are placed in phone mounts, cup holders, or consoles with arbitrary 3D mounting orientations that must be estimated causally at runtime.

---

## 3. Current Architecture

The NAVRIS classical navigation pipeline consists of modular subsystems:

```
[Raw Smartphone Data] ---> [Ingestion & Sanitization] ---> [Stationary Leveling (Method A)]
                                       |                                    |
                                       v                                    v
                            [10 Hz Synchronization]          [Mounting Alignment (Method B)]
                                       |                                    |
                                       v                                    v
                           [Sparse GNSS Fix Extraction]   [Orthonormal DCM R_s^v (Method D)]
                                       |                                    |
                                       +------------------+-----------------+
                                                          |
                                                          v
                                            [Phase 2.3A Frozen ESKF Core]
                                            - 16-State Error-State Filter
                                            - True Earth Gravity Integration
                                            - Chi-Square Innovation Gating
                                                          |
                                                          v
                                            [Estimated Position, Velocity, Attitude]
```

- **Coordinates & Projections (`src/navris/coords.py`):** WGS-84 geodetic coordinates to local East-North-Up (ENU) tangent-plane projections.
- **Ingestion & Sanitization (`src/navris/ingest.py`):** Canonical units, timestamp monotonicity validation, and correct speed unit handling (raw m/s).
- **Offline Benchmark Synchronization (`src/navris/sync.py`):** Dynamic cross-correlation lag estimation aligning reference CAN/VBOX data with smartphone streams.
- **Causal Extrinsic Calibration (`src/navris/calibration.py`):**
  - *Method A:* Stationary gravity leveling with variance gating to reject dynamic braking transients.
  - *Method B:* Dynamic horizontal forward-axis alignment during straight-line cruise.
  - *Method C:* Planar kinematic turn cross-product ($\mathbf{a}_{\text{dyn}} \times \boldsymbol{\omega}$).
  - *Method D:* Right-handed orthonormal basis transformation ($R_{\mathcal{S}}^{\mathcal{V}}$) and gyroscope axis mapping.
- **Strapdown Mechanization (`src/navris/inertial/`):** Full 3D quaternion-based attitude propagation, Coriolis and normal gravity modeling (`gravity.py`), and cold-start leveling.
- **Error-State Kalman Filter (`src/navris/eskf/`):** 16-dimensional continuous-discrete error-state formulation tracking position ($3$), velocity ($3$), attitude error ($3$), accelerometer bias ($3$), gyroscope bias ($3$), and scalar gravity parameter ($1$).

---

## 4. Repository Structure

```text
NAVRIS/
├── src/
│   └── navris/
│       ├── coords.py        # Geodetic / local ENU projections
│       ├── schema.py        # Canonical sensor schemas & metadata
│       ├── ingest.py        # Raw data parsing & unit sanitization
│       ├── sync.py          # Lag estimation & 10 Hz interpolation
│       ├── windowing.py     # Causal temporal sliding windows
│       ├── splitting.py     # Driver-isolated train/val/test splits
│       ├── pipeline.py      # End-to-end dataset processing
│       ├── calibration.py   # Methods A, B, C, D causal extrinsic calibration
│       ├── inertial/        # Mechanization, frames, gravity, strapdown
│       └── eskf/            # 16-state Error-State Kalman Filter core (Frozen)
│
├── tests/                   # Deterministic test suite (84 tests passing)
├── scripts/                 # Reproducibility, audit, and benchmark scripts
├── docs/                    # Complete research reports (Phase 0 -> Gate 2.1)
├── data/
│   ├── README.md            # Dataset acquisition and structure guide
│   └── manifest/            # Benchmark catalog (recordings_manifest.csv)
│
├── pyproject.toml           # Standard Python package configuration
├── requirements.txt         # Runtime and development dependencies
├── LICENSE                  # Apache License 2.0
└── README.md
```

---

## 5. Current Research Status

> **Current Milestone: Phase 2.3B Gate 2.1 — CONDITIONAL PASS / PARTIAL OBSERVABILITY**

The project has rigorously audited and frozen the classical navigation baseline before introducing any machine learning components:
- **Phase 0 (Dataset Forensics):** Audited the 564 raw CSV files of the IO-VNBD dataset.
- **Phase 1 (Data Ingestion & Pipeline):** Implemented coordinate projection, sanitization, and 10 Hz synchronization.
- **Phase 2.1 (Sensor Truth Validation):** Confirmed smartphone sensor vertical axis and channel characteristics.
- **Phase 2.2 (Raw Inertial Baseline):** Measured unconstrained 3D strapdown dead reckoning (A0 baseline) divergence across multi-hour drives.
- **Phase 2.3A (ESKF Core):** Implemented and synthetically verified the 16-state ESKF.
- **Phase 2.3B Gate 1.5A & 1.5B (Forensics & Synchronization):** Discovered and corrected a 3.6x speed ingestion defect and multi-second acquisition time lags across recordings.
- **Phase 2.3B Gate 2.1 (Causal Sensor-Frame Calibration):** Demonstrated causal gravity leveling and horizontal mounting alignment on benchmark recording S1.

---

## 6. Dataset

NAVRIS is benchmarked against the public **IO-VNBD (Input-Output Vehicle Navigation Benchmark Dataset)**:
- **Source:** [onyekpeu/IO-VNBD (GitHub)](https://github.com/onyekpeu/IO-VNBD)
- **Reference Ground Truth:** Ford Fiesta VBOX Video HD2 logger and CAN bus at 10 Hz.
- **Smartphone Devices:** Consumer smartphones (Huawei P20 Pro, Samsung Galaxy S8, Motorola Moto G7 Power) recording IMU, magnetometer, and GNSS at 10 Hz across the UK, France, and Nigeria.

Raw and bulk interpolated dataset files are external and excluded from git tracking. See [`data/README.md`](data/README.md) for instructions on downloading and configuring the dataset.

---

## 7. Current Results (Gate 2.1 S1 Ablation)

On benchmark recording `S1` ($t_0 = 156.0\text{ s}$), the controlled 4-mode ablation yielded:

| Mode | Initial Speed | 60s Horiz RMSE | 60s Max Error | 60s Accepted Fixes | Early Turn Fix 5 Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mode A: Baseline (Uncalibrated)** | $11.94\text{ m/s}$ | $1,761.11\text{ m}$ | $5,188.09\text{ m}$ | $4 / 7$ ($57\%$) | REJECTED (Diverged, err $= 486.8\text{ m}$) |
| **Mode B: Causal Calibration (Method D)** | $11.94\text{ m/s}$ | **$48.83\text{ m}$** | **$178.28\text{ m}$** | **$7 / 7$ ($100\%$)** | **ACCEPTED (Recovered, err $= 3.34\text{ m}$)** |
| **Mode C: Causal Calib + Disp Velocity** | $13.95\text{ m/s}$ | $50.35\text{ m}$ | $180.09\text{ m}$ | **$7 / 7$ ($100\%$)** | **ACCEPTED (Recovered, err $= 3.60\text{ m}$)** |
| **Mode D: E2 Oracle Reference** | $11.94\text{ m/s}$ | $48.33\text{ m}$ | $170.91\text{ m}$ | **$7 / 7$ ($100\%$)** | **ACCEPTED (Reference, err $= 3.56\text{ m}$)** |

**Key Findings:**
1. Causal sensor-frame calibration and gyro remapping account for **$97.2\%$** of the baseline error reduction ($1,761\text{ m} \to 48.8\text{ m}$).
2. Catastrophic divergence during the early dynamic turn ($t=192\text{ s}$) was caused by unmapped gyroscope axes tilting pitch by $-68^\circ$. With corrected calibration, nominal attitude remains stable ($[-2.8^\circ, +10.3^\circ]$) and all 5 early fixes are accepted with NIS $\le 2.39$.
3. Initial velocity magnitude variation has a negligible effect ($< 1.5\text{ m}$ difference between Mode B and Mode C).

---

## 8. Reproducibility

### Installation

Clone the repository and set up a Python 3.10+ virtual environment:

```bash
git clone https://github.com/the-jaypatel/NAVRIS.git
cd NAVRIS
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

### Running Tests

Execute the deterministic test suite:

```bash
python -m pytest -q
```

All 84 tests should pass without external data dependencies.

### Running Controlled Experiments

Reproduce the Gate 2.1 S1 calibration and ablation (requires `S1_sync.parquet`):

```bash
python scripts/run_gate2_1_s1.py
```

---

## 9. Research Roadmap

- [x] **Phase 0:** Raw IO-VNBD dataset empirical validation and manifest generation.
- [x] **Phase 1:** Standardized geodetic-to-local ingestion, unit sanitization, and 10 Hz synchronization.
- [x] **Phase 2.1:** Sensor truth and gravity vector verification.
- [x] **Phase 2.2:** Pure 3D strapdown dead reckoning divergence characterization.
- [x] **Phase 2.3A:** 16-State ESKF mathematical core implementation and synthetic verification.
- [x] **Phase 2.3B Gate 1.5:** Speed unit forensics and offline benchmark timeline lag estimation.
- [x] **Phase 2.3B Gate 2.1:** Causal sensor-frame calibration and dynamic turn ablation on S1.
- [ ] **Phase 2.3B Gate 2.2:** Multi-recording generalization of classical calibration.
- [ ] **Phase 2.5:** Measurement constraint aiding: Non-Holonomic Constraints (NHC) and Zero Velocity Updates (ZUPT).
- [ ] **Phase 3:** Machine Learning (AI-ML) augmentation for learned pseudo-measurements and adaptive IMU error compensation.

---

## 10. Disclaimer & Scientific Status

NAVRIS is active scientific research. It has **NOT** solved GNSS-denied navigation, and AI/ML models are not yet integrated into the runtime filter. 

Classical loose GNSS/IMU integration with 0.1 Hz updates remains sensitive to long-term drift in the absence of Non-Holonomic Constraints (NHC) or wheel-speed updates. All documented results represent verified empirical baselines established strictly within the audited constraints.
