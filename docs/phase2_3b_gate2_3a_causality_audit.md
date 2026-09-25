# NAVRIS — Phase 2.3B Gate 2.3A: Causality Audit Report

**Date**: September 2026  
**Auditor**: NAVRIS Scientific Verification Team  
**Scope**: Verification of strictly causal execution for Phase 2.3B Gate 2.3A Causal ZUPT Module  
**Status**: **PASS (100% Causal Integrity Verified)**  

---

## 1. Source-Level Inspection of Runtime ZUPT Components

### 1.1 Causal Stationary Detector (`src/navris/zupt.py::CausalStationaryDetector`)
- **Sample Ingestion**: The detector method `update(f_meas_b, omega_meas_b)` takes strictly the instantaneous 3-vector measurements from the current time step $t_k$.
- **Window Buffering**: Storage is implemented via Python's standard `collections.deque(maxlen=window_size)`. Samples are appended causally on arrival (`self.acc_window.append(f)`).
- **Zero Future Indexing**: All statistics (`np.mean`, `np.std`) are evaluated strictly over the trailing window:
  $$\mathcal{W}_k = \{ \mathbf{f}_{k - W + 1}, \dots, \mathbf{f}_k \}$$
  There is zero forward indexing (`idx > k` or `k + 1`), zero reverse indexing into future arrays, and zero centered window filtering.
- **Dwell Counter**: The dwell accumulation counter `consecutive_entry_hits` is incremented sequentially upon meeting entry thresholds. If any exit threshold is tripped at sample $k$, the counter resets immediately to 0 and `is_stationary` drops to `False`.

### 1.2 ZUPT Measurement Updater (`src/navris/zupt.py::apply_zupt_update`)
- **Measurement Inputs**:
  - The measurement is a constant physical zero-velocity vector: $\mathbf{z} = [0, 0, 0]^T\text{ m/s}$.
  - The predicted state is taken strictly from the current filter nominal state: $\hat{\mathbf{v}} = \text{eskf.state.v}$.
  - The innovation residual is strictly causal: $\mathbf{r} = -\hat{\mathbf{v}}$.
- **Measurement Jacobian $\mathbf{H}$**:
  - Evaluated at the current time step: $\mathbf{H} = [\mathbf{0}_{3\times 3}, \mathbf{I}_{3\times 3}, \mathbf{0}_{3\times 9}]$.
- **No Evaluation / Ground-Truth Contamination**:
  - Neither VBOX reference velocity (`ref_speed_mps`), reference position (`ref_east_m`, `ref_north_m`), nor future evaluation outcomes are passed into or accessible by `apply_zupt_update`.

---

## 2. Temporal & Calibration Boundary Audit

1. **Calibration Precedence**:
   - For all recordings, causal calibration decision time satisfies:
     $$t_{\text{decision}} \le t_{\text{eval\_start}}$$
   - Sensor leveling ($\mathbf{q}_{\text{level}}$), mounting yaw ($\psi_{\text{mount}}$), and gyro mapping are computed strictly from data where $t \le t_{\text{decision}}$.
2. **Evaluation Window Invariance**:
   - The evaluation interval $[t_{\text{eval\_start}}, t_{\text{eval\_end}}]$ is identical between baseline and ZUPT configurations.
3. **Monotonic Progression**:
   - The filter processes IMU samples and ZUPT updates in strictly non-decreasing timestamp order ($t_k \ge t_{k-1}$).

---

## 3. Strict Code Separation Audit

- **Offline Dataset Characterization**:
  The Phase 1 dataset stationary audit (`data/processed/phase2_3b/gate2_3a/stationary_audit.csv`) utilized VBOX reference speed (`ref_speed_mps < 0.2`) strictly for offline characterization.
- **Runtime Code Isolation**:
  The runtime module `src/navris/zupt.py` has **zero imports** or dependencies on:
  - `ref_speed_mps`
  - VBOX ground truth files
  - Future stationary labels
  - Post-hoc evaluation metrics
- **Frozen Parameters**:
  Both `ZUPTDetectorConfig` thresholds and `DEFAULT_R_ZUPT` are pre-declared constants frozen before real-data benchmarking.

---

## 4. Machine-Checkable Causality Verification Checklist

| Criterion | Requirement | Verification Method | Status |
| :--- | :--- | :--- | :---: |
| **No Future IMU Indexing** | Trailing window only ($t \le t_k$) | Source audit of `CausalStationaryDetector.update` | **PASS** |
| **No Centered Windows** | Backward-looking buffer only | Deque maxlen inspection; trailing stats only | **PASS** |
| **No Reference Contamination** | Zero VBOX / ground-truth in runtime | Symbol search: 0 occurrences in `src/navris/zupt.py` | **PASS** |
| **No Future Labels** | Purely sensor-driven detection | Logic inspection of `update` | **PASS** |
| **Monotonic Updates** | $t_k \ge t_{k-1}$ enforced | Strapdown loop inspection | **PASS** |
| **Pre-Declared Covariance** | $\mathbf{R}_{\text{zupt}}$ frozen prior to evaluation | Static constant `DEFAULT_R_ZUPT` | **PASS** |
| **Frozen ESKF Core** | `src/navris/eskf/*` untouched | `git diff -- src/navris/eskf/` produces 0 lines | **PASS** |

**OVERALL CAUSALITY AUDIT VERDICT: PASS**
