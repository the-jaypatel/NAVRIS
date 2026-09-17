# NAVRIS Phase 1: Data Ingestion, Cleaning, Normalization & Windowing Pipeline

**Project:** NAVRIS — Intelligent Navigation & Inertial System  
**SIH 2026 Problem Statement:** 26168 — AI-ML based Intelligent Dead Reckoning system for seamless navigation  
**Dataset Base:** IO-VNBD (Coventry University, 2020)  
**Package:** `src/navris/`  
**Automated Tests:** `tests/` (20 passed tests)  
**Pipeline CLI:** `scripts/run_phase1_pipeline.py`

---

## 1. Raw → Standardized Transformation

The NAVRIS ingestion pipeline converts raw IO-VNBD CSV files into a uniform, physically sound SI representation without modifying raw data.

### 1.1 Ingestion & File Handling (`src/navris/ingest.py`)
* **Encoding Resilience:** Raw files are dynamically decoded using `latin1` / `cp1252` to preserve degree symbols (`°`), superscript two (`²` in `m/s²`), and micro (`μ` in `μT`) without Unicode decode failures.
* **Header Sanitization:** Column headers have whitespace stripped, and naming inconsistencies (e.g. `GYROSCOPE X/Y/Z` vs. `GYROSCOPE Yaw/Pitch/Roll`, `ORIENTATION (Azimuth)` vs. `ORIENTATION (Yaw)`) are normalized using case-insensitive regular expressions.
* **Corrupted CSV Edge-Cases:** The anomalous file `S-A4.csv` (which contains an unquoted empty comma at column position 6 shifting subsequent values) is automatically sanitized during loading.
* **Satellite String Sanitization:** Telemetry strings in `GPS SATELLITES IN RANGE` that were corrupted by Microsoft Excel into calendar dates (e.g. `"10-Nov"` for 10 used / 11 in view, `"Aug-20"` for 8 used / 20 in view) are parsed and converted back into numerical counts (`phone_gps_satellites_used`, `phone_gps_satellites_in_view`).

### 1.2 Physical Unit Conversions
All measurements are strictly converted to standard SI units:
* **Speed / Velocity:** Converted from km/h to m/s (speed_mps = speed_kmh / 3.6).
* **Acceleration:** Converted from g to m/s² (a_mps2 = a_g * 9.80665).
* **Angles & Headings:** Converted from degrees to radians in the standard range [0, 2pi) or [-pi, pi] (theta_rad = theta_deg * pi / 180.0).
* **Angular Velocities:** Converted from deg/s to rad/s (omega_radps = omega_degps * pi / 180.0).
* **Elevation / Altitude:** The CAN field `Height (km)` was empirically proven in Phase 0 to be stored in meters (e.g. 110.19 m in Coventry); it is correctly mapped to `ref_alt_m` without erroneous scaling.
* **Throttle:** Converted from the mislabeled `Accelerator Pedal Position (0 or 1)` into percentage [0, 100%].

---

## 2. Coordinate System & Geodetic Transformation

To enable linear dynamics and Newtonian dead reckoning, geodetic coordinates are projected into a local Cartesian tangent plane.

### 2.1 Convention: Local Cartesian ENU (East-North-Up)
NAVRIS adopts the standard **Local Cartesian ENU** coordinate system (ISO 80000-3 / IEEE / ROS REP-103):
* **X-axis (E):** Points East along local parallel [meters].
* **Y-axis (N):** Points North along local meridian [meters].
* **Z-axis (U):** Points Up along ellipsoidal normal [meters].

### 2.2 Mathematical Transformation (`src/navris/coords.py`)
Using the standard WGS-84 reference ellipsoid parameters (a = 6378137.0 m, f = 1/298.257223563, e² = 2f - f²):

1. **Origin Selection:** The first valid position (lat0, lon0, alt0) in the recording is designated as the local reference origin, mapping to (E=0, N=0, U=0).
2. **Geodetic to ECEF:**
   N(phi) = a / sqrt(1 - e² * sin²(phi))
   X = (N + h) * cos(phi) * cos(lambda)
   Y = (N + h) * cos(phi) * sin(lambda)
   Z = (N * (1 - e²) + h) * sin(phi)
3. **ECEF to Local ENU:**
   Rotate (X - X0, Y - Y0, Z - Z0) by rotation matrix R_ECEF->ENU determined by (lat0, lon0).
4. **Sub-millimeter Closed-Form Roundtrips:** Verified by `test_geodetic_ecef_roundtrip` and `test_enu_geodetic_roundtrip` with < 1 mm numerical closure error.

---

## 3. Timestamp Synchronization

### 3.1 Common 10 Hz Timeline (`src/navris/sync.py`)
* Vehicle CAN logging and smartphone IMU are both sampled at 10 Hz.
* The synchronized pipeline establishes a uniform time grid t_k = k * 0.1 s over the overlapping interval [max(t_ref_0, t_phone_0), min(t_ref_end, t_phone_end)].
* **Continuous Kinematics:** Accelerations, angular rates, wheel speeds, and positions are aligned via linear interpolation.
* **Angular State (Heading):** Continuous phase unwrapping is applied prior to interpolation to eliminate artificial 0 <-> 2pi wrap-around discontinuities.
* **Discrete States:** Transmission gear, handbrake, and GNSS status flags are mapped using nearest-neighbor lookup.

---

## 4. Sparse GNSS Handling & Sample-and-Hold Detection

One of the most critical empirical findings of Phase 0 was that smartphone GPS does **not** update at 10 Hz, nor is it consistently 1 Hz across drivers. In several recordings (Drivers A, B, D), the smartphone OS only pushes position updates every ~9–10 seconds, holding the coordinates constant in between.

### 4.1 Strict Preservation of Sparse Observations
* **No Artificial GPS Interpolation:** The pipeline **never** interpolates GPS coordinates across update gaps.
* **Sample-and-Hold Telemetry (`phone_gps_is_new_fix`):**
  * `phone_gps_is_new_fix = True` exclusively on epochs where the smartphone hardware delivered a fresh position fix.
  * `phone_gps_is_new_fix = False` during inter-update sample-and-hold epochs.
* **Sparse Observation Channels (`phone_gps_obs_east_m`, `phone_gps_obs_north_m`, `phone_gps_obs_speed_mps`):**
  * Contain the valid measurement values on new fix arrivals.
  * Are populated with `NaN` during all intermediate epochs.
  * This allows downstream Kalman Filters (EKF/UKF) and ML models to perform sparse measurement updates without falsely assuming 10 Hz GPS availability.

---

## 5. Driver F Compatibility (18-Column Support)

* Recordings from Driver F in France (`S-T1` through `S-T9`) were captured with a Motorola Moto G7 Power where magnetometer and fused orientation angles were unrecorded.
* `ingest_smartphone_data` automatically detects the column schema:
  * For 18-column files, `phone_has_mag` and `phone_has_orientation` are set to `False`.
  * Magnetometer (`phone_mag_x/y/z_uT`) and orientation (`phone_orient_azimuth/pitch/roll_rad`) are explicitly populated with `NaN`.
  * IMU accelerometer, gyroscope, gravity, and GPS channels are ingested without interruption.
  * Verified by `test_driver_f_compatibility`.

---

## 6. Reference Trajectory & Signal Definition

Data from the Racelogic VBOX Video HD2 CAN logger is designated as the **reference trajectory**:
* **Signals:**
  * `ref_east_m`, `ref_north_m`, `ref_up_m`: High-precision reference position.
  * `ref_speed_mps`, `ref_vertical_speed_mps`: Doppler radar GPS velocity.
  * `ref_heading_rad`: GPS true course over ground.
  * `ref_yaw_rate_radps`: Chassis gyro yaw rate.
  * `ref_wheel_speed_fl/fr/rl/rr_radps`: Four-wheel encoder angular velocities.
* **Terminology & Limitations:**
  * This signal is termed the **reference trajectory**, *not* "absolute ground truth".
  * Known physical limitations include:
    1. **Antenna Lever-Arm:** Roof-mounted VBOX antenna vs. dashboard-mounted phone introduces a ~1.5–2.0 m spatial offset.
    2. **GNSS Dropouts:** In 12 vehicle files, tall buildings/foliage cause total satellite loss (`No of GPS Satellites Available == 0.0`), detected and flagged by `ref_gps_dropout = True`.

---

## 7. Causal Windowing System (`src/navris/windowing.py`)

For subsequent dead-reckoning ML models (e.g. velocity regressors or step-length estimators), a configurable causal sliding window generator is provided.

### 7.1 Configuration & Causality
* **Default Parameters:** Window Length W = 20 samples (2.0 s at 10 Hz), Stride S = 10 samples (1.0 s at 10 Hz).
* **Strict Causality:** Window i spanning indices [k - W + 1, ..., k] contains exclusively past and present observations. No future samples are included in the feature tensor.
* **Feature Tensor:** Shape (N_windows, 20, 12) containing 3-axis accelerometer, 3-axis gyroscope, 3-axis gravity, and 3-axis magnetometer.
* **Supervised Targets:**
  * `disp_east_m`, `disp_north_m`, `disp_up_m`: Total displacement vector over the 2.0s window.
  * `speed_mps`: Instantaneous vehicle speed at window termination.
  * `heading_rad`: Vehicle heading at window termination.
  * `delta_heading_rad`: Angular yaw displacement over the window wrapped to [-pi, pi].
* **Metadata Retention:** Every window stores `recording_id`, `driver`, `vehicle`, `phone`, `region`, `split`, `start_time_s`, and `end_time_s`.
* **Storage:** Windows are serialized as compressed `.npz` files in `data/processed/windows/`.

---

## 8. Group-Based Dataset Splitting & Leakage Prevention (`src/navris/splitting.py`)

### 8.1 Zero-Leakage Guarantee
* Splitting is performed strictly at the **recording level**.
* All windows belonging to a given recording ID are placed into the same partition.
* Downstream tests enforce that:
  Train intersect Val = Empty, Train intersect Test = Empty, Val intersect Test = Empty.
* `verify_no_split_leakage()` raises a `ValueError` if an overlap is detected.

### 8.2 Partition Strategies
1. **Standard Recording Split:** Deterministic cryptographic hashing (MD5) partitions recordings into 70% Train, 15% Validation, 15% Test.
2. **Unseen-Driver Split:** Trains on Drivers E, A, C; validates on B, D; tests on out-of-distribution Drivers F (France), G (Nigeria), and H (Volvo UK).
3. **Unseen-Region Split:** Trains on UK Midlands; validates on Peak District/Oxford; tests on international roads in France and Nigeria.
4. **Unseen-Vehicle/Phone Split:** Trains on Ford Fiesta (Huawei P20); tests on Renault Megane (Motorola Moto G7) and Volvo XC70 (Blackberry Priv).

---

## 9. Automated Test Suite (`tests/`)

The test suite contains 20 automated unit and integration tests executed with `pytest`:

| Test Module | Test Case | Purpose | Status |
|:---|:---|:---|:---:|
| `test_coords.py` | `test_geodetic_ecef_roundtrip` | Verifies WGS-84 to ECEF and reverse closed-form accuracy (< 1 mm) | PASSED |
| `test_coords.py` | `test_enu_origin_maps_to_zero` | Verifies origin maps to exact (0, 0, 0) in ENU | PASSED |
| `test_coords.py` | `test_enu_directional_monotonicity` | Verifies North increases with latitude, East with longitude | PASSED |
| `test_coords.py` | `test_enu_geodetic_roundtrip` | Verifies local ENU to WGS-84 round-trip | PASSED |
| `test_units_and_schema.py` | `test_speed_conversion_factors` | Verifies km/h to m/s conversion and roundtrip | PASSED |
| `test_units_and_schema.py` | `test_acceleration_conversion` | Verifies g to m/s² conversion (1g = 9.80665 m/s²) | PASSED |
| `test_units_and_schema.py` | `test_angle_conversion` | Verifies degree to radian conversion | PASSED |
| `test_units_and_schema.py` | `test_schema_definitions_integrity` | Verifies schema column counts and required channel presence | PASSED |
| `test_ingest.py` | `test_parse_satellite_string_standard` | Verifies parsing of '27 / 28' satellite format | PASSED |
| `test_ingest.py` | `test_parse_satellite_string_excel_date_corruption` | Verifies recovery of Excel date corrupted strings ('10-Nov', 'Aug-20') | PASSED |
| `test_ingest.py` | `test_parse_satellite_string_empty` | Verifies graceful NaN handling for empty satellite fields | PASSED |
| `test_ingest.py` | `test_ingest_reference_v_s1` | Tests full ingestion of reference VBOX data | PASSED |
| `test_ingest.py` | `test_ingest_smartphone_s_s1` | Tests smartphone 24-col ingestion and sample-and-hold detection | PASSED |
| `test_ingest.py` | `test_driver_f_compatibility` | Verifies 18-col Driver F file ingestion with explicit NaN masking | PASSED |
| `test_sync.py` | `test_sync_v_s1_timeline` | Verifies strict 10 Hz monotonicity and sparse GNSS preservation | PASSED |
| `test_windowing.py` | `test_causal_window_generation` | Verifies causal window shapes, strides, and displacement targets | PASSED |
| `test_splitting_and_leakage.py` | `test_leakage_detection_raises_error` | Asserts that `verify_no_split_leakage` raises error on overlap | PASSED |
| `test_splitting_and_leakage.py` | `test_standard_split_zero_leakage` | Verifies standard 70/15/15 split has zero recording overlap | PASSED |
| `test_splitting_and_leakage.py` | `test_unseen_driver_split_zero_leakage` | Verifies unseen-driver split isolation | PASSED |
| `test_pipeline_e2e.py` | `test_pipeline_single_sync_recording` | End-to-end integration test creating Parquet and NPZ | PASSED |

---

## 10. Pipeline Usage & CLI Commands

### 10.1 Running the Automated Test Suite
```powershell
python -m pytest -v
```

### 10.2 Executing the Preprocessing Pipeline
```powershell
# Run pipeline on a subset of recordings
python scripts/run_phase1_pipeline.py --limit 10

# Run pipeline with a specific group split strategy
python scripts/run_phase1_pipeline.py --strategy unseen_driver

# Run pipeline with custom window size (e.g. 3.0s window, 1.5s stride)
python scripts/run_phase1_pipeline.py --window-len 30 --stride 15
```

### 10.3 Programmatic API Usage
```python
import pandas as pd
from navris.ingest import ingest_smartphone_data, ingest_reference_data
from navris.sync import synchronize_recordings
from navris.windowing import generate_causal_windows, WindowConfig

# 1. Ingest reference and phone data
df_ref, origin = ingest_reference_data('data/raw/IO-VNBD/.../V-S1.csv')
df_phone, _ = ingest_smartphone_data('data/raw/IO-VNBD/.../S-S1.csv', origin_geodetic=origin)

# 2. Synchronize to 10 Hz common timeline
df_sync = synchronize_recordings(df_ref, df_phone, dt=0.1)

# 3. Generate causal windows
config = WindowConfig(window_length=20, stride=10, dt=0.1)
windows = generate_causal_windows(df_sync, {'recording_id': 'S1', 'split': 'train'}, config)
```

---

## 11. Known Limitations & Recommendations for Phase 2

1. **VBOX Roof vs. Phone Dashboard Lever-Arm:** The ~1.5–2.0 m lever arm between the roof GNSS antenna and internal phone mount introduces minor positional offsets that Phase 2 EKF state estimators can model as a static extrinsic translation.
2. **Sparse GNSS Latency:** Because smartphone GPS updates are held for up to 9–10 seconds in some recordings, dead reckoning integration in Phase 2 must rely on high-rate inertial velocity/displacement estimation during GPS-deprived periods.
3. **Driver F Incomplete IMU:** Models requiring magnetometer orientation cannot be evaluated on Driver F recordings `S-T1`–`S-T9`; evaluation protocols must gate orientation features on `phone_has_mag` / `phone_has_orientation`.
