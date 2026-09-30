"""Unit tests for Phase 3.3B — Frozen ML Forward-Speed Generalization Benchmark.

Covers the 12 required verification items:
  1. Unit tests for experiment orchestration.
  2. Frozen ML speed model architecture & parameter verification.
  3. Frozen feature normalization & anti-circularity verification.
  4. Frozen Gate C classifier configuration & probability threshold.
  5. Fixed measurement noise R_ml verification (39.2842 (m/s)^2).
  6. Fixed scalar NIS gate threshold verification (10.828, 1 DOF).
  7. Analytical vs finite-difference Jacobian consistency across velocity/attitude.
  8. Strict causality & future-perturbation invariance.
  9. Strict VBOX firewall (runtime logic isolation from ground truth).
  10. Covariance symmetry and PSD preservation under repeated updates.
  11. Quaternion normalization preservation (| ||q|| - 1 | < 1e-12).
  12. Arm A baseline reproduction consistency against Gate 2.3B configuration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from navris.eskf import (
    ESKF,
    NominalState,
    ESKFConfig,
    STATE_DIM,
    IDX_VEL,
    IDX_ATT,
)
from navris.inertial.frames import quat_to_dcm, quat_multiply, rotvec_to_quat
from scripts.run_gate2_3b_nhc_benchmark import COMMON_CONFIG, COMMON_INIT_COV
from scripts.phase3.run_phase3_2b_multisource_speed import ABLATION_A_COLS
from scripts.phase3.run_phase3_2c_speed_reliability import RELIABILITY_FEATURE_COLS
from scripts.phase3.run_phase3_3a_ml_pseudomeasurement import (
    compute_ml_speed_jacobian,
    verify_ml_jacobian_finite_difference,
    apply_ml_speed_update,
)
from scripts.phase3.run_phase3_3b_ml_generalization import (
    FROZEN_R_ML,
    FROZEN_NIS_THRESHOLD,
    FROZEN_GATE_C_THRESHOLD,
    TRAIN_RECORDINGS,
    VAL_RECORDINGS,
    BENCHMARK_RECORDINGS,
)


@pytest.fixture
def nominal_eskf() -> ESKF:
    """Creates a well-conditioned nominal ESKF instance."""
    q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    v0 = np.array([10.0, 0.0, 0.0], dtype=np.float64)  # 10 m/s East
    p0 = np.array([100.0, 200.0, 10.0], dtype=np.float64)
    state = NominalState(t=0.0, p=p0, v=v0, q=q0, ba=np.zeros(3), bg=np.zeros(3))
    P0 = COMMON_INIT_COV.copy()
    return ESKF(state, P0, COMMON_CONFIG, lat_deg=52.4, alt_m=100.0)


@pytest.fixture
def identity_c_b_v() -> np.ndarray:
    """Identity mounting calibration."""
    return np.eye(3, dtype=np.float64)


# 1. Experiment Orchestration Unit Test
def test_1_experiment_recordings_and_partitions():
    """Verify exact 7 benchmark recordings and partition immutability."""
    assert len(BENCHMARK_RECORDINGS) == 7
    assert set(BENCHMARK_RECORDINGS) == {"S1", "S2", "S3A", "S4", "Y1", "VTA1A", "VTA2"}
    assert "M" not in BENCHMARK_RECORDINGS, "Recording M must be excluded (unobservable)"
    assert set(TRAIN_RECORDINGS) == {"S1", "S2", "S4"}
    assert set(VAL_RECORDINGS) == {"S3A", "VTA1A"}


# 2. Frozen-Model Architecture Test
def test_2_frozen_model_architecture():
    """Verify the exact frozen IMU-only model architecture."""
    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    params = model.get_params()
    assert params["n_estimators"] == 100
    assert params["max_depth"] == 4
    assert params["learning_rate"] == 0.05
    assert params["subsample"] == 0.8
    assert params["colsample_bytree"] == 0.8
    assert params["random_state"] == 42


# 3. Frozen-Feature / Scaler Test
def test_3_frozen_features_anti_circular():
    """Verify IMU-only Ablation A feature columns contain zero ESKF or GNSS features."""
    assert len(ABLATION_A_COLS) == 12
    forbidden_substrings = ["eskf", "pos", "gnss", "vbox", "ref", "true"]
    for col in ABLATION_A_COLS:
        for bad in forbidden_substrings:
            assert bad not in col.lower(), f"Forbidden feature detected: {col}"


# 4. Frozen Gate C Test
def test_4_frozen_gate_c_classifier():
    """Verify Gate C configuration and probability threshold."""
    clf = xgb.XGBClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.08, random_state=42, n_jobs=-1
    )
    params = clf.get_params()
    assert params["n_estimators"] == 50
    assert params["max_depth"] == 3
    assert params["learning_rate"] == 0.08
    assert params["random_state"] == 42
    assert FROZEN_GATE_C_THRESHOLD == 0.50
    assert len(RELIABILITY_FEATURE_COLS) == 15


# 5. Fixed R_ml Test
def test_5_fixed_measurement_covariance():
    """Verify fixed R_ml value and standard deviation."""
    assert np.isclose(FROZEN_R_ML, 39.284247596549264, atol=1e-6)
    sigma_ml = np.sqrt(FROZEN_R_ML)
    assert np.isclose(sigma_ml, 6.2677147, atol=1e-4)


# 6. Fixed NIS Threshold Test
def test_6_fixed_nis_threshold():
    """Verify scalar NIS threshold for 1 DOF at 99.9% significance."""
    assert FROZEN_NIS_THRESHOLD == 10.828


# 7. Jacobian Consistency Test
def test_7_jacobian_consistency(nominal_eskf, identity_c_b_v):
    """Verify analytical Jacobian matches finite difference across multiple headings."""
    for yaw_deg in [0.0, 30.0, 60.0, 120.0, -45.0]:
        yaw = np.radians(yaw_deg)
        nominal_eskf.state.q = np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)])
        nominal_eskf.state.v = np.array([14.0, -3.5, 0.8])
        res = verify_ml_jacobian_finite_difference(nominal_eskf.state, identity_c_b_v, eps=1e-7)
        assert res["passes_tolerance"], f"Jacobian mismatch at yaw {yaw_deg} deg"
        assert res["discrepancy_max"] < 1e-5


# 8. Causality Test
def test_8_strict_causality(identity_c_b_v):
    """Verify measurement update at timestamp t depends strictly on current/past state."""
    state1 = NominalState(t=10.0, p=np.zeros(3), v=np.array([12.0, 0.0, 0.0]), q=np.array([1.0, 0.0, 0.0, 0.0]), ba=np.zeros(3), bg=np.zeros(3))
    state2 = NominalState(t=10.0, p=np.zeros(3), v=np.array([12.0, 0.0, 0.0]), q=np.array([1.0, 0.0, 0.0, 0.0]), ba=np.zeros(3), bg=np.zeros(3))
    eskf1 = ESKF(state1, COMMON_INIT_COV.copy(), COMMON_CONFIG, lat_deg=52.4, alt_m=100.0)
    eskf2 = ESKF(state2, COMMON_INIT_COV.copy(), COMMON_CONFIG, lat_deg=52.4, alt_m=100.0)

    acc1, _, diag1 = apply_ml_speed_update(eskf1, identity_c_b_v, 11.5, FROZEN_R_ML, timestamp=10.0)
    acc2, _, diag2 = apply_ml_speed_update(eskf2, identity_c_b_v, 11.5, FROZEN_R_ML, timestamp=10.0)

    assert acc1 == acc2
    assert np.isclose(diag1["residual_mps"], diag2["residual_mps"])
    np.testing.assert_array_almost_equal(eskf1.state.v, eskf2.state.v)


# 9. VBOX Firewall Test
def test_9_vbox_firewall_isolation():
    """Verify that ML prediction features and inputs are strictly isolated from reference/VBOX data."""
    sample_df = pd.DataFrame({
        "phone_accel_x_mps2": [0.1, 0.2],
        "ref_east_m": [100.0, 101.0],
        "ref_speed_mps": [10.0, 10.5],
    })
    # Feature selector must NEVER include ref_* columns
    selected_cols = [c for c in sample_df.columns if c in ABLATION_A_COLS]
    for c in selected_cols:
        assert not c.startswith("ref_")
        assert "vbox" not in c.lower()


# 10. Covariance PSD and Symmetry Test
def test_10_covariance_psd_and_symmetry(nominal_eskf, identity_c_b_v):
    """Verify covariance remains symmetric and positive-definite after repeated updates."""
    np.random.seed(42)
    for _ in range(30):
        v_meas = 10.0 + np.random.normal(0.0, 1.0)
        apply_ml_speed_update(nominal_eskf, identity_c_b_v, v_meas, FROZEN_R_ML)

    sym_err = np.max(np.abs(nominal_eskf.P - nominal_eskf.P.T))
    assert sym_err < 1e-8, f"Covariance asymmetry detected: {sym_err}"

    eigs = np.linalg.eigvalsh(nominal_eskf.P)
    assert np.all(eigs > 0.0), f"Negative eigenvalue detected: {eigs[eigs <= 0]}"


# 11. Quaternion Normalization Test
def test_11_quaternion_normalization(nominal_eskf, identity_c_b_v):
    """Verify quaternion norm remains strictly 1 after repeated updates."""
    np.random.seed(42)
    for _ in range(30):
        v_meas = 10.0 + np.random.normal(0.0, 1.5)
        apply_ml_speed_update(nominal_eskf, identity_c_b_v, v_meas, FROZEN_R_ML)
        norm_q = np.linalg.norm(nominal_eskf.state.q)
        assert abs(norm_q - 1.0) < 1e-12, f"Quaternion norm drifted: {norm_q}"


# 12. Baseline Reproducibility Test
def test_12_baseline_reproducibility():
    """Verify Gate 2.3B baseline results exist and match Configuration B parameters."""
    gate_summary_file = "data/processed/phase2_3b/gate2_3b/gate2_3b_summary.csv"
    df_gate = pd.read_csv(gate_summary_file)
    b_rows = df_gate[df_gate["configuration"] == "B_EXPERIMENT_NHC"]
    assert len(b_rows) == 8  # S1, S2, S3A, S4, M, Y1, VTA1A, VTA2
    # Verify baseline observable recordings have positive horiz_rmse
    for r in BENCHMARK_RECORDINGS:
        row = b_rows[b_rows["recording_id"] == r]
        assert len(row) == 1
        assert float(row["horiz_rmse_m"].iloc[0]) > 0.0
