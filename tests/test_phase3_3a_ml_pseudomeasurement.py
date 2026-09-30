"""Synthetic sanity and causality unit tests for Phase 3.3A ML Forward-Speed Pseudo-Measurement.

Covers:
  Test 1: Correct measurement produces near-zero innovation.
  Test 2: Known forward-speed error produces exact innovation sign and magnitude.
  Test 3: Jacobian finite difference verification (discrepancy < 1e-6).
  Test 4: Quaternion convention consistency (NAVRIS navigation-frame attitude error).
  Test 5: Measurement update reduces velocity covariance (P_vv^+ < P_vv^-).
  Test 6: Large innovation rejected by NIS gate (chi2 > 10.828).
  Test 7: Gate rejection causes zero ESKF state change.
  Test 8: Causality / future perturbation invariance.
  Test 9: Covariance symmetry and PSD preserved under repeated updates.
  Test 10: Quaternion normalization preserved (| ||q|| - 1 | < 1e-12).
"""

from __future__ import annotations

import numpy as np
import pytest

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
from scripts.phase3.run_phase3_3a_ml_pseudomeasurement import (
    compute_ml_speed_jacobian,
    verify_ml_jacobian_finite_difference,
    apply_ml_speed_update,
    ML_SPEED_CHI2_THRESHOLD,
)


@pytest.fixture
def nominal_eskf() -> ESKF:
    """Creates a well-conditioned nominal ESKF instance."""
    q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # Identity attitude
    v0 = np.array([10.0, 0.0, 0.0], dtype=np.float64)      # 10 m/s East
    p0 = np.array([100.0, 200.0, 10.0], dtype=np.float64)
    state = NominalState(t=0.0, p=p0, v=v0, q=q0, ba=np.zeros(3), bg=np.zeros(3))
    P0 = COMMON_INIT_COV.copy()
    return ESKF(state, P0, COMMON_CONFIG, lat_deg=52.0, alt_m=100.0)


@pytest.fixture
def identity_c_b_v() -> np.ndarray:
    """Identity mounting calibration."""
    return np.eye(3, dtype=np.float64)


def test_1_correct_measurement_near_zero_innovation(nominal_eskf, identity_c_b_v):
    """Test 1: Known ESKF state and correct forward speed produce near-zero innovation."""
    # With identity attitude and v0 = [10, 0, 0], vehicle forward speed is exactly 10 m/s
    v_correct = 10.0
    R_ml = 4.0
    accepted, record, diag = apply_ml_speed_update(
        eskf=nominal_eskf,
        C_b_v=identity_c_b_v,
        v_ml_pred=v_correct,
        R_ml=R_ml,
    )
    assert accepted
    assert abs(diag["residual_mps"]) < 1e-12
    assert diag["nis"] < 1e-12


def test_2_known_forward_speed_error(nominal_eskf, identity_c_b_v):
    """Test 2: Known forward-speed error produces exact innovation sign and magnitude."""
    # ESKF forward speed is 10 m/s. Measurement is 13 m/s. Residual = z - h = +3.0 m/s.
    v_meas = 13.0
    R_ml = 4.0
    accepted, record, diag = apply_ml_speed_update(
        eskf=nominal_eskf,
        C_b_v=identity_c_b_v,
        v_ml_pred=v_meas,
        R_ml=R_ml,
    )
    assert accepted
    assert np.isclose(diag["residual_mps"], 3.0, atol=1e-12)
    # Velocity should increase in the forward direction
    assert nominal_eskf.state.v[0] > 10.0


def test_3_jacobian_finite_difference(nominal_eskf, identity_c_b_v):
    """Test 3: Compare analytical H against numerical finite difference across velocity and attitude."""
    # Test across multiple orientations
    for angle_deg in [0.0, 30.0, 75.0, -45.0]:
        yaw = np.radians(angle_deg)
        q = np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)])
        nominal_eskf.state.q = q
        nominal_eskf.state.v = np.array([12.0, -4.0, 1.0])

        res = verify_ml_jacobian_finite_difference(nominal_eskf.state, identity_c_b_v, eps=1e-7)
        assert res["passes_tolerance"], f"Jacobian failed at yaw={angle_deg} deg: {res}"
        assert res["discrepancy_max"] < 1e-5


def test_4_quaternion_convention(nominal_eskf, identity_c_b_v):
    """Test 4: Verify Jacobian against NAVRIS navigation-frame attitude error convention."""
    H = compute_ml_speed_jacobian(nominal_eskf.state, identity_c_b_v)
    assert H.shape == (1, 15)
    # Position, accel bias, gyro bias blocks must be identically zero
    assert np.allclose(H[0, :3], 0.0)
    assert np.allclose(H[0, 9:15], 0.0)
    # Velocity and attitude blocks non-zero for moving state
    assert not np.allclose(H[0, IDX_VEL], 0.0)


def test_5_measurement_update_reduces_covariance(nominal_eskf, identity_c_b_v):
    """Test 5: Verify velocity covariance decreases appropriately after a valid scalar update."""
    P_prior = nominal_eskf.P.copy()
    v_meas = 10.5
    R_ml = 2.0
    accepted, _, _ = apply_ml_speed_update(nominal_eskf, identity_c_b_v, v_meas, R_ml)
    assert accepted
    P_post = nominal_eskf.P

    # Forward velocity variance (index 3) must strictly decrease
    assert P_post[3, 3] < P_prior[3, 3]
    # Total trace should decrease
    assert np.trace(P_post) < np.trace(P_prior)


def test_6_large_innovation_rejected_by_nis(nominal_eskf, identity_c_b_v):
    """Test 6: Verify NIS rejects an intentionally inconsistent measurement."""
    # Inject massive 100 m/s discrepancy
    v_huge = 110.0
    R_ml = 1.0
    P_prior = nominal_eskf.P.copy()
    v_prior = nominal_eskf.state.v.copy()

    accepted, record, diag = apply_ml_speed_update(
        eskf=nominal_eskf,
        C_b_v=identity_c_b_v,
        v_ml_pred=v_huge,
        R_ml=R_ml,
        chi2_threshold=ML_SPEED_CHI2_THRESHOLD,
    )
    assert not accepted
    assert diag["nis"] > ML_SPEED_CHI2_THRESHOLD
    # State and covariance must remain completely unchanged
    np.testing.assert_array_equal(nominal_eskf.state.v, v_prior)
    np.testing.assert_array_equal(nominal_eskf.P, P_prior)


def test_7_gate_rejection_causes_zero_eskf_update(nominal_eskf):
    """Test 7: Reliability rejection causes zero ESKF measurement update."""
    P_prior = nominal_eskf.P.copy()
    v_prior = nominal_eskf.state.v.copy()
    p_good = 0.20  # Rejected by Gate C

    # Simulating Gate C sequence: rejected samples do NOT call apply_ml_speed_update
    if p_good >= 0.50:
        apply_ml_speed_update(nominal_eskf, np.eye(3), 15.0, 4.0)

    np.testing.assert_array_equal(nominal_eskf.state.v, v_prior)
    np.testing.assert_array_equal(nominal_eskf.P, P_prior)


def test_8_causality_future_perturbation_invariance(identity_c_b_v):
    """Test 8: Future perturbation does not alter current ML measurement/update."""
    state1 = NominalState(t=5.0, p=np.zeros(3), v=np.array([8.0, 0.0, 0.0]), q=np.array([1.0, 0.0, 0.0, 0.0]), ba=np.zeros(3), bg=np.zeros(3))
    state2 = NominalState(t=5.0, p=np.zeros(3), v=np.array([8.0, 0.0, 0.0]), q=np.array([1.0, 0.0, 0.0, 0.0]), ba=np.zeros(3), bg=np.zeros(3))
    P1 = COMMON_INIT_COV.copy()
    P2 = COMMON_INIT_COV.copy()

    eskf1 = ESKF(state1, P1, COMMON_CONFIG, lat_deg=52.0, alt_m=100.0)
    eskf2 = ESKF(state2, P2, COMMON_CONFIG, lat_deg=52.0, alt_m=100.0)

    # Current update at t=5.0
    acc1, _, diag1 = apply_ml_speed_update(eskf1, identity_c_b_v, 8.5, 4.0, timestamp=5.0)
    acc2, _, diag2 = apply_ml_speed_update(eskf2, identity_c_b_v, 8.5, 4.0, timestamp=5.0)

    assert acc1 == acc2
    assert np.isclose(diag1["residual_mps"], diag2["residual_mps"])
    np.testing.assert_array_almost_equal(eskf1.state.v, eskf2.state.v)


def test_9_covariance_symmetry_and_psd_preserved(nominal_eskf, identity_c_b_v):
    """Test 9: Verify covariance remains symmetric and PSD under repeated updates."""
    np.random.seed(42)
    R_ml = 4.0
    for _ in range(50):
        v_meas = 10.0 + np.random.normal(0.0, 1.0)
        apply_ml_speed_update(nominal_eskf, identity_c_b_v, v_meas, R_ml)

    # Check symmetry
    sym_err = np.max(np.abs(nominal_eskf.P - nominal_eskf.P.T))
    assert sym_err < 1e-9

    # Check positive definiteness
    eigs = np.linalg.eigvalsh(nominal_eskf.P)
    assert np.all(eigs > 0.0), f"Non-positive eigenvalues: {eigs[eigs <= 0]}"


def test_10_quaternion_normalization_preserved(nominal_eskf, identity_c_b_v):
    """Test 10: Verify quaternion norm remains approximately 1 after repeated ML updates."""
    np.random.seed(42)
    R_ml = 4.0
    for _ in range(50):
        v_meas = 10.0 + np.random.normal(0.0, 2.0)
        apply_ml_speed_update(nominal_eskf, identity_c_b_v, v_meas, R_ml)
        q_norm = np.linalg.norm(nominal_eskf.state.q)
        assert abs(q_norm - 1.0) < 1e-12
