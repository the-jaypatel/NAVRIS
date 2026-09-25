"""
Unit and synthetic validation tests for NAVRIS Causal Non-Holonomic Constraints (NHC) Module.

Phase 2.3B Gate 2.3B-2 Synthetic Validation Suite.
Validates:
1. Constant forward speed (10 m/s) with zero errors -> Innovations identically zero
2. Pure forward acceleration (0 to 25 m/s) -> Zero lateral/vertical residuals
3. Pure braking deceleration (25 to 0 m/s) -> Zero lateral residuals and low-speed inhibitor
4. Linear motion at 45° azimuth in ENU -> Velocity rotated correctly to vehicle frame
5. Induced +2° yaw error at 20 m/s -> H_theta couples into lateral residual and corrects yaw
6. Induced +2° yaw error at 0 m/s -> H_theta is zero (no false correction, speed inhibitor)
7. Induced +2° pitch error at 20 m/s -> Vertical constraint couples into pitch and corrects pitch
8. Induced +5° roll error at 20 m/s -> Vehicle roll is in the null space of H_nhc
9. Simulated lateral side-slip (v_lat = 2.0 m/s) -> NIS exceeds 13.82, update rejected
10. Simulated vertical pothole shock (v_vert = 1.5 m/s) -> NIS rejected
11. Constant steady-state turn (15 deg/s) -> Detector inhibits NHC execution
12. Reverse driving (v_fwd = -5 m/s) -> Detector inhibits NHC execution
13. Covariance symmetry and positive definiteness across 1,000 continuous updates
14. Minimum eigenvalue lambda_min(P) > 1e-6 (no covariance collapse)
15. Joseph update equivalence against standard Kalman formulation (< 1e-10)
16. Quaternion normalization preserved (|q| = 1.00000000000000)
17. Uncorrected gyro bias -> NHC prevents unbounded lateral velocity runaway
18. Uncorrected accel bias -> NHC bounds velocity drift in lateral plane
19. Mounting misalignment (+5°) -> Triggers chi-square gating at cruise speed
20. Mandatory finite-difference Jacobian verification (< 1e-6 error) & strict causality check
"""

import pytest
import numpy as np

from navris.eskf import (
    ESKF,
    NominalState,
    ESKFConfig,
    STATE_DIM,
    IDX_POS,
    IDX_VEL,
    IDX_ATT,
    IDX_ACC_BIAS,
    IDX_GYR_BIAS,
    compute_kalman_gain,
    joseph_covariance_update,
)
from navris.eskf.dynamics import skew
from navris.inertial.frames import (
    quat_to_dcm,
    rotvec_to_quat,
    quat_multiply,
    euler_to_quat,
)
from navris.nhc import (
    NHCConfig,
    NHCDetectorConfig,
    CausalNHCDetector,
    compute_nhc_jacobian,
    apply_nhc_update,
    DEFAULT_R_NHC,
    DEFAULT_SIGMA_LAT,
    DEFAULT_SIGMA_VERT,
    NHC_CHI2_THRESHOLD,
    M_NHC,
)


@pytest.fixture
def cruising_eskf():
    """Returns an ESKF instance cruising forward at 20 m/s along East (+X_n)."""
    init_state = NominalState(
        t=0.0,
        p=np.zeros(3),
        v=np.array([20.0, 0.0, 0.0]),  # 20 m/s East
        q=np.array([1.0, 0.0, 0.0, 0.0]),  # Aligned with navigation frame
        ba=np.zeros(3),
        bg=np.zeros(3),
    )
    init_cov = np.eye(STATE_DIM) * 0.001
    init_cov[IDX_ATT, IDX_ATT] = (np.radians(0.2))**2
    init_cov[IDX_VEL, IDX_VEL] = 0.05**2
    config = ESKFConfig(
        sigma_acc=0.20,
        sigma_gyr=0.02,
        sigma_acc_bias=1e-3,
        sigma_gyr_bias=1e-4,
    )
    return ESKF(init_state, init_cov, config=config, g_n=np.array([0.0, 0.0, -9.80665]))


# -----------------------------------------------------------------------------
# 1. Constant forward speed (10 m/s) with zero errors
# -----------------------------------------------------------------------------
def test_01_constant_forward_speed_zero_innovations():
    """Test 01: Constant forward speed with zero error produces identically zero innovations."""
    state = NominalState(
        t=0.0,
        p=np.zeros(3),
        v=np.array([10.0, 0.0, 0.0]),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        ba=np.zeros(3),
        bg=np.zeros(3),
    )
    cov = np.eye(STATE_DIM) * 0.1
    eskf = ESKF(state, cov, ESKFConfig())
    C_b_v = np.eye(3)

    detector = CausalNHCDetector()
    f_meas = np.array([0.0, 0.0, 9.80665])
    w_meas = np.zeros(3)

    # Detector must accept valid cruising conditions
    should_apply = detector.update(eskf.state, C_b_v, f_meas, w_meas, is_stationary=False)
    assert should_apply is True

    accepted, record = apply_nhc_update(eskf, C_b_v, timestamp=1.0)
    assert accepted is True
    assert record.sensor_type == "nhc"
    np.testing.assert_allclose(record.residual, np.zeros(2), atol=1e-12)
    assert record.mahalanobis_sq == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(eskf.state.v, np.array([10.0, 0.0, 0.0]), atol=1e-12)


# -----------------------------------------------------------------------------
# 2. Pure forward acceleration (0 to 25 m/s)
# -----------------------------------------------------------------------------
def test_02_pure_forward_acceleration_zero_residuals():
    """Test 02: Pure forward acceleration produces zero lateral and vertical residuals."""
    C_b_v = np.eye(3)
    detector = CausalNHCDetector()
    f_meas = np.array([2.0, 0.0, 9.80665])  # 2 m/s^2 forward accel
    w_meas = np.zeros(3)

    for v_fwd in np.linspace(5.0, 25.0, 10):
        state = NominalState(
            t=0.0,
            p=np.zeros(3),
            v=np.array([v_fwd, 0.0, 0.0]),
            q=np.array([1.0, 0.0, 0.0, 0.0]),
            ba=np.zeros(3),
            bg=np.zeros(3),
        )
        eskf = ESKF(state, np.eye(STATE_DIM) * 0.1, ESKFConfig())
        assert detector.update(eskf.state, C_b_v, f_meas, w_meas, is_stationary=False) is True

        accepted, record = apply_nhc_update(eskf, C_b_v)
        assert accepted is True
        np.testing.assert_allclose(record.residual, np.zeros(2), atol=1e-12)


# -----------------------------------------------------------------------------
# 3. Pure braking deceleration (25 to 0 m/s)
# -----------------------------------------------------------------------------
def test_03_pure_braking_deceleration_and_speed_cutoff():
    """Test 03: Pure braking deceleration maintains zero residuals and cuts off at speed <= 1.5 m/s."""
    C_b_v = np.eye(3)
    detector = CausalNHCDetector()
    f_meas = np.array([-3.0, 0.0, 9.80665])
    w_meas = np.zeros(3)

    # 10 m/s: Active and zero residual
    state_high = NominalState(t=0.0, p=np.zeros(3), v=np.array([10.0, 0.0, 0.0]), q=np.array([1.0, 0.0, 0.0, 0.0]), ba=np.zeros(3), bg=np.zeros(3))
    eskf_high = ESKF(state_high, np.eye(STATE_DIM) * 0.1, ESKFConfig())
    assert detector.update(eskf_high.state, C_b_v, f_meas, w_meas, is_stationary=False) is True
    accepted, record = apply_nhc_update(eskf_high, C_b_v)
    assert accepted is True
    np.testing.assert_allclose(record.residual, np.zeros(2), atol=1e-12)

    # 1.0 m/s: Below threshold 1.5 m/s -> detector inhibits NHC
    state_low = NominalState(t=0.0, p=np.zeros(3), v=np.array([1.0, 0.0, 0.0]), q=np.array([1.0, 0.0, 0.0, 0.0]), ba=np.zeros(3), bg=np.zeros(3))
    assert detector.update(state_low, C_b_v, f_meas, w_meas, is_stationary=False) is False


# -----------------------------------------------------------------------------
# 4. Linear motion at 45° azimuth in ENU
# -----------------------------------------------------------------------------
def test_04_linear_motion_at_45_deg_azimuth_enu():
    """Test 04: Linear motion at 45° azimuth rotates correctly into vehicle frame."""
    yaw_45 = np.pi / 4.0
    q_45 = euler_to_quat(0.0, 0.0, yaw_45)
    speed = 15.0
    v_nav = np.array([speed * np.cos(yaw_45), speed * np.sin(yaw_45), 0.0])

    state = NominalState(t=0.0, p=np.zeros(3), v=v_nav, q=q_45, ba=np.zeros(3), bg=np.zeros(3))
    eskf = ESKF(state, np.eye(STATE_DIM) * 0.1, ESKFConfig())
    C_b_v = np.eye(3)

    detector = CausalNHCDetector()
    f_meas = np.array([0.0, 0.0, 9.80665])
    w_meas = np.zeros(3)
    assert detector.update(eskf.state, C_b_v, f_meas, w_meas, is_stationary=False) is True

    accepted, record = apply_nhc_update(eskf, C_b_v)
    assert accepted is True
    np.testing.assert_allclose(record.residual, np.zeros(2), atol=1e-12)
    assert record.mahalanobis_sq == pytest.approx(0.0, abs=1e-12)


# -----------------------------------------------------------------------------
# 5. Induced +2° yaw error at 20 m/s
# -----------------------------------------------------------------------------
def test_05_induced_yaw_error_at_20_mps_attitude_coupling():
    """Test 05: Induced +2° yaw error at 20 m/s couples into lateral residual and corrects yaw."""
    yaw_err = np.radians(2.0)
    dq = rotvec_to_quat(np.array([0.0, 0.0, yaw_err]))
    q_err = dq  # nominal orientation has yaw error

    v_nav = np.array([20.0, 0.0, 0.0])
    state = NominalState(t=0.0, p=np.zeros(3), v=v_nav, q=q_err, ba=np.zeros(3), bg=np.zeros(3))
    cov = np.eye(STATE_DIM) * 0.1
    cov[IDX_VEL, IDX_VEL] = 0.01
    cov[IDX_ATT, IDX_ATT] = 0.05

    eskf = ESKF(state, cov, ESKFConfig())
    C_b_v = np.eye(3)

    H = compute_nhc_jacobian(state, C_b_v)
    H_theta = H[:, IDX_ATT]
    # Lateral row (index 0) must have non-zero coupling with Z-axis rotation
    assert abs(H_theta[0, 2]) > 10.0  # Speed is 20 m/s, coupling magnitude is ~20

    accepted, record = apply_nhc_update(eskf, C_b_v)
    assert accepted is True
    assert record.residual[0] > 0.5  # Lateral residual is non-zero
    assert record.mahalanobis_sq < NHC_CHI2_THRESHOLD

    # Verify yaw error reduction after multiplicative injection
    post_yaw = 2.0 * np.arctan2(eskf.state.q[3], eskf.state.q[0])
    assert abs(post_yaw) < abs(yaw_err)
    assert abs(post_yaw) < np.radians(1.5)  # Corrected from 2.0° down towards 1.3°


# -----------------------------------------------------------------------------
# 6. Induced +2° yaw error at 0 m/s
# -----------------------------------------------------------------------------
def test_06_induced_yaw_error_at_zero_speed_zero_coupling():
    """Test 06: Induced +2° yaw error at 0 m/s produces zero H_theta and is rejected by detector."""
    yaw_err = np.radians(2.0)
    dq = rotvec_to_quat(np.array([0.0, 0.0, yaw_err]))

    state = NominalState(t=0.0, p=np.zeros(3), v=np.zeros(3), q=dq, ba=np.zeros(3), bg=np.zeros(3))
    C_b_v = np.eye(3)

    # Jacobian check: H_theta must be identically zero when v = 0
    H = compute_nhc_jacobian(state, C_b_v)
    np.testing.assert_allclose(H[:, IDX_ATT], np.zeros((2, 3)), atol=1e-15)

    # Detector check: Must reject because speed <= 1.5 m/s
    detector = CausalNHCDetector()
    f_meas = np.array([0.0, 0.0, 9.80665])
    w_meas = np.zeros(3)
    assert detector.update(state, C_b_v, f_meas, w_meas, is_stationary=False) is False


# -----------------------------------------------------------------------------
# 7. Induced +2° pitch error at 20 m/s
# -----------------------------------------------------------------------------
def test_07_induced_pitch_error_at_20_mps_vertical_coupling():
    """Test 07: Induced +2° pitch error at 20 m/s couples into vertical constraint and corrects pitch."""
    pitch_err = np.radians(2.0)
    dq = rotvec_to_quat(np.array([0.0, pitch_err, 0.0]))

    v_nav = np.array([20.0, 0.0, 0.0])
    state = NominalState(t=0.0, p=np.zeros(3), v=v_nav, q=dq, ba=np.zeros(3), bg=np.zeros(3))
    cov = np.eye(STATE_DIM) * 0.1
    cov[IDX_VEL, IDX_VEL] = 0.01
    cov[IDX_ATT, IDX_ATT] = 0.05

    eskf = ESKF(state, cov, ESKFConfig())
    C_b_v = np.eye(3)

    H = compute_nhc_jacobian(state, C_b_v)
    H_theta = H[:, IDX_ATT]
    # Vertical row (index 1) must have non-zero coupling with Y-axis rotation
    assert abs(H_theta[1, 1]) > 10.0

    accepted, record = apply_nhc_update(eskf, C_b_v)
    assert accepted is True
    assert record.residual[1] < -0.5  # Vertical residual is non-zero
    assert record.mahalanobis_sq < NHC_CHI2_THRESHOLD

    # Verify pitch error reduction
    post_pitch = 2.0 * np.arcsin(eskf.state.q[2])
    assert abs(post_pitch) < abs(pitch_err)
    assert abs(post_pitch) < np.radians(1.0)


# -----------------------------------------------------------------------------
# 8. Induced +5° roll error in null space
# -----------------------------------------------------------------------------
def test_08_induced_roll_error_in_null_space():
    """Test 08: Induced vehicle roll error is strictly in the null space of H_nhc."""
    state = NominalState(
        t=0.0,
        p=np.zeros(3),
        v=np.array([20.0, 0.0, 0.0]),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        ba=np.zeros(3),
        bg=np.zeros(3),
    )
    C_b_v = np.eye(3)
    H = compute_nhc_jacobian(state, C_b_v)
    H_theta = H[:, IDX_ATT]

    # Vehicle longitudinal forward vector in navigation frame: u_fwd = [1, 0, 0]^T
    u_fwd_nav = np.array([1.0, 0.0, 0.0])
    coupling = H_theta @ u_fwd_nav
    np.testing.assert_allclose(coupling, np.zeros(2), atol=1e-14)


# -----------------------------------------------------------------------------
# 9. Simulated lateral side-slip gating rejection
# -----------------------------------------------------------------------------
def test_09_lateral_side_slip_gating_rejection(cruising_eskf):
    """Test 09: Significant lateral side-slip (2.0 m/s) is rejected by 2-DOF Chi-square gating."""
    # Prior state has 2.0 m/s lateral velocity
    cruising_eskf.state.v = np.array([20.0, 2.0, 0.0])
    C_b_v = np.eye(3)

    accepted, record = apply_nhc_update(cruising_eskf, C_b_v)
    assert accepted is False
    assert record.sensor_type == "nhc"
    assert record.mahalanobis_sq > NHC_CHI2_THRESHOLD
    # State must not be updated
    assert cruising_eskf.state.v[1] == pytest.approx(2.0)


# -----------------------------------------------------------------------------
# 10. Simulated vertical pothole shock gating rejection
# -----------------------------------------------------------------------------
def test_10_vertical_pothole_shock_gating_rejection(cruising_eskf):
    """Test 10: Simulated vertical pothole velocity shock (1.5 m/s) is rejected by Chi-square gating."""
    cruising_eskf.state.v = np.array([20.0, 0.0, 1.5])
    C_b_v = np.eye(3)

    accepted, record = apply_nhc_update(cruising_eskf, C_b_v)
    assert accepted is False
    assert record.mahalanobis_sq > NHC_CHI2_THRESHOLD
    assert cruising_eskf.state.v[2] == pytest.approx(1.5)


# -----------------------------------------------------------------------------
# 11. Steady-state turn inhibitor
# -----------------------------------------------------------------------------
def test_11_steady_state_turn_inhibitor(cruising_eskf):
    """Test 11: Angular rate exceeding 5 deg/s inhibits NHC execution."""
    detector = CausalNHCDetector()
    C_b_v = np.eye(3)
    f_meas = np.array([0.0, 0.0, 9.80665])

    # 15 deg/s yaw rate = 0.2618 rad/s > 0.087 rad/s
    w_turn = np.array([0.0, 0.0, np.radians(15.0)])
    assert detector.update(cruising_eskf.state, C_b_v, f_meas, w_turn, is_stationary=False) is False

    # 2 deg/s yaw rate = 0.0349 rad/s <= 0.087 rad/s -> allowed
    w_straight = np.array([0.0, 0.0, np.radians(2.0)])
    assert detector.update(cruising_eskf.state, C_b_v, f_meas, w_straight, is_stationary=False) is True


# -----------------------------------------------------------------------------
# 12. Reverse driving inhibitor
# -----------------------------------------------------------------------------
def test_12_reverse_driving_inhibitor():
    """Test 12: Reverse driving (v_fwd < 0) is explicitly rejected by detector."""
    detector = CausalNHCDetector()
    C_b_v = np.eye(3)
    f_meas = np.array([0.0, 0.0, 9.80665])
    w_meas = np.zeros(3)

    state_reverse = NominalState(
        t=0.0,
        p=np.zeros(3),
        v=np.array([-5.0, 0.0, 0.0]),  # Backing up at 5 m/s
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        ba=np.zeros(3),
        bg=np.zeros(3),
    )
    assert detector.update(state_reverse, C_b_v, f_meas, w_meas, is_stationary=False) is False


# -----------------------------------------------------------------------------
# 13. Covariance symmetry and PSD across 1,000 updates
# -----------------------------------------------------------------------------
def test_13_covariance_symmetry_and_psd_1000_updates():
    """Test 13: Covariance remains strictly symmetric and positive definite across 1,000 continuous updates."""
    state = NominalState(t=0.0, p=np.zeros(3), v=np.array([15.0, 0.0, 0.0]), q=np.array([1.0, 0.0, 0.0, 0.0]), ba=np.zeros(3), bg=np.zeros(3))
    cov = np.eye(STATE_DIM) * 1.0
    eskf = ESKF(state, cov, ESKFConfig(), g_n=np.array([0.0, 0.0, -9.80665]))
    C_b_v = np.eye(3)

    f_meas = np.array([0.0, 0.0, 9.80665])
    w_meas = np.zeros(3)

    for step in range(1000):
        eskf.predict(f_meas, w_meas, 0.01)
        apply_nhc_update(eskf, C_b_v, timestamp=(step + 1) * 0.01)

    sym_diff = np.max(np.abs(eskf.P - eskf.P.T))
    eigs = np.linalg.eigvalsh(eskf.P)

    assert sym_diff < 1e-12
    assert np.all(eigs > 0.0)
    assert not np.any(np.isnan(eskf.P))
    assert not np.any(np.isinf(eskf.P))


# -----------------------------------------------------------------------------
# 14. Minimum eigenvalue bound (no covariance collapse)
# -----------------------------------------------------------------------------
def test_14_minimum_eigenvalue_bound():
    """Test 14: Minimum eigenvalue remains well above numerical safety floor (> 1e-6) after 1,000 updates."""
    state = NominalState(t=0.0, p=np.zeros(3), v=np.array([15.0, 0.0, 0.0]), q=np.array([1.0, 0.0, 0.0, 0.0]), ba=np.zeros(3), bg=np.zeros(3))
    cov = np.eye(STATE_DIM) * 0.5
    eskf = ESKF(state, cov, ESKFConfig(), g_n=np.array([0.0, 0.0, -9.80665]))
    C_b_v = np.eye(3)

    f_meas = np.array([0.0, 0.0, 9.80665])
    w_meas = np.zeros(3)

    for step in range(1000):
        eskf.predict(f_meas, w_meas, 0.01)
        apply_nhc_update(eskf, C_b_v, timestamp=(step + 1) * 0.01)

    min_eig = np.min(np.linalg.eigvalsh(eskf.P))
    assert min_eig > 1e-6


# -----------------------------------------------------------------------------
# 15. Joseph update equivalence against standard Kalman formulation
# -----------------------------------------------------------------------------
def test_15_joseph_update_equivalence():
    """Test 15: Joseph update is numerically equivalent to standard formulation on well-conditioned state."""
    state = NominalState(t=0.0, p=np.zeros(3), v=np.array([12.0, 0.0, 0.0]), q=np.array([1.0, 0.0, 0.0, 0.0]), ba=np.zeros(3), bg=np.zeros(3))
    C_b_v = np.eye(3)
    H = compute_nhc_jacobian(state, C_b_v)
    P = np.eye(STATE_DIM) * 0.5
    R = DEFAULT_R_NHC

    K, S = compute_kalman_gain(P, H, R)
    P_joseph = joseph_covariance_update(P, K, H, R)
    I_KH = np.eye(STATE_DIM) - K @ H
    P_std = 0.5 * (I_KH @ P + (I_KH @ P).T)

    max_diff = np.max(np.abs(P_joseph - P_std))
    assert max_diff < 1e-10


# -----------------------------------------------------------------------------
# 16. Quaternion normalization preserved
# -----------------------------------------------------------------------------
def test_16_quaternion_normalization_preserved(cruising_eskf):
    """Test 16: Quaternion remains exactly normalized (|q| = 1.0) after NHC updates."""
    # Induce small lateral velocity to force non-zero correction
    cruising_eskf.state.v[1] = 0.1
    C_b_v = np.eye(3)

    for _ in range(50):
        apply_nhc_update(cruising_eskf, C_b_v)
        q_norm = float(np.linalg.norm(cruising_eskf.state.q))
        assert abs(q_norm - 1.0) < 1e-14


# -----------------------------------------------------------------------------
# 17. Uncorrected gyro bias velocity bounding
# -----------------------------------------------------------------------------
def test_17_uncorrected_gyro_bias_velocity_bounding():
    """Test 17: NHC prevents unbounded lateral velocity drift caused by uncorrected gyro bias."""
    dt = 0.1
    C_b_v = np.eye(3)

    # Filter 1: Without NHC
    state_no_nhc = NominalState(t=0.0, p=np.zeros(3), v=np.array([10.0, 0.0, 0.0]), q=np.array([1.0, 0.0, 0.0, 0.0]), ba=np.zeros(3), bg=np.zeros(3))
    cov_no_nhc = np.eye(STATE_DIM) * 0.01
    cov_no_nhc[IDX_ATT, IDX_ATT] = 0.01
    eskf_no_nhc = ESKF(state_no_nhc, cov_no_nhc, ESKFConfig(), g_n=np.array([0.0, 0.0, -9.80665]))

    # Filter 2: With NHC
    state_nhc = NominalState(t=0.0, p=np.zeros(3), v=np.array([10.0, 0.0, 0.0]), q=np.array([1.0, 0.0, 0.0, 0.0]), ba=np.zeros(3), bg=np.zeros(3))
    cov_nhc = np.eye(STATE_DIM) * 0.01
    cov_nhc[IDX_ATT, IDX_ATT] = 0.01
    eskf_nhc = ESKF(state_nhc, cov_nhc, ESKFConfig(), g_n=np.array([0.0, 0.0, -9.80665]))

    f_meas = np.array([0.0, 0.0, 9.80665])
    omega_meas = np.array([0.0, 0.0, 0.005])  # Uncorrected 0.005 rad/s yaw rate bias

    for step in range(40):
        t = step * dt
        eskf_no_nhc.predict(f_meas, omega_meas, dt)
        eskf_nhc.predict(f_meas, omega_meas, dt)
        apply_nhc_update(eskf_nhc, C_b_v, timestamp=t + dt)

    v_veh_no_nhc = (C_b_v @ quat_to_dcm(eskf_no_nhc.state.q).T @ eskf_no_nhc.state.v)[1]
    v_veh_nhc = (C_b_v @ quat_to_dcm(eskf_nhc.state.q).T @ eskf_nhc.state.v)[1]

    assert abs(v_veh_no_nhc) > 0.15  # Unbounded drift reaches > 0.15 m/s
    assert abs(v_veh_nhc) < 0.01     # NHC keeps vehicle lateral velocity bounded below 0.01 m/s


# -----------------------------------------------------------------------------
# 18. Uncorrected accel bias velocity bounding
# -----------------------------------------------------------------------------
def test_18_uncorrected_accel_bias_velocity_bounding():
    """Test 18: NHC bounds lateral velocity error caused by uncorrected accelerometer bias."""
    dt = 0.1
    C_b_v = np.eye(3)

    state_no_nhc = NominalState(t=0.0, p=np.zeros(3), v=np.array([10.0, 0.0, 0.0]), q=np.array([1.0, 0.0, 0.0, 0.0]), ba=np.zeros(3), bg=np.zeros(3))
    cov_no_nhc = np.eye(STATE_DIM) * 0.01
    eskf_no_nhc = ESKF(state_no_nhc, cov_no_nhc, ESKFConfig(), g_n=np.array([0.0, 0.0, -9.80665]))

    state_nhc = NominalState(t=0.0, p=np.zeros(3), v=np.array([10.0, 0.0, 0.0]), q=np.array([1.0, 0.0, 0.0, 0.0]), ba=np.zeros(3), bg=np.zeros(3))
    cov_nhc = np.eye(STATE_DIM) * 0.01
    eskf_nhc = ESKF(state_nhc, cov_nhc, ESKFConfig(), g_n=np.array([0.0, 0.0, -9.80665]))

    # Lateral acceleration bias of 0.05 m/s^2 on Y
    f_meas = np.array([0.0, 0.05, 9.80665])
    omega_meas = np.zeros(3)

    for step in range(30):
        t = step * dt
        eskf_no_nhc.predict(f_meas, omega_meas, dt)
        eskf_nhc.predict(f_meas, omega_meas, dt)
        apply_nhc_update(eskf_nhc, C_b_v, timestamp=t + dt)

    v_lat_no_nhc = eskf_no_nhc.state.v[1]
    v_lat_nhc = eskf_nhc.state.v[1]

    assert abs(v_lat_nhc) < abs(v_lat_no_nhc)
    assert abs(v_lat_no_nhc) == pytest.approx(0.15, abs=0.01)
    assert abs(v_lat_nhc) < 0.08


# -----------------------------------------------------------------------------
# 19. Mounting misalignment (+5°) rejection
# -----------------------------------------------------------------------------
def test_19_mounting_misalignment_rejection():
    """Test 19: 5° mounting misalignment at highway cruise triggers Chi-square gating rejection."""
    state = NominalState(t=0.0, p=np.zeros(3), v=np.array([20.0, 0.0, 0.0]), q=np.array([1.0, 0.0, 0.0, 0.0]), ba=np.zeros(3), bg=np.zeros(3))
    cov = np.eye(STATE_DIM) * 0.001
    cov[IDX_ATT, IDX_ATT] = np.radians(0.2)**2
    cov[IDX_VEL, IDX_VEL] = 0.05**2
    eskf = ESKF(state, cov, ESKFConfig())

    # 5 degree mounting yaw error
    theta = np.radians(5.0)
    C_b_v = np.array([
        [np.cos(theta), np.sin(theta), 0.0],
        [-np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])

    accepted, record = apply_nhc_update(eskf, C_b_v)
    assert accepted is False
    assert record.mahalanobis_sq > NHC_CHI2_THRESHOLD


# -----------------------------------------------------------------------------
# 20. Mandatory finite-difference Jacobian verification & strict causality check
# -----------------------------------------------------------------------------
def test_20_mandatory_finite_difference_jacobian_and_strict_causality():
    """
    Test 20: Mandatory verification of analytical Jacobian against central finite differences.
    Validates:
    - Error state ordering [pos(3), vel(3), att(3), ba(3), bg(3)]
    - NAVRIS convention: q_true = rotvec_to_quat(delta_theta^n) (x) q_nominal
    - Discrepancy strictly bounded: ||H_analytical - H_num||_inf < 1.0e-6
    - Strict causality: detector operates solely on current-epoch inputs.
    """
    np.random.seed(2026)

    for trial in range(5):
        # Generate random nominal state
        q_rand = np.random.randn(4)
        q_rand /= np.linalg.norm(q_rand)
        v_rand = np.random.uniform(-25.0, 25.0, size=3)

        state = NominalState(
            t=float(trial),
            p=np.random.uniform(-100.0, 100.0, size=3),
            v=v_rand,
            q=q_rand,
            ba=np.random.uniform(-0.1, 0.1, size=3),
            bg=np.random.uniform(-0.01, 0.01, size=3),
        )

        # Random SO(3) mounting matrix
        rot_axis = np.random.randn(3)
        rot_axis /= np.linalg.norm(rot_axis)
        rot_angle = np.random.uniform(-np.pi, np.pi)
        C_b_v = quat_to_dcm(rotvec_to_quat(rot_axis * rot_angle))

        # Analytical Jacobian
        H_analytical = compute_nhc_jacobian(state, C_b_v)
        assert H_analytical.shape == (2, STATE_DIM)

        # Numerical Jacobian via central finite differences (epsilon = 1e-7)
        eps = 1e-7
        H_numerical = np.zeros((2, STATE_DIM), dtype=np.float64)

        def eval_h(nom_v: np.ndarray, nom_q: np.ndarray) -> np.ndarray:
            C_bn = quat_to_dcm(nom_q)
            C_nv = C_b_v @ C_bn.T
            return M_NHC @ (C_nv @ nom_v)

        # Position block (0:3) -> must be zero
        for i in range(3):
            dp = np.zeros(3)
            dp[i] = eps
            # Position does not enter h
            H_numerical[:, IDX_POS.start + i] = 0.0

        # Velocity block (3:6)
        for i in range(3):
            v_plus = state.v.copy()
            v_plus[i] += eps
            h_plus = eval_h(v_plus, state.q)

            v_minus = state.v.copy()
            v_minus[i] -= eps
            h_minus = eval_h(v_minus, state.q)

            H_numerical[:, IDX_VEL.start + i] = (h_plus - h_minus) / (2.0 * eps)

        # Attitude block (6:9): q_pert = rotvec_to_quat(delta_theta^n) (x) q_nominal
        for i in range(3):
            dtheta_plus = np.zeros(3)
            dtheta_plus[i] = eps
            q_plus = quat_multiply(rotvec_to_quat(dtheta_plus), state.q)
            h_plus = eval_h(state.v, q_plus)

            dtheta_minus = np.zeros(3)
            dtheta_minus[i] = -eps
            q_minus = quat_multiply(rotvec_to_quat(dtheta_minus), state.q)
            h_minus = eval_h(state.v, q_minus)

            H_numerical[:, IDX_ATT.start + i] = (h_plus - h_minus) / (2.0 * eps)

        # Accel and Gyro bias blocks (9:15) -> must be zero
        for i in range(3):
            H_numerical[:, IDX_ACC_BIAS.start + i] = 0.0
            H_numerical[:, IDX_GYR_BIAS.start + i] = 0.0

        # Discrepancy evaluation
        max_error = float(np.max(np.abs(H_analytical - H_numerical)))
        assert max_error < 1.0e-6, f"Trial {trial}: Jacobian discrepancy {max_error:.4e} exceeds 1.0e-6"

    # Strict causality audit on CausalNHCDetector
    detector = CausalNHCDetector()
    f_sample = np.array([0.0, 0.0, 9.80665])
    w_sample = np.zeros(3)
    c_bv = np.eye(3)
    nom = NominalState(t=100.0, p=np.zeros(3), v=np.array([10.0, 0.0, 0.0]), q=np.array([1.0, 0.0, 0.0, 0.0]), ba=np.zeros(3), bg=np.zeros(3))

    # Single-sample evaluation without storing or requiring future samples
    res = detector.update(nom, c_bv, f_sample, w_sample, is_stationary=False)
    assert res is True
    assert detector.total_nhc_attempts == 1
    assert detector.total_nhc_accepted_conditions == 1
