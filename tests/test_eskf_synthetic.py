"""
NAVRIS Phase 2.3A Synthetic Unit Test Suite.

Automated verification of the 15-state GNSS-Aided Error-State Kalman Filter (ESKF)
under controlled synthetic conditions.

Covers:
1. Stationary stability
2. Constant velocity propagation
3. Constant acceleration propagation
4. Observable accelerometer-bias estimation
5. Observable gyroscope-bias experiment (under dynamic turning excitation)
6. GNSS position correction
7. GNSS outlier rejection (Chi-square gate)
8. Duplicate GNSS rejection (sample-and-hold check)
9. Attitude error injection and reset consistency (observability via velocity/dynamics)
10. Quaternion convention consistency (matching frames.py)
11. Covariance symmetry maintenance
12. Covariance numerical validity (positive semi-definiteness without artificial clipping)
13. Measurement gating threshold behavior
14. Strict causality / zero future leakage
"""

import numpy as np
import pytest
import scipy.linalg

from navris.eskf.state import (
    NominalState,
    ESKFConfig,
    STATE_DIM,
    IDX_POS,
    IDX_VEL,
    IDX_ATT,
    IDX_ACC_BIAS,
    IDX_GYR_BIAS,
)
from navris.eskf.dynamics import (
    continuous_f_matrix,
    continuous_g_matrix,
    continuous_qc_matrix,
    discretize_dynamics_van_loan,
    skew,
)
from navris.eskf.propagation import (
    propagate_nominal,
    propagate_covariance,
)
from navris.eskf.gating import (
    gate_measurement,
    compute_mahalanobis_sq,
)
from navris.eskf.update import (
    compute_kalman_gain,
    joseph_covariance_update,
)
from navris.eskf.reset import inject_and_reset
from navris.eskf.filter import ESKF
from navris.inertial.frames import (
    quat_to_dcm,
    quat_multiply,
    quat_normalize,
    rotvec_to_quat,
    rotate_vector,
    euler_to_quat,
    quat_to_euler,
)
from navris.inertial.gravity import gravity_vector_enu


# =============================================================================
# Helper Fixtures
# =============================================================================

@pytest.fixture
def standard_gravity():
    return np.array([0.0, 0.0, -9.80665], dtype=np.float64)


@pytest.fixture
def default_config():
    return ESKFConfig(
        sigma_acc=0.05,
        sigma_gyr=0.005,
        sigma_acc_bias=1e-4,
        sigma_gyr_bias=1e-5,
        gnss_pos_std_horiz=1.5,
        gnss_pos_std_vert=3.0,
        gnss_vel_std_horiz=0.1,
        gnss_vel_std_vert=0.2,
        chi2_threshold_pos=16.27,
        chi2_threshold_vel=16.27,
    )


@pytest.fixture
def healthy_initial_cov():
    cov_diag = np.concatenate([
        np.full(3, 4.0),       # pos std 2m -> var 4
        np.full(3, 0.25),      # vel std 0.5 m/s -> var 0.25
        np.full(3, (np.deg2rad(2.0)) ** 2),  # att std 2 deg
        np.full(3, 0.04),      # acc bias std 0.2 m/s^2
        np.full(3, (np.deg2rad(0.5)) ** 2),  # gyr bias std 0.5 deg/s
    ])
    return np.diag(cov_diag)


# =============================================================================
# Test 1: Stationary Stability
# =============================================================================
def test_test_1_stationary_stability(standard_gravity, default_config, healthy_initial_cov):
    """
    Measurable criteria:
    - Stationary phone with true gravity specific force f_b = [0, 0, +g].
    - Filter propagates for 100 seconds (1000 steps at dt=0.1 s).
    - Position drift < 1e-4 m.
    - Velocity drift < 1e-5 m/s.
    - Quaternion norm == 1.0 within 1e-12.
    """
    g = -standard_gravity[2]
    init_state = NominalState(
        t=0.0,
        p=np.zeros(3),
        v=np.zeros(3),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        ba=np.zeros(3),
        bg=np.zeros(3)
    )
    filter_inst = ESKF(init_state, healthy_initial_cov, default_config, g_n=standard_gravity)

    f_b = np.array([0.0, 0.0, g], dtype=np.float64)
    omega_b = np.zeros(3, dtype=np.float64)
    dt = 0.1

    for _ in range(1000):
        ok = filter_inst.predict(f_b, omega_b, dt)
        assert ok, "Propagation step must succeed"

    pos_drift = np.linalg.norm(filter_inst.state.p)
    vel_drift = np.linalg.norm(filter_inst.state.v)
    quat_norm = np.linalg.norm(filter_inst.state.q)

    assert pos_drift < 1e-4, f"Stationary position drift too high: {pos_drift} m"
    assert vel_drift < 1e-5, f"Stationary velocity drift too high: {vel_drift} m/s"
    assert np.isclose(quat_norm, 1.0, atol=1e-12), f"Quaternion norm changed: {quat_norm}"


# =============================================================================
# Test 2: Constant Velocity Propagation
# =============================================================================
def test_test_2_constant_velocity(standard_gravity, default_config, healthy_initial_cov):
    """
    Measurable criteria:
    - Vehicle cruising with constant velocity v0 = [12.0, -8.0, 0.0] m/s.
    - Specific force is strictly reaction against gravity: f_b = [0, 0, +g].
    - Filter propagates for 50 seconds.
    - Velocity error < 1e-5 m/s.
    - Position matches p(t) = v0 * t within 1e-3 m.
    """
    g = -standard_gravity[2]
    v0 = np.array([12.0, -8.0, 0.0], dtype=np.float64)
    init_state = NominalState(
        t=0.0,
        p=np.zeros(3),
        v=v0.copy(),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        ba=np.zeros(3),
        bg=np.zeros(3)
    )
    filter_inst = ESKF(init_state, healthy_initial_cov, default_config, g_n=standard_gravity)

    f_b = np.array([0.0, 0.0, g], dtype=np.float64)
    omega_b = np.zeros(3, dtype=np.float64)
    dt = 0.1
    duration = 50.0
    steps = int(duration / dt)

    for _ in range(steps):
        filter_inst.predict(f_b, omega_b, dt)

    expected_pos = v0 * duration
    pos_err = np.linalg.norm(filter_inst.state.p - expected_pos)
    vel_err = np.linalg.norm(filter_inst.state.v - v0)

    assert vel_err < 1e-5, f"Velocity deviated under constant velocity: {vel_err} m/s"
    assert pos_err < 1e-3, f"Position deviated from v0 * t: {pos_err} m"


# =============================================================================
# Test 3: Constant Acceleration Propagation
# =============================================================================
def test_test_3_constant_acceleration(standard_gravity, default_config, healthy_initial_cov):
    """
    Measurable criteria:
    - Constant forward acceleration along East: a_net = [2.0, 0.0, 0.0] m/s^2.
    - Specific force: f_b = [2.0, 0.0, +g].
    - Duration = 20 seconds.
    - Expected velocity = a_net * t = [40.0, 0.0, 0.0] m/s.
    - Expected position = 0.5 * a_net * t^2 = [400.0, 0.0, 0.0] m.
    - Velocity error < 1e-4 m/s.
    - Position error < 1e-3 m.
    """
    g = -standard_gravity[2]
    a_net = np.array([2.0, 0.0, 0.0], dtype=np.float64)
    f_b = a_net + np.array([0.0, 0.0, g], dtype=np.float64)
    omega_b = np.zeros(3, dtype=np.float64)

    init_state = NominalState(
        t=0.0,
        p=np.zeros(3),
        v=np.zeros(3),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        ba=np.zeros(3),
        bg=np.zeros(3)
    )
    filter_inst = ESKF(init_state, healthy_initial_cov, default_config, g_n=standard_gravity)

    dt = 0.1
    duration = 20.0
    steps = int(duration / dt)

    for _ in range(steps):
        filter_inst.predict(f_b, omega_b, dt)

    expected_v = a_net * duration
    expected_p = 0.5 * a_net * (duration ** 2)

    vel_err = np.linalg.norm(filter_inst.state.v - expected_v)
    pos_err = np.linalg.norm(filter_inst.state.p - expected_p)

    assert vel_err < 1e-4, f"Velocity error under constant accel: {vel_err}"
    assert pos_err < 1e-3, f"Position error under constant accel: {pos_err}"


# =============================================================================
# Test 4: Observable Accelerometer-Bias Estimation
# =============================================================================
def test_test_4_observable_accel_bias_estimation(standard_gravity, default_config, healthy_initial_cov):
    """
    Observability Experiment:
    - True vehicle is stationary at origin.
    - Accelerometer has a known constant bias ba_true = [0.25, -0.15, 0.0] m/s^2.
    - Raw IMU measures specific force f_meas = [0, 0, g] + ba_true.
    - GNSS position measurements arrive at 1 Hz with sigma = 0.5 m (true position = [0, 0, 0]).
    - Filter initializes ba = [0, 0, 0].
    Measurable criteria:
    - Filter reduces accelerometer bias error by > 60% over 60 seconds.
    - Estimated bias converges toward ba_true.
    """
    g = -standard_gravity[2]
    ba_true = np.array([0.25, -0.15, 0.0], dtype=np.float64)
    f_meas = np.array([0.0, 0.0, g], dtype=np.float64) + ba_true
    omega_b = np.zeros(3, dtype=np.float64)

    # Configure GNSS noise tighter to clearly observe bias in this experiment
    config = ESKFConfig(
        sigma_acc=0.05,
        sigma_gyr=0.005,
        sigma_acc_bias=1e-3,  # allow bias to track
        sigma_gyr_bias=1e-5,
        gnss_pos_std_horiz=0.3,
        gnss_pos_std_vert=0.5,
        chi2_threshold_pos=25.0,
    )

    init_state = NominalState(
        t=0.0,
        p=np.zeros(3),
        v=np.zeros(3),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        ba=np.zeros(3),
        bg=np.zeros(3)
    )
    # Increase initial bias uncertainty
    init_cov = healthy_initial_cov.copy()
    init_cov[IDX_ACC_BIAS, IDX_ACC_BIAS] = 0.5 * np.eye(3)

    filter_inst = ESKF(init_state, init_cov, config, g_n=standard_gravity)

    dt = 0.1
    total_time = 60.0
    steps = int(total_time / dt)

    initial_bias_error = np.linalg.norm(filter_inst.state.ba - ba_true)

    for i in range(steps):
        filter_inst.predict(f_meas, omega_b, dt)

        # 1 Hz GNSS position update
        if (i + 1) % 10 == 0:
            # True position is [0, 0, 0]
            filter_inst.update_gnss_pos(np.zeros(3), is_new_fix=True)

    final_bias_error = np.linalg.norm(filter_inst.state.ba - ba_true)
    bias_reduction = (initial_bias_error - final_bias_error) / initial_bias_error

    assert bias_reduction > 0.60, f"Accel bias error reduction insufficient: {bias_reduction:.2%}"
    assert final_bias_error < 0.12, f"Final accel bias error too large: {final_bias_error} m/s^2"


# =============================================================================
# Test 5: Observable Gyroscope-Bias Experiment
# =============================================================================
def test_test_5_observable_gyro_bias_experiment(standard_gravity, healthy_initial_cov):
    """
    Scientific Correction #3: Observability under dynamic turning excitation.
    - GNSS position alone cannot observe gyro bias without turning / kinematic excitation.
    - Synthetic scenario: Vehicle undergoes a circular turn at speed v=10 m/s, yaw rate omega_z = 0.1 rad/s.
    - Gyroscope has a true bias bg_true = [0.0, 0.0, 0.03] rad/s (~1.7 deg/s).
    - Gyro measures: omega_meas = [0, 0, 0.1] + bg_true.
    - True vehicle position follows a known circle:
        x(t) = R * sin(omega * t), y(t) = R * (1 - cos(omega * t)), R = v / omega = 100 m.
    - Centripetal acceleration: a_c = v * omega = 1.0 m/s^2 along body lateral axis.
    - GNSS position fixes arrive at 2 Hz along true circle.
    - Filter initializes bg = [0, 0, 0].
    Measurable criteria:
    - Gyro bias error ||bg_hat - bg_true|| reduces by > 50% over 60 seconds of turning.
    """
    g = -standard_gravity[2]
    v_mag = 10.0
    omega_yaw = 0.1  # rad/s
    R_circle = v_mag / omega_yaw  # 100 m
    bg_true = np.array([0.0, 0.0, 0.03], dtype=np.float64)

    config = ESKFConfig(
        sigma_acc=0.05,
        sigma_gyr=0.005,
        sigma_acc_bias=1e-4,
        sigma_gyr_bias=1e-3,  # Allow gyro bias tracking
        gnss_pos_std_horiz=0.2,
        gnss_pos_std_vert=0.5,
        chi2_threshold_pos=30.0,
    )

    # Initial heading is pointing East (+X), velocity along East
    init_state = NominalState(
        t=0.0,
        p=np.zeros(3),
        v=np.array([v_mag, 0.0, 0.0]),
        q=np.array([1.0, 0.0, 0.0, 0.0]),  # identity orientation
        ba=np.zeros(3),
        bg=np.zeros(3)
    )
    init_cov = healthy_initial_cov.copy()
    init_cov[IDX_GYR_BIAS, IDX_GYR_BIAS] = 0.01 * np.eye(3)

    filter_inst = ESKF(init_state, init_cov, config, g_n=standard_gravity)

    dt = 0.05  # 20 Hz IMU
    total_time = 60.0
    steps = int(total_time / dt)

    initial_bias_error = np.linalg.norm(filter_inst.state.bg - bg_true)

    current_yaw = 0.0
    for i in range(steps):
        t = (i + 1) * dt
        # True attitude and kinematics
        current_yaw += omega_yaw * dt
        # In ENU: yaw 0 = East (+X). After yaw turning towards North (+Y):
        # true velocity in ENU
        v_true = np.array([v_mag * np.cos(current_yaw), v_mag * np.sin(current_yaw), 0.0])
        # true position in ENU
        p_true = np.array([R_circle * np.sin(current_yaw), R_circle * (1.0 - np.cos(current_yaw)), 0.0])

        # Specific force in body frame (forward, lateral centripetal, vertical gravity):
        # Body frame: x=lateral, y=forward, z=up (or matching orientation)
        # With q identity at t=0 and yawing around Z:
        q_true = rotvec_to_quat(np.array([0.0, 0.0, current_yaw]))
        C_b_n_true = quat_to_dcm(q_true)
        # Net acceleration in nav: a_net = [-v*omega*sin(yaw), v*omega*cos(yaw), 0]
        a_net_nav = np.array([-v_mag * omega_yaw * np.sin(current_yaw), v_mag * omega_yaw * np.cos(current_yaw), 0.0])
        f_nav_true = a_net_nav - standard_gravity
        f_body_true = C_b_n_true.T @ f_nav_true

        # Sensor readings corrupted by true gyro bias
        omega_meas = np.array([0.0, 0.0, omega_yaw]) + bg_true

        filter_inst.predict(f_body_true, omega_meas, dt)

        # 2 Hz GNSS updates
        if (i + 1) % 10 == 0:
            filter_inst.update_gnss_pos(p_true, is_new_fix=True)

    final_bias_error = np.linalg.norm(filter_inst.state.bg - bg_true)
    bias_reduction = (initial_bias_error - final_bias_error) / initial_bias_error

    assert bias_reduction > 0.50, f"Gyro bias error reduction insufficient: {bias_reduction:.2%}"
    assert final_bias_error < 0.015, f"Final gyro bias error too large: {final_bias_error} rad/s"


# =============================================================================
# Test 6: GNSS Position Correction
# =============================================================================
def test_test_6_gnss_position_correction(standard_gravity, default_config, healthy_initial_cov):
    """
    Measurable criteria:
    - Inject initial position error of 15 meters: p_init = [10.0, -10.0, 5.0].
    - True vehicle is stationary at origin [0, 0, 0].
    - On receiving 1 Hz GNSS position measurements at [0, 0, 0]:
      Filter pulls nominal position back to origin.
    - After 5 updates (5 s), position error < 1.0 m.
    """
    g = -standard_gravity[2]
    init_state = NominalState(
        t=0.0,
        p=np.array([10.0, -10.0, 5.0]),  # 15 m error
        v=np.zeros(3),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        ba=np.zeros(3),
        bg=np.zeros(3)
    )
    init_cov = healthy_initial_cov.copy()
    init_cov[IDX_POS, IDX_POS] = 100.0 * np.eye(3)  # std 10m to match 15m initial error
    filter_inst = ESKF(init_state, init_cov, default_config, g_n=standard_gravity)

    f_b = np.array([0.0, 0.0, g])
    omega_b = np.zeros(3)
    dt = 0.1

    for step in range(50):  # 5 seconds
        filter_inst.predict(f_b, omega_b, dt)
        if (step + 1) % 10 == 0:
            accepted = filter_inst.update_gnss_pos(np.zeros(3), is_new_fix=True)
            assert accepted, "GNSS update within expected covariance must be accepted"

    final_pos_err = np.linalg.norm(filter_inst.state.p)
    assert final_pos_err < 1.0, f"Position failed to converge: {final_pos_err} m"


# =============================================================================
# Test 7: GNSS Outlier Rejection (Chi-square gate)
# =============================================================================
def test_test_7_gnss_outlier_rejection(standard_gravity, default_config, healthy_initial_cov):
    """
    Measurable criteria:
    - Filter tracking stationary target at origin.
    - Inject an unphysical 200-meter GNSS spike: z_outlier = [200.0, 0.0, 0.0].
    - Filter Chi-square gate rejects outlier (accepted == False).
    - State position remains within 0.1 m of origin.
    - Covariance is NOT corrupted.
    """
    g = -standard_gravity[2]
    init_state = NominalState(t=0.0, p=np.zeros(3), v=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]))
    filter_inst = ESKF(init_state, healthy_initial_cov, default_config, g_n=standard_gravity)

    # Let filter settle on origin with 2 good fixes
    filter_inst.update_gnss_pos(np.zeros(3), is_new_fix=True)
    filter_inst.update_gnss_pos(np.zeros(3), is_new_fix=True)

    pos_before = filter_inst.state.p.copy()
    cov_before = filter_inst.P.copy()

    # Now introduce huge outlier
    z_outlier = np.array([200.0, 0.0, 0.0])
    accepted = filter_inst.update_gnss_pos(z_outlier, is_new_fix=True)

    assert not accepted, "Outlier must be rejected by Chi-square gate"
    assert np.allclose(filter_inst.state.p, pos_before, atol=1e-12), "State was corrupted by outlier"
    assert np.allclose(filter_inst.P, cov_before, atol=1e-12), "Covariance was corrupted by outlier"
    assert filter_inst.innovations[-1].mahalanobis_sq > default_config.chi2_threshold_pos


# =============================================================================
# Test 8: Duplicate GNSS Rejection (Sample-and-hold check)
# =============================================================================
def test_test_8_duplicate_gnss_rejection(standard_gravity, default_config, healthy_initial_cov):
    """
    Measurable criteria:
    - Feed identical position with is_new_fix == False (mimicking smartphone 10 Hz sample-and-hold).
    - Filter returns False.
    - Innovation record logs is_new_fix == False.
    - Covariance P does NOT artificially deflate.
    """
    init_state = NominalState(t=0.0, p=np.zeros(3), v=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]))
    filter_inst = ESKF(init_state, healthy_initial_cov, default_config, g_n=standard_gravity)

    cov_before = filter_inst.P.copy()
    accepted = filter_inst.update_gnss_pos(np.zeros(3), is_new_fix=False)

    assert not accepted, "Duplicate sample-and-hold measurement must NOT trigger update"
    assert np.allclose(filter_inst.P, cov_before), "Covariance changed on duplicate measurement"
    assert not filter_inst.innovations[-1].is_new_fix


# =============================================================================
# Test 9: Attitude Error / Injection / Reset Consistency
# =============================================================================
def test_test_9_attitude_error_injection_and_observability(standard_gravity, default_config, healthy_initial_cov):
    """
    Scientific Correction #4 & #5:
    Part A: Mathematical verification of multiplicative attitude injection and reset Jacobian.
    Part B: Observability of heading error under vehicle motion using GNSS velocity updates.
    """
    # Part A: Direct injection math
    nom = NominalState(
        t=0.0,
        p=np.zeros(3),
        v=np.zeros(3),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        ba=np.zeros(3),
        bg=np.zeros(3)
    )
    # 5 degree yaw error around Z in navigation frame
    yaw_err = np.deg2rad(5.0)
    delta_x = np.zeros(STATE_DIM)
    delta_x[IDX_ATT] = np.array([0.0, 0.0, yaw_err])

    P_post = healthy_initial_cov.copy()
    nom_updated, P_reset = inject_and_reset(nom, delta_x, P_post)

    roll, pitch, yaw = quat_to_euler(nom_updated.q)
    assert np.isclose(yaw, yaw_err, atol=1e-6), f"Attitude injection yaw mismatch: {yaw} vs {yaw_err}"
    assert np.allclose(P_reset, P_reset.T), "Reset covariance not symmetric"
    min_eig = np.min(np.linalg.eigvalsh(P_reset))
    assert min_eig > 0.0, "Reset covariance not positive definite"

    # Part B: Tilt observability under gravity coupling with GNSS velocity updates
    # Under stationary conditions, gravity g^n = [0, 0, -g] couples pitch and roll
    # directly into horizontal velocity errors via F[VEL, ATT] = -[g^n]_x.
    # A 5-degree pitch error projects ~g * sin(5 deg) ~ 0.85 m/s^2 into horizontal velocity.
    # GNSS velocity updates (true v = [0, 0, 0]) observe this velocity residual and
    # correct the tilt error via Kalman gain.
    pitch_err = np.deg2rad(5.0)
    g = -standard_gravity[2]
    nom_tilted = NominalState(
        t=0.0,
        p=np.zeros(3),
        v=np.zeros(3),
        q=euler_to_quat(0.0, pitch_err, 0.0),  # 5 deg pitch error
        ba=np.zeros(3),
        bg=np.zeros(3)
    )
    init_cov_tilt = healthy_initial_cov.copy()
    init_cov_tilt[IDX_ATT, IDX_ATT] = (np.deg2rad(10.0) ** 2) * np.eye(3)

    filter_inst = ESKF(nom_tilted, init_cov_tilt, default_config, g_n=standard_gravity)

    initial_pitch_err = abs(quat_to_euler(filter_inst.state.q)[1])

    dt = 0.1
    f_b_stat = np.array([0.0, 0.0, g])
    omega_b_stat = np.zeros(3)

    # Propagate with gravity and apply velocity updates at 10 Hz for 20 steps (2 seconds)
    for _ in range(20):
        filter_inst.predict(f_b_stat, omega_b_stat, dt)
        filter_inst.update_gnss_vel(np.zeros(3), is_new_fix=True)

    final_pitch_err = abs(quat_to_euler(filter_inst.state.q)[1])
    pitch_reduction = (initial_pitch_err - final_pitch_err) / initial_pitch_err

    assert pitch_reduction > 0.80, f"Tilt error was not sufficiently reduced: {pitch_reduction:.2%}"
    assert final_pitch_err < np.deg2rad(1.0), f"Final pitch error too high: {np.rad2deg(final_pitch_err)} deg"


# =============================================================================
# Test 10: Quaternion Convention Consistency
# =============================================================================
def test_test_10_quaternion_convention_consistency():
    """
    Verifies that ESKF quaternion conventions match frames.py:
    - Scalar-first Hamilton: [qw, qx, qy, qz].
    - Body-to-Navigation rotation: v^n = C_b^n(q) @ v^b.
    - Navigation error convention: q_true = Delta_q(delta_theta^n) (x) q.
    """
    # 90-degree yaw rotation around Z
    q_z90 = euler_to_quat(0.0, 0.0, np.pi / 2.0)
    C = quat_to_dcm(q_z90)

    # Body vector along body X should map to Nav Y (East rotated 90 deg -> North)
    v_b = np.array([1.0, 0.0, 0.0])
    v_n = C @ v_b
    assert np.allclose(v_n, [0.0, 1.0, 0.0], atol=1e-12)

    # Nav error delta_theta along Z of 10 degrees
    delta_theta = np.array([0.0, 0.0, np.deg2rad(10.0)])
    delta_q = rotvec_to_quat(delta_theta)
    q_comp = quat_normalize(quat_multiply(delta_q, q_z90))

    roll, pitch, yaw = quat_to_euler(q_comp)
    assert np.isclose(yaw, np.pi / 2.0 + np.deg2rad(10.0), atol=1e-6)


# =============================================================================
# Test 11: Covariance Symmetry
# =============================================================================
def test_test_11_covariance_symmetry(standard_gravity, default_config, healthy_initial_cov):
    """
    Measurable criteria:
    - Run 300 cycles of propagation and measurement updates with random variations.
    - After every cycle, verify |P - P^T| < 1e-12.
    """
    g = -standard_gravity[2]
    filter_inst = ESKF(
        NominalState(t=0.0, p=np.zeros(3), v=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0])),
        healthy_initial_cov,
        default_config,
        g_n=standard_gravity
    )

    np.random.seed(42)
    dt = 0.1

    for step in range(300):
        f_b = np.array([0.0, 0.0, g]) + np.random.randn(3) * 0.02
        omega_b = np.random.randn(3) * 0.001
        filter_inst.predict(f_b, omega_b, dt)

        if step % 10 == 0:
            z_pos = filter_inst.state.p + np.random.randn(3) * 0.2
            filter_inst.update_gnss_pos(z_pos, is_new_fix=True)

        sym_err = np.max(np.abs(filter_inst.P - filter_inst.P.T))
        assert sym_err < 1e-12, f"Covariance asymmetry detected at step {step}: {sym_err}"


# =============================================================================
# Test 12: Covariance Numerical Validity (Positive Semi-Definiteness)
# =============================================================================
def test_test_12_covariance_numerical_validity(standard_gravity, default_config, healthy_initial_cov):
    """
    Scientific Correction #1:
    - No artificial eigenvalue clipping.
    - Verify that all eigenvalues of P remain strictly positive (within numerical precision tol -1e-14).
    - Verify all diagonal elements P_ii > 0.
    """
    g = -standard_gravity[2]
    filter_inst = ESKF(
        NominalState(t=0.0, p=np.zeros(3), v=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0])),
        healthy_initial_cov,
        default_config,
        g_n=standard_gravity
    )

    dt = 0.1
    for step in range(200):
        filter_inst.predict(np.array([0.0, 0.0, g]), np.zeros(3), dt)
        if step % 10 == 0:
            filter_inst.update_gnss_pos(np.zeros(3), is_new_fix=True)

        eigs = np.linalg.eigvalsh(filter_inst.P)
        min_eig = np.min(eigs)
        assert min_eig > -1e-14, f"Negative eigenvalue detected: {min_eig}"
        assert np.all(np.diag(filter_inst.P) > 0.0), "Non-positive diagonal in covariance"


# =============================================================================
# Test 13: Measurement Gating Threshold Behavior
# =============================================================================
def test_test_13_measurement_gating_threshold_behavior(default_config, healthy_initial_cov):
    """
    Explicit threshold test:
    - For 3 DOF Chi-square with threshold gamma = 16.27:
      residual yielding d^2 = 15.0 -> accepted
      residual yielding d^2 = 17.0 -> rejected
    """
    S = np.eye(3, dtype=np.float64) * 4.0  # std 2.0
    threshold = default_config.chi2_threshold_pos

    # Target d^2 = 15.0 (below 16.27) -> r = sqrt(15.0 * 4.0 / 3) * ones(3)
    target_d2_pass = 15.0
    r_pass = np.full(3, np.sqrt(target_d2_pass * 4.0 / 3.0))
    accepted_pass, d2_computed_pass = gate_measurement(r_pass, S, threshold)
    assert accepted_pass, f"Expected accept for d2={d2_computed_pass}"
    assert np.isclose(d2_computed_pass, target_d2_pass, atol=1e-6)

    # Target d^2 = 17.0 (above 16.27) -> r = sqrt(17.0 * 4.0 / 3) * ones(3)
    target_d2_fail = 17.0
    r_fail = np.full(3, np.sqrt(target_d2_fail * 4.0 / 3.0))
    accepted_fail, d2_computed_fail = gate_measurement(r_fail, S, threshold)
    assert not accepted_fail, f"Expected reject for d2={d2_computed_fail}"
    assert np.isclose(d2_computed_fail, target_d2_fail, atol=1e-6)


# =============================================================================
# Test 14: Strict Causality / Zero Future Leakage
# =============================================================================
def test_test_14_strict_causality_zero_future_leakage(standard_gravity, default_config, healthy_initial_cov):
    """
    Measurable criteria:
    - Filter trajectory up to step k.
    - Run separate filter instance up to step N (where N > k).
    - State at step k must be bitwise identical between both runs (zero future information leakage).
    """
    g = -standard_gravity[2]
    np.random.seed(123)
    N = 100
    k = 50
    dt = 0.1

    # Generate synthetic input sequence
    accel_seq = [np.array([0.0, 0.0, g]) + np.random.randn(3) * 0.01 for _ in range(N)]
    gyro_seq = [np.random.randn(3) * 0.001 for _ in range(N)]
    gnss_seq = [np.random.randn(3) * 0.5 if i % 10 == 0 else None for i in range(N)]

    # Run filter 1 up to k
    filter_short = ESKF(
        NominalState(t=0.0, p=np.zeros(3), v=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0])),
        healthy_initial_cov,
        default_config,
        g_n=standard_gravity
    )
    for i in range(k):
        filter_short.predict(accel_seq[i], gyro_seq[i], dt)
        if gnss_seq[i] is not None:
            filter_short.update_gnss_pos(gnss_seq[i], is_new_fix=True)

    state_at_k = filter_short.state.copy()
    P_at_k = filter_short.P.copy()

    # Run filter 2 up to N, snapshot at k
    filter_long = ESKF(
        NominalState(t=0.0, p=np.zeros(3), v=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0])),
        healthy_initial_cov,
        default_config,
        g_n=standard_gravity
    )
    state_long_at_k = None
    P_long_at_k = None
    for i in range(N):
        filter_long.predict(accel_seq[i], gyro_seq[i], dt)
        if gnss_seq[i] is not None:
            filter_long.update_gnss_pos(gnss_seq[i], is_new_fix=True)
        if i == k - 1:
            state_long_at_k = filter_long.state.copy()
            P_long_at_k = filter_long.P.copy()

    assert np.array_equal(state_at_k.p, state_long_at_k.p), "Position causality violation"
    assert np.array_equal(state_at_k.v, state_long_at_k.v), "Velocity causality violation"
    assert np.array_equal(state_at_k.q, state_long_at_k.q), "Quaternion causality violation"
    assert np.array_equal(state_at_k.ba, state_long_at_k.ba), "Accel bias causality violation"
    assert np.array_equal(state_at_k.bg, state_long_at_k.bg), "Gyro bias causality violation"
    assert np.array_equal(P_at_k, P_long_at_k), "Covariance causality violation"
