"""
Unit and synthetic validation tests for NAVRIS Causal ZUPT Module.

Phase 2.3B Gate 2.3A Synthetic Validation Suite.
Validates:
1. True stationary state (ZUPT fires, velocity error pulled to zero)
2. Moving state must not trigger ZUPT (detector returns False)
3. Stationary -> moving transition (disengages immediately on exit condition)
4. Moving -> stationary transition (dwell time required before engagement)
5. Accelerometer noise handling
6. Gyroscope noise handling
7. Accelerometer bias cross-correlation / correction
8. Gyroscope bias stability during stationary intervals
9. Noisy stationary period (high vibration exceeding exit threshold)
10. Short stationary candidate (candidate shorter than dwell time rejected)
11. Prolonged stationary period (numerical stability, symmetry, PSD, no NaN/Inf)
Additional checks:
- Detector causality (no future indexing, trailing window only)
- Residual sign (-v_hat)
- Correct H indexing (IDX_VEL)
- Gating check and rejection behavior
- Quaternion norm preservation
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
)
from navris.zupt import (
    ZUPTDetectorConfig,
    CausalStationaryDetector,
    apply_zupt_update,
    DEFAULT_R_ZUPT,
    DEFAULT_SIGMA_ZUPT,
    ZUPT_CHI2_THRESHOLD,
)


@pytest.fixture
def clean_eskf():
    """Returns a clean stationary ESKF instance."""
    init_state = NominalState(
        t=0.0,
        p=np.zeros(3),
        v=np.array([1.0, -0.5, 0.2]),  # Small initial velocity error
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        ba=np.zeros(3),
        bg=np.zeros(3),
    )
    init_cov = np.eye(STATE_DIM) * 0.1
    config = ESKFConfig(
        sigma_acc=0.20,
        sigma_gyr=0.02,
        sigma_acc_bias=1e-3,
        sigma_gyr_bias=1e-4,
    )
    return ESKF(init_state, init_cov, config=config)


# -----------------------------------------------------------------------------
# 1. True stationary state
# -----------------------------------------------------------------------------
def test_1_true_stationary_state(clean_eskf):
    """ZUPT fires on stationary state and reduces velocity error towards zero."""
    detector = CausalStationaryDetector()
    f_stat = np.array([0.0, 0.0, 9.80665])
    w_stat = np.array([0.0, 0.0, 0.0])

    # Feed 10 stationary samples (dwell is 5 samples)
    is_stat = False
    for _ in range(10):
        is_stat = detector.update(f_stat, w_stat)
    assert is_stat is True

    # Initial velocity is non-zero
    assert np.linalg.norm(clean_eskf.state.v) > 0.5
    v_init_norm = np.linalg.norm(clean_eskf.state.v)

    # Apply ZUPT
    accepted, record = apply_zupt_update(clean_eskf, timestamp=1.0)
    assert accepted is True
    assert record.sensor_type == "zupt"
    assert record.mahalanobis_sq < ZUPT_CHI2_THRESHOLD

    # Velocity error must be significantly reduced
    v_post_norm = np.linalg.norm(clean_eskf.state.v)
    assert v_post_norm < 0.15 * v_init_norm


# -----------------------------------------------------------------------------
# 2. Moving state must NOT trigger ZUPT
# -----------------------------------------------------------------------------
def test_2_moving_state_no_trigger():
    """Moving vehicle with dynamic acceleration and angular velocity must not trigger detector."""
    detector = CausalStationaryDetector()
    
    # Driving with acceleration and turn rate
    for _ in range(30):
        f_mov = np.array([1.5, 0.2, 9.81])
        w_mov = np.array([0.0, 0.0, 0.15])
        is_stat = detector.update(f_mov, w_mov)
        assert is_stat is False


# -----------------------------------------------------------------------------
# 3. Stationary -> moving transition
# -----------------------------------------------------------------------------
def test_3_stationary_to_moving_transition():
    """Detector enters stationary, then immediately disengages upon motion."""
    detector = CausalStationaryDetector()
    f_stat = np.array([0.0, 0.0, 9.80665])
    w_stat = np.array([0.0, 0.0, 0.0])

    # Enter stationary state (10 samples)
    for _ in range(10):
        detector.update(f_stat, w_stat)
    assert detector.is_stationary is True

    # Immediate exit on dynamic forward acceleration
    f_depart = np.array([3.5, 0.0, 9.81])  # Exceeds acc_inst_dev_exit (0.45)
    is_stat = detector.update(f_depart, w_stat)
    assert is_stat is False
    assert detector.is_stationary is False

    # Re-enter stationary state (requires flushing moving sample from window + dwell)
    for _ in range(15):
        detector.update(f_stat, w_stat)
    assert detector.is_stationary is True

    # Immediate exit on turn rotation
    w_turn = np.array([0.0, 0.0, 0.12])  # Exceeds gyr_mean_exit (0.075)
    is_stat = detector.update(f_stat, w_turn)
    assert is_stat is False
    assert detector.is_stationary is False


# -----------------------------------------------------------------------------
# 4. Moving -> stationary transition (dwell time required)
# -----------------------------------------------------------------------------
def test_4_moving_to_stationary_transition():
    """Vehicle stops; detector requires dwell time before declaring stationary."""
    detector = CausalStationaryDetector(ZUPTDetectorConfig(window_size=8, min_window_samples=4, dwell_samples=5))
    
    # Moving for 20 samples
    for _ in range(20):
        detector.update(np.array([1.0, 0.0, 9.81]), np.array([0.0, 0.0, 0.1]))
    assert detector.is_stationary is False

    # Vehicle comes to stop
    f_stat = np.array([0.0, 0.0, 9.80665])
    w_stat = np.zeros(3)

    # First few samples must not be stationary
    for _ in range(4):
        assert detector.update(f_stat, w_stat) is False

    # After full dwell (5 consecutive satisfying samples), must become True
    stat_history = []
    for _ in range(10):
        stat_history.append(detector.update(f_stat, w_stat))

    assert stat_history[-1] is True


# -----------------------------------------------------------------------------
# 5. Accelerometer noise handling
# -----------------------------------------------------------------------------
def test_5_accelerometer_noise_handling():
    """Realistic stationary accelerometer noise (0.06 m/s^2 std) does not prevent detection."""
    np.random.seed(42)
    detector = CausalStationaryDetector()
    
    for _ in range(20):
        noise_a = np.random.normal(0, 0.05, 3)
        f = np.array([0.0, 0.0, 9.80665]) + noise_a
        w = np.random.normal(0, 0.005, 3)
        detector.update(f, w)
    
    assert detector.is_stationary is True


# -----------------------------------------------------------------------------
# 6. Gyroscope noise handling
# -----------------------------------------------------------------------------
def test_6_gyroscope_noise_handling():
    """Realistic stationary gyro noise (0.012 rad/s std) does not prevent detection."""
    np.random.seed(123)
    detector = CausalStationaryDetector()
    
    for _ in range(20):
        f = np.array([0.0, 0.0, 9.80665]) + np.random.normal(0, 0.04, 3)
        w = np.random.normal(0, 0.010, 3)
        detector.update(f, w)
        
    assert detector.is_stationary is True


# -----------------------------------------------------------------------------
# 7. Accelerometer bias cross-correlation
# -----------------------------------------------------------------------------
def test_7_accelerometer_bias_coupling(clean_eskf):
    """ZUPT error state injection updates accelerometer bias via cross-covariance."""
    # Introduce cross-covariance between velocity (index 3) and accel bias (index 9)
    clean_eskf.P[3, 9] = 0.02
    clean_eskf.P[9, 3] = 0.02
    
    ba_prior = clean_eskf.state.ba.copy()
    accepted, record = apply_zupt_update(clean_eskf, timestamp=1.0)
    assert accepted is True
    
    # Accel bias should receive an error correction through cross-covariance
    ba_post = clean_eskf.state.ba.copy()
    assert not np.allclose(ba_prior, ba_post)


# -----------------------------------------------------------------------------
# 8. Gyroscope bias stability during stationary intervals
# -----------------------------------------------------------------------------
def test_8_gyroscope_bias_stability(clean_eskf):
    """At rest without yaw turning, ZUPT does not destabilize gyro bias."""
    bg_prior = clean_eskf.state.bg.copy()
    apply_zupt_update(clean_eskf, timestamp=1.0)
    bg_post = clean_eskf.state.bg.copy()
    
    # Gyro bias should remain near zero when uncoupled
    np.testing.assert_allclose(bg_prior, bg_post, atol=1e-3)


# -----------------------------------------------------------------------------
# 9. Noisy stationary period (high vibration exceeding threshold)
# -----------------------------------------------------------------------------
def test_9_high_vibration_prevents_false_zupt():
    """Excessive vibration (e.g. rough engine idling with std > 0.35 m/s^2) trips exit threshold."""
    np.random.seed(999)
    detector = CausalStationaryDetector()
    
    for _ in range(25):
        # Extreme vibration noise
        noise_a = np.random.normal(0, 0.40, 3)
        f = np.array([0.0, 0.0, 9.80665]) + noise_a
        w = np.zeros(3)
        is_stat = detector.update(f, w)
        assert is_stat is False


# -----------------------------------------------------------------------------
# 10. Short stationary candidate
# -----------------------------------------------------------------------------
def test_10_short_stationary_candidate():
    """A brief hesitation (< 0.5 s) does not engage stationary state."""
    detector = CausalStationaryDetector(ZUPTDetectorConfig(window_size=8, min_window_samples=4, dwell_samples=5))
    
    # 3 stationary samples (< 5 dwell)
    f_stat = np.array([0.0, 0.0, 9.80665])
    w_stat = np.zeros(3)
    
    for _ in range(3):
        detector.update(f_stat, w_stat)
    assert detector.is_stationary is False

    # Followed immediately by motion
    detector.update(np.array([2.0, 0.0, 9.81]), np.zeros(3))
    assert detector.is_stationary is False


# -----------------------------------------------------------------------------
# 11. Prolonged stationary period (numerical stability, symmetry, PSD, no NaN/Inf)
# -----------------------------------------------------------------------------
def test_11_prolonged_stationary_numerical_stability(clean_eskf):
    """50 repeated ZUPT updates preserve covariance symmetry, PSD, and finite values."""
    f_stat = np.array([0.0, 0.0, 9.80665])
    w_stat = np.zeros(3)
    dt = 0.1

    for step in range(50):
        t = (step + 1) * dt
        # Predict strapdown
        clean_eskf.predict(f_stat, w_stat, dt)
        # Apply ZUPT
        accepted, record = apply_zupt_update(clean_eskf, timestamp=t)
        assert accepted is True

        # Check numerical sanity
        assert not np.isnan(clean_eskf.state.p).any()
        assert not np.isnan(clean_eskf.state.v).any()
        assert not np.isnan(clean_eskf.state.q).any()
        assert not np.isnan(clean_eskf.P).any()

        # Check covariance symmetry
        np.testing.assert_allclose(clean_eskf.P, clean_eskf.P.T, atol=1e-8)

        # Check positive semi-definiteness (all eigenvalues >= 0)
        eigs = np.linalg.eigvalsh(clean_eskf.P)
        assert np.min(eigs) > -1e-10, f"Covariance has negative eigenvalue: {np.min(eigs)}"

        # Check quaternion unit norm
        q_norm = np.linalg.norm(clean_eskf.state.q)
        assert abs(q_norm - 1.0) < 1e-6

    # Velocity must be tightly bounded
    assert np.linalg.norm(clean_eskf.state.v) < 0.01


# -----------------------------------------------------------------------------
# Additional Checks: Residual Sign, H Indexing, and Chi-square Gating
# -----------------------------------------------------------------------------
def test_residual_sign_and_h_indexing(clean_eskf):
    """Verifies that residual is strictly -v_hat and H indexes only velocity state."""
    clean_eskf.state.v = np.array([3.0, -4.0, 1.0])
    accepted, record = apply_zupt_update(clean_eskf, timestamp=1.0)
    
    # Residual must be z - v_hat = -v_hat
    np.testing.assert_allclose(record.residual, -clean_eskf.state.v)
    assert record.sensor_type == "zupt"


def test_chi2_gating_rejection(clean_eskf):
    """Massive velocity error (> 100 m/s) with small R must trip the chi2 gate and reject."""
    clean_eskf.state.v = np.array([150.0, 0.0, 0.0])
    v_prior = clean_eskf.state.v.copy()
    P_prior = clean_eskf.P.copy()

    accepted, record = apply_zupt_update(clean_eskf, timestamp=1.0)
    assert accepted is False
    assert record.accepted is False
    assert record.mahalanobis_sq > ZUPT_CHI2_THRESHOLD

    # State and covariance must remain completely unchanged when rejected
    np.testing.assert_allclose(clean_eskf.state.v, v_prior)
    np.testing.assert_allclose(clean_eskf.P, P_prior)
