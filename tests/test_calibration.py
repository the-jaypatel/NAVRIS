"""
Unit Tests for NAVRIS Causal Sensor-Frame / Extrinsic Calibration.

Verifies:
1. Synthetic gravity tilt recovery (known roll and pitch).
2. Standstill detection on stationary vs dynamic data.
3. Forward mounting yaw angle recovery on synthetic acceleration.
4. Turn kinematics cross-product behavior under pure lateral vs longitudinal contamination.
5. Gyro channel mapping identification.
6. Strict causality: zero lookahead leakage.
7. Deterministic convention verification: known -45 deg mounting angle with explicit frame assertions.
"""

import numpy as np
import pandas as pd
import pytest

from navris.calibration import (
    detect_standstill_intervals,
    estimate_gravity_leveling,
    estimate_mounting_yaw_from_motion,
    evaluate_turn_cross_product,
    identify_gyro_mapping,
    calibrate_causal_s1,
    dcm_to_quat
)
from navris.inertial.frames import (
    euler_to_quat,
    quat_to_euler,
    rotate_vector,
    quat_to_dcm
)


def test_gravity_leveling_recovery_known_tilt():
    """Verifies that known roll and pitch tilts are accurately recovered from specific force."""
    g = 9.80665
    
    # Test cases: (known_roll_deg, known_pitch_deg)
    cases = [
        (0.0, 0.0),
        (3.5, 0.0),
        (0.0, -4.2),
        (-2.8, 5.1),
        (1.5, 2.0)
    ]
    
    for roll_deg, pitch_deg in cases:
        r_rad = np.radians(roll_deg)
        p_rad = np.radians(pitch_deg)
        
        q_true = euler_to_quat(r_rad, p_rad, 0.0)
        C = quat_to_dcm(q_true) # body to nav
        f_b = C.T @ np.array([0.0, 0.0, g])
        
        n = 50
        df = pd.DataFrame({
            'time_s': np.linspace(0, 5, n),
            'phone_accel_x_mps2': np.full(n, f_b[0]),
            'phone_accel_y_mps2': np.full(n, f_b[1]),
            'phone_accel_z_mps2': np.full(n, f_b[2]),
            'phone_gyro_x_radps': np.zeros(n),
            'phone_gyro_y_radps': np.zeros(n),
            'phone_gyro_z_radps': np.zeros(n)
        })
        
        res = estimate_gravity_leveling(df)
        assert res.is_valid
        assert res.quality == 'HIGH'
        assert abs(res.residual_g_mps2) < 1e-4
        
        f_leveled = rotate_vector(res.q_level, f_b)
        assert abs(f_leveled[0]) < 1e-5
        assert abs(f_leveled[1]) < 1e-5
        assert abs(f_leveled[2] - g) < 1e-5
        
        assert abs(res.roll_deg - roll_deg) < 0.05
        assert abs(res.pitch_deg - pitch_deg) < 0.05


def test_standstill_detection_synthetic():
    """Verifies standstill intervals are identified on quiet sensor data and rejected on dynamic data."""
    n_stat = 100
    n_dyn = 100
    dt = 0.1
    t = np.arange(n_stat + n_dyn) * dt

    np.random.seed(42)
    ax_stat = np.random.normal(0.0, 0.02, n_stat)
    ay_stat = np.random.normal(0.0, 0.02, n_stat)
    az_stat = np.random.normal(9.81, 0.02, n_stat)
    gx_stat = np.random.normal(0.0, 0.005, n_stat)
    gy_stat = np.random.normal(0.0, 0.005, n_stat)
    gz_stat = np.random.normal(0.0, 0.005, n_stat)
    spd_stat = np.zeros(n_stat)

    ax_dyn = np.random.normal(1.0, 0.5, n_dyn)
    ay_dyn = np.random.normal(-0.5, 0.5, n_dyn)
    az_dyn = np.random.normal(9.81, 0.5, n_dyn)
    gx_dyn = np.random.normal(0.1, 0.08, n_dyn)
    gy_dyn = np.random.normal(0.3, 0.08, n_dyn)
    gz_dyn = np.random.normal(0.2, 0.08, n_dyn)
    spd_dyn = np.full(n_dyn, 12.0)

    df = pd.DataFrame({
        'time_s': t,
        'phone_accel_x_mps2': np.concatenate([ax_stat, ax_dyn]),
        'phone_accel_y_mps2': np.concatenate([ay_stat, ay_dyn]),
        'phone_accel_z_mps2': np.concatenate([az_stat, az_dyn]),
        'phone_gyro_x_radps': np.concatenate([gx_stat, gx_dyn]),
        'phone_gyro_y_radps': np.concatenate([gy_stat, gy_dyn]),
        'phone_gyro_z_radps': np.concatenate([gz_stat, gz_dyn]),
        'phone_gps_speed_mps': np.concatenate([spd_stat, spd_dyn])
    })

    intervals = detect_standstill_intervals(df, dt=dt, window_len_s=2.0)
    assert len(intervals) >= 1
    assert intervals[0].start_time_s == 0.0
    assert intervals[0].end_time_s <= (n_stat * dt)
    assert intervals[0].duration_s >= 5.0


def test_forward_mounting_yaw_recovery_synthetic():
    """Verifies that forward mounting angle is recovered from straight forward acceleration."""
    known_angles_deg = [0.0, 45.0, -127.0, 180.0, -45.0]
    
    for phi_deg in known_angles_deg:
        phi_rad = np.radians(phi_deg)
        fwd_vec = np.array([np.cos(phi_rad), np.sin(phi_rad), 0.0])
        
        n = 30
        dt = 0.1
        t = np.arange(n) * dt
        a_fwd = 1.5
        ax = fwd_vec[0] * a_fwd
        ay = fwd_vec[1] * a_fwd
        az = 9.80665
        
        df = pd.DataFrame({
            'time_s': t,
            'phone_accel_x_mps2': np.full(n, ax),
            'phone_accel_y_mps2': np.full(n, ay),
            'phone_accel_z_mps2': np.full(n, az),
            'phone_gyro_x_radps': np.zeros(n),
            'phone_gyro_y_radps': np.zeros(n),
            'phone_gyro_z_radps': np.zeros(n),
            'phone_gps_speed_mps': np.linspace(0, 4.5, n)
        })
        
        q_level = np.array([1.0, 0.0, 0.0, 0.0])
        res = estimate_mounting_yaw_from_motion(df, q_level=q_level, min_accel_mps2=0.5)
        
        assert res.is_valid
        diff = (res.forward_angle_phone_frame_deg - phi_deg + 180.0) % 360.0 - 180.0
        assert abs(diff) < 0.1
        assert abs(res.mounting_yaw_vehicle_from_phone_deg - (-phi_deg) + 180.0) % 360.0 - 180.0 < 0.1


def test_deterministic_convention_synthetic():
    """
    DETERMINISTIC CONVENTION TEST:
    Explicitly defines:
    - Phone Frame (s): +X right, +Y top, +Z face-up.
    - Vehicle Frame (v): FLU (+X Forward, +Y Left, +Z Up).
    - True Phone-to-Vehicle Mounting Yaw: -45.0 deg.
      That is: vehicle forward axis in phone frame is at angle -45 deg:
      u_fwd_s = [cos(-45°), sin(-45°), 0] = [1/sqrt(2), -1/sqrt(2), 0].
    
    Verifies that:
    1. forward_angle_phone_frame_deg == -45.0 deg.
    2. mounting_yaw_vehicle_from_phone_deg == +45.0 deg (rotation mapping phone to vehicle).
    3. R_s_v @ u_fwd_s == [1, 0, 0]^T (+X_v Forward).
    4. R_s_v @ u_left_s == [0, 1, 0]^T (+Y_v Left).
    5. R_s_v @ u_up_s == [0, 0, 1]^T (+Z_v Up).
    6. det(R_s_v) == +1.0 and R @ R.T == I.
    7. Catches +45 vs -45, +135 vs -135, R vs R.T, body->vehicle vs vehicle->body.
    """
    g = 9.80665
    true_yaw_deg = -45.0
    true_yaw_rad = np.radians(true_yaw_deg)
    
    # 1. Physical forward vector in phone frame:
    # Points at angle -45 deg in the horizontal plane: [1/sqrt(2), -1/sqrt(2), 0]
    u_fwd_true_s = np.array([np.cos(true_yaw_rad), np.sin(true_yaw_rad), 0.0], dtype=np.float64)
    u_up_true_s = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    u_left_true_s = np.cross(u_up_true_s, u_fwd_true_s) # [1/sqrt(2), 1/sqrt(2), 0]

    # Expected true transformation R_s^v (Row 0: u_fwd, Row 1: u_left, Row 2: u_up)
    R_s_v_expected = np.array([
        [ np.cos(true_yaw_rad),  np.sin(true_yaw_rad), 0.0],
        [-np.sin(true_yaw_rad),  np.cos(true_yaw_rad), 0.0],
        [ 0.0,                   0.0,                  1.0]
    ], dtype=np.float64)
    
    # Check that expected R maps u_fwd to [1, 0, 0]
    np.testing.assert_allclose(R_s_v_expected @ u_fwd_true_s, [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(R_s_v_expected @ u_left_true_s, [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(R_s_v_expected @ u_up_true_s, [0.0, 0.0, 1.0], atol=1e-12)

    # 2. Build synthetic data:
    # Standstill interval: t in [0, 20.0s] (200 samples)
    dt = 0.1
    n_stat = 200
    t_stat = np.arange(n_stat) * dt
    f_stat = np.array([0.0, 0.0, g]) # specific force opposite gravity (Up)

    # Motion interval: t in [20.0, 40.0s] (200 samples)
    # Forward acceleration along vehicle +X_v with 2.0 m/s^2 for 5 seconds (t in [25, 30])
    n_mot = 200
    t_mot = 20.0 + np.arange(n_mot) * dt
    
    a_phone_mot = np.tile(f_stat, (n_mot, 1))
    speed_mot = np.zeros(n_mot)
    
    # Acceleration event at t in [25, 30] (idx 50 to 100 in motion window)
    a_lon = 2.0 # m/s^2
    # In phone frame, forward acceleration is a_lon * u_fwd_true_s:
    a_fwd_s = a_lon * u_fwd_true_s
    for idx in range(50, 100):
        a_phone_mot[idx] += a_fwd_s
        speed_mot[idx] = (idx - 50) * dt * a_lon
    for idx in range(100, n_mot):
        speed_mot[idx] = 50 * dt * a_lon

    df_synth = pd.DataFrame({
        'time_s': np.concatenate([t_stat, t_mot]),
        'phone_accel_x_mps2': np.concatenate([np.full(n_stat, f_stat[0]), a_phone_mot[:, 0]]),
        'phone_accel_y_mps2': np.concatenate([np.full(n_stat, f_stat[1]), a_phone_mot[:, 1]]),
        'phone_accel_z_mps2': np.concatenate([np.full(n_stat, f_stat[2]), a_phone_mot[:, 2]]),
        'phone_gyro_x_radps': np.zeros(n_stat + n_mot),
        'phone_gyro_y_radps': np.zeros(n_stat + n_mot),
        'phone_gyro_z_radps': np.zeros(n_stat + n_mot),
        'phone_gps_speed_mps': np.concatenate([np.zeros(n_stat), speed_mot]),
        'phone_gps_is_new_fix': np.zeros(n_stat + n_mot, dtype=bool)
    })

    # 3. Run calibration
    calib = calibrate_causal_s1(df_synth, t_decision=35.0)

    # 4. Strict assertions:
    assert calib.is_calibrated
    # (a) Forward angle in phone frame must be -45.0 deg (NOT +45.0 deg, NOT -135 deg)
    assert abs(calib.forward_angle_phone_frame_deg - true_yaw_deg) < 0.1
    # (b) Mounting yaw (active phone to vehicle) must be +45.0 deg
    assert abs(calib.mounting_yaw_vehicle_from_phone_deg - (-true_yaw_deg)) < 0.1
    # (c) R_body_vehicle must match expected R_s_v
    np.testing.assert_allclose(calib.R_body_vehicle, R_s_v_expected, atol=1e-3)
    # (d) Check orthonormal properties:
    np.testing.assert_allclose(calib.R_body_vehicle @ calib.R_body_vehicle.T, np.eye(3), atol=1e-6)
    assert abs(np.linalg.det(calib.R_body_vehicle) - 1.0) < 1e-6
    # (e) Check directional alignment:
    # Forward vector rotated to vehicle frame MUST be [1, 0, 0]^T
    v_fwd_v = calib.R_body_vehicle @ u_fwd_true_s
    np.testing.assert_allclose(v_fwd_v, [1.0, 0.0, 0.0], atol=1e-3)
    # Left vector rotated to vehicle frame MUST be [0, 1, 0]^T
    v_left_v = calib.R_body_vehicle @ u_left_true_s
    np.testing.assert_allclose(v_left_v, [0.0, 1.0, 0.0], atol=1e-3)
    # Up vector rotated to vehicle frame MUST be [0, 0, 1]^T
    v_up_v = calib.R_body_vehicle @ u_up_true_s
    np.testing.assert_allclose(v_up_v, [0.0, 0.0, 1.0], atol=1e-3)
    # (f) Check transpose: R_vehicle_body must be R_body_vehicle.T
    np.testing.assert_allclose(calib.R_vehicle_body, calib.R_body_vehicle.T, atol=1e-12)


def test_turn_cross_product_pure_vs_contaminated():
    """Verifies turn cross-product yields exact forward under pure lateral, and bias under braking."""
    fwd_angle = -127.0
    fwd_rad = np.radians(fwd_angle)
    u_fwd = np.array([np.cos(fwd_rad), np.sin(fwd_rad), 0.0])
    u_lat = np.array([-np.sin(fwd_rad), np.cos(fwd_rad), 0.0]) # 90 deg CCW (left)
    
    n = 20
    dt = 0.1
    t = np.arange(n) * dt
    wz = 0.3 # rad/s turning left
    
    # 1. Pure lateral acceleration (a_lat = wz * v = 3.0 m/s^2 along u_lat)
    a_pure = 3.0 * u_lat + np.array([0, 0, 9.80665])
    df_pure = pd.DataFrame({
        'time_s': t,
        'phone_accel_x_mps2': np.full(n, a_pure[0]),
        'phone_accel_y_mps2': np.full(n, a_pure[1]),
        'phone_accel_z_mps2': np.full(n, a_pure[2]),
        'phone_gyro_x_radps': np.zeros(n),
        'phone_gyro_y_radps': np.full(n, wz),
        'phone_gyro_z_radps': np.zeros(n)
    })
    
    q_level = np.array([1.0, 0.0, 0.0, 0.0])
    res_pure = evaluate_turn_cross_product(df_pure, q_level, "Pure Turn", 0.0, 2.0, fwd_angle)
    assert res_pure.is_valid
    assert abs(res_pure.bias_from_forward_deg) < 0.1
    assert res_pure.support_classification == 'SUPPORTED'

    # 2. Contaminated with braking (-1.5 m/s^2 along u_fwd)
    a_contam = (3.0 * u_lat - 1.5 * u_fwd) + np.array([0, 0, 9.80665])
    df_contam = pd.DataFrame({
        'time_s': t,
        'phone_accel_x_mps2': np.full(n, a_contam[0]),
        'phone_accel_y_mps2': np.full(n, a_contam[1]),
        'phone_accel_z_mps2': np.full(n, a_contam[2]),
        'phone_gyro_x_radps': np.zeros(n),
        'phone_gyro_y_radps': np.full(n, wz),
        'phone_gyro_z_radps': np.zeros(n)
    })
    
    res_contam = evaluate_turn_cross_product(df_contam, q_level, "Braking Turn", 0.0, 2.0, fwd_angle)
    assert res_contam.is_valid
    assert abs(res_contam.bias_from_forward_deg - 26.56) < 1.0
    assert res_contam.support_classification == 'PARTIALLY_SUPPORTED'


def test_strict_causality_zero_lookahead():
    """Verifies calibration at t_decision is completely independent of data after t_decision."""
    np.random.seed(123)
    n = 200
    dt = 0.1
    t = np.arange(n) * dt
    
    df1 = pd.DataFrame({
        'time_s': t,
        'phone_accel_x_mps2': np.random.normal(0, 0.1, n),
        'phone_accel_y_mps2': np.random.normal(0, 0.1, n),
        'phone_accel_z_mps2': np.random.normal(9.81, 0.05, n),
        'phone_gyro_x_radps': np.random.normal(0, 0.01, n),
        'phone_gyro_y_radps': np.random.normal(0, 0.01, n),
        'phone_gyro_z_radps': np.random.normal(0, 0.01, n),
        'phone_gps_speed_mps': np.zeros(n),
        'phone_gps_is_new_fix': np.zeros(n, dtype=bool)
    })
    
    df2 = df1.copy()
    future_mask = df2['time_s'] > 10.0
    df2.loc[future_mask, 'phone_accel_x_mps2'] += 100.0
    df2.loc[future_mask, 'phone_accel_y_mps2'] -= 50.0
    df2.loc[future_mask, 'phone_gyro_y_radps'] += 5.0
    
    res1 = calibrate_causal_s1(df1, t_decision=10.0)
    res2 = calibrate_causal_s1(df2, t_decision=10.0)
    
    np.testing.assert_array_equal(res1.leveling.q_level, res2.leveling.q_level)
    np.testing.assert_array_equal(res1.R_body_vehicle, res2.R_body_vehicle)
    np.testing.assert_array_equal(res1.initial_attitude_q, res2.initial_attitude_q)
    assert res1.mounting_yaw_deg == res2.mounting_yaw_deg
