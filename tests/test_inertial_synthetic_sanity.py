"""
NAVRIS Phase 2.2 Forensic Validation: Deterministic Synthetic Sanity Tests.

Tests:
1. Stationary phone at arbitrary 3D orientation -> horizontal acceleration ~ 0.
2. Stationary phone -> position remains stationary over time.
3. Known constant horizontal acceleration -> exact v(t) = a*t and p(t) = 0.5*a*t^2.
4. Pure yaw rotation -> verifies yaw direction and sign.
5. Pure pitch/roll rotation -> verifies gravity projection direction.
6. Quaternion norm -> remains 1.0 under continuous dynamic rotations.
7. Body -> ENU known rotations -> verifies mapping of all 3 principal axes.
8. Timestamp gap -> verifies zero integration across dt > 0.25 s.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from navris.inertial.frames import (
    quat_normalize,
    quat_multiply,
    rotvec_to_quat,
    quat_to_dcm,
    rotate_vector,
    quat_to_euler,
    euler_to_quat
)
from navris.inertial.gravity import normal_gravity_wgs84, gravity_vector_enu
from navris.inertial.strapdown import propagate_strapdown_segment
from navris.inertial.init import align_leveling_from_gravity


def test_sanity_1_stationary_phone_arbitrary_orientation():
    """1. Stationary phone at arbitrary 3D orientation: expected horizontal accel ~ 0."""
    lat = 52.4
    g_mag = normal_gravity_wgs84(lat)
    g_vec = gravity_vector_enu(lat)
    
    # Test across 12 distinct 3D orientations (roll, pitch, yaw)
    for roll in np.linspace(-np.pi/3, np.pi/3, 3):
        for pitch in np.linspace(-np.pi/3, np.pi/3, 2):
            for yaw in np.linspace(0, 2*np.pi, 2):
                q = euler_to_quat(roll, pitch, yaw)
                C_b_n = quat_to_dcm(q)
                C_n_b = C_b_n.T
                
                # In ENU, gravity is [0, 0, -g], so resting reaction is [0, 0, +g]
                f_n_ideal = np.array([0.0, 0.0, g_mag])
                f_b = C_n_b @ f_n_ideal
                
                # Rotate back to ENU
                f_n_rot = rotate_vector(q, f_b)
                a_kinematic = f_n_rot + g_vec
                
                assert np.allclose(a_kinematic[:2], [0.0, 0.0], atol=1e-12)
                assert np.isclose(a_kinematic[2], 0.0, atol=1e-12)


def test_sanity_2_stationary_phone_position_remains_stationary():
    """2. Stationary phone: position should remain stationary."""
    lat = 52.4
    g_mag = normal_gravity_wgs84(lat)
    
    n = 200
    t = np.arange(n) * 0.1 # 20 seconds
    
    # Body tilted by 25 deg roll, 15 deg pitch
    q0 = euler_to_quat(np.radians(25), np.radians(15), np.radians(45))
    C_n_b = quat_to_dcm(q0).T
    f_b_sample = C_n_b @ np.array([0.0, 0.0, g_mag])
    
    accel_b = np.tile(f_b_sample, (n, 1))
    gyro_b = np.zeros((n, 3))
    
    p0 = np.array([100.0, 200.0, 50.0])
    v0 = np.array([0.0, 0.0, 0.0])
    
    traj = propagate_strapdown_segment(
        time_s=t, accel_body=accel_b, gyro_body=gyro_b,
        p0=p0, v0=v0, q0=q0, lat_deg=lat, alt_m=0.0
    )
    
    # Position across all 20 seconds must remain p0
    for k in range(n):
        assert np.allclose(traj.pos_enu[k], p0, atol=1e-10)
        assert np.allclose(traj.vel_enu[k], [0.0, 0.0, 0.0], atol=1e-10)


def test_sanity_3_known_constant_horizontal_acceleration():
    """3. Known constant horizontal acceleration: verify expected v(t) = a*t and p(t) = 0.5*a*t^2."""
    lat = 52.4
    g_mag = normal_gravity_wgs84(lat, alt_m=0.0)
    
    n = 101
    dt = 0.1
    t = np.arange(n) * dt # 10 seconds
    
    # Phone aligned with ENU: X_b = East, Y_b = North, Z_b = Up
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    
    # Accelerating East at 2.0 m/s^2 and North at 1.5 m/s^2
    a_east = 2.0
    a_north = 1.5
    
    accel_b = np.zeros((n, 3))
    accel_b[:, 0] = a_east
    accel_b[:, 1] = a_north
    accel_b[:, 2] = g_mag # gravity reaction
    
    gyro_b = np.zeros((n, 3))
    
    p0 = np.array([10.0, 20.0, 0.0])
    v0 = np.array([1.0, -0.5, 0.0])
    
    traj = propagate_strapdown_segment(
        time_s=t, accel_body=accel_b, gyro_body=gyro_b,
        p0=p0, v0=v0, q0=q0, lat_deg=lat, alt_m=0.0
    )
    
    T = t[-1]
    expected_v_east = v0[0] + a_east * T
    expected_v_north = v0[1] + a_north * T
    expected_p_east = p0[0] + v0[0] * T + 0.5 * a_east * (T ** 2)
    expected_p_north = p0[1] + v0[1] * T + 0.5 * a_north * (T ** 2)
    
    assert np.isclose(traj.vel_enu[-1, 0], expected_v_east, atol=1e-8)
    assert np.isclose(traj.vel_enu[-1, 1], expected_v_north, atol=1e-8)
    assert np.isclose(traj.pos_enu[-1, 0], expected_p_east, atol=1e-8)
    assert np.isclose(traj.pos_enu[-1, 1], expected_p_north, atol=1e-8)
    assert np.isclose(traj.pos_enu[-1, 2], 0.0, atol=1e-8)


def test_sanity_4_pure_yaw_rotation_sign_and_direction():
    """4. Pure yaw rotation: verify yaw direction and sign."""
    # Start facing North (azimuth 0 deg, ENU yaw = pi/2 rad = 90 deg)
    q0 = euler_to_quat(0.0, 0.0, np.pi / 2.0)
    
    # Rotate CCW around +Z by +pi/2 (90 deg) at rate 1.0 rad/s for pi/2 seconds
    duration = np.pi / 2.0
    t = np.linspace(0.0, duration, 1571)
    n = len(t)
    
    gyro_b = np.zeros((n, 3))
    gyro_b[:, 2] = 1.0 # +1 rad/s around body Z (which points Up)
    accel_b = np.zeros((n, 3))
    accel_b[:, 2] = 9.80665
    
    traj = propagate_strapdown_segment(
        time_s=t, accel_body=accel_b, gyro_body=gyro_b,
        p0=np.zeros(3), v0=np.zeros(3), q0=q0
    )
    
    # Final ENU yaw should be 90 + 90 = 180 deg (facing West, -X)
    roll_f, pitch_f, yaw_f = quat_to_euler(traj.quat_b_n[-1])
    assert np.isclose(roll_f, 0.0, atol=1e-8)
    assert np.isclose(pitch_f, 0.0, atol=1e-8)
    assert np.isclose(np.abs(yaw_f), np.pi, atol=1e-4) # 180 deg
    
    # Azimuth clockwise from North: North (0) -> West (270 deg = 3*pi/2 rad)
    assert np.isclose(traj.heading_rad[-1], 3.0 * np.pi / 2.0, atol=1e-4)


def test_sanity_5_pure_pitch_roll_gravity_projection():
    """5. Pure pitch/roll rotation: verify gravity projection direction."""
    lat = 52.4
    g_mag = normal_gravity_wgs84(lat)
    g_vec = gravity_vector_enu(lat)
    
    # Tilted pitch: nose pitched UP by +30 degrees
    # Body X points upward by 30 deg. Gravity reaction [0, 0, +g] in nav frame
    # will project into body -X: f_x^b = -g * sin(30 deg)
    pitch_angle = np.radians(30.0)
    q_pitched = euler_to_quat(0.0, pitch_angle, 0.0)
    C_n_b = quat_to_dcm(q_pitched).T
    
    f_b_pitched = C_n_b @ np.array([0.0, 0.0, g_mag])
    assert np.isclose(f_b_pitched[0], -g_mag * np.sin(pitch_angle), atol=1e-10)
    assert np.isclose(f_b_pitched[2], g_mag * np.cos(pitch_angle), atol=1e-10)
    
    # Rotate back to nav: must cancel g_vec exactly
    f_n_recovered = rotate_vector(q_pitched, f_b_pitched)
    assert np.allclose(f_n_recovered + g_vec, [0.0, 0.0, 0.0], atol=1e-12)


def test_sanity_6_quaternion_norm_under_dynamic_integration():
    """6. Quaternion norm: remains ~1.0 under heavy continuous rotation."""
    np.random.seed(999)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    
    for _ in range(1000):
        # High dynamic angular rates up to 5 rad/s (~286 deg/s)
        w = np.random.uniform(-5.0, 5.0, size=3)
        dt = 0.01
        dq = rotvec_to_quat(w * dt)
        q = quat_normalize(quat_multiply(q, dq))
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-14)


def test_sanity_7_body_to_enu_known_principal_axis_rotations():
    """7. Body -> ENU known rotation: verify all principal axes."""
    # (a) 90 deg rotation around X (Roll = +90 deg):
    # Rotates Y_b -> +Z_n, and Z_b -> -Y_n
    q_x = euler_to_quat(np.pi / 2.0, 0.0, 0.0)
    assert np.allclose(rotate_vector(q_x, [1, 0, 0]), [1, 0, 0], atol=1e-12)
    assert np.allclose(rotate_vector(q_x, [0, 1, 0]), [0, 0, 1], atol=1e-12)
    assert np.allclose(rotate_vector(q_x, [0, 0, 1]), [0, -1, 0], atol=1e-12)
    
    # (b) 90 deg rotation around Y (Pitch = +90 deg):
    # Rotates X_b -> -Z_n, and Z_b -> +X_n
    q_y = euler_to_quat(0.0, np.pi / 2.0, 0.0)
    assert np.allclose(rotate_vector(q_y, [1, 0, 0]), [0, 0, -1], atol=1e-12)
    assert np.allclose(rotate_vector(q_y, [0, 1, 0]), [0, 1, 0], atol=1e-12)
    assert np.allclose(rotate_vector(q_y, [0, 0, 1]), [1, 0, 0], atol=1e-12)
    
    # (c) 90 deg rotation around Z (Yaw = +90 deg):
    # Rotates X_b -> +Y_n, and Y_b -> -X_n
    q_z = euler_to_quat(0.0, 0.0, np.pi / 2.0)
    assert np.allclose(rotate_vector(q_z, [1, 0, 0]), [0, 1, 0], atol=1e-12)
    assert np.allclose(rotate_vector(q_z, [0, 1, 0]), [-1, 0, 0], atol=1e-12)
    assert np.allclose(rotate_vector(q_z, [0, 0, 1]), [0, 0, 1], atol=1e-12)


def test_sanity_8_timestamp_gap_halt_strictness():
    """8. Timestamp gap: verify no integration occurs across dt > 0.25s."""
    t = np.array([0.0, 0.1, 0.2, 0.3, 10.0, 10.1, 10.2])
    n = len(t)
    accel_b = np.tile([1.0, 2.0, 9.80665], (n, 1))
    gyro_b = np.tile([0.01, 0.02, 0.03], (n, 1))
    
    traj = propagate_strapdown_segment(
        time_s=t, accel_body=accel_b, gyro_body=gyro_b,
        p0=np.zeros(3), v0=np.zeros(3), q0=np.array([1.0, 0.0, 0.0, 0.0]),
        max_dt=0.25
    )
    
    # Gap is at index 4 (from t=0.3 to t=10.0 is 9.7s > 0.25s)
    assert traj.gap_indices == [4]
    # Pos at index 4 and beyond must be exactly frozen at pos[3]
    for k in range(4, n):
        assert np.array_equal(traj.pos_enu[k], traj.pos_enu[3])
        assert np.array_equal(traj.vel_enu[k], traj.vel_enu[3])


def test_sanity_9_compound_nonorthogonal_rotation_scipy_crosscheck():
    """9. Compound non-orthogonal rotation cross-check against scipy.spatial.transform.Rotation.
    Simultaneously excites non-orthogonal roll, pitch, and yaw over multiple integration steps.
    Cross-checks quaternion propagation, DCM generation, and vector rotations."""
    from scipy.spatial.transform import Rotation
    
    np.random.seed(2026)
    
    # Initialize from arbitrary non-trivial attitude
    euler_init = [0.3, -0.4, 1.2] # roll, pitch, yaw
    q_navris = euler_to_quat(euler_init[0], euler_init[1], euler_init[2])
    
    # In scipy, intrinsic 'xyz' (or extrinsic 'ZYX') corresponds to roll around X, pitch around Y', yaw around Z''
    r_scipy = Rotation.from_euler('xyz', [euler_init[0], euler_init[1], euler_init[2]])
    
    # Compare initial DCM
    C_navris = quat_to_dcm(q_navris)
    C_scipy = r_scipy.as_matrix()
    assert np.allclose(C_navris, C_scipy, atol=1e-12)
    
    dt = 0.01
    # 50 steps of simultaneous 3D rates
    for step in range(50):
        # Non-orthogonal angular rate with all components non-zero
        w_body = np.array([
            0.5 * np.sin(0.2 * step),
            -0.3 * np.cos(0.15 * step),
            0.8 * np.sin(0.1 * step + 0.5)
        ])
        
        rotvec = w_body * dt
        dq_navris = rotvec_to_quat(rotvec)
        q_navris = quat_normalize(quat_multiply(q_navris, dq_navris))
        
        # In scipy, body-fixed rotation multiplies on the right: r_next = r * delta_r
        dr_scipy = Rotation.from_rotvec(rotvec)
        r_scipy = r_scipy * dr_scipy
        
        # Verify DCM matches scipy to floating-point precision
        C_navris = quat_to_dcm(q_navris)
        C_scipy = r_scipy.as_matrix()
        assert np.allclose(C_navris, C_scipy, atol=1e-12), f"DCM mismatch at step {step}"
        
        # Verify vector rotation of arbitrary 3D test vector
        v_test_body = np.array([1.23, -4.56, 7.89])
        v_rot_navris = rotate_vector(q_navris, v_test_body)
        v_rot_scipy = r_scipy.apply(v_test_body)
        assert np.allclose(v_rot_navris, v_rot_scipy, atol=1e-12), f"Vector rotation mismatch at step {step}"

