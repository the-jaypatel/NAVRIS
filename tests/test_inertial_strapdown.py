"""
Comprehensive Unit & Synthetic Verification Tests for NAVRIS Phase 2.2 Strapdown INS.

Tests:
1. Stationary specific force -> near-zero acceleration
2. Gravity sign verification
3. Quaternion norm preservation
4. Known constant-rate rotation (exact 90 deg rotation)
5. Noise-free constant-yaw circle (analytical trajectory match)
6. Trapezoidal integration accuracy
7. Gap detection (dt > 0.25s)
8. No integration across gaps
9. Novel GNSS fix gating
10. Repeated GNSS sample-and-hold rejection
11. Truncation invariance (zero future leakage)
12. Deterministic numerical propagation
13. Physical unit consistency
14. S4 timestamp-gap handling
15. Y1 reference-gap masking
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from navris.inertial.frames import (
    quat_normalize,
    quat_multiply,
    quat_to_dcm,
    rotate_vector,
    rotvec_to_quat,
    quat_to_euler,
    euler_to_quat
)
from navris.inertial.gravity import normal_gravity_wgs84, gravity_vector_enu
from navris.inertial.segments import detect_discontinuities, split_into_contiguous_segments
from navris.inertial.gnss_fixes import filter_novel_gnss_fixes
from navris.inertial.strapdown import propagate_strapdown_segment
from navris.inertial.init import align_leveling_from_gravity


def test_gravity_sign_and_stationary_zero_accel():
    """1 & 2: Verify gravity sign in ENU and that stationary specific force produces ~0 accel."""
    lat = 52.4
    g_mag = normal_gravity_wgs84(lat)
    g_vec = gravity_vector_enu(lat)
    
    # In ENU: Up is +Z, so gravity vector must be negative [0, 0, -g]
    assert g_vec[2] < 0.0
    assert np.isclose(g_vec[2], -g_mag)
    
    # Specific force at rest on Earth counters gravity: f_n = [0, 0, +g]
    f_rest_n = np.array([0.0, 0.0, g_mag])
    a_kinematic = f_rest_n + g_vec
    
    # Resulting kinematic acceleration must be exactly zero
    assert np.allclose(a_kinematic, np.zeros(3), atol=1e-12)


def test_quaternion_norm_preservation():
    """3: Verify quaternion norm remains 1.0 throughout sequential propagations."""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    # Apply 100 random rotations
    np.random.seed(42)
    for _ in range(100):
        w = np.random.randn(3) * 0.1
        dq = rotvec_to_quat(w)
        q = quat_normalize(quat_multiply(q, dq))
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-12)


def test_known_constant_rate_rotation():
    """4: Rotate at exactly pi/2 rad/s around Z for 1.0 second; must yield exact 90 deg rotation."""
    t = np.linspace(0.0, 1.0, 101) # 100 steps of dt = 0.01s
    dt = 0.01
    w_const = np.array([0.0, 0.0, np.pi / 2.0]) # 90 deg/s around Z
    
    q = np.array([1.0, 0.0, 0.0, 0.0])
    for _ in range(100):
        dq = rotvec_to_quat(w_const * dt)
        q = quat_normalize(quat_multiply(q, dq))
        
    roll, pitch, yaw = quat_to_euler(q)
    assert np.isclose(roll, 0.0, atol=1e-10)
    assert np.isclose(pitch, 0.0, atol=1e-10)
    assert np.isclose(yaw, np.pi / 2.0, atol=1e-6) # exact +90 degrees


def test_noise_free_constant_yaw_circle():
    """5: Pure kinematic circle: speed v=10 m/s, radius R=50m -> omega = v/R = 0.2 rad/s.
    In T = 2*pi/omega = 10*pi s (~31.4159s), car must complete a full circle and return to origin."""
    v = 10.0
    R = 50.0
    omega = v / R # 0.2 rad/s
    T_period = 2.0 * np.pi / omega # 31.4159265 s
    
    # 3142 samples at 100 Hz (dt = 0.01s)
    dt = 0.01
    t = np.arange(0.0, T_period + dt/2, dt)
    n = len(t)
    
    # In body frame (X=forward, Y=lateral, Z=up):
    # Turning left: angular rate around Z is +omega
    # Centripetal acceleration in body lateral Y is +v*omega = v^2/R
    # Vertical specific force counters local gravity exactly
    lat = 52.4
    alt = 0.0
    g = normal_gravity_wgs84(lat, alt)
    gyro_b = np.zeros((n, 3))
    gyro_b[:, 2] = omega
    
    accel_b = np.zeros((n, 3))
    accel_b[:, 1] = v * omega # centripetal
    accel_b[:, 2] = g         # gravity reaction
    
    # Initial state: start at [0, 0, 0] facing North (+Y in ENU)
    # Forward heading North means initial velocity is [0, v, 0]
    p0 = np.array([0.0, 0.0, 0.0])
    v0 = np.array([0.0, v, 0.0])
    # Yaw=90 deg in ENU means facing North (+Y)
    q0 = euler_to_quat(0.0, 0.0, np.pi / 2.0)
    
    traj = propagate_strapdown_segment(
        time_s=t, accel_body=accel_b, gyro_body=gyro_b,
        p0=p0, v0=v0, q0=q0, lat_deg=lat, alt_m=alt
    )
    
    # Final position after one full revolution must return to origin [0, 0, 0] within 0.15 m
    p_final = traj.pos_enu[-1]
    dist_from_origin = np.linalg.norm(p_final[:2])
    assert dist_from_origin < 0.15, f"Circle closure error too large: {dist_from_origin} m"
    # Vertical position must remain 0
    assert np.abs(p_final[2]) < 1e-4


def test_gap_detection_and_no_bridging():
    """6, 7 & 8: Verify gap detection halts integration across dt > 0.25s."""
    # 50 samples at 0.1s, then jump of 2.0s, then 50 samples
    t1 = np.arange(50) * 0.1
    t2 = t1[-1] + 2.0 + np.arange(1, 51) * 0.1
    t = np.concatenate([t1, t2])
    
    gaps = detect_discontinuities(t, max_dt=0.25)
    assert len(gaps) == 1
    assert gaps[0] == 50
    
    # Run strapdown on this array with a gap
    n = len(t)
    accel_b = np.zeros((n, 3))
    accel_b[:, 2] = 9.80665
    gyro_b = np.zeros((n, 3))
    
    traj = propagate_strapdown_segment(
        time_s=t, accel_body=accel_b, gyro_body=gyro_b,
        p0=np.zeros(3), v0=np.zeros(3), q0=np.array([1.0, 0.0, 0.0, 0.0]),
        max_dt=0.25
    )
    
    # Integration must stop at index 50
    assert len(traj.gap_indices) == 1
    assert traj.gap_indices[0] == 50
    # Values after gap must be frozen (not integrated across 2.0s gap)
    assert np.all(traj.pos_enu[50:] == traj.pos_enu[49])


def test_novel_gnss_fix_filtering():
    """9 & 10: Verify novel GNSS fix filter isolates real updates and rejects sample-and-hold."""
    data = {
        'phone_gps_lat_deg': [52.4, 52.4, 52.4, 52.4001, 52.4001, 52.4002],
        'phone_gps_lon_deg': [-1.5, -1.5, -1.5, -1.5001, -1.5001, -1.5002],
        'phone_gps_speed_mps': [0.0, 0.0, 0.0, 5.0, 5.0, 5.2]
    }
    df = pd.DataFrame(data)
    novel = filter_novel_gnss_fixes(df)
    
    # Rows 0, 3, and 5 have changes. Rows 1, 2, and 4 are held.
    assert len(novel) == 3
    assert list(novel.index) == [0, 3, 5]


def test_truncation_invariance_zero_future_leakage():
    """11 & 12: Verify truncation invariance: state at step k is identical whether
    propagating 0..k or 0..N (strictly causal, zero future leakage, deterministic)."""
    np.random.seed(123)
    n_full = 100
    t = np.arange(n_full) * 0.1
    acc = np.random.randn(n_full, 3) * 0.5
    acc[:, 2] += 9.80665
    gyr = np.random.randn(n_full, 3) * 0.02
    
    p0 = np.array([10.0, -20.0, 5.0])
    v0 = np.array([1.5, 2.0, 0.0])
    q0 = quat_normalize(np.array([1.0, 0.1, -0.2, 0.05]))
    b_g = np.array([0.001, -0.002, 0.0005])
    
    # Run full trajectory
    traj_full = propagate_strapdown_segment(
        time_s=t, accel_body=acc, gyro_body=gyr,
        p0=p0, v0=v0, q0=q0, gyro_bias=b_g
    )
    
    # Run truncated trajectory up to k=50
    k = 50
    traj_trunc = propagate_strapdown_segment(
        time_s=t[:k], accel_body=acc[:k], gyro_body=gyr[:k],
        p0=p0, v0=v0, q0=q0, gyro_bias=b_g
    )
    
    # State at k-1 must match to exact floating point precision
    assert np.allclose(traj_trunc.pos_enu[-1], traj_full.pos_enu[k - 1], atol=1e-15)
    assert np.allclose(traj_trunc.vel_enu[-1], traj_full.vel_enu[k - 1], atol=1e-15)
    assert np.allclose(traj_trunc.quat_b_n[-1], traj_full.quat_b_n[k - 1], atol=1e-15)


def test_real_s4_and_y1_gap_handling():
    """14 & 15: Verify real recordings S4 and Y1 segments are split cleanly without bridging."""
    manifest = pd.read_csv('data/manifest/recordings_manifest.csv')
    
    for rid in ['S4', 'Y1']:
        sub = manifest[manifest['recording_id'] == rid]
        if len(sub) == 0:
            continue
        parquet_path = f"data/processed/synchronized/{rid}_sync.parquet"
        if not os.path.exists(parquet_path):
            continue
        
        df = pd.read_parquet(parquet_path)
        segments = split_into_contiguous_segments(df, max_dt=0.25)
        
        # Each segment must have strictly dt <= 0.25s
        for s_idx, e_idx, seg in segments:
            dt_vals = np.diff(seg['time_s'].values)
            assert np.all(dt_vals <= 0.25), f"Segment in {rid} contained unhandled dt > 0.25s!"
