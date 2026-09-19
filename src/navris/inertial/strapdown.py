"""
NAVRIS 3D Quaternion Strapdown INS Mechanization.

Mechanizes pure dead-reckoning from smartphone IMU signals in local Cartesian ENU:
- Body-to-Navigation attitude propagation via exponential map quaternion update
- Specific force rotation into navigation frame: f^n = C_b^n(q) @ f^b
- Gravity compensation: a^n = f^n + g^n, with g^n = [0, 0, -g]
- Trapezoidal integration for velocity and position using actual per-sample dt
- Zero access to reference/VBOX data during propagation
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List

from navris.inertial.frames import (
    quat_normalize,
    quat_multiply,
    rotvec_to_quat,
    rotate_vector,
    quat_to_euler
)
from navris.inertial.gravity import normal_gravity_wgs84, gravity_vector_enu


@dataclass
class StrapdownTrajectory:
    time_s: np.ndarray             # (N,) timestamps in seconds
    pos_enu: np.ndarray            # (N, 3) [East, North, Up] in meters
    vel_enu: np.ndarray            # (N, 3) [vE, vN, vU] in m/s
    quat_b_n: np.ndarray           # (N, 4) unit quaternions [qw, qx, qy, qz]
    accel_enu: np.ndarray          # (N, 3) compensated kinematic acceleration
    heading_rad: np.ndarray        # (N,) navigational azimuth (clockwise from North)
    gap_indices: List[int]         # Indices where dt > max_dt halted integration
    meta: Dict[str, Any]           # Diagnostics, bias, initial conditions


def propagate_strapdown_segment(
    time_s: np.ndarray,
    accel_body: np.ndarray,        # (N, 3) m/s^2 specific force
    gyro_body: np.ndarray,         # (N, 3) rad/s angular rate
    p0: np.ndarray,                # (3,) initial position [E, N, U]
    v0: np.ndarray,                # (3,) initial velocity [vE, vN, vU]
    q0: np.ndarray,                # (4,) initial attitude quaternion [qw, qx, qy, qz]
    gyro_bias: Optional[np.ndarray] = None, # (3,) rad/s
    accel_scale_factor: float = 1.0,        # optional scale ablation
    lat_deg: float = 52.4,
    alt_m: float = 100.0,
    max_dt: float = 0.25           # maximum allowed dt before stopping
) -> StrapdownTrajectory:
    """
    Executes forward strapdown mechanization across a single contiguous segment.
    Strictly causal, pure IMU input, zero reference trajectory access.
    """
    n = len(time_s)
    if n == 0:
        raise ValueError("Cannot propagate empty segment")

    # Allocate trajectory arrays
    pos = np.zeros((n, 3), dtype=np.float64)
    vel = np.zeros((n, 3), dtype=np.float64)
    quat = np.zeros((n, 4), dtype=np.float64)
    acc_n = np.zeros((n, 3), dtype=np.float64)
    heading = np.zeros(n, dtype=np.float64)

    # Initial state
    pos[0] = np.asarray(p0, dtype=np.float64)
    vel[0] = np.asarray(v0, dtype=np.float64)
    quat[0] = quat_normalize(q0)

    b_g = np.zeros(3, dtype=np.float64) if gyro_bias is None else np.asarray(gyro_bias, dtype=np.float64)

    # Precompute local normal gravity in ENU
    g_vec = gravity_vector_enu(lat_deg=lat_deg, alt_m=alt_m)

    # Initial acceleration & heading
    f0_b = accel_body[0] * accel_scale_factor
    f0_n = rotate_vector(quat[0], f0_b)
    acc_n[0] = f0_n + g_vec

    _, _, yaw0 = quat_to_euler(quat[0])
    # Convert ENU yaw (polar) to navigational azimuth (clockwise from North)
    heading[0] = (np.pi / 2.0 - yaw0) % (2.0 * np.pi)

    gap_indices = []

    for k in range(n - 1):
        dt = time_s[k + 1] - time_s[k]

        # Strict gap check: never integrate across discontinuities > max_dt
        if dt > max_dt or dt <= 0.0:
            gap_indices.append(k + 1)
            # Freeze/stop integration at gap boundary
            pos[k + 1:] = pos[k]
            vel[k + 1:] = vel[k]
            quat[k + 1:] = quat[k]
            acc_n[k + 1:] = 0.0
            heading[k + 1:] = heading[k]
            break

        # 1. Bias-corrected trapezoidal angular velocity
        w_k = gyro_body[k] - b_g
        w_kp1 = gyro_body[k + 1] - b_g
        w_bar = 0.5 * (w_k + w_kp1)

        # 2. Attitude propagation via quaternion exponential map
        rotvec = w_bar * dt
        dq = rotvec_to_quat(rotvec)
        q_next = quat_normalize(quat_multiply(quat[k], dq))
        quat[k + 1] = q_next

        # 3. Specific force rotation to navigation frame
        f_k_b = accel_body[k] * accel_scale_factor
        f_kp1_b = accel_body[k + 1] * accel_scale_factor

        f_k_n = rotate_vector(quat[k], f_k_b)
        f_kp1_n = rotate_vector(q_next, f_kp1_b)
        f_bar_n = 0.5 * (f_k_n + f_kp1_n)

        # 4. Gravity compensation in ENU navigation frame
        # Specific force counters gravity: at rest f_n = [0, 0, +g]^T.
        # Adding g_vec = [0, 0, -g]^T yields kinematically true acceleration.
        a_kinematic_n = f_bar_n + g_vec
        acc_n[k + 1] = a_kinematic_n

        # 5. Velocity integration (trapezoidal)
        v_next = vel[k] + a_kinematic_n * dt
        vel[k + 1] = v_next

        # 6. Position integration (trapezoidal velocity)
        p_next = pos[k] + 0.5 * (vel[k] + v_next) * dt
        pos[k + 1] = p_next

        # 7. Extract navigational heading (azimuth)
        _, _, yaw_k = quat_to_euler(q_next)
        heading[k + 1] = (np.pi / 2.0 - yaw_k) % (2.0 * np.pi)

    return StrapdownTrajectory(
        time_s=time_s,
        pos_enu=pos,
        vel_enu=vel,
        quat_b_n=quat,
        accel_enu=acc_n,
        heading_rad=heading,
        gap_indices=gap_indices,
        meta={
            'gyro_bias': b_g.tolist(),
            'accel_scale_factor': accel_scale_factor,
            'g_mag': float(-g_vec[2]),
            'num_samples': n,
            'duration_s': float(time_s[-1] - time_s[0]) if n > 0 else 0.0
        }
    )


def run_strapdown_dr(
    df: pd.DataFrame,
    p0: np.ndarray,
    v0: np.ndarray,
    q0: np.ndarray,
    gyro_bias: Optional[np.ndarray] = None,
    accel_scale: str = 'none',          # 'none' or 'normalize_stationary_norm'
    f_stationary_norm: Optional[float] = None,
    max_dt: float = 0.25
) -> StrapdownTrajectory:
    """
    Convenience wrapper extracting IMU columns from synchronized DataFrame and
    executing strapdown dead-reckoning.
    """
    t = df['time_s'].values
    ax = df['phone_accel_x_mps2'].values
    ay = df['phone_accel_y_mps2'].values
    az = df['phone_accel_z_mps2'].values
    accel_b = np.column_stack([ax, ay, az])

    wx = df['phone_gyro_x_radps'].values
    wy = df['phone_gyro_y_radps'].values
    wz = df['phone_gyro_z_radps'].values
    gyro_b = np.column_stack([wx, wy, wz])

    scale_factor = 1.0
    if accel_scale == 'normalize_stationary_norm' and f_stationary_norm is not None:
        g_ref = 9.80665
        if f_stationary_norm > 1.0:
            scale_factor = g_ref / f_stationary_norm

    lat_ref = float(df['ref_lat_deg'].iloc[0]) if 'ref_lat_deg' in df.columns else 52.4

    return propagate_strapdown_segment(
        time_s=t,
        accel_body=accel_b,
        gyro_body=gyro_b,
        p0=p0,
        v0=v0,
        q0=q0,
        gyro_bias=gyro_bias,
        accel_scale_factor=scale_factor,
        lat_deg=lat_ref,
        max_dt=max_dt
    )
