"""
NAVRIS Planar 2D Dead-Reckoning Control Baseline (A-planar).

A 2D simplified kinematic baseline executing planar dead reckoning:
- Assumes vehicle motion is constrained to the 2D horizontal navigation plane
- Propagates single-axis heading from yaw rate
- Integrates forward acceleration along heading
- Exists specifically to contrast 3D tilt/gravity leakage against 2D vehicle-oriented assumptions
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List

from navris.inertial.frames import wrap_angle_2pi


@dataclass
class PlanarTrajectory:
    time_s: np.ndarray             # (N,) timestamps in seconds
    pos_enu: np.ndarray            # (N, 3) [East, North, Up=0] in meters
    vel_enu: np.ndarray            # (N, 3) [vE, vN, 0] in m/s
    speed_mps: np.ndarray          # (N,) forward speed
    heading_rad: np.ndarray        # (N,) navigational azimuth
    gap_indices: List[int]
    meta: Dict[str, Any]


def run_planar_dr(
    df: pd.DataFrame,
    p0: np.ndarray,
    v0_mps: float,
    heading0_rad: float,
    yaw_rate_bias: float = 0.0,
    max_dt: float = 0.25
) -> PlanarTrajectory:
    """
    Executes planar dead reckoning.
    Uses candidate yaw rate channel (phone_gyro_y_radps) and dynamic horizontal acceleration.
    """
    n = len(df)
    t = df['time_s'].values

    pos = np.zeros((n, 3), dtype=np.float64)
    vel = np.zeros((n, 3), dtype=np.float64)
    speed = np.zeros(n, dtype=np.float64)
    heading = np.zeros(n, dtype=np.float64)

    pos[0] = np.asarray(p0, dtype=np.float64)
    speed[0] = float(v0_mps)
    heading[0] = float(heading0_rad)
    vel[0] = np.array([
        speed[0] * np.sin(heading[0]),
        speed[0] * np.cos(heading[0]),
        0.0
    ], dtype=np.float64)

    # In Phase 2.1, candidate yaw-rate channel was phone_gyro_y_radps
    w_yaw_raw = df['phone_gyro_y_radps'].values

    # Dynamic horizontal acceleration from accelerometer norm deviation
    ax = df['phone_accel_x_mps2'].values
    ay = df['phone_accel_y_mps2'].values
    az = df['phone_accel_z_mps2'].values
    a_mag = np.sqrt(ax**2 + ay**2 + az**2)
    # Planar approximation: dynamic acceleration magnitude along heading
    # (Subtract mean stationary g ~ 9.807)
    g_ref = 9.80665
    a_long_approx = a_mag - g_ref

    gap_indices = []

    for k in range(n - 1):
        dt = t[k + 1] - t[k]
        if dt > max_dt or dt <= 0.0:
            gap_indices.append(k + 1)
            pos[k + 1:] = pos[k]
            vel[k + 1:] = vel[k]
            speed[k + 1:] = speed[k]
            heading[k + 1:] = heading[k]
            break

        # 1. Heading propagation
        w_k = w_yaw_raw[k] - yaw_rate_bias
        w_kp1 = w_yaw_raw[k + 1] - yaw_rate_bias
        w_bar = 0.5 * (w_k + w_kp1)

        # Yaw rate in clockwise azimuth: heading decreases with CCW rotation, increases with CW
        heading_next = wrap_angle_2pi(heading[k] + w_bar * dt)
        heading[k + 1] = heading_next

        # 2. Speed propagation (clamp non-negative for forward vehicle motion)
        a_bar = 0.5 * (a_long_approx[k] + a_long_approx[k + 1])
        spd_next = max(0.0, speed[k] + a_bar * dt)
        speed[k + 1] = spd_next

        # 3. Velocity in ENU
        v_next = np.array([
            spd_next * np.sin(heading_next),
            spd_next * np.cos(heading_next),
            0.0
        ], dtype=np.float64)
        vel[k + 1] = v_next

        # 4. Position in ENU (trapezoidal)
        pos[k + 1] = pos[k] + 0.5 * (vel[k] + v_next) * dt

    return PlanarTrajectory(
        time_s=t,
        pos_enu=pos,
        vel_enu=vel,
        speed_mps=speed,
        heading_rad=heading,
        gap_indices=gap_indices,
        meta={'type': 'Planar_2D_Control', 'yaw_bias': yaw_rate_bias}
    )
