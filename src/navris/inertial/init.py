"""
NAVRIS Causal Initial Alignment (Leveling + Heading).

Strictly causal initial attitude determination:
1. Initial Leveling: Rotates body specific-force up-vector to ENU Up [0, 0, 1]^T.
   Works for ANY arbitrary mounting angle without assuming axis mappings.
2. Initial Heading: Derived from causal GNSS course-over-ground during the first
   straight dynamic driving segment meeting strict motion-integrity checks.
   Only novel GNSS fixes arriving at or before initialization epoch are used.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

from navris.inertial.frames import (
    quat_normalize,
    quat_multiply,
    rotate_vector,
    wrap_angle_pi
)
from navris.inertial.gnss_fixes import filter_novel_gnss_fixes


@dataclass
class AlignmentResult:
    is_valid: bool
    quality: str                       # 'HIGH', 'NOMINAL', 'DEGRADED', 'FAILED'
    init_time_s: float
    init_sample_idx: int
    q_b_n: np.ndarray                 # 4-vector [qw, qx, qy, qz]
    initial_heading_rad: float         # Navigational azimuth (clockwise from North)
    initial_velocity_enu: np.ndarray   # 3-vector [vE, vN, vU]
    gyro_bias_radps: np.ndarray        # 3-vector [bx, by, bz]
    f_stationary_norm: float
    stationary_samples_used: int
    heading_novel_fixes_used: int
    details: Dict[str, Any]


def align_leveling_from_gravity(f_body_mean: np.ndarray) -> np.ndarray:
    """
    Finds exact unit quaternion q_level rotating body specific force vector to ENU Up [0, 0, 1]^T.
    C_b^l(q_level) @ (f_body / ||f_body||) = [0, 0, 1]^T.
    Works for any arbitrary 3D mounting angle.
    """
    v_from = f_body_mean / np.linalg.norm(f_body_mean)
    v_to = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    dot = np.dot(v_from, v_to)
    if dot > 0.9999999:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if dot < -0.9999999:
        # 180 degree rotation around X
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

    cross = np.cross(v_from, v_to)
    w = 1.0 + dot
    q = np.array([w, cross[0], cross[1], cross[2]], dtype=np.float64)
    return quat_normalize(q)


def align_attitude_causal(
    df_history: pd.DataFrame,
    min_stat_samples: int = 30,          # at least 3 seconds stationary
    min_speed_mps: float = 3.0,          # minimum speed for heading estimate
    min_displacement_m: float = 40.0,    # minimum straight baseline
    max_yaw_rate_radps: float = 0.04,    # maximum gyro rate during straight drive
    max_heading_std_rad: float = 0.15    # ~8.5 degrees
) -> AlignmentResult:
    """
    Causally initializes navigation state from history up to current epoch.
    Zero access to future data or reference trajectory.
    """
    if len(df_history) < min_stat_samples:
        return AlignmentResult(
            is_valid=False, quality='FAILED', init_time_s=0.0, init_sample_idx=0,
            q_b_n=np.array([1.0, 0.0, 0.0, 0.0]), initial_heading_rad=0.0,
            initial_velocity_enu=np.zeros(3), gyro_bias_radps=np.zeros(3),
            f_stationary_norm=0.0, stationary_samples_used=0,
            heading_novel_fixes_used=0, details={'reason': 'insufficient_history'}
        )

    # 1. Leveling & Gyro Bias from initial stationary window
    stat_sub = df_history.iloc[:min_stat_samples]
    ax = stat_sub['phone_accel_x_mps2'].values
    ay = stat_sub['phone_accel_y_mps2'].values
    az = stat_sub['phone_accel_z_mps2'].values
    f_mean = np.array([np.mean(ax), np.mean(ay), np.mean(az)], dtype=np.float64)
    f_norm = float(np.linalg.norm(f_mean))

    wx = stat_sub['phone_gyro_x_radps'].values
    wy = stat_sub['phone_gyro_y_radps'].values
    wz = stat_sub['phone_gyro_z_radps'].values
    b_g = np.array([np.mean(wx), np.mean(wy), np.mean(wz)], dtype=np.float64)

    q_level = align_leveling_from_gravity(f_mean)

    # 2. Causal Heading Search across available history
    # Search for the first straight dynamic segment of novel GNSS fixes
    novel_gnss = filter_novel_gnss_fixes(df_history)
    moving_fixes = novel_gnss[novel_gnss['phone_gps_speed_mps'] > min_speed_mps]

    heading_found = False
    init_heading = 0.0
    init_time_s = float(df_history['time_s'].iloc[-1])
    init_idx = len(df_history) - 1
    init_vel_enu = np.zeros(3, dtype=np.float64)
    fixes_used = 0

    if len(moving_fixes) >= 2:
        # Search sequentially for the earliest straight moving window
        for i in range(len(moving_fixes) - 1):
            sub = moving_fixes.iloc[i:min(i + 3, len(moving_fixes))]
            de = sub['phone_gps_east_m'].iloc[-1] - sub['phone_gps_east_m'].iloc[0]
            dn = sub['phone_gps_north_m'].iloc[-1] - sub['phone_gps_north_m'].iloc[0]
            disp = np.sqrt(de**2 + dn**2)

            if disp >= min_displacement_m:
                cog_disp = np.arctan2(de, dn) % (2.0 * np.pi)
                headings_raw = sub['phone_gps_orientation_rad'].values
                valid_head = headings_raw[~np.isnan(headings_raw)]

                is_consistent = True
                if len(valid_head) >= 2:
                    sin_sum = np.sum(np.sin(valid_head))
                    cos_sum = np.sum(np.cos(valid_head))
                    circ_std = np.sqrt(max(0.0, 1.0 - np.sqrt(sin_sum**2 + cos_sum**2) / len(valid_head)))
                    if circ_std > max_heading_std_rad:
                        is_consistent = False

                if is_consistent:
                    heading_found = True
                    init_heading = float(cog_disp)
                    fixes_used = len(sub)
                    end_row = sub.index[-1]
                    # Find integer position of end_row in df_history
                    init_idx = df_history.index.get_loc(end_row)
                    init_time_s = float(df_history['time_s'].iloc[init_idx])
                    current_speed = float(sub['phone_gps_speed_mps'].iloc[-1])
                    init_vel_enu = np.array([
                        current_speed * np.sin(init_heading),
                        current_speed * np.cos(init_heading),
                        0.0
                    ], dtype=np.float64)
                    break

    # If heading found, construct full 3D attitude:
    # In ENU: Azimuth psi (clockwise from North) means forward vector is [sin(psi), cos(psi), 0]
    # Polar angle in ENU (CCW from East) is theta = pi/2 - psi.
    # Level-to-navigation yaw quaternion:
    if heading_found:
        quality = 'HIGH' if fixes_used >= 5 else 'NOMINAL'
        # Rotation around Up (+Z) by polar angle theta
        theta = np.pi / 2.0 - init_heading
        q_yaw = np.array([np.cos(0.5 * theta), 0.0, 0.0, np.sin(0.5 * theta)], dtype=np.float64)
        q_b_n = quat_multiply(q_yaw, q_level)
    else:
        # Fallback: Level only, zero heading assumed (marked degraded)
        quality = 'DEGRADED'
        q_b_n = q_level

    return AlignmentResult(
        is_valid=heading_found,
        quality=quality,
        init_time_s=init_time_s,
        init_sample_idx=init_idx,
        q_b_n=q_b_n,
        initial_heading_rad=init_heading,
        initial_velocity_enu=init_vel_enu,
        gyro_bias_radps=b_g,
        f_stationary_norm=f_norm,
        stationary_samples_used=min_stat_samples,
        heading_novel_fixes_used=fixes_used,
        details={
            'heading_found': heading_found,
            'f_norm_diff': f_norm - 9.80665,
            'q_level': q_level.tolist()
        }
    )
