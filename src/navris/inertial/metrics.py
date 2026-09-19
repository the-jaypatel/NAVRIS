"""
NAVRIS Dead-Reckoning Navigation Performance Metrics.

Computes comprehensive error metrics against the reference VBOX trajectory:
- Horizontal 2D and 3D Position Error
- RMSE, MAE, Median, P90, P95, Max Error, Final Displacement
- Along-Track & Cross-Track Error Projections
- Heading Error (circular wrapping) & Velocity RMSE
- Time-to-50m and Time-to-100m Drift Thresholds
- Vertical Drift Statistics
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple

from navris.inertial.frames import wrap_angle_pi


@dataclass
class TrajectoryMetrics:
    duration_s: float
    distance_traveled_m: float
    horiz_rmse_m: float
    horiz_mae_m: float
    horiz_median_m: float
    horiz_p90_m: float
    horiz_p95_m: float
    horiz_max_m: float
    final_error_m: float
    drift_rate_mps: float
    drift_pct_distance: float
    pos_3d_rmse_m: float
    vel_rmse_mps: float
    speed_mae_mps: float
    heading_rmse_deg: float
    along_track_rmse_m: float
    cross_track_rmse_m: float
    vertical_drift_final_m: float
    vertical_vel_drift_final_mps: float
    time_to_50m_s: Optional[float]
    time_to_100m_s: Optional[float]
    details: Dict[str, Any]


def compute_along_cross_track_errors(
    pos_dr: np.ndarray,      # (N, 2) [East, North]
    pos_ref: np.ndarray,     # (N, 2) [East, North]
    heading_ref: np.ndarray  # (N,) azimuth in radians (clockwise from North)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects 2D horizontal position errors into vehicle reference along-track
    and cross-track directions.
    
    Forward vector in ENU: [sin(psi), cos(psi)]
    Right (starboard) vector in ENU: [cos(psi), -sin(psi)]
    """
    de = pos_dr[:, 0] - pos_ref[:, 0]
    dn = pos_dr[:, 1] - pos_ref[:, 1]

    sin_psi = np.sin(heading_ref)
    cos_psi = np.cos(heading_ref)

    # Along-track (forward displacement error)
    along_track = de * sin_psi + dn * cos_psi
    # Cross-track (lateral displacement error)
    cross_track = de * cos_psi - dn * sin_psi

    return along_track, cross_track


def compute_dr_metrics(
    time_s: np.ndarray,
    pos_dr_enu: np.ndarray,      # (N, 3)
    vel_dr_enu: np.ndarray,      # (N, 3)
    heading_dr_rad: np.ndarray,  # (N,)
    df_ref: pd.DataFrame
) -> TrajectoryMetrics:
    """
    Computes rigorous metrics between DR trajectory and reference trajectory.
    Handles reference dropouts and invalid rows gracefully.
    """
    n = len(time_s)
    ref_e = df_ref['ref_east_m'].values[:n]
    ref_n = df_ref['ref_north_m'].values[:n]
    ref_u = df_ref['ref_up_m'].values[:n] if 'ref_up_m' in df_ref.columns else np.zeros(n)

    ref_spd = df_ref['ref_speed_mps'].values[:n]
    ref_head = df_ref['ref_heading_rad'].values[:n]

    # Valid mask (non-NaN reference coordinates)
    valid = ~np.isnan(ref_e) & ~np.isnan(ref_n) & ~np.isnan(ref_spd) & ~np.isnan(ref_head)
    if 'ref_gps_dropout' in df_ref.columns:
        valid = valid & (~df_ref['ref_gps_dropout'].values[:n])

    if np.sum(valid) < 5:
        raise ValueError("Insufficient valid reference points for metric computation")

    t_valid = time_s[valid]
    p_dr_val = pos_dr_enu[valid]
    v_dr_val = vel_dr_enu[valid]
    h_dr_val = heading_dr_rad[valid]

    re_val = ref_e[valid]
    rn_val = ref_n[valid]
    ru_val = ref_u[valid]
    rspd_val = ref_spd[valid]
    rhead_val = ref_head[valid]

    # Horizontal position errors
    de = p_dr_val[:, 0] - re_val
    dn = p_dr_val[:, 1] - rn_val
    du = p_dr_val[:, 2] - ru_val
    horiz_errors = np.sqrt(de**2 + dn**2)
    errors_3d = np.sqrt(de**2 + dn**2 + du**2)

    # Duration and distance traveled
    duration_s = float(t_valid[-1] - t_valid[0])
    dist_traveled = float(np.sum(np.abs(rspd_val[:-1]) * np.diff(t_valid)))

    # Basic statistics
    horiz_rmse = float(np.sqrt(np.mean(horiz_errors**2)))
    horiz_mae = float(np.mean(horiz_errors))
    horiz_median = float(np.median(horiz_errors))
    horiz_p90 = float(np.percentile(horiz_errors, 90))
    horiz_p95 = float(np.percentile(horiz_errors, 95))
    horiz_max = float(np.max(horiz_errors))
    final_error = float(horiz_errors[-1])

    drift_rate = final_error / duration_s if duration_s > 0.1 else 0.0
    drift_pct = (final_error / dist_traveled * 100.0) if dist_traveled > 10.0 else 0.0

    # 3D and Velocity
    pos_3d_rmse = float(np.sqrt(np.mean(errors_3d**2)))

    dr_spd = np.sqrt(v_dr_val[:, 0]**2 + v_dr_val[:, 1]**2)
    speed_mae = float(np.mean(np.abs(dr_spd - rspd_val)))

    ref_v_e = rspd_val * np.sin(rhead_val)
    ref_v_n = rspd_val * np.cos(rhead_val)
    vel_rmse = float(np.sqrt(np.mean((v_dr_val[:, 0] - ref_v_e)**2 + (v_dr_val[:, 1] - ref_v_n)**2)))

    # Heading error with circular wrapping
    head_diff = wrap_angle_pi(h_dr_val - rhead_val)
    heading_rmse_deg = float(np.sqrt(np.mean(head_diff**2)) * (180.0 / np.pi))

    # Along-track & cross-track
    along, cross = compute_along_cross_track_errors(p_dr_val[:, :2], np.column_stack([re_val, rn_val]), rhead_val)
    along_rmse = float(np.sqrt(np.mean(along**2)))
    cross_rmse = float(np.sqrt(np.mean(cross**2)))

    # Vertical drift
    vert_drift_final = float(du[-1])
    vert_vel_drift_final = float(v_dr_val[-1, 2])

    # Time to threshold
    t_to_50 = None
    t_to_100 = None
    idx_50 = np.where(horiz_errors >= 50.0)[0]
    if len(idx_50) > 0:
        t_to_50 = float(t_valid[idx_50[0]] - t_valid[0])

    idx_100 = np.where(horiz_errors >= 100.0)[0]
    if len(idx_100) > 0:
        t_to_100 = float(t_valid[idx_100[0]] - t_valid[0])

    return TrajectoryMetrics(
        duration_s=duration_s,
        distance_traveled_m=dist_traveled,
        horiz_rmse_m=horiz_rmse,
        horiz_mae_m=horiz_mae,
        horiz_median_m=horiz_median,
        horiz_p90_m=horiz_p90,
        horiz_p95_m=horiz_p95,
        horiz_max_m=horiz_max,
        final_error_m=final_error,
        drift_rate_mps=drift_rate,
        drift_pct_distance=drift_pct,
        pos_3d_rmse_m=pos_3d_rmse,
        vel_rmse_mps=vel_rmse,
        speed_mae_mps=speed_mae,
        heading_rmse_deg=heading_rmse_deg,
        along_track_rmse_m=along_rmse,
        cross_track_rmse_m=cross_rmse,
        vertical_drift_final_m=vert_drift_final,
        vertical_vel_drift_final_mps=vert_vel_drift_final,
        time_to_50m_s=t_to_50,
        time_to_100m_s=t_to_100,
        details={
            'horiz_errors': horiz_errors,
            'along_track': along,
            'cross_track': cross,
            't_rel_s': t_valid - t_valid[0]
        }
    )
