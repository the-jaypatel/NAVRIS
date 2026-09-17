"""
NAVRIS Timeline Synchronization and Sensor Fusion Alignment.

Synchronizes Vehicle Reference CAN/VBOX and Smartphone Data onto a common 10 Hz timebase:
- IMU and continuous kinematic signals are aligned to a regular 10 Hz grid (dt = 0.1 s).
- Angles and headings are interpolated using continuous phase unwrapping.
- Discrete states (gears, handbrake, dropout flags) use nearest-neighbor alignment.
- Smartphone GNSS observations are strictly preserved as SPARSE intermittent arrivals:
  - phone_gps_is_new_fix: True ONLY at epochs where a new GPS measurement arrived.
  - phone_gps_obs_east_m, phone_gps_obs_north_m, phone_gps_obs_speed_mps: NaN during
    inter-update periods, non-NaN ONLY on new arrivals.
  - No high-frequency GNSS points are ever invented.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any

from navris.schema import SYNCHRONIZED_SCHEMA_COLUMNS


def synchronize_recordings(
    df_ref: pd.DataFrame,
    df_phone: pd.DataFrame,
    dt: float = 0.1
) -> pd.DataFrame:
    """
    Synchronizes a reference DataFrame and a smartphone DataFrame onto a common 10 Hz timeline.
    
    Args:
        df_ref: Normalized reference DataFrame from ingest_reference_data
        df_phone: Normalized smartphone DataFrame from ingest_smartphone_data
        dt: Target sampling interval in seconds (default 0.1 s = 10 Hz)
        
    Returns:
        Synchronized DataFrame adhering to SYNCHRONIZED_SCHEMA_COLUMNS.
    """
    t_ref = df_ref['ref_time_s'].values
    t_phone = df_phone['phone_time_s'].values

    # Determine overlapping time range
    t_start = max(float(t_ref[0]), float(t_phone[0]))
    t_end = min(float(t_ref[-1]), float(t_phone[-1]))

    if t_end <= t_start:
        raise ValueError(f"No temporal overlap between reference [{t_ref[0]}, {t_ref[-1]}] and phone [{t_phone[0]}, {t_phone[-1]}]")

    # Construct regular 10 Hz timeline
    common_time = np.arange(t_start, t_end + 1e-6, dt)
    n_samples = len(common_time)

    out = pd.DataFrame()
    out['time_s'] = np.round(common_time - common_time[0], 4)

    # 1. Interpolate Continuous Reference Signals
    ref_continuous_cols = [
        'ref_lat_deg', 'ref_lon_deg', 'ref_alt_m',
        'ref_east_m', 'ref_north_m', 'ref_up_m',
        'ref_speed_mps', 'ref_vertical_speed_mps',
        'ref_yaw_rate_radps', 'ref_steering_angle_rad',
        'ref_wheel_speed_fl_radps', 'ref_wheel_speed_fr_radps',
        'ref_wheel_speed_rl_radps', 'ref_wheel_speed_rr_radps',
        'ref_accel_long_mps2', 'ref_accel_lat_mps2',
        'ref_indicated_speed_mps', 'ref_engine_speed_rpm',
        'ref_brake_pressure_psi', 'ref_throttle_pct'
    ]

    for col in ref_continuous_cols:
        if col in df_ref.columns:
            out[col] = np.interp(common_time, t_ref, df_ref[col].values)

    # Reference Heading: Unwrap phase before interpolating to avoid 0/360 wrap-around jumps
    if 'ref_heading_rad' in df_ref.columns:
        unwrapped_heading = np.unwrap(df_ref['ref_heading_rad'].values)
        interp_unwrapped = np.interp(common_time, t_ref, unwrapped_heading)
        interp_heading = interp_unwrapped % (2.0 * np.pi)
        out['ref_heading_rad'] = interp_heading
        out['ref_heading_deg'] = np.degrees(interp_heading)

    # Reference Discrete / Flag Signals (Nearest-Neighbor)
    ref_discrete_cols = [
        'ref_gear', 'ref_handbrake',
        'ref_gps_satellites_raw', 'ref_gps_satellites_count',
        'ref_gps_is_dgps', 'ref_gps_dropout'
    ]
    nearest_ref_indices = np.searchsorted(t_ref, common_time, side='left')
    nearest_ref_indices = np.clip(nearest_ref_indices, 0, len(t_ref) - 1)

    for col in ref_discrete_cols:
        if col in df_ref.columns:
            out[col] = df_ref[col].values[nearest_ref_indices]

    # 2. Interpolate Continuous Smartphone Signals (IMU, Gravity, Magnetometer)
    phone_continuous_cols = [
        'phone_accel_x_mps2', 'phone_accel_y_mps2', 'phone_accel_z_mps2',
        'phone_gravity_x_mps2', 'phone_gravity_y_mps2', 'phone_gravity_z_mps2',
        'phone_gyro_x_radps', 'phone_gyro_y_radps', 'phone_gyro_z_radps',
    ]
    for col in phone_continuous_cols:
        if col in df_phone.columns:
            out[col] = np.interp(common_time, t_phone, df_phone[col].values)

    # Magnetometer (Driver F might have NaNs)
    has_mag = bool(df_phone['phone_has_mag'].iloc[0]) if 'phone_has_mag' in df_phone.columns else False
    out['phone_has_mag'] = has_mag
    for axis in ['x', 'y', 'z']:
        col = f'phone_mag_{axis}_uT'
        if has_mag and col in df_phone.columns and not df_phone[col].isna().all():
            out[col] = np.interp(common_time, t_phone, df_phone[col].values)
        else:
            out[col] = np.nan

    # Orientation (Driver F might have NaNs)
    has_orient = bool(df_phone['phone_has_orientation'].iloc[0]) if 'phone_has_orientation' in df_phone.columns else False
    out['phone_has_orientation'] = has_orient
    for ang in ['azimuth', 'pitch', 'roll']:
        col = f'phone_orient_{ang}_rad'
        if has_orient and col in df_phone.columns and not df_phone[col].isna().all():
            unwrapped_ang = np.unwrap(df_phone[col].values)
            out[col] = (np.interp(common_time, t_phone, unwrapped_ang) + np.pi) % (2.0 * np.pi) - np.pi
        else:
            out[col] = np.nan

    # 3. Smartphone GNSS: Strict Sparse Arrival Preservation
    # Nearest phone sample for sample-and-hold telemetry
    nearest_phone_idx = np.searchsorted(t_phone, common_time, side='left')
    nearest_phone_idx = np.clip(nearest_phone_idx, 0, len(t_phone) - 1)

    out['phone_gps_lat_deg'] = df_phone['phone_gps_lat_deg'].values[nearest_phone_idx]
    out['phone_gps_lon_deg'] = df_phone['phone_gps_lon_deg'].values[nearest_phone_idx]
    out['phone_gps_alt_m'] = df_phone['phone_gps_alt_m'].values[nearest_phone_idx]
    out['phone_gps_east_m'] = df_phone['phone_gps_east_m'].values[nearest_phone_idx]
    out['phone_gps_north_m'] = df_phone['phone_gps_north_m'].values[nearest_phone_idx]
    out['phone_gps_up_m'] = df_phone['phone_gps_up_m'].values[nearest_phone_idx]
    out['phone_gps_speed_mps'] = df_phone['phone_gps_speed_mps'].values[nearest_phone_idx]
    out['phone_gps_accuracy_m'] = df_phone['phone_gps_accuracy_m'].values[nearest_phone_idx]
    out['phone_gps_orientation_rad'] = df_phone['phone_gps_orientation_rad'].values[nearest_phone_idx]
    out['phone_gps_satellites_str'] = df_phone['phone_gps_satellites_str'].values[nearest_phone_idx]
    out['phone_gps_satellites_used'] = df_phone['phone_gps_satellites_used'].values[nearest_phone_idx]
    out['phone_gps_satellites_in_view'] = df_phone['phone_gps_satellites_in_view'].values[nearest_phone_idx]
    out['phone_gps_is_valid'] = df_phone['phone_gps_is_valid'].values[nearest_phone_idx]

    # Map original new fixes into the common timeline bins
    # A common grid bin k corresponds to [common_time[k] - dt/2, common_time[k] + dt/2)
    is_new_grid = np.zeros(n_samples, dtype=bool)
    orig_new_indices = np.where(df_phone['phone_gps_is_new_fix'].values)[0]

    for idx in orig_new_indices:
        t_orig = t_phone[idx]
        if t_start - dt/2 <= t_orig <= t_end + dt/2:
            k = int(np.round((t_orig - t_start) / dt))
            if 0 <= k < n_samples:
                is_new_grid[k] = True

    out['phone_gps_is_new_fix'] = is_new_grid

    # Add sparse observation channels (non-NaN ONLY when new fix arrives)
    out['phone_gps_obs_east_m'] = np.where(is_new_grid, out['phone_gps_east_m'], np.nan)
    out['phone_gps_obs_north_m'] = np.where(is_new_grid, out['phone_gps_north_m'], np.nan)
    out['phone_gps_obs_speed_mps'] = np.where(is_new_grid, out['phone_gps_speed_mps'], np.nan)

    return out
