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


def estimate_benchmark_lag(
    df_ref: pd.DataFrame,
    df_phone: pd.DataFrame,
    min_speed_mps: float = 3.0,
    min_yaw_rate_radps: float = 0.05,
    max_lag_s: float = 25.0,
    dt: float = 0.1,
    n_segments: int = 3,
    phone_yaw_channel: str = 'phone_gyro_y_radps'
) -> Dict[str, Any]:
    """
    Estimates deterministic offline time lag between vehicle reference and smartphone data.
    
    Uses vehicle reference yaw rate (VBOX/CAN) and smartphone gyro yaw channel with
    speed and yaw-rate excitation gating to find the constant offset tau maximizing Pearson correlation:
        t_phone_aligned = t_phone + tau
        
    Args:
        df_ref: Ingested vehicle reference DataFrame
        df_phone: Ingested smartphone DataFrame
        min_speed_mps: Speed excitation threshold in m/s (default 3.0)
        min_yaw_rate_radps: Yaw-rate excitation threshold in rad/s (default 0.05)
        max_lag_s: Maximum lag search range in seconds (+/- max_lag_s, default 25.0)
        dt: Lag search resolution in seconds (default 0.1)
        n_segments: Number of temporal segments for stability validation (default 3)
        phone_yaw_channel: Smartphone column name representing yaw rate (default 'phone_gyro_y_radps')
        
    Returns:
        Dictionary containing estimated_lag_s, zero_lag_corr, aligned_corr,
        zero_lag_slope, aligned_slope, segment_lags, segment_spread_s,
        is_valid, n_eval_samples, and status.
    """
    req_ref = ['ref_time_s', 'ref_speed_mps', 'ref_yaw_rate_radps']
    req_phone = ['phone_time_s', phone_yaw_channel]
    
    if not all(c in df_ref.columns for c in req_ref) or not all(c in df_phone.columns for c in req_phone):
        return {
            'estimated_lag_s': 0.0,
            'zero_lag_corr': float('nan'),
            'aligned_corr': float('nan'),
            'zero_lag_slope': float('nan'),
            'aligned_slope': float('nan'),
            'segment_lags': [],
            'segment_spread_s': 0.0,
            'is_valid': False,
            'n_eval_samples': 0,
            'status': 'FAILED: Missing required signal columns'
        }

    t_ref = df_ref['ref_time_s'].values
    v_ref = df_ref['ref_speed_mps'].values
    r_ref = df_ref['ref_yaw_rate_radps'].values
    t_phone = df_phone['phone_time_s'].values
    r_phone = df_phone[phone_yaw_channel].values

    if len(t_ref) == 0 or len(t_phone) == 0:
        return {
            'estimated_lag_s': 0.0,
            'zero_lag_corr': float('nan'),
            'aligned_corr': float('nan'),
            'zero_lag_slope': float('nan'),
            'aligned_slope': float('nan'),
            'segment_lags': [],
            'segment_spread_s': 0.0,
            'is_valid': False,
            'n_eval_samples': 0,
            'status': 'FAILED: Empty DataFrame'
        }

    lags = np.round(np.arange(-max_lag_s, max_lag_s + dt / 2, dt), 4)
    corrs = []
    
    for tau in lags:
        t_p = t_phone + tau
        r_interp = np.interp(t_ref, t_p, r_phone)
        mask = (
            (t_ref >= t_p[0]) & (t_ref <= t_p[-1]) &
            (v_ref > min_speed_mps) & (np.abs(r_ref) > min_yaw_rate_radps) &
            (~np.isnan(r_ref)) & (~np.isnan(r_interp))
        )
        if np.sum(mask) >= 30 and np.std(r_interp[mask]) > 1e-5 and np.std(r_ref[mask]) > 1e-5:
            c = float(np.corrcoef(r_interp[mask], r_ref[mask])[0, 1])
        else:
            c = -1.0
        corrs.append(c)

    best_idx = int(np.argmax(corrs))
    max_c = float(corrs[best_idx])
    
    # Check if sufficient excitation was present
    if max_c <= 0.0:
        return {
            'estimated_lag_s': 0.0,
            'zero_lag_corr': float('nan'),
            'aligned_corr': float('nan'),
            'zero_lag_slope': float('nan'),
            'aligned_slope': float('nan'),
            'segment_lags': [],
            'segment_spread_s': 0.0,
            'is_valid': False,
            'n_eval_samples': 0,
            'status': 'FAILED: Insufficient excitation or no positive correlation'
        }

    best_lag = float(lags[best_idx])
    
    # Zero-lag correlation and slope
    zero_idx = int(np.argmin(np.abs(lags)))
    zero_corr = float(corrs[zero_idx])
    
    t_p_0 = t_phone
    r_p_0 = np.interp(t_ref, t_p_0, r_phone)
    mask_0 = (
        (t_ref >= t_p_0[0]) & (t_ref <= t_p_0[-1]) &
        (v_ref > min_speed_mps) & (np.abs(r_ref) > min_yaw_rate_radps) &
        (~np.isnan(r_ref)) & (~np.isnan(r_p_0))
    )
    if np.sum(mask_0) >= 30 and np.std(r_ref[mask_0]) > 1e-5 and np.std(r_p_0[mask_0]) > 1e-5:
        zero_slope = float(np.polyfit(r_ref[mask_0], r_p_0[mask_0], 1)[0])
    else:
        zero_slope = float('nan')

    # Aligned slope and sample count
    t_p_best = t_phone + best_lag
    r_p_best = np.interp(t_ref, t_p_best, r_phone)
    mask_best = (
        (t_ref >= t_p_best[0]) & (t_ref <= t_p_best[-1]) &
        (v_ref > min_speed_mps) & (np.abs(r_ref) > min_yaw_rate_radps) &
        (~np.isnan(r_ref)) & (~np.isnan(r_p_best))
    )
    n_eval_samples = int(np.sum(mask_best))
    if n_eval_samples >= 30 and np.std(r_ref[mask_best]) > 1e-5 and np.std(r_p_best[mask_best]) > 1e-5:
        aligned_slope = float(np.polyfit(r_ref[mask_best], r_p_best[mask_best], 1)[0])
    else:
        aligned_slope = float('nan')

    # Multi-segment stability validation
    t_edges = np.linspace(t_ref[0], t_ref[-1], n_segments + 1)
    seg_lags = []
    for s in range(n_segments):
        seg_mask = (
            (t_ref >= t_edges[s]) & (t_ref < t_edges[s + 1]) &
            (v_ref > min_speed_mps) & (np.abs(r_ref) > min_yaw_rate_radps) &
            (~np.isnan(r_ref))
        )
        if np.sum(seg_mask) < 20:
            seg_lags.append(float('nan'))
            continue
        seg_corrs = []
        for tau in lags:
            t_p = t_phone + tau
            r_interp = np.interp(t_ref, t_p, r_phone)
            o_mask = seg_mask & (t_ref >= t_p[0]) & (t_ref <= t_p[-1]) & (~np.isnan(r_interp))
            if np.sum(o_mask) >= 20 and np.std(r_interp[o_mask]) > 1e-5 and np.std(r_ref[o_mask]) > 1e-5:
                seg_corrs.append(float(np.corrcoef(r_interp[o_mask], r_ref[o_mask])[0, 1]))
            else:
                seg_corrs.append(-1.0)
        seg_best_idx = int(np.argmax(seg_corrs))
        if seg_corrs[seg_best_idx] > 0.0:
            seg_lags.append(float(round(lags[seg_best_idx], 4)))
        else:
            seg_lags.append(float('nan'))

    valid_segs = [x for x in seg_lags if not np.isnan(x)]
    seg_spread = float(round(max(valid_segs) - min(valid_segs), 4)) if len(valid_segs) >= 2 else 0.0

    return {
        'estimated_lag_s': float(round(best_lag, 4)),
        'zero_lag_corr': float(round(zero_corr, 4)),
        'aligned_corr': float(round(max_c, 4)),
        'zero_lag_slope': float(round(zero_slope, 4)) if not np.isnan(zero_slope) else float('nan'),
        'aligned_slope': float(round(aligned_slope, 4)) if not np.isnan(aligned_slope) else float('nan'),
        'segment_lags': seg_lags,
        'segment_spread_s': seg_spread,
        'is_valid': True,
        'n_eval_samples': n_eval_samples,
        'status': 'SUCCESS'
    }


def synchronize_recordings(
    df_ref: pd.DataFrame,
    df_phone: pd.DataFrame,
    dt: float = 0.1,
    phone_time_offset_s: float = 0.0,
    auto_align_benchmark: bool = False,
    benchmark_yaw_channel: str = 'phone_gyro_y_radps'
) -> pd.DataFrame:
    """
    Synchronizes a reference DataFrame and a smartphone DataFrame onto a common 10 Hz timeline.
    
    Args:
        df_ref: Normalized reference DataFrame from ingest_reference_data
        df_phone: Normalized smartphone DataFrame from ingest_smartphone_data
        dt: Target sampling interval in seconds (default 0.1 s = 10 Hz)
        phone_time_offset_s: Constant offset added to phone timeline: t_phone_aligned = t_phone + offset
        auto_align_benchmark: If True, automatically estimates benchmark lag via estimate_benchmark_lag
        benchmark_yaw_channel: Phone channel used for auto lag estimation (default 'phone_gyro_y_radps')
        
    Returns:
        Synchronized DataFrame adhering to SYNCHRONIZED_SCHEMA_COLUMNS, with sync metadata in df.attrs.
    """
    sync_metadata: Dict[str, Any] = {}

    if auto_align_benchmark:
        lag_res = estimate_benchmark_lag(
            df_ref=df_ref,
            df_phone=df_phone,
            dt=dt,
            phone_yaw_channel=benchmark_yaw_channel
        )
        if lag_res['is_valid']:
            phone_time_offset_s = lag_res['estimated_lag_s']
            sync_metadata = {
                'method': 'auto_benchmark_lag',
                **lag_res
            }
        else:
            phone_time_offset_s = 0.0
            sync_metadata = {
                'method': 'auto_benchmark_lag_failed',
                **lag_res
            }
    else:
        sync_metadata = {
            'method': 'manual_offset' if phone_time_offset_s != 0.0 else 'zero_lag',
            'estimated_lag_s': float(round(phone_time_offset_s, 4)),
            'is_valid': True
        }

    t_ref = df_ref['ref_time_s'].values
    t_phone = df_phone['phone_time_s'].values + phone_time_offset_s

    # Determine overlapping time range
    t_start = max(float(t_ref[0]), float(t_phone[0]))
    t_end = min(float(t_ref[-1]), float(t_phone[-1]))

    if t_end <= t_start:
        raise ValueError(
            f"No temporal overlap between reference [{t_ref[0]}, {t_ref[-1]}] "
            f"and aligned phone [{t_phone[0]}, {t_phone[-1]}]"
        )

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

    out.attrs['sync_metadata'] = sync_metadata

    return out
