"""
NAVRIS Raw Data Ingestion and Normalization.
Loads raw CSV files from IO-VNBD, sanitizes anomalies, converts units to standard SI,
and outputs standardized pandas DataFrames.
"""

import os
import re
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List

from navris.schema import (
    KMH_TO_MPS, G_TO_MPS2, DEG_TO_RAD,
    PHONE_SCHEMA_COLUMNS, REFERENCE_SCHEMA_COLUMNS
)
from navris.coords import geodetic_to_enu

# Month name mapping for Excel date-corrupted satellite strings
MONTH_MAP = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}


def parse_satellite_string(val: Any) -> Tuple[str, float, float]:
    """
    Sanitizes and parses raw smartphone satellite string.
    Handles standard 'used / in_view' strings as well as Excel date-converted artifacts
    (e.g., '10-Nov' -> 10 used, 11 in view; 'Aug-20' -> 8 used, 20 in view).
    
    Returns: (cleaned_str, satellites_used, satellites_in_view)
    """
    if pd.isna(val) or val is None:
        return "N/A", np.nan, np.nan

    s = str(val).strip()
    if not s:
        return "N/A", np.nan, np.nan

    # Standard format: e.g. '27 / 28' or '17 / 23'
    slash_match = re.match(r'^(\d+)\s*/\s*(\d+)$', s)
    if slash_match:
        u = float(slash_match.group(1))
        v = float(slash_match.group(2))
        return s, u, v

    # Single number
    single_num = re.match(r'^(\d+)$', s)
    if single_num:
        num = float(single_num.group(1))
        return s, num, num

    # Excel date conversion: e.g. '10-Nov', '09-Oct', 'Aug-20'
    m_day_month = re.match(r'^(\d+)-([A-Za-z]{3})$', s)
    if m_day_month:
        u = float(m_day_month.group(1))
        mon_str = m_day_month.group(2).lower()
        if mon_str in MONTH_MAP:
            v = float(MONTH_MAP[mon_str])
            return f"{int(u)} / {int(v)}", u, v

    m_month_day = re.match(r'^([A-Za-z]{3})-(\d+)$', s)
    if m_month_day:
        mon_str = m_month_day.group(1).lower()
        v = float(m_month_day.group(2))
        if mon_str in MONTH_MAP:
            u = float(MONTH_MAP[mon_str])
            return f"{int(u)} / {int(v)}", u, v

    return s, np.nan, np.nan


def load_raw_csv_robust(filepath: str) -> pd.DataFrame:
    """
    Loads raw CSV handling character encodings and column variations.
    """
    encodings = ['latin1', 'cp1252', 'utf-8']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            break
        except Exception:
            continue

    if df is None:
        raise IOError(f"Failed to read CSV at {filepath} with any supported encoding.")

    # Clean whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]

    # Special handling for known anomalous file S-A4.csv (extra unquoted comma at position 6)
    if os.path.basename(filepath) == 'S-A4.csv' and 'Unnamed: 24' in df.columns:
        # Re-parse cleanly by reading lines and fixing the empty column
        with open(filepath, 'r', encoding='latin1') as fp:
            lines = fp.readlines()
        headers = [h.strip() for h in lines[0].strip().split(',') if h.strip()]
        rows = []
        for line in lines[1:]:
            parts = [p.strip() for p in line.strip().split(',')]
            # remove empty element at position 6 if present
            if len(parts) > len(headers):
                # find the empty part
                cleaned = [p for i, p in enumerate(parts) if p != '' or i != 6]
                rows.append(cleaned[:len(headers)])
            else:
                rows.append(parts[:len(headers)])
        df = pd.DataFrame(rows, columns=headers)
        # Infer types safely in pandas 3.0
        for col_name in df.columns:
            try:
                df[col_name] = pd.to_numeric(df[col_name])
            except (ValueError, TypeError):
                pass

    return df


def causal_unwrap_timestamps(
    raw_time_ms: np.ndarray,
    date_series: Optional[pd.Series] = None,
    default_dt_ms: float = 100.0
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Causally unwraps smartphone timestamps subject to logger timer resets.
    
    Strictly causal: For each epoch i, only data at or before epoch i is used.
    If a backward timer reset (diff < -1000 ms) is detected:
    - If the wall-clock DATE column is available and valid, elapsed time between
      date[i] and date[i-1] is used to bridge the reset causally.
    - Otherwise, a causal default step (100 ms = 10 Hz) is applied.
    - Strict forward monotonicity (diff > 0) is enforced.
    
    Returns:
        (corrected_time_s, list_of_resets_detected)
    """
    n = len(raw_time_ms)
    if n == 0:
        return np.array([], dtype=float), []

    # Parse date series causally if available
    date_dt = None
    if date_series is not None and len(date_series) > 0:
        try:
            # IO-VNBD format: '2019-09-08 12:03:51:741' -> replace last colon with period
            s_clean = date_series.astype(str).str.replace(r':(\d{3})$', r'.\1', regex=True)
            date_dt = pd.to_datetime(s_clean, format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        except Exception:
            date_dt = None

    corrected_ms = np.zeros(n, dtype=float)
    t0_ms = raw_time_ms[0] if (not np.isnan(raw_time_ms[0])) else 0.0
    cum_offset_ms = -t0_ms
    resets_detected = []

    corrected_ms[0] = 0.0
    for i in range(1, n):
        cur_raw = raw_time_ms[i]
        prev_raw = raw_time_ms[i - 1]

        diff = cur_raw - prev_raw
        if diff < -1000.0:  # Timer reset detected (> 1 second backwards jump)
            step_ms = default_dt_ms
            if date_dt is not None and pd.notna(date_dt.iloc[i]) and pd.notna(date_dt.iloc[i - 1]):
                date_diff_s = (date_dt.iloc[i] - date_dt.iloc[i - 1]).total_seconds()
                if 0 < date_diff_s < 86400:  # Causally valid positive wall-clock delta
                    step_ms = date_diff_s * 1000.0

            cum_offset_ms = corrected_ms[i - 1] + step_ms - cur_raw
            resets_detected.append({
                'index': i,
                'prev_raw_ms': prev_raw,
                'cur_raw_ms': cur_raw,
                'step_ms': step_ms,
                'offset_ms': cum_offset_ms
            })

        cur_corr = cur_raw + cum_offset_ms
        # Enforce strict forward progression
        if cur_corr <= corrected_ms[i - 1]:
            cur_corr = corrected_ms[i - 1] + 1.0  # +1 ms minimum step
            cum_offset_ms = cur_corr - cur_raw

        corrected_ms[i] = cur_corr

    return corrected_ms / 1000.0, resets_detected


def ingest_smartphone_data(
    filepath: str,
    origin_geodetic: Optional[Tuple[float, float, float]] = None
) -> Tuple[pd.DataFrame, Tuple[float, float, float]]:
    """
    Ingests and normalizes smartphone CSV (both 24-column and 18-column Driver F variants).
    
    Converts units to standard SI:
    - Speed km/h -> m/s
    - Angles deg -> rad
    - Computes local ENU relative to origin_geodetic (or first valid GPS fix)
    - Detects sample-and-hold GPS (phone_gps_is_new_fix)
    
    Returns: (normalized_dataframe, (lat0, lon0, alt0))
    """
    df_raw = load_raw_csv_robust(filepath)

    out = pd.DataFrame()

    # 1. Time
    time_col = [c for c in df_raw.columns if 'time since start' in c.lower()]
    date_col = [c for c in df_raw.columns if 'date' in c.lower()]
    if time_col:
        raw_time_ms = pd.to_numeric(df_raw[time_col[0]], errors='coerce').values.astype(float)
        date_series = df_raw[date_col[0]] if date_col else None
        corr_time_s, _ = causal_unwrap_timestamps(raw_time_ms, date_series=date_series)
        out['phone_time_raw_ms'] = raw_time_ms
        out['phone_time_s'] = corr_time_s
    else:
        out['phone_time_raw_ms'] = np.arange(len(df_raw)) * 100.0
        out['phone_time_s'] = np.arange(len(df_raw)) * 0.1

    # 2. Accelerometer (m/s^2)
    for axis in ['x', 'y', 'z']:
        col = [c for c in df_raw.columns if re.search(r'accelerometer\s+' + axis, c, re.I)]
        out[f'phone_accel_{axis}_mps2'] = pd.to_numeric(df_raw[col[0]], errors='coerce') if col else np.nan

    # 3. Gravity (m/s^2)
    for axis in ['x', 'y', 'z']:
        col = [c for c in df_raw.columns if re.search(r'gravity\s+' + axis, c, re.I)]
        out[f'phone_gravity_{axis}_mps2'] = pd.to_numeric(df_raw[col[0]], errors='coerce') if col else np.nan

    # 4. Gyroscope (rad/s)
    gyro_x_col = [c for c in df_raw.columns if re.search(r'gyroscope\s+(x|yaw)\b', c, re.I)]
    gyro_y_col = [c for c in df_raw.columns if re.search(r'gyroscope\s+(y|pitch)\b', c, re.I)]
    gyro_z_col = [c for c in df_raw.columns if re.search(r'gyroscope\s+(z|roll)\b', c, re.I)]

    out['phone_gyro_x_radps'] = pd.to_numeric(df_raw[gyro_x_col[0]], errors='coerce') if gyro_x_col else np.nan
    out['phone_gyro_y_radps'] = pd.to_numeric(df_raw[gyro_y_col[0]], errors='coerce') if gyro_y_col else np.nan
    out['phone_gyro_z_radps'] = pd.to_numeric(df_raw[gyro_z_col[0]], errors='coerce') if gyro_z_col else np.nan

    # 5. Magnetometer (uT) - Driver F compatibility (missing in S-T1 to S-T9)
    mag_cols = [c for c in df_raw.columns if 'magnetic field' in c.lower()]
    has_mag = len(mag_cols) >= 3
    out['phone_has_mag'] = has_mag

    for axis in ['x', 'y', 'z']:
        col = [c for c in df_raw.columns if re.search(r'magnetic\s*field\s+' + axis, c, re.I)]
        out[f'phone_mag_{axis}_uT'] = pd.to_numeric(df_raw[col[0]], errors='coerce') if col else np.nan

    # 6. Orientation (Euler angles, deg -> rad) - Driver F compatibility
    orient_cols = [c for c in df_raw.columns if 'orientation' in c.lower() and 'gps' not in c.lower()]
    has_orient = len(orient_cols) >= 3
    out['phone_has_orientation'] = has_orient

    az_col = [c for c in orient_cols if 'azimuth' in c.lower() or 'yaw' in c.lower()]
    pi_col = [c for c in orient_cols if 'pitch' in c.lower()]
    ro_col = [c for c in orient_cols if 'roll' in c.lower()]

    out['phone_orient_azimuth_rad'] = (pd.to_numeric(df_raw[az_col[0]], errors='coerce') * DEG_TO_RAD) if az_col else np.nan
    out['phone_orient_pitch_rad'] = (pd.to_numeric(df_raw[pi_col[0]], errors='coerce') * DEG_TO_RAD) if pi_col else np.nan
    out['phone_orient_roll_rad'] = (pd.to_numeric(df_raw[ro_col[0]], errors='coerce') * DEG_TO_RAD) if ro_col else np.nan

    # 7. GPS Signals
    lat_col = [c for c in df_raw.columns if 'gps' in c.lower() and 'lat' in c.lower()][0]
    lon_col = [c for c in df_raw.columns if 'gps' in c.lower() and 'long' in c.lower()][0]
    alt_col = [c for c in df_raw.columns if 'gps' in c.lower() and 'alt' in c.lower()]
    spd_col = [c for c in df_raw.columns if 'gps' in c.lower() and 'speed' in c.lower()]
    acc_col = [c for c in df_raw.columns if 'gps' in c.lower() and 'acc' in c.lower()]
    ori_col = [c for c in df_raw.columns if 'gps' in c.lower() and 'orientation' in c.lower()]
    sat_col = [c for c in df_raw.columns if 'sat' in c.lower()]

    lats = pd.to_numeric(df_raw[lat_col], errors='coerce').values
    lons = pd.to_numeric(df_raw[lon_col], errors='coerce').values
    alts = pd.to_numeric(df_raw[alt_col[0]], errors='coerce').values if alt_col else np.zeros(len(df_raw))

    out['phone_gps_lat_deg'] = lats
    out['phone_gps_lon_deg'] = lons
    out['phone_gps_alt_m'] = alts

    # Speed: Raw column 'GPS SPEED (Kmh)' is empirically verified to be logged natively in m/s
    # (Gate 1.5A & E3 forensic audits demonstrated raw/VBOX ratio is 0.997 +/- 0.005 across all 8 recordings).
    # No /3.6 conversion is applied.
    if spd_col:
        out['phone_gps_speed_mps'] = pd.to_numeric(df_raw[spd_col[0]], errors='coerce')
    else:
        out['phone_gps_speed_mps'] = np.nan

    # Accuracy (m)
    out['phone_gps_accuracy_m'] = pd.to_numeric(df_raw[acc_col[0]], errors='coerce') if acc_col else np.nan

    # GPS Orientation (deg -> rad)
    if ori_col:
        out['phone_gps_orientation_rad'] = pd.to_numeric(df_raw[ori_col[0]], errors='coerce') * DEG_TO_RAD
    else:
        out['phone_gps_orientation_rad'] = np.nan

    # Satellites info parsing
    sat_strs, sat_used, sat_view = [], [], []
    if sat_col:
        for val in df_raw[sat_col[0]]:
            cleaned_s, u, v = parse_satellite_string(val)
            sat_strs.append(cleaned_s)
            sat_used.append(u)
            sat_view.append(v)
    else:
        sat_strs = ['N/A'] * len(df_raw)
        sat_used = [np.nan] * len(df_raw)
        sat_view = [np.nan] * len(df_raw)

    out['phone_gps_satellites_str'] = sat_strs
    out['phone_gps_satellites_used'] = sat_used
    out['phone_gps_satellites_in_view'] = sat_view

    # Validity & Sample-and-Hold / New Fix Detection
    valid_fix = (np.abs(lats) > 0.1) & (np.abs(lons) > 0.1) & (~np.isnan(lats)) & (~np.isnan(lons))
    out['phone_gps_is_valid'] = valid_fix

    # A new fix occurs ONLY when coordinates change from previous sample
    is_new = np.zeros(len(df_raw), dtype=bool)
    if len(df_raw) > 0 and valid_fix[0]:
        is_new[0] = True
    for i in range(1, len(df_raw)):
        if valid_fix[i]:
            if not valid_fix[i-1]:
                is_new[i] = True
            elif lats[i] != lats[i-1] or lons[i] != lons[i-1]:
                is_new[i] = True
    out['phone_gps_is_new_fix'] = is_new

    # 8. Local ENU Coordinate Transformation
    if origin_geodetic is None:
        valid_indices = np.where(valid_fix)[0]
        if len(valid_indices) > 0:
            idx0 = valid_indices[0]
            origin_geodetic = (float(lats[idx0]), float(lons[idx0]), float(alts[idx0]))
        else:
            origin_geodetic = (0.0, 0.0, 0.0)

    e, n, u = geodetic_to_enu(
        lats, lons, alts,
        origin_geodetic[0], origin_geodetic[1], origin_geodetic[2]
    )
    out['phone_gps_east_m'] = e
    out['phone_gps_north_m'] = n
    out['phone_gps_up_m'] = u

    # Ensure schema column order
    return out[PHONE_SCHEMA_COLUMNS], origin_geodetic


def ingest_reference_data(
    filepath: str,
    origin_geodetic: Optional[Tuple[float, float, float]] = None
) -> Tuple[pd.DataFrame, Tuple[float, float, float]]:
    """
    Ingests and normalizes Vehicle CAN / VBOX reference CSV.
    
    Converts units to standard SI:
    - Speed km/h -> m/s
    - Heading deg -> rad [0, 2pi)
    - Yaw rate deg/s -> rad/s
    - Steering angle deg -> rad
    - Longitudinal/lateral acceleration g -> m/s^2
    - Height km -> interpreted properly as meters
    - Computes local ENU relative to origin_geodetic
    - Parses 0x80 DGPS bitmask and detects satellite dropout epochs
    
    Returns: (normalized_dataframe, (lat0, lon0, alt0))
    """
    df_raw = load_raw_csv_robust(filepath)

    out = pd.DataFrame()

    # 1. Time
    time_col = [c for c in df_raw.columns if 'time' in c.lower()][0]
    raw_time_s = pd.to_numeric(df_raw[time_col], errors='coerce').values
    t0 = raw_time_s[0] if len(raw_time_s) > 0 else 0.0
    out['ref_time_s'] = raw_time_s - t0

    # 2. Coordinates
    lat_col = [c for c in df_raw.columns if 'lat' in c.lower()][0]
    lon_col = [c for c in df_raw.columns if 'long' in c.lower()][0]
    alt_col = [c for c in df_raw.columns if 'height' in c.lower()][0]

    lats = pd.to_numeric(df_raw[lat_col], errors='coerce').values
    lons = pd.to_numeric(df_raw[lon_col], errors='coerce').values
    alts = pd.to_numeric(df_raw[alt_col], errors='coerce').values  # True altitude in meters

    out['ref_lat_deg'] = lats
    out['ref_lon_deg'] = lons
    out['ref_alt_m'] = alts

    # Establish origin if not provided
    if origin_geodetic is None:
        valid_coords = (np.abs(lats) > 0.1) & (np.abs(lons) > 0.1) & (~np.isnan(lats))
        valid_idx = np.where(valid_coords)[0]
        if len(valid_idx) > 0:
            idx0 = valid_idx[0]
            origin_geodetic = (float(lats[idx0]), float(lons[idx0]), float(alts[idx0]))
        else:
            origin_geodetic = (0.0, 0.0, 0.0)

    e, n, u = geodetic_to_enu(
        lats, lons, alts,
        origin_geodetic[0], origin_geodetic[1], origin_geodetic[2]
    )
    out['ref_east_m'] = e
    out['ref_north_m'] = n
    out['ref_up_m'] = u

    # 3. Speeds (km/h -> m/s)
    vel_col = [c for c in df_raw.columns if c.lower().startswith('velocity')][0]
    out['ref_speed_mps'] = pd.to_numeric(df_raw[vel_col], errors='coerce') * KMH_TO_MPS

    vvel_col = [c for c in df_raw.columns if 'vertical' in c.lower()][0]
    out['ref_vertical_speed_mps'] = pd.to_numeric(df_raw[vvel_col], errors='coerce') * KMH_TO_MPS

    # 4. Heading (deg -> rad)
    head_col = [c for c in df_raw.columns if 'heading' in c.lower()][0]
    head_deg = pd.to_numeric(df_raw[head_col], errors='coerce').values
    out['ref_heading_deg'] = head_deg
    out['ref_heading_rad'] = np.radians(head_deg) % (2.0 * np.pi)

    # 5. Kinematics & Chassis
    yaw_col = [c for c in df_raw.columns if 'yaw rate' in c.lower()][0]
    out['ref_yaw_rate_radps'] = np.radians(pd.to_numeric(df_raw[yaw_col], errors='coerce'))

    steer_col = [c for c in df_raw.columns if 'steering' in c.lower()][0]
    out['ref_steering_angle_rad'] = np.radians(pd.to_numeric(df_raw[steer_col], errors='coerce'))

    # Wheel Speeds (rad/s)
    ws_fl = [c for c in df_raw.columns if 'wheel speed front left' in c.lower()][0]
    ws_fr = [c for c in df_raw.columns if 'wheel speed front right' in c.lower()][0]
    ws_rl = [c for c in df_raw.columns if 'wheel speed rear left' in c.lower()][0]
    ws_rr = [c for c in df_raw.columns if 'wheel speed rear right' in c.lower()][0]

    out['ref_wheel_speed_fl_radps'] = pd.to_numeric(df_raw[ws_fl], errors='coerce')
    out['ref_wheel_speed_fr_radps'] = pd.to_numeric(df_raw[ws_fr], errors='coerce')
    out['ref_wheel_speed_rl_radps'] = pd.to_numeric(df_raw[ws_rl], errors='coerce')
    out['ref_wheel_speed_rr_radps'] = pd.to_numeric(df_raw[ws_rr], errors='coerce')

    # Accelerations (g -> m/s^2)
    acc_long = [c for c in df_raw.columns if 'longitudinal acceleration' in c.lower()][0]
    acc_lat = [c for c in df_raw.columns if 'lateral acceleration' in c.lower()][0]
    out['ref_accel_long_mps2'] = pd.to_numeric(df_raw[acc_long], errors='coerce') * G_TO_MPS2
    out['ref_accel_lat_mps2'] = pd.to_numeric(df_raw[acc_lat], errors='coerce') * G_TO_MPS2

    # Speedometer speed (km/h -> m/s)
    ind_spd = [c for c in df_raw.columns if 'indicated vehicle speed' in c.lower()][0]
    out['ref_indicated_speed_mps'] = pd.to_numeric(df_raw[ind_spd], errors='coerce') * KMH_TO_MPS

    # Engine & Powertrain
    eng_spd = [c for c in df_raw.columns if 'engine speed' in c.lower()][0]
    out['ref_engine_speed_rpm'] = pd.to_numeric(df_raw[eng_spd], errors='coerce')

    brk_prs = [c for c in df_raw.columns if 'brake pressure' in c.lower()][0]
    out['ref_brake_pressure_psi'] = pd.to_numeric(df_raw[brk_prs], errors='coerce')

    throt = [c for c in df_raw.columns if 'accelerator pedal' in c.lower()][0]
    out['ref_throttle_pct'] = pd.to_numeric(df_raw[throt], errors='coerce')

    gear = [c for c in df_raw.columns if c.strip().lower() == 'gear (number fof gear employed 1-5)'][0]
    out['ref_gear'] = pd.to_numeric(df_raw[gear], errors='coerce')

    hb = [c for c in df_raw.columns if 'handbrake' in c.lower()][0]
    out['ref_handbrake'] = pd.to_numeric(df_raw[hb], errors='coerce')

    # 6. Satellites & Dropout Detection
    sat_col = [c for c in df_raw.columns if 'satellite' in c.lower()][0]
    raw_sats = pd.to_numeric(df_raw[sat_col], errors='coerce').values
    out['ref_gps_satellites_raw'] = raw_sats

    # If raw >= 128 (0x80 bit set), it's DGPS / valid lock
    is_dgps = raw_sats >= 128.0
    true_sats = np.where(is_dgps, raw_sats - 128.0, raw_sats)
    out['ref_gps_is_dgps'] = is_dgps
    out['ref_gps_satellites_count'] = true_sats
    out['ref_gps_dropout'] = true_sats == 0.0

    return out[REFERENCE_SCHEMA_COLUMNS], origin_geodetic
