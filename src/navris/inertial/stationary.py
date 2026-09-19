"""
NAVRIS Phone-Only Stationary Detection & Bias Estimation.

Identifies vehicle rest epochs strictly using smartphone IMU measurements (and optional novel GNSS speed).
Estimates initial up vector, stationary gyro bias, and noise statistics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple


def detect_stationary_epochs(
    df: pd.DataFrame,
    window_samples: int = 50,           # ~5 seconds at 10 Hz
    gyro_mag_thresh: float = 0.05,       # rad/s
    accel_norm_std_thresh: float = 0.5,  # m/s^2 variability threshold
    gps_speed_thresh: float = 0.5        # m/s (used only on novel fixes)
) -> np.ndarray:
    """
    Causally detects whether each sample belongs to a stationary period.
    Phone-only detector: relies on accelerometer variance and gyro magnitude.
    
    Returns:
        Boolean numpy array of shape (N,) indicating stationary status.
    """
    n = len(df)
    if n == 0:
        return np.array([], dtype=bool)

    ax = df['phone_accel_x_mps2'].values
    ay = df['phone_accel_y_mps2'].values
    az = df['phone_accel_z_mps2'].values
    a_norm = np.sqrt(ax**2 + ay**2 + az**2)

    wx = df['phone_gyro_x_radps'].values
    wy = df['phone_gyro_y_radps'].values
    wz = df['phone_gyro_z_radps'].values
    w_norm = np.sqrt(wx**2 + wy**2 + wz**2)

    # Rolling statistics over causal window
    is_stat = np.zeros(n, dtype=bool)

    # Convert to pandas series for rolling computation
    a_series = pd.Series(a_norm)
    w_series = pd.Series(w_norm)

    roll_a_std = a_series.rolling(window=window_samples, min_periods=min(10, window_samples)).std().fillna(0.0).values
    roll_w_mean = w_series.rolling(window=window_samples, min_periods=min(10, window_samples)).mean().fillna(0.0).values

    # Base IMU condition
    imu_stat = (roll_a_std < accel_norm_std_thresh) & (roll_w_mean < gyro_mag_thresh)

    # Optional GNSS speed check (ONLY on novel fixes)
    if 'phone_gps_speed_mps' in df.columns and 'phone_gps_is_new_fix' in df.columns:
        gps_spd = df['phone_gps_speed_mps'].values
        is_new = df['phone_gps_is_new_fix'].fillna(False).astype(bool).values

        # Forward fill novel GPS speed causally
        novel_spd_causal = np.full(n, np.nan)
        last_spd = 0.0
        for i in range(n):
            if is_new[i] and not np.isnan(gps_spd[i]):
                last_spd = gps_spd[i]
            novel_spd_causal[i] = last_spd

        gps_stat = (novel_spd_causal < gps_speed_thresh)
        is_stat = imu_stat & gps_stat
    else:
        is_stat = imu_stat

    return is_stat


def estimate_stationary_imu_stats(
    df: pd.DataFrame,
    start_idx: int = 0,
    num_samples: int = 50
) -> Dict[str, Any]:
    """
    Computes mean specific force, gyro bias, and noise statistics over a confirmed stationary window.
    """
    sub = df.iloc[start_idx:start_idx + num_samples]
    if len(sub) < 10:
        raise ValueError(f"Insufficient samples for stationary stats: {len(sub)} < 10")

    ax = sub['phone_accel_x_mps2'].values
    ay = sub['phone_accel_y_mps2'].values
    az = sub['phone_accel_z_mps2'].values

    wx = sub['phone_gyro_x_radps'].values
    wy = sub['phone_gyro_y_radps'].values
    wz = sub['phone_gyro_z_radps'].values

    f_mean = np.array([np.mean(ax), np.mean(ay), np.mean(az)], dtype=np.float64)
    f_norm = float(np.linalg.norm(f_mean))
    f_std = np.array([np.std(ax), np.std(ay), np.std(az)], dtype=np.float64)

    w_mean = np.array([np.mean(wx), np.mean(wy), np.mean(wz)], dtype=np.float64)
    w_std = np.array([np.std(wx), np.std(wy), np.std(wz)], dtype=np.float64)

    return {
        'num_samples': len(sub),
        'f_body_mean': f_mean,
        'f_body_norm': f_norm,
        'f_body_std': f_std,
        'gyro_bias': w_mean,
        'gyro_noise_std': w_std,
        'up_vector_body': f_mean / f_norm if f_norm > 1e-6 else np.array([0.0, 0.0, 1.0])
    }
