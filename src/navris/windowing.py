"""
NAVRIS Causal Window Generation for Inertial Navigation & Dead Reckoning.

Generates temporal windows for classical and machine-learning dead reckoning pipelines.
Features:
- Configurable window length (default: 20 samples = 2.0 s at 10 Hz)
- Configurable stride (default: 10 samples = 1.0 s at 10 Hz)
- Strictly causal: window index range [start, end] contains no future samples.
- Window targets: displacement dE, dN, dU, delta heading, and end-of-window velocities.
- Metadata retention: recording ID, driver, vehicle, phone, region, timestamps, split.
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class WindowConfig:
    window_length: int = 20    # Number of samples (20 samples = 2.0 s @ 10 Hz)
    stride: int = 10           # Number of samples step (10 samples = 1.0 s @ 10 Hz)
    dt: float = 0.1            # Nominal sampling interval [seconds]


def generate_causal_windows(
    df_sync: pd.DataFrame,
    metadata: Dict[str, Any],
    config: Optional[WindowConfig] = None
) -> Dict[str, Any]:
    """
    Generates causal sliding windows from a synchronized 10 Hz dataframe.
    
    Args:
        df_sync: Synchronized 10 Hz DataFrame
        metadata: Dictionary containing recording_id, driver, vehicle, phone, region, split
        config: WindowConfig instance (default: 20 samples length, 10 samples stride)
        
    Returns:
        Dictionary containing:
        - 'features': np.ndarray of shape (N_windows, window_length, N_features)
        - 'feature_names': list of feature column names
        - 'targets': dict of np.ndarrays for supervised displacement, velocity, heading
        - 'metadata': list of dicts per window
    """
    if config is None:
        config = WindowConfig()

    W = config.window_length
    S = config.stride
    n_rows = len(df_sync)

    if n_rows < W:
        return {
            'features': np.empty((0, W, 0), dtype=np.float32),
            'feature_names': [],
            'targets': {},
            'metadata': []
        }

    # Standard inertial input features (always available across all phones)
    imu_feature_cols = [
        'phone_accel_x_mps2', 'phone_accel_y_mps2', 'phone_accel_z_mps2',
        'phone_gyro_x_radps', 'phone_gyro_y_radps', 'phone_gyro_z_radps',
        'phone_gravity_x_mps2', 'phone_gravity_y_mps2', 'phone_gravity_z_mps2',
    ]

    # Additional features if available
    has_mag = bool(df_sync['phone_has_mag'].iloc[0]) if 'phone_has_mag' in df_sync.columns else False
    if has_mag and 'phone_mag_x_uT' in df_sync.columns and not df_sync['phone_mag_x_uT'].isna().all():
        imu_feature_cols.extend(['phone_mag_x_uT', 'phone_mag_y_uT', 'phone_mag_z_uT'])

    feature_data = df_sync[imu_feature_cols].values.astype(np.float32)

    # Reference signals for target extraction
    has_ref = 'ref_east_m' in df_sync.columns
    if has_ref:
        ref_east = df_sync['ref_east_m'].values
        ref_north = df_sync['ref_north_m'].values
        ref_up = df_sync['ref_up_m'].values
        ref_speed = df_sync['ref_speed_mps'].values
        ref_heading = df_sync['ref_heading_rad'].values
        ref_yaw_rate = df_sync['ref_yaw_rate_radps'].values
    else:
        ref_east = df_north = ref_up = ref_speed = ref_heading = ref_yaw_rate = None

    time_s = df_sync['time_s'].values

    # Window slice indices
    starts = list(range(0, n_rows - W + 1, S))
    num_windows = len(starts)

    features = np.zeros((num_windows, W, len(imu_feature_cols)), dtype=np.float32)

    disp_east = np.zeros(num_windows, dtype=np.float32)
    disp_north = np.zeros(num_windows, dtype=np.float32)
    disp_up = np.zeros(num_windows, dtype=np.float32)
    target_speed = np.zeros(num_windows, dtype=np.float32)
    target_heading = np.zeros(num_windows, dtype=np.float32)
    delta_heading = np.zeros(num_windows, dtype=np.float32)
    target_yaw_rate = np.zeros(num_windows, dtype=np.float32)

    window_meta = []

    for i, s_idx in enumerate(starts):
        e_idx = s_idx + W - 1  # inclusive end index

        features[i] = feature_data[s_idx:e_idx + 1]

        if has_ref:
            disp_east[i] = ref_east[e_idx] - ref_east[s_idx]
            disp_north[i] = ref_north[e_idx] - ref_north[s_idx]
            disp_up[i] = ref_up[e_idx] - ref_up[s_idx]
            target_speed[i] = ref_speed[e_idx]
            target_heading[i] = ref_heading[e_idx]
            # Wrap delta heading to [-pi, pi]
            dh = (ref_heading[e_idx] - ref_heading[s_idx] + np.pi) % (2.0 * np.pi) - np.pi
            delta_heading[i] = dh
            target_yaw_rate[i] = ref_yaw_rate[e_idx]

        window_meta.append({
            'window_idx': i,
            'recording_id': metadata.get('recording_id'),
            'driver': metadata.get('driver'),
            'vehicle': metadata.get('vehicle'),
            'phone': metadata.get('phone_model'),
            'country': metadata.get('country'),
            'region': metadata.get('region'),
            'tyre_pressure_code': metadata.get('tyre_pressure_code'),
            'split': metadata.get('split', 'unassigned'),
            'start_idx': s_idx,
            'end_idx': e_idx,
            'start_time_s': float(time_s[s_idx]),
            'end_time_s': float(time_s[e_idx]),
            'duration_s': float(time_s[e_idx] - time_s[s_idx])
        })

    targets = {
        'disp_east_m': disp_east,
        'disp_north_m': disp_north,
        'disp_up_m': disp_up,
        'speed_mps': target_speed,
        'heading_rad': target_heading,
        'delta_heading_rad': delta_heading,
        'yaw_rate_radps': target_yaw_rate
    }

    return {
        'features': features,
        'feature_names': imu_feature_cols,
        'targets': targets,
        'metadata': window_meta
    }


def save_windows_npz(
    window_dict: Dict[str, Any],
    output_filepath: str
) -> None:
    """
    Saves generated window features, targets, and metadata into a compact, fast NPZ file.
    """
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df_meta = pd.DataFrame(window_dict['metadata'])
    
    np.savez_compressed(
        output_filepath,
        features=window_dict['features'],
        feature_names=np.array(window_dict['feature_names']),
        disp_east_m=window_dict['targets']['disp_east_m'],
        disp_north_m=window_dict['targets']['disp_north_m'],
        disp_up_m=window_dict['targets']['disp_up_m'],
        speed_mps=window_dict['targets']['speed_mps'],
        heading_rad=window_dict['targets']['heading_rad'],
        delta_heading_rad=window_dict['targets']['delta_heading_rad'],
        yaw_rate_radps=window_dict['targets']['yaw_rate_radps'],
        metadata_json=df_meta.to_json(orient='records')
    )
