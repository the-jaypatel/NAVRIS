import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from navris.windowing import generate_causal_windows, WindowConfig

def test_causal_window_generation():
    # Synthetic 10 Hz DataFrame with 100 samples
    n_samples = 100
    t = np.arange(n_samples) * 0.1
    df_sync = pd.DataFrame({
        'time_s': t,
        'phone_accel_x_mps2': np.sin(t),
        'phone_accel_y_mps2': np.cos(t),
        'phone_accel_z_mps2': np.ones(n_samples) * 9.81,
        'phone_gyro_x_radps': np.zeros(n_samples),
        'phone_gyro_y_radps': np.zeros(n_samples),
        'phone_gyro_z_radps': np.ones(n_samples) * 0.1,
        'phone_gravity_x_mps2': np.zeros(n_samples),
        'phone_gravity_y_mps2': np.zeros(n_samples),
        'phone_gravity_z_mps2': np.ones(n_samples) * 9.81,
        'phone_has_mag': False,
        'phone_has_orientation': False,
        'ref_east_m': t * 5.0,     # moving East at 5 m/s
        'ref_north_m': t * 2.0,    # moving North at 2 m/s
        'ref_up_m': np.zeros(n_samples),
        'ref_speed_mps': np.ones(n_samples) * np.sqrt(5**2 + 2**2),
        'ref_heading_rad': np.ones(n_samples) * 0.5,
        'ref_yaw_rate_radps': np.zeros(n_samples)
    })

    meta = {'recording_id': 'TEST_REC', 'driver': 'Driver A', 'split': 'train'}
    config = WindowConfig(window_length=20, stride=10, dt=0.1)

    win = generate_causal_windows(df_sync, meta, config)

    # Window count check: (100 - 20) // 10 + 1 = 9 windows
    assert len(win['features']) == 9
    assert win['features'].shape == (9, 20, 9)

    # Causal span check: window 0 spans indices 0 to 19 (duration = 1.9s)
    w0_meta = win['metadata'][0]
    assert w0_meta['start_idx'] == 0
    assert w0_meta['end_idx'] == 19
    assert np.isclose(w0_meta['duration_s'], 1.9, atol=1e-5)

    # Targets check: displacement over 20 samples (1.9s span)
    # dx = 5.0 m/s * 1.9s = 9.5 m
    assert np.isclose(win['targets']['disp_east_m'][0], 9.5, atol=1e-4)
    # dy = 2.0 m/s * 1.9s = 3.8 m
    assert np.isclose(win['targets']['disp_north_m'][0], 3.8, atol=1e-4)

    # Check window 1 starts at index 10 (stride 10)
    w1_meta = win['metadata'][1]
    assert w1_meta['start_idx'] == 10
    assert w1_meta['end_idx'] == 29
