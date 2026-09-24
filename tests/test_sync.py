import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from navris.ingest import ingest_smartphone_data, ingest_reference_data
from navris.sync import synchronize_recordings

def test_sync_v_s1_timeline():
    ref_path = 'data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/V-Dataset/V-S1.csv'
    phone_path = 'data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset/S-S1.csv'
    if not (os.path.exists(ref_path) and os.path.exists(phone_path)):
        pytest.skip('Dataset not found locally')

    df_ref, origin = ingest_reference_data(ref_path)
    df_phone, _ = ingest_smartphone_data(phone_path, origin_geodetic=origin)

    df_sync = synchronize_recordings(df_ref, df_phone, dt=0.1)

    # 1. Timeline monotonicity and uniform 10 Hz step
    t = df_sync['time_s'].values
    dt = np.diff(t)
    assert np.all(dt > 0), "Timestamps must be strictly monotonically increasing"
    assert np.allclose(dt, 0.1, atol=1e-4), "Time step must be constant 0.1s (10 Hz)"

    # 2. Sparse GPS observation preservation
    new_fixes_count = df_sync['phone_gps_is_new_fix'].sum()
    sparse_obs_count = df_sync['phone_gps_obs_east_m'].notna().sum()
    assert new_fixes_count == sparse_obs_count
    # Verify that GPS is NOT updated every 0.1s
    assert new_fixes_count < len(df_sync) / 5

    # 3. Reference trajectory integrity
    assert 'ref_east_m' in df_sync.columns
    assert 'ref_north_m' in df_sync.columns
    assert 'ref_speed_mps' in df_sync.columns
    assert not df_sync['ref_east_m'].isna().any()


from navris.sync import estimate_benchmark_lag


def _create_synthetic_test_data(lag_s: float = 0.0, speed_mps: float = 12.0, yaw_amp: float = 0.2):
    """Helper to generate synthetic paired reference and phone data with known lag."""
    t_ref = np.arange(0.0, 100.05, 0.1)
    ref_speed = np.full_like(t_ref, speed_mps)
    # Sinusoidal yaw rate active between 10s and 90s
    ref_yaw = yaw_amp * np.sin(0.2 * t_ref) * ((t_ref > 10.0) & (t_ref < 90.0))

    df_ref = pd.DataFrame({
        'ref_time_s': t_ref,
        'ref_speed_mps': ref_speed,
        'ref_yaw_rate_radps': ref_yaw,
        'ref_east_m': t_ref * speed_mps,
        'ref_north_m': t_ref * 2.0,
        'ref_up_m': np.zeros_like(t_ref),
        'ref_heading_rad': np.zeros_like(t_ref),
        'ref_gear': np.ones_like(t_ref, dtype=int),
        'ref_handbrake': np.zeros_like(t_ref, dtype=int)
    })

    t_phone = np.arange(0.0, 100.05, 0.1)
    # Signal shifted so that t_phone + lag_s matches t_ref
    shifted_time = t_phone + lag_s
    phone_yaw = yaw_amp * np.sin(0.2 * shifted_time) * ((shifted_time > 10.0) & (shifted_time < 90.0))

    n_p = len(t_phone)
    is_new = (np.arange(n_p) % 10 == 0) # 1 Hz sparse fixes on 10 Hz stream

    df_phone = pd.DataFrame({
        'phone_time_s': t_phone,
        'phone_gyro_x_radps': np.zeros(n_p),
        'phone_gyro_y_radps': phone_yaw,
        'phone_gyro_z_radps': np.zeros(n_p),
        'phone_accel_x_mps2': np.zeros(n_p),
        'phone_accel_y_mps2': np.zeros(n_p),
        'phone_accel_z_mps2': np.full(n_p, 9.81),
        'phone_gravity_x_mps2': np.zeros(n_p),
        'phone_gravity_y_mps2': np.zeros(n_p),
        'phone_gravity_z_mps2': np.full(n_p, 9.81),
        'phone_gps_lat_deg': np.full(n_p, 52.4),
        'phone_gps_lon_deg': np.full(n_p, -1.5),
        'phone_gps_alt_m': np.full(n_p, 100.0),
        'phone_gps_east_m': t_phone * speed_mps,
        'phone_gps_north_m': t_phone * 2.0,
        'phone_gps_up_m': np.zeros(n_p),
        'phone_gps_speed_mps': np.full(n_p, speed_mps),
        'phone_gps_accuracy_m': np.full(n_p, 5.0),
        'phone_gps_orientation_rad': np.zeros(n_p),
        'phone_gps_satellites_str': ['10'] * n_p,
        'phone_gps_satellites_used': np.full(n_p, 10),
        'phone_gps_satellites_in_view': np.full(n_p, 12),
        'phone_gps_is_valid': np.ones(n_p, dtype=bool),
        'phone_gps_is_new_fix': is_new
    })

    return df_ref, df_phone


def test_estimate_benchmark_lag_positive():
    """Verify estimate_benchmark_lag correctly detects known positive lag (+2.0 s)."""
    df_ref, df_phone = _create_synthetic_test_data(lag_s=2.0)
    res = estimate_benchmark_lag(df_ref, df_phone)
    assert res['is_valid'] is True
    assert abs(res['estimated_lag_s'] - 2.0) < 1e-3
    assert res['aligned_corr'] > 0.99
    assert abs(res['aligned_slope'] - 1.0) < 0.05
    assert res['segment_spread_s'] < 0.05


def test_estimate_benchmark_lag_negative():
    """Verify estimate_benchmark_lag correctly detects known negative lag (-2.0 s)."""
    df_ref, df_phone = _create_synthetic_test_data(lag_s=-2.0)
    res = estimate_benchmark_lag(df_ref, df_phone)
    assert res['is_valid'] is True
    assert abs(res['estimated_lag_s'] - (-2.0)) < 1e-3
    assert res['aligned_corr'] > 0.99
    assert abs(res['aligned_slope'] - 1.0) < 0.05
    assert res['segment_spread_s'] < 0.05


def test_estimate_benchmark_lag_zero():
    """Verify estimate_benchmark_lag correctly detects zero lag (0.0 s)."""
    df_ref, df_phone = _create_synthetic_test_data(lag_s=0.0)
    res = estimate_benchmark_lag(df_ref, df_phone)
    assert res['is_valid'] is True
    assert abs(res['estimated_lag_s'] - 0.0) < 1e-3
    assert res['aligned_corr'] > 0.99
    assert abs(res['aligned_slope'] - 1.0) < 0.05
    assert res['segment_spread_s'] < 0.05


def test_estimate_benchmark_lag_insufficient_excitation():
    """Verify estimate_benchmark_lag rejects scenarios with insufficient motion excitation."""
    # Stationary vehicle (speed = 0.0, yaw_rate = 0.0)
    df_ref, df_phone = _create_synthetic_test_data(lag_s=1.0, speed_mps=0.0, yaw_amp=0.0)
    res = estimate_benchmark_lag(df_ref, df_phone)
    assert res['is_valid'] is False
    assert res['estimated_lag_s'] == 0.0
    assert 'FAILED' in res['status']


def test_estimate_benchmark_lag_deterministic():
    """Verify lag estimation produces strictly identical results across multiple calls."""
    df_ref, df_phone = _create_synthetic_test_data(lag_s=1.5)
    res1 = estimate_benchmark_lag(df_ref, df_phone)
    res2 = estimate_benchmark_lag(df_ref, df_phone)
    assert res1['estimated_lag_s'] == res2['estimated_lag_s']
    assert res1['aligned_corr'] == res2['aligned_corr']
    assert res1['segment_spread_s'] == res2['segment_spread_s']


@pytest.mark.parametrize("target_lag", [0.1, -0.1, 0.2, -0.2])
def test_estimate_benchmark_lag_fine_resolutions(target_lag):
    """Verify estimator resolves fine sub-second lags (+/-0.1s, +/-0.2s) at 0.1s grid."""
    df_ref, df_phone = _create_synthetic_test_data(lag_s=target_lag)
    res = estimate_benchmark_lag(df_ref, df_phone)
    assert res['is_valid'] is True
    assert abs(res['estimated_lag_s'] - target_lag) < 1e-3
    assert res['aligned_corr'] > 0.99


def test_estimate_benchmark_lag_noisy_gyro():
    """Verify estimator recovers lag within 0.1s under additive sensor noise."""
    df_ref, df_phone = _create_synthetic_test_data(lag_s=1.0)
    # Add Gaussian noise with std = 0.02 rad/s (~10% of signal amplitude)
    np.random.seed(1234)
    df_phone['phone_gyro_y_radps'] += np.random.normal(0.0, 0.02, size=len(df_phone))
    res = estimate_benchmark_lag(df_ref, df_phone)
    assert res['is_valid'] is True
    assert abs(res['estimated_lag_s'] - 1.0) <= 0.1 + 1e-4
    assert res['aligned_corr'] > 0.90


@pytest.mark.parametrize("scale", [0.5, 1.5])
def test_estimate_benchmark_lag_amplitude_scaling(scale):
    """Verify Pearson correlation lag estimation is invariant to amplitude scaling."""
    df_ref, df_phone = _create_synthetic_test_data(lag_s=-0.5, yaw_amp=0.2 * scale)
    res = estimate_benchmark_lag(df_ref, df_phone)
    assert res['is_valid'] is True
    assert abs(res['estimated_lag_s'] - (-0.5)) < 1e-3
    assert res['aligned_corr'] > 0.99


def test_estimate_benchmark_lag_partial_excitation():
    """Verify estimator recovers lag when motion excitation occurs in only a fraction of the drive."""
    df_ref, df_phone = _create_synthetic_test_data(lag_s=0.8)
    # Zero out yaw rate in the last 70% of the drive (active excitation only in first 30%)
    n_cutoff = int(len(df_ref) * 0.3)
    df_ref.loc[n_cutoff:, 'ref_yaw_rate_radps'] = 0.0
    df_phone.loc[n_cutoff:, 'phone_gyro_y_radps'] = 0.0

    res = estimate_benchmark_lag(df_ref, df_phone)
    assert res['is_valid'] is True
    assert abs(res['estimated_lag_s'] - 0.8) < 1e-3
    assert res['aligned_corr'] > 0.95


def test_estimate_benchmark_lag_periodic_multimodal_ambiguity():
    """Verify periodic/harmonic maneuvers produce multiple candidate peaks across segments."""
    t_ref = np.arange(0.0, 100.05, 0.1)
    # Pure 0.5 Hz sinusoid has period T = 2.0s
    ref_yaw = 0.2 * np.sin(2.0 * np.pi * 0.5 * t_ref)
    df_ref = pd.DataFrame({
        'ref_time_s': t_ref,
        'ref_speed_mps': np.full_like(t_ref, 12.0),
        'ref_yaw_rate_radps': ref_yaw
    })
    t_p = np.arange(0.0, 100.05, 0.1)
    df_phone = pd.DataFrame({
        'phone_time_s': t_p,
        'phone_gyro_y_radps': 0.2 * np.sin(2.0 * np.pi * 0.5 * (t_p + 0.5))
    })
    res = estimate_benchmark_lag(df_ref, df_phone, max_lag_s=5.0)
    # The estimated lag will be congruent modulo 2.0s to 0.5s (e.g. 0.5, 2.5, -1.5, -3.5)
    mod_error = (res['estimated_lag_s'] - 0.5) % 2.0
    assert abs(mod_error) < 1e-3 or abs(mod_error - 2.0) < 1e-3
    assert res['aligned_corr'] > 0.99


def test_synchronize_recordings_auto_align_sparsity_and_monotonicity():
    """Verify synchronize_recordings preserves GNSS sparsity and timeline monotonicity."""
    df_ref, df_phone = _create_synthetic_test_data(lag_s=1.0, speed_mps=12.0)
    df_sync = synchronize_recordings(df_ref, df_phone, dt=0.1, auto_align_benchmark=True)

    # 1. Monotonicity
    t = df_sync['time_s'].values
    dt_arr = np.diff(t)
    assert np.all(dt_arr > 0), "Timestamps must be strictly monotonic"
    assert np.allclose(dt_arr, 0.1, atol=1e-4), "Sample step must be 0.1s"

    # 2. GNSS Sparsity
    n_new = df_sync['phone_gps_is_new_fix'].sum()
    n_sparse_obs = df_sync['phone_gps_obs_east_m'].notna().sum()
    assert n_new == n_sparse_obs, "phone_gps_obs_east_m must be non-NaN only when is_new_fix is True"
    assert df_sync['phone_gps_obs_speed_mps'].isna().sum() > 0, "GPS observations must be sparse"
    assert n_new < len(df_sync) / 5, "GPS arrivals must be ~1 Hz, not 10 Hz"

    # 3. Metadata preservation
    assert 'sync_metadata' in df_sync.attrs
    meta = df_sync.attrs['sync_metadata']
    assert meta['method'] == 'auto_benchmark_lag'
    assert abs(meta['estimated_lag_s'] - 1.0) < 1e-3


def test_synchronize_recordings_speed_preserved():
    """Verify phone_gps_speed_mps is preserved without artificial scaling."""
    df_ref, df_phone = _create_synthetic_test_data(lag_s=0.0, speed_mps=15.5)
    df_sync = synchronize_recordings(df_ref, df_phone, dt=0.1, auto_align_benchmark=False)
    # GPS speed in df_sync should match the unscaled speed (15.5 m/s)
    valid_speeds = df_sync['phone_gps_obs_speed_mps'].dropna().values
    assert len(valid_speeds) > 0
    assert np.allclose(valid_speeds, 15.5, atol=1e-3)
