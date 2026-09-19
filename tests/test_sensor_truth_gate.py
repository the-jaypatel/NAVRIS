"""
Unit tests for NAVRIS Phase 2.1: Sensor Truth Gate.
Tests timestamp monotonicity, causal timer-reset unwrapping, DATE consistency,
sparse GNSS new-fix gating, accelerometer specific-force semantics,
gravity semantics, and Driver F compatibility.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from navris.ingest import (
    causal_unwrap_timestamps,
    ingest_smartphone_data,
    ingest_reference_data
)
from navris.schema import PHONE_SCHEMA_COLUMNS


def test_causal_unwrap_synthetic_reset():
    """Test causal timer-reset unwrapping on synthetic data without DATE."""
    # 100 samples at 100ms: 0 to 4900 ms, then timer resets to 10 ms at index 50
    t_part1 = np.arange(50) * 100.0
    t_part2 = np.arange(50) * 100.0 + 10.0
    raw_ms = np.concatenate([t_part1, t_part2])
    
    corr_s, resets = causal_unwrap_timestamps(raw_ms, date_series=None, default_dt_ms=100.0)
    
    assert len(resets) == 1
    assert resets[0]['index'] == 50
    assert resets[0]['prev_raw_ms'] == 4900.0
    assert resets[0]['cur_raw_ms'] == 10.0
    
    # Must be strictly monotonic
    diffs = np.diff(corr_s)
    assert np.all(diffs > 0), "Corrected timestamps must be strictly monotonic"
    # At reset, step should be default 100ms = 0.1s
    assert np.isclose(corr_s[50] - corr_s[49], 0.1, atol=1e-4)


def test_causal_unwrap_with_date():
    """Test causal unwrapping using wall-clock DATE series with known elapsed time."""
    # Synthetic 5-sample sequence with 2.5s gap at index 3 (reset diff < -1000ms)
    raw_ms = np.array([0.0, 1000.0, 2000.0, 10.0, 110.0])
    date_strs = pd.Series([
        '2019-09-08 12:00:00:000',
        '2019-09-08 12:00:01:000',
        '2019-09-08 12:00:02:000',
        '2019-09-08 12:00:04:500',  # 2.5s later in wall-clock time
        '2019-09-08 12:00:04:600',
    ])
    
    corr_s, resets = causal_unwrap_timestamps(raw_ms, date_series=date_strs)
    
    assert len(resets) == 1
    assert resets[0]['index'] == 3
    assert np.isclose(resets[0]['step_ms'], 2500.0)
    assert np.all(np.diff(corr_s) > 0)
    assert np.isclose(corr_s[3] - corr_s[2], 2.5, atol=1e-3)


def test_real_recording_s2_timer_reset():
    """Test real IO-VNBD recording S2 known timer reset at sample 1863."""
    manifest = pd.read_csv('data/manifest/recordings_manifest.csv')
    sub = manifest[manifest['recording_id'] == 'S2']
    if len(sub) == 0:
        pytest.skip("S2 not in manifest")
    
    s_path = os.path.join('data/raw/IO-VNBD', sub.iloc[0]['s_file_path'])
    if not os.path.exists(s_path):
        pytest.skip(f"S2 raw file not found at {s_path}")
        
    df_phone, _ = ingest_smartphone_data(s_path)
    
    # Raw timestamp must be preserved
    assert 'phone_time_raw_ms' in df_phone.columns
    raw_diffs = np.diff(df_phone['phone_time_raw_ms'].values)
    assert np.any(raw_diffs < -1000.0), "Raw data must contain the original negative timer jump"
    
    # Corrected timestamp must be strictly monotonic
    corr_diffs = np.diff(df_phone['phone_time_s'].values)
    assert np.all(corr_diffs > 0), "Corrected phone_time_s must be strictly monotonic"
    
    # Total span must match true recording span (~9388s, not truncated 186s or 9201s)
    total_span = df_phone['phone_time_s'].iloc[-1]
    assert 9380.0 < total_span < 9395.0, f"Unexpected total span: {total_span}"


def test_real_recording_y1_multiple_resets():
    """Test real IO-VNBD recording Y1 with 3 timer resets and app pause gaps."""
    manifest = pd.read_csv('data/manifest/recordings_manifest.csv')
    sub = manifest[manifest['recording_id'] == 'Y1']
    if len(sub) == 0:
        pytest.skip("Y1 not in manifest")
    
    s_path = os.path.join('data/raw/IO-VNBD', sub.iloc[0]['s_file_path'])
    if not os.path.exists(s_path):
        pytest.skip(f"Y1 raw file not found at {s_path}")
        
    df_phone, _ = ingest_smartphone_data(s_path)
    corr_diffs = np.diff(df_phone['phone_time_s'].values)
    assert np.all(corr_diffs > 0), "Y1 corrected timeline must be strictly monotonic throughout"
    assert df_phone['phone_time_s'].iloc[-1] > 7000.0


def test_smartphone_gnss_new_fix_detection():
    """Verify that new fixes are strictly isolated and sample-and-hold epochs are flagged False."""
    manifest = pd.read_csv('data/manifest/recordings_manifest.csv')
    sub = manifest[manifest['recording_id'] == 'S1']
    if len(sub) == 0:
        pytest.skip("S1 not in manifest")
    
    s_path = os.path.join('data/raw/IO-VNBD', sub.iloc[0]['s_file_path'])
    if not os.path.exists(s_path):
        pytest.skip(f"S1 raw file not found at {s_path}")
        
    df_phone, _ = ingest_smartphone_data(s_path)
    
    n_samples = len(df_phone)
    new_fixes = df_phone['phone_gps_is_new_fix'].sum()
    
    # In S1: 51,746 samples, exactly 532 new fixes (~1 fix every 9.7s)
    assert new_fixes < n_samples * 0.05, f"Expected sparse fixes (<5%), got {new_fixes}/{n_samples}"
    assert new_fixes > 500, f"Expected ~532 new fixes, got {new_fixes}"
    
    # Verify that duplicate coordinates result in phone_gps_is_new_fix == False
    is_new = df_phone['phone_gps_is_new_fix'].values
    lats = df_phone['phone_gps_lat_deg'].values
    lons = df_phone['phone_gps_lon_deg'].values
    
    for i in range(1, 200):
        if lats[i] == lats[i-1] and lons[i] == lons[i-1]:
            assert not is_new[i], f"Sample-and-hold epoch {i} was falsely marked as new fix"


def test_accelerometer_specific_force_semantics():
    """Verify that accelerometer measures specific force (magnitude ~ 9.81 m/s^2 when stationary)."""
    manifest = pd.read_csv('data/manifest/recordings_manifest.csv')
    sub = manifest[manifest['recording_id'] == 'S1']
    if len(sub) == 0:
        pytest.skip("S1 not in manifest")
    
    s_path = os.path.join('data/raw/IO-VNBD', sub.iloc[0]['s_file_path'])
    if not os.path.exists(s_path):
        pytest.skip(f"S1 raw file not found at {s_path}")
        
    df_phone, _ = ingest_smartphone_data(s_path)
    
    # S1 begins stationary (first 100 samples)
    stat = df_phone.iloc[:100]
    ax = stat['phone_accel_x_mps2'].values
    ay = stat['phone_accel_y_mps2'].values
    az = stat['phone_accel_z_mps2'].values
    
    norm = np.sqrt(ax**2 + ay**2 + az**2)
    mean_norm = float(np.mean(norm))
    
    # Specific force at rest on Earth should be ~9.81 m/s^2 (+/- 2% due to sensor calibration/vibration)
    assert 9.6 < mean_norm < 10.1, f"Expected specific force magnitude ~9.81 m/s^2, got {mean_norm}"
    # In S1, phone is mounted flat (screen up), so z contains almost the entire gravity reaction
    assert float(np.mean(az)) > 9.5, f"Expected az to carry gravity reaction force, got {np.mean(az)}"


def test_gravity_semantics():
    """Verify that Android gravity sensor has unit norm ~ 9.807 m/s^2."""
    manifest = pd.read_csv('data/manifest/recordings_manifest.csv')
    sub = manifest[manifest['recording_id'] == 'S1']
    if len(sub) == 0:
        pytest.skip("S1 not in manifest")
    
    s_path = os.path.join('data/raw/IO-VNBD', sub.iloc[0]['s_file_path'])
    if not os.path.exists(s_path):
        pytest.skip(f"S1 raw file not found at {s_path}")
        
    df_phone, _ = ingest_smartphone_data(s_path)
    
    gx = df_phone['phone_gravity_x_mps2'].iloc[:100].values
    gy = df_phone['phone_gravity_y_mps2'].iloc[:100].values
    gz = df_phone['phone_gravity_z_mps2'].iloc[:100].values
    
    g_norm = np.sqrt(gx**2 + gy**2 + gz**2)
    mean_g_norm = float(np.mean(g_norm))
    
    # Android gravity sensor norm should exactly match standard gravity ~9.807 m/s^2
    assert np.isclose(mean_g_norm, 9.80665, atol=0.05), f"Expected Android gravity norm ~9.807, got {mean_g_norm}"


def test_driver_f_missing_fields_handling():
    """Verify Driver F files (no magnetometer, no orientation) ingest cleanly without error."""
    manifest = pd.read_csv('data/manifest/recordings_manifest.csv')
    sub = manifest[manifest['recording_id'] == 'T1']
    if len(sub) == 0:
        pytest.skip("T1 not in manifest")
        
    s_path = os.path.join('data/raw/IO-VNBD', sub.iloc[0]['s_file_path'])
    if not os.path.exists(s_path):
        pytest.skip(f"T1 raw file not found at {s_path}")
        
    df_phone, _ = ingest_smartphone_data(s_path)
    
    assert not df_phone['phone_has_mag'].iloc[0]
    assert not df_phone['phone_has_orientation'].iloc[0]
    assert df_phone['phone_mag_x_uT'].isna().all()
    assert df_phone['phone_orient_azimuth_rad'].isna().all()
    
    # IMU and GPS must be populated
    assert not df_phone['phone_accel_x_mps2'].isna().all()
    assert not df_phone['phone_gyro_x_radps'].isna().all()
    assert not df_phone['phone_gps_lat_deg'].isna().all()
