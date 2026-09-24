import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from navris.ingest import (
    parse_satellite_string,
    ingest_smartphone_data,
    ingest_reference_data
)

def test_parse_satellite_string_standard():
    s, used, in_view = parse_satellite_string('27 / 28')
    assert s == '27 / 28'
    assert used == 27.0
    assert in_view == 28.0

def test_parse_satellite_string_excel_date_corruption():
    # '10-Nov' corresponds to 10 used, 11 in view
    s, used, in_view = parse_satellite_string('10-Nov')
    assert used == 10.0
    assert in_view == 11.0

    # 'Aug-20' corresponds to 8 used, 20 in view
    s, used, in_view = parse_satellite_string('Aug-20')
    assert used == 8.0
    assert in_view == 20.0

def test_parse_satellite_string_empty():
    s, used, in_view = parse_satellite_string('')
    assert np.isnan(used)
    assert np.isnan(in_view)

def test_ingest_reference_v_s1():
    ref_path = 'data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/V-Dataset/V-S1.csv'
    if not os.path.exists(ref_path):
        pytest.skip('Dataset not found locally')

    df_ref, origin = ingest_reference_data(ref_path)
    assert len(df_ref) > 1000
    assert 'ref_speed_mps' in df_ref.columns
    assert 'ref_heading_rad' in df_ref.columns
    assert 'ref_east_m' in df_ref.columns
    # Check origin is (lat0, lon0, alt0)
    assert np.isclose(origin[0], 52.4017, atol=0.01)
    assert np.isclose(df_ref['ref_east_m'].iloc[0], 0.0, atol=1e-5)
    # Check speed is reasonable (between 0 and 50 m/s)
    assert df_ref['ref_speed_mps'].min() >= 0.0
    assert df_ref['ref_speed_mps'].max() < 50.0

def test_ingest_smartphone_s_s1():
    phone_path = 'data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset/S-S1.csv'
    if not os.path.exists(phone_path):
        pytest.skip('Dataset not found locally')

    df_phone, _ = ingest_smartphone_data(phone_path)
    assert len(df_phone) > 1000
    assert df_phone['phone_has_mag'].iloc[0] == True
    assert df_phone['phone_has_orientation'].iloc[0] == True
    # Verify sample-and-hold GPS detection
    new_fixes = df_phone['phone_gps_is_new_fix'].sum()
    assert 0 < new_fixes < len(df_phone)  # Sparse, not 10 Hz!

def test_driver_f_compatibility():
    driver_f_path = 'data/raw/IO-VNBD/Unsynchronised V and S Dataset/Uncategorised IOVNB (V and S) Dataset/S-Dataset/S-T1.csv'
    if not os.path.exists(driver_f_path):
        pytest.skip('Dataset not found locally')

    df_f, _ = ingest_smartphone_data(driver_f_path)
    assert len(df_f) > 100
    assert df_f['phone_has_mag'].iloc[0] == False
    assert df_f['phone_has_orientation'].iloc[0] == False
    assert df_f['phone_mag_x_uT'].isna().all()
    assert df_f['phone_orient_azimuth_rad'].isna().all()
    # Accelerometer and Gyroscope must still be present and valid
    assert not df_f['phone_accel_x_mps2'].isna().all()
    assert not df_f['phone_gyro_x_radps'].isna().all()


def test_phone_gps_speed_unscaled_mps():
    """
    Gate 1.5A Regression: Verifies that raw 'GPS SPEED (Kmh)' values are ingested
    directly as m/s without applying an erroneous /3.6 conversion.
    """
    phone_path = 'data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset/S-S1.csv'
    if not os.path.exists(phone_path):
        pytest.skip('Dataset not found locally')

    df_phone, _ = ingest_smartphone_data(phone_path)
    # At t = 156.0s (row ~1560), raw CSV value is 11.94, reference speed is ~12.03 m/s
    # Ingested speed must be ~11.94 m/s (not 3.32 m/s!)
    spd_156 = df_phone.loc[(df_phone['phone_time_s'] >= 155.9) & (df_phone['phone_time_s'] <= 156.1), 'phone_gps_speed_mps'].dropna().iloc[0]
    assert np.isclose(spd_156, 11.94, atol=0.1)
    assert spd_156 > 10.0  # Must not be compressed to ~3.3 m/s!
