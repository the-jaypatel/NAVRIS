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
