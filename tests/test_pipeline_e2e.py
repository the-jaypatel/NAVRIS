import pytest
import os
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from navris.pipeline import process_single_recording
from navris.windowing import WindowConfig

def test_pipeline_single_sync_recording(tmp_path):
    manifest_path = 'data/manifest/recordings_manifest.csv'
    if not os.path.exists(manifest_path):
        pytest.skip('Manifest not found')

    df_manifest = pd.read_csv(manifest_path)
    entry = df_manifest[df_manifest['recording_id'] == 'S1'].iloc[0].to_dict()
    entry['split'] = 'train'

    out_dir = str(tmp_path / 'processed')
    config = WindowConfig(window_length=20, stride=10, dt=0.1)

    res = process_single_recording(
        rec_id='S1',
        manifest_entry=entry,
        output_dir=out_dir,
        window_config=config
    )

    assert res['status'] == 'SUCCESS'
    assert res['num_samples'] > 1000
    assert res['num_windows'] > 100
    assert os.path.exists(res['sync_parquet_path'])
    assert os.path.exists(res['windows_npz_path'])

    # Read back generated Parquet
    df_sync = pd.read_parquet(res['sync_parquet_path'])
    assert 'ref_east_m' in df_sync.columns
    assert 'phone_accel_x_mps2' in df_sync.columns
    assert 'phone_gps_is_new_fix' in df_sync.columns

    # Read back NPZ
    npz = np.load(res['windows_npz_path'])
    features = npz['features']
    disp_east = npz['disp_east_m']
    assert len(features) == len(disp_east)
    assert features.shape[1] == 20  # window length 20
