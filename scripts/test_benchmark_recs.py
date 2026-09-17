import sys, os
sys.path.insert(0, 'src')
import pandas as pd
from navris.pipeline import process_single_recording
from navris.windowing import WindowConfig

manifest = pd.read_csv('data/manifest/recordings_manifest.csv')
sample_ids = ['S1', 'S2', 'S3A', 'S4', 'M', 'Y1', 'VTA1A', 'VTA2', 'T1', 'T10', 'I', 'A1', 'ST1']

print('Testing diverse benchmark recordings:')
results = []
for rid in sample_ids:
    sub = manifest[manifest['recording_id'] == rid]
    if len(sub) == 0:
        continue
    entry = sub.iloc[0].to_dict()
    entry['split'] = 'train'
    res = process_single_recording(rid, entry)
    print(f"[{res['status']}] {rid:6s} | Sync={res['is_synchronized']} | Samples={res['num_samples']:6d} | Windows={res['num_windows']:6d} | Duration={res['duration_s']:7.1f}s | Fixes={res['num_gps_fixes']:4d}")
    results.append(res)
