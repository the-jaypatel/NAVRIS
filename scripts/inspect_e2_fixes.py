import json
import pandas as pd
import numpy as np

with open('data/processed/phase2_3b/experiments/E2/E2_diagnostics.json', 'r') as f:
    diag = json.load(f)

print(f"{'Fix':<5} {'Time':<8} {'Acc':<6} {'NIS':<8} {'ResNorm':<10} {'HErr':<10} {'PitchPost':<10}")
for d in diag:
    print(f"{d['fix_idx']:<5} {d['time_s']:<8.1f} {str(d['accepted']):<6} {d['nis']:<8.2f} {d['residual_norm']:<10.2f} {d['horiz_err_ref']:<10.2f} {d['pitch_post_deg']:<10.2f}")

# Check when the first rejected fix occurs in E2
df_res = pd.read_parquet('data/processed/phase2_3b/experiments/E2/E2_results.parquet')
novel = df_res[df_res['phone_gps_is_new_fix']].copy().reset_index()
print("\nFirst 12 novel fixes in E2 results:")
cols = ['index', 'time_s', 'gnss_accepted', 'gnss_nis', 'horiz_pos_err_m', 'vel_err_mps', 'eskf_pitch_deg', 'pos_std_3d_m']
print(novel[cols].head(12).to_string())

print(f"\nTotal novel fixes: {len(novel)}")
print(f"Accepted novel fixes: {novel['gnss_accepted'].sum()}/{len(novel)}")
print(f"Indices of accepted novel fixes:")
print(novel[novel['gnss_accepted']][['index', 'time_s', 'gnss_nis', 'horiz_pos_err_m']].to_string())
