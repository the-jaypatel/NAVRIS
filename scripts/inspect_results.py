import pandas as pd
import numpy as np

res = pd.read_parquet('data/processed/phase2_3b/S1_eskf_results.parquet')
metrics = pd.read_csv('data/processed/phase2_3b/S1_eskf_metrics.csv')

print('--- SUMMARY METRICS ---')
print(metrics.T.to_string())

print('\n--- ERROR STATS ---')
h_err = res['horiz_pos_err_m']
v_err = res['vel_err_mps']
print(f'Horizontal Error: Mean={h_err.mean():.2f}m, Median={h_err.median():.2f}m, Max={h_err.max():.2f}m, Final={h_err.iloc[-1]:.2f}m')
print(f'Velocity Error: Mean={v_err.mean():.2f}m/s, Median={v_err.median():.2f}m/s, Max={v_err.max():.2f}m/s, Final={v_err.iloc[-1]:.2f}m/s')
moving = res[res['ref_speed_mps'] > 2.0]
print(f"Heading Error (moving): Mean={moving['heading_err_deg'].mean():.2f} deg, Median={moving['heading_err_deg'].median():.2f} deg")

print('\n--- GNSS GATING SUMMARY ---')
print(f'Total epochs: {len(res)}')
print(f"Duplicate sample-and-hold: {(res['gnss_reject_reason'] == 'duplicate_sample_and_hold').sum()}")
print(f"Novel fixes: {res['phone_gps_is_new_fix'].sum()}")
print(f"Accepted novel fixes: {res['gnss_accepted'].sum()}")
print(f"Rejected novel fixes: {(res['phone_gps_is_new_fix'] & (~res['gnss_accepted'])).sum()}")

print('\n--- ACCEPTED FIXES DETAILS ---')
acc = res[res['gnss_accepted']]
for idx, r in acc.iterrows():
    print(f"t={r['time_s']:.1f}s (idx {idx}): NIS={r['gnss_nis']:.2f}, horiz_err={r['horiz_pos_err_m']:.2f}m, 3d_err={r['pos_err_3d_m']:.2f}m, 3sigma={3*r['pos_std_3d_m']:.2f}m, roll={r['eskf_roll_deg']:.2f}, pitch={r['eskf_pitch_deg']:.2f}, yaw={r['eskf_yaw_deg']:.2f}")

print('\n--- REJECTED FIXES SAMPLE ---')
rej = res[res['phone_gps_is_new_fix'] & (~res['gnss_accepted'])]
print(rej[['time_s', 'gnss_nis', 'horiz_pos_err_m', 'pos_std_3d_m', 'eskf_roll_deg', 'eskf_pitch_deg']].head(10).to_string())

print('\n--- FIRST 5 FIXES DETAILS ---')
novel_df = res[res['phone_gps_is_new_fix']].copy().reset_index()
for i in range(min(5, len(novel_df))):
    r = novel_df.iloc[i]
    print(f"Fix {i} at t={r['time_s']:.1f}s (step {r['index']}):")
    print(f"  Accepted: {r['gnss_accepted']}, Reason: {r['gnss_reject_reason']}")
    print(f"  NIS: {r['gnss_nis']:.2f}")
    print(f"  Ref Pos: [{r['ref_east_m']:.1f}, {r['ref_north_m']:.1f}, {r['ref_up_m']:.1f}]")
    print(f"  ESKF Pos: [{r['eskf_east_m']:.1f}, {r['eskf_north_m']:.1f}, {r['eskf_up_m']:.1f}]")
    print(f"  ESKF Vel: [{r['eskf_vel_east_mps']:.1f}, {r['eskf_vel_north_mps']:.1f}, {r['eskf_vel_up_mps']:.1f}]")
    print(f"  Ref Speed: {r['ref_speed_mps']:.2f} m/s, ESKF Speed: {r['eskf_speed_mps']:.2f} m/s")
    print(f"  Euler [R, P, Y]: [{r['eskf_roll_deg']:.2f}, {r['eskf_pitch_deg']:.2f}, {r['eskf_yaw_deg']:.2f}] deg")
    print(f"  Pos 3D Std: {r['pos_std_3d_m']:.2f} m")

