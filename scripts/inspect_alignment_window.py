import sys
import os
sys.path.insert(0, os.path.abspath('src'))
import pandas as pd
import numpy as np
from navris.inertial.gnss_fixes import filter_novel_gnss_fixes

df = pd.read_parquet('data/processed/synchronized/S1_sync.parquet')
history = df.iloc[:3500]
novel = filter_novel_gnss_fixes(history)
moving = novel[novel['phone_gps_speed_mps'] > 1.5]

for i in range(len(moving) - 1):
    sub = moving.iloc[i:min(i + 3, len(moving))]
    de = sub['phone_gps_east_m'].iloc[-1] - sub['phone_gps_east_m'].iloc[0]
    dn = sub['phone_gps_north_m'].iloc[-1] - sub['phone_gps_north_m'].iloc[0]
    disp = np.sqrt(de**2 + dn**2)
    headings_raw = sub['phone_gps_orientation_rad'].values
    valid_head = headings_raw[~np.isnan(headings_raw)]
    sin_sum = np.sum(np.sin(valid_head))
    cos_sum = np.sum(np.cos(valid_head))
    circ_std = np.sqrt(max(0.0, 1.0 - np.sqrt(sin_sum**2 + cos_sum**2) / len(valid_head))) if len(valid_head) >= 2 else 0.0
    passes = (disp >= 100.0) and (circ_std <= 0.2)
    t0 = sub['time_s'].iloc[0]
    t1 = sub['time_s'].iloc[-1]
    print(f"Window {i}: t=[{t0:.1f} to {t1:.1f}], disp={disp:.1f}m, circ_std={circ_std:.4f}, pass={passes}")
    if passes:
        print("  --> SELECTED WINDOW:")
        print(sub[['time_s', 'phone_gps_east_m', 'phone_gps_north_m', 'phone_gps_speed_mps', 'phone_gps_orientation_rad']])
        dt = t1 - t0
        print(f"  Displacement over window ({dt:.1f}s): de={de:.2f}m, dn={dn:.2f}m -> v=[{de/dt:.3f}, {dn/dt:.3f}], speed={disp/dt:.3f} m/s")
        dt_last = sub['time_s'].iloc[-1] - sub['time_s'].iloc[-2]
        de_last = sub['phone_gps_east_m'].iloc[-1] - sub['phone_gps_east_m'].iloc[-2]
        dn_last = sub['phone_gps_north_m'].iloc[-1] - sub['phone_gps_north_m'].iloc[-2]
        disp_last = np.sqrt(de_last**2 + dn_last**2)
        print(f"  Displacement over last 2 fixes ({dt_last:.1f}s): de={de_last:.2f}m, dn={dn_last:.2f}m -> v=[{de_last/dt_last:.3f}, {dn_last/dt_last:.3f}], speed={disp_last/dt_last:.3f} m/s")
        break

