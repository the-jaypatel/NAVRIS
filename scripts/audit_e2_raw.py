import pandas as pd
import numpy as np
import re

# 1. Load synchronized data
df = pd.read_parquet('data/processed/synchronized/S1_sync.parquet')

# 2. Load raw data for first 1000 rows to see raw column values
df_raw = pd.read_csv('data/raw/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S1/S-S1.csv', encoding='latin1', nrows=1000)
df_raw.columns = [c.strip() for c in df_raw.columns]

print("=" * 80)
print("E2-A: RAW DATA AUDIT - COLUMN HEADERS AND VALUES")
print("=" * 80)
for col in df_raw.columns:
    sample_val = df_raw[col].iloc[0]
    dtype = df_raw[col].dtype
    print(f"  {col:<35} | Sample: {sample_val} | Dtype: {dtype}")

print("\n" + "=" * 80)
print("E2-A: GPS SPEED AUDIT (RAW vs VBOX vs PROCESSED)")
print("=" * 80)
# Check multiple rows across S1
indices = [0, 90, 529, 620, 1560, 1650, 1740, 5000, 10000, 20000]
print(f"{'Idx':<8} {'Time(s)':<8} {'Raw GPS Speed':<15} {'Proc GPS Speed':<16} {'VBOX Speed m/s':<16} {'VBOX Speed km/h':<16} {'Ratio Raw/VBOX':<16}")
for idx in indices:
    if idx < len(df):
        t = df['time_s'].iloc[idx]
        proc_spd = df['phone_gps_speed_mps'].iloc[idx]
        vbox_spd = df['ref_speed_mps'].iloc[idx]
        raw_spd = df_raw['GPS SPEED (Kmh)'].iloc[idx] if idx < len(df_raw) else proc_spd * 3.6
        vbox_kmh = vbox_spd * 3.6
        ratio = raw_spd / vbox_spd if vbox_spd > 0.5 else np.nan
        print(f"{idx:<8} {t:<8.1f} {raw_spd:<15.3f} {proc_spd:<16.3f} {vbox_spd:<16.3f} {vbox_kmh:<16.3f} {ratio:<16.3f}")

print("\n" + "=" * 80)
print("E2-B: GYROSCOPE TO VBOX YAW RATE CORRELATIONS (FULL RUN)")
print("=" * 80)
vbox_yaw_rate = df['ref_yaw_rate_radps'].values
gx = df['phone_gyro_x_radps'].values
gy = df['phone_gyro_y_radps'].values
gz = df['phone_gyro_z_radps'].values

moving = (df['ref_speed_mps'] > 2.0) & (~np.isnan(vbox_yaw_rate))
print(f"Moving samples (speed > 2 m/s): {moving.sum()}/{len(df)}")

for name, g in [("phone_gyro_x (Raw: GYROSCOPE Yaw)", gx),
                ("phone_gyro_y (Raw: GYROSCOPE Pitch)", gy),
                ("phone_gyro_z (Raw: GYROSCOPE Roll)", gz)]:
    corr = np.corrcoef(g[moving], vbox_yaw_rate[moving])[0, 1]
    # Linear regression slope: g = slope * vbox_yaw_rate
    slope = np.cov(g[moving], vbox_yaw_rate[moving])[0, 1] / np.var(vbox_yaw_rate[moving])
    print(f"  {name:<42}: Corr = {corr:+.4f}, Slope = {slope:+.4f}")

print("\n" + "=" * 80)
print("E2-B: ACCELEROMETER CORRELATIONS WITH VBOX (FULL RUN)")
print("=" * 80)
ax = df['phone_accel_x_mps2'].values
ay = df['phone_accel_y_mps2'].values
az = df['phone_accel_z_mps2'].values

# VBOX longitudinal and lateral accelerations (converted to m/s^2)
vbox_ax = df['ref_accel_lat_mps2'].values   # Lateral
vbox_ay = df['ref_accel_long_mps2'].values  # Longitudinal

for a_name, a_val in [("phone_accel_x", ax), ("phone_accel_y", ay), ("phone_accel_z", az)]:
    mean_val = np.nanmean(a_val)
    std_val = np.nanstd(a_val)
    corr_lat = np.corrcoef(a_val[moving], vbox_ax[moving])[0, 1]
    corr_long = np.corrcoef(a_val[moving], vbox_ay[moving])[0, 1]
    print(f"  {a_name:<16}: Mean={mean_val:+.3f}, Std={std_val:.3f} | Corr with VBOX Lat={corr_lat:+.4f}, Corr with VBOX Long={corr_long:+.4f}")

print("\n" + "=" * 80)
print("E2-C: SEGMENT GENERALIZATION WITHIN S1")
print("=" * 80)
segments = [
    ("0-60s", 0.0, 60.0),
    ("140-200s", 140.0, 200.0),
    ("500-560s", 500.0, 560.0),
    ("1000-1100s", 1000.0, 1100.0),
    ("2000-2100s", 2000.0, 2100.0),
    ("4000-4100s", 4000.0, 4100.0)
]

for seg_name, t_start, t_end in segments:
    mask = (df['time_s'] >= t_start) & (df['time_s'] <= t_end) & moving
    if mask.sum() > 20:
        c_gx = np.corrcoef(gx[mask], vbox_yaw_rate[mask])[0, 1]
        c_gy = np.corrcoef(gy[mask], vbox_yaw_rate[mask])[0, 1]
        c_gz = np.corrcoef(gz[mask], vbox_yaw_rate[mask])[0, 1]
        slope_gy = np.cov(gy[mask], vbox_yaw_rate[mask])[0, 1] / np.var(vbox_yaw_rate[mask])
        c_ax_lat = np.corrcoef(ax[mask], vbox_ax[mask])[0, 1]
        c_ay_long = np.corrcoef(ay[mask], vbox_ay[mask])[0, 1]
        print(f"Segment {seg_name:<12} (N={mask.sum():<4}): Gyro_Y vs YawRate Corr={c_gy:+.3f} (slope={slope_gy:+.2f}) | Acc_X vs Lat={c_ax_lat:+.3f}, Acc_Y vs Long={c_ay_long:+.3f}")
    else:
        print(f"Segment {seg_name:<12}: Insufficient dynamic samples")
