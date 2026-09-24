import pandas as pd
import numpy as np

df = pd.read_parquet("data/processed/synchronized/S1_sync.parquet")

print("--- DATA SUMMARY FOR S1_sync.parquet ---")
print(f"Total rows: {len(df)}")
t0, t1 = df["time_s"].iloc[0], df["time_s"].iloc[-1]
print(f"Duration: {t1 - t0:.2f} s ({t0:.2f} to {t1:.2f})")

dt = np.diff(df["time_s"].values)
print(f"dt: min={dt.min():.4f}, max={dt.max():.4f}, mean={dt.mean():.4f}, std={dt.std():.6f}")

# Check new fix flag
is_new = df["phone_gps_is_new_fix"].fillna(False).astype(bool)
print(f"Total GNSS rows: {len(df)}")
print(f"New GNSS fixes: {is_new.sum()}")
print(f"Sample-and-hold duplicates: {len(df) - is_new.sum()}")

# Check GNSS update intervals
gnss_times = df.loc[is_new, "time_s"].values
gnss_dt = np.diff(gnss_times)
print(f"GNSS update intervals (s): min={gnss_dt.min():.2f}, max={gnss_dt.max():.2f}, median={np.median(gnss_dt):.2f}, mean={gnss_dt.mean():.2f}")

# Check sensor statistics
print("--- SENSOR STATISTICS ---")
cols = [
    "phone_accel_x_mps2", "phone_accel_y_mps2", "phone_accel_z_mps2",
    "phone_gravity_x_mps2", "phone_gravity_y_mps2", "phone_gravity_z_mps2",
    "phone_gyro_x_radps", "phone_gyro_y_radps", "phone_gyro_z_radps",
    "phone_gps_east_m", "phone_gps_north_m", "phone_gps_up_m",
    "phone_gps_speed_mps", "phone_gps_accuracy_m",
    "phone_gps_satellites_used", "phone_gps_satellites_in_view",
    "ref_east_m", "ref_north_m", "ref_up_m", "ref_speed_mps"
]

for col in cols:
    if col in df.columns:
        s = df[col].dropna()
        print(f"{col:<28}: min={s.min():9.3f}, max={s.max():9.3f}, mean={s.mean():9.3f}, std={s.std():9.3f}, NaNs={df[col].isna().sum()}")
