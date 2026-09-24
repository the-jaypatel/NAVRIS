import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/synchronized/S1_sync.parquet')

# Let's inspect variances and ranges of the 3 gyro channels:
g_raw_yaw = df['phone_gyro_x_radps'].values    # raw col 16: GYROSCOPE Yaw
g_raw_pitch = df['phone_gyro_y_radps'].values  # raw col 17: GYROSCOPE Pitch
g_raw_roll = df['phone_gyro_z_radps'].values   # raw col 18: GYROSCOPE Roll

print("Standard Deviations across S1 (rad/s):")
print(f"  raw Yaw   (phone_gyro_x): {np.nanstd(g_raw_yaw):.4f} rad/s")
print(f"  raw Pitch (phone_gyro_y): {np.nanstd(g_raw_pitch):.4f} rad/s")
print(f"  raw Roll  (phone_gyro_z): {np.nanstd(g_raw_roll):.4f} rad/s")

vbox_yaw_std = np.nanstd(df['ref_yaw_rate_radps'].values)
print(f"  VBOX Yaw Rate std:        {vbox_yaw_std:.4f} rad/s")

print("\nNotice:")
print(f"  raw Pitch std ({np.nanstd(g_raw_pitch):.4f}) matches VBOX Yaw Rate std ({vbox_yaw_std:.4f}) almost exactly!")
print(f"  raw Yaw std   ({np.nanstd(g_raw_yaw):.4f}) is tiny (noise level)!")
print(f"  raw Roll std  ({np.nanstd(g_raw_roll):.4f}) is small!")
