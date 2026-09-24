import pandas as pd
import numpy as np

# Test gyro permutation options
# Let's inspect the correlation and variance of gyro channels under permutations
df = pd.read_parquet('data/processed/synchronized/S1_sync.parquet')
gx = df['phone_gyro_x_radps'].values
gy = df['phone_gyro_y_radps'].values
gz = df['phone_gyro_z_radps'].values

# What if gyro_accel = [gx, -gz, gy] or [gz, gx, gy] or [-gz, gx, gy]?
# In accelerometer frame:
# az is Up (+Z)
# ax is right-ish, ay is forward-ish (or combination)
# If az is Up (+Z), then rotation around Up (+Z) is YAW.
# We know gy is Yaw rate (+0.93 correlation with VBOX yaw rate)!
# So omega_z = gy!
print("Raw gyro variances:")
print(f"  gx: {np.nanvar(gx):.6f}")
print(f"  gy: {np.nanvar(gy):.6f} (Yaw rate)")
print(f"  gz: {np.nanvar(gz):.6f}")
