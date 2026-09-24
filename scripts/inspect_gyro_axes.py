import pandas as pd
import numpy as np

# Load raw S-S1.csv
df = pd.read_parquet('data/processed/synchronized/S1_sync.parquet')

# Let's inspect the gyro and accel during turning and braking
# When turning:
# VBOX yaw rate > 0.1 rad/s
turn = df[df['ref_yaw_rate_radps'] > 0.1]
print("During Left Turns (VBOX yaw rate > 0.1 rad/s):")
print("  VBOX yaw rate mean:     ", turn['ref_yaw_rate_radps'].mean())
print("  phone_gyro_x (raw Yaw):  ", turn['phone_gyro_x_radps'].mean())
print("  phone_gyro_y (raw Pitch):", turn['phone_gyro_y_radps'].mean())
print("  phone_gyro_z (raw Roll): ", turn['phone_gyro_z_radps'].mean())

# Now what about vehicle pitch?
# When vehicle brakes (negative longitudinal acceleration, ref_accel_long < -1.5 m/s^2)
# The front dips down (negative pitch). Pitch rate d(pitch)/dt should have a pulse!
# Let's check gyro during braking transients:
brake = df[(df['ref_accel_long_mps2'] < -1.5) & (np.abs(df['ref_yaw_rate_radps']) < 0.02)]
print("\nDuring Straight Braking (ref_accel_long < -1.5 m/s^2, yaw_rate ~ 0):")
print("  VBOX long accel mean:   ", brake['ref_accel_long_mps2'].mean())
print("  phone_gyro_x (raw Yaw):  ", brake['phone_gyro_x_radps'].mean())
print("  phone_gyro_y (raw Pitch):", brake['phone_gyro_y_radps'].mean())
print("  phone_gyro_z (raw Roll): ", brake['phone_gyro_z_radps'].mean())
print("  phone_accel_x:          ", brake['phone_accel_x_mps2'].mean())
print("  phone_accel_y:          ", brake['phone_accel_y_mps2'].mean())
print("  phone_accel_z:          ", brake['phone_accel_z_mps2'].mean())
