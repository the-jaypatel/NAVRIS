import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/synchronized/S1_sync.parquet')

# Filter for straight driving with positive acceleration (engine accelerating forward)
accel = df[(df['ref_accel_long_mps2'] > 1.0) & (np.abs(df['ref_yaw_rate_radps']) < 0.02)]
# Filter for straight driving with braking (decelerating)
brake = df[(df['ref_accel_long_mps2'] < -1.0) & (np.abs(df['ref_yaw_rate_radps']) < 0.02)]

print("Forward Acceleration:")
print("  ref_accel_long mean:  ", accel['ref_accel_long_mps2'].mean())
print("  phone_accel_x mean:   ", accel['phone_accel_x_mps2'].mean())
print("  phone_accel_y mean:   ", accel['phone_accel_y_mps2'].mean())

print("\nBraking Deceleration:")
print("  ref_accel_long mean:  ", brake['ref_accel_long_mps2'].mean())
print("  phone_accel_x mean:   ", brake['phone_accel_x_mps2'].mean())
print("  phone_accel_y mean:   ", brake['phone_accel_y_mps2'].mean())

# Notice: Specific force f = a - g.
# When vehicle accelerates forward with acceleration +a_long:
# Accelerometer in forward direction measures +a_long (inertial reaction pushes proof mass backward).
# When braking (-a_long), accelerometer measures -a_long.
print("\nDifference (Accel - Brake):")
print("  ref_long diff: ", accel['ref_accel_long_mps2'].mean() - brake['ref_accel_long_mps2'].mean())
print("  phone_x diff:  ", accel['phone_accel_x_mps2'].mean() - brake['phone_accel_x_mps2'].mean())
print("  phone_y diff:  ", accel['phone_accel_y_mps2'].mean() - brake['phone_accel_y_mps2'].mean())

# Let's check left turn vs right turn:
left = df[(df['ref_yaw_rate_radps'] > 0.1) & (df['ref_speed_mps'] > 5.0)]
right = df[(df['ref_yaw_rate_radps'] < -0.1) & (df['ref_speed_mps'] > 5.0)]

print("\nDifference (Left Turn - Right Turn):")
print("  ref_lat diff:  ", left['ref_accel_lat_mps2'].mean() - right['ref_accel_lat_mps2'].mean())
print("  phone_x diff:  ", left['phone_accel_x_mps2'].mean() - right['phone_accel_x_mps2'].mean())
print("  phone_y diff:  ", left['phone_accel_y_mps2'].mean() - right['phone_accel_y_mps2'].mean())
