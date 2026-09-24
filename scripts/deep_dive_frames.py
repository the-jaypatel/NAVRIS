import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/synchronized/S1_sync.parquet')

# 1. Stationary Analysis (t in [0, 5] s)
stat = df[df['time_s'] <= 5.0]
print("=" * 80)
print("1. STATIONARY GRAVITY & ACCELEROMETER ANALYSIS (t <= 5s)")
print("=" * 80)
print("Accel Mean (m/s^2):", stat[['phone_accel_x_mps2', 'phone_accel_y_mps2', 'phone_accel_z_mps2']].mean().to_dict())
print("Gravity Mean (m/s^2):", stat[['phone_gravity_x_mps2', 'phone_gravity_y_mps2', 'phone_gravity_z_mps2']].mean().to_dict())
print("Gyro Mean (rad/s):", stat[['phone_gyro_x_radps', 'phone_gyro_y_radps', 'phone_gyro_z_radps']].mean().to_dict())
if 'phone_orient_pitch_rad' in stat.columns:
    print("Orientation Mean (deg):")
    print("  Azimuth:", np.degrees(stat['phone_orient_azimuth_rad'].mean()))
    print("  Pitch:  ", np.degrees(stat['phone_orient_pitch_rad'].mean()))
    print("  Roll:   ", np.degrees(stat['phone_orient_roll_rad'].mean()))

# 2. Dynamic Longitudinal Acceleration Analysis (straight driving)
# Find intervals where vehicle is moving, driving straight (|yaw_rate| < 0.02 rad/s), with strong longitudinal acceleration
moving_straight = (df['ref_speed_mps'] > 5.0) & (np.abs(df['ref_yaw_rate_radps']) < 0.02)
accel_events = moving_straight & (df['ref_accel_long_mps2'] > 1.0)
brake_events = moving_straight & (df['ref_accel_long_mps2'] < -1.0)

print("\n" + "=" * 80)
print("2. LONGITUDINAL MOTION (VEHICLE FORWARD/BRAKING)")
print("=" * 80)
print(f"Forward Acceleration Events (N={accel_events.sum()}):")
print("  VBOX Long Accel Mean:", df.loc[accel_events, 'ref_accel_long_mps2'].mean())
print("  Phone Accel X Mean:  ", df.loc[accel_events, 'phone_accel_x_mps2'].mean())
print("  Phone Accel Y Mean:  ", df.loc[accel_events, 'phone_accel_y_mps2'].mean())
print("  Phone Accel Z Mean:  ", df.loc[accel_events, 'phone_accel_z_mps2'].mean() - 9.81)

print(f"\nBraking Deceleration Events (N={brake_events.sum()}):")
print("  VBOX Long Accel Mean:", df.loc[brake_events, 'ref_accel_long_mps2'].mean())
print("  Phone Accel X Mean:  ", df.loc[brake_events, 'phone_accel_x_mps2'].mean())
print("  Phone Accel Y Mean:  ", df.loc[brake_events, 'phone_accel_y_mps2'].mean())
print("  Phone Accel Z Mean:  ", df.loc[brake_events, 'phone_accel_z_mps2'].mean() - 9.81)

# 3. Dynamic Lateral Acceleration & Yaw Rate Analysis (turning)
left_turns = (df['ref_speed_mps'] > 5.0) & (df['ref_yaw_rate_radps'] > 0.05)
right_turns = (df['ref_speed_mps'] > 5.0) & (df['ref_yaw_rate_radps'] < -0.05)

print("\n" + "=" * 80)
print("3. LATERAL MOTION & TURNING (LEFT vs RIGHT TURNS)")
print("=" * 80)
print(f"Left Turns (N={left_turns.sum()}):")
print("  VBOX Yaw Rate Mean:   ", df.loc[left_turns, 'ref_yaw_rate_radps'].mean())
print("  VBOX Lat Accel Mean:  ", df.loc[left_turns, 'ref_accel_lat_mps2'].mean())
print("  Phone Gyro X Mean:    ", df.loc[left_turns, 'phone_gyro_x_radps'].mean())
print("  Phone Gyro Y Mean:    ", df.loc[left_turns, 'phone_gyro_y_radps'].mean())
print("  Phone Gyro Z Mean:    ", df.loc[left_turns, 'phone_gyro_z_radps'].mean())
print("  Phone Accel X Mean:   ", df.loc[left_turns, 'phone_accel_x_mps2'].mean())
print("  Phone Accel Y Mean:   ", df.loc[left_turns, 'phone_accel_y_mps2'].mean())

print(f"\nRight Turns (N={right_turns.sum()}):")
print("  VBOX Yaw Rate Mean:   ", df.loc[right_turns, 'ref_yaw_rate_radps'].mean())
print("  VBOX Lat Accel Mean:  ", df.loc[right_turns, 'ref_accel_lat_mps2'].mean())
print("  Phone Gyro X Mean:    ", df.loc[right_turns, 'phone_gyro_x_radps'].mean())
print("  Phone Gyro Y Mean:    ", df.loc[right_turns, 'phone_gyro_y_radps'].mean())
print("  Phone Gyro Z Mean:    ", df.loc[right_turns, 'phone_gyro_z_radps'].mean())
print("  Phone Accel X Mean:   ", df.loc[right_turns, 'phone_accel_x_mps2'].mean())
print("  Phone Accel Y Mean:   ", df.loc[right_turns, 'phone_accel_y_mps2'].mean())
