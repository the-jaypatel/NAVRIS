import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/synchronized/S1_sync.parquet')

# Let's inspect the relationship between angular rates and vehicle pitch/roll
# In VBOX, we have ref_yaw_rate_radps.
# Can we estimate vehicle pitch rate and roll rate from VBOX or vehicle chassis?
# In S1, the road has elevation changes and vehicle has suspension pitch/roll during braking and turning.
# But even simpler: let's look at the correlation between the gyro channels and the derivatives of the accelerometer components!
# If the vehicle tilts, gravity shifts between Z and (X, Y).
# Specifically: d(f_x)/dt ~ g * omega_y_accel, d(f_y)/dt ~ -g * omega_x_accel!

# Let's compute finite differences of phone_accel_x, phone_accel_y, phone_accel_z
dt = 0.1
df_x = np.gradient(df['phone_accel_x_mps2'].values, dt)
df_y = np.gradient(df['phone_accel_y_mps2'].values, dt)
df_z = np.gradient(df['phone_accel_z_mps2'].values, dt)

gx = df['phone_gyro_x_radps'].values
gy = df['phone_gyro_y_radps'].values
gz = df['phone_gyro_z_radps'].values

moving = (df['ref_speed_mps'] > 2.0)

print("Correlation of gyro channels with derivative of accel components:")
for g_name, g_val in [("gyro_x", gx), ("gyro_y", gy), ("gyro_z", gz)]:
    c_dfx = np.corrcoef(g_val[moving], df_x[moving])[0, 1]
    c_dfy = np.corrcoef(g_val[moving], df_y[moving])[0, 1]
    c_dfz = np.corrcoef(g_val[moving], df_z[moving])[0, 1]
    print(f"  {g_name}: with d(ax)/dt = {c_dfx:+.4f}, with d(ay)/dt = {c_dfy:+.4f}, with d(az)/dt = {c_dfz:+.4f}")
