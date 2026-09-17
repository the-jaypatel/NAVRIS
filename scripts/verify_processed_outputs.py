import pandas as pd
import numpy as np

# Check S1_sync.parquet
df_sync = pd.read_parquet('data/processed/synchronized/S1_sync.parquet')
print('=== S1_sync.parquet Analysis ===')
print('Shape:', df_sync.shape)
print('Ref East range: [', round(df_sync['ref_east_m'].min(), 1), ',', round(df_sync['ref_east_m'].max(), 1), 'm ]')
print('Ref North range: [', round(df_sync['ref_north_m'].min(), 1), ',', round(df_sync['ref_north_m'].max(), 1), 'm ]')
print('Ref Speed range: [', round(df_sync['ref_speed_mps'].min(), 1), ',', round(df_sync['ref_speed_mps'].max(), 1), 'm/s ]')
print('Ref Speed mean:', round(df_sync['ref_speed_mps'].mean(), 2), 'm/s (~' + str(round(df_sync['ref_speed_mps'].mean()*3.6, 1)) + ' km/h)')
print('Phone Accel Z mean:', round(df_sync['phone_accel_z_mps2'].mean(), 3), 'm/s^2')
print('Phone GPS is_new_fix count:', df_sync['phone_gps_is_new_fix'].sum())
print('Phone GPS obs East non-null count:', df_sync['phone_gps_obs_east_m'].notna().sum())

# Check S1_windows.npz
npz = np.load('data/processed/windows/S1_windows.npz')
print('\n=== S1_windows.npz Analysis ===')
print('Keys:', list(npz.keys()))
features = npz['features']
print('Features shape:', features.shape)
print('Feature names:', npz['feature_names'])
print('Speed targets shape:', npz['speed_mps'].shape)
print('East disp targets range: [', round(float(npz['disp_east_m'].min()), 2), ',', round(float(npz['disp_east_m'].max()), 2), 'm ]')
print('North disp targets range: [', round(float(npz['disp_north_m'].min()), 2), ',', round(float(npz['disp_north_m'].max()), 2), 'm ]')
