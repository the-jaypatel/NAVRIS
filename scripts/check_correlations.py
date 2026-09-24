import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/synchronized/S1_sync.parquet')

# Check raw S-S1.csv column order vs Android conventions
df_raw = pd.read_csv('data/raw/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S1/S-S1.csv', encoding='latin1', nrows=100)
df_raw.columns = [c.strip() for c in df_raw.columns]

# Print gyro and accel columns in raw data
print("Raw columns 9 to 18:")
for c in df_raw.columns[9:18]:
    print(f"  {c}")

# Let's check the correlations of all combinations of (accel_x, accel_y) with (lat_accel, long_accel)
# and all combinations of (gyro_x, gyro_y, gyro_z) with (yaw_rate, d_pitch, d_roll)
moving = (df['ref_speed_mps'] > 5.0)

# VBOX has:
# ref_yaw_rate_radps
# ref_accel_long_mps2
# ref_accel_lat_mps2
print("\n" + "=" * 80)
print("CORRELATION MATRIX: PHONE SENSORS vs VBOX VEHICLE DYNAMICS")
print("=" * 80)
vbox_vars = {
    'VBOX_Lat_Accel': df['ref_accel_lat_mps2'].values,
    'VBOX_Long_Accel': df['ref_accel_long_mps2'].values,
    'VBOX_Yaw_Rate': df['ref_yaw_rate_radps'].values
}

phone_vars = {
    'phone_accel_x': df['phone_accel_x_mps2'].values,
    'phone_accel_y': df['phone_accel_y_mps2'].values,
    'phone_accel_z': df['phone_accel_z_mps2'].values,
    'phone_gyro_x': df['phone_gyro_x_radps'].values,
    'phone_gyro_y': df['phone_gyro_y_radps'].values,
    'phone_gyro_z': df['phone_gyro_z_radps'].values
}

corr_table = pd.DataFrame(index=phone_vars.keys(), columns=vbox_vars.keys())

for p_name, p_val in phone_vars.items():
    for v_name, v_val in vbox_vars.items():
        valid = moving & (~np.isnan(p_val)) & (~np.isnan(v_val))
        corr = np.corrcoef(p_val[valid], v_val[valid])[0, 1]
        corr_table.loc[p_name, v_name] = f"{corr:+.4f}"

print(corr_table.to_string())
