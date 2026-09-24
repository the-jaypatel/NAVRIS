import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/synchronized/S1_sync.parquet')

print("Gravity Z mean across entire S1:", df['phone_gravity_z_mps2'].mean())
print("Gravity Z std:                  ", df['phone_gravity_z_mps2'].std())
print("Gravity Y mean:                 ", df['phone_gravity_y_mps2'].mean())
print("Gravity X mean:                 ", df['phone_gravity_x_mps2'].mean())

print("\nAccel Z mean across entire S1:  ", df['phone_accel_z_mps2'].mean())
print("Accel Y mean:                   ", df['phone_accel_y_mps2'].mean())
print("Accel X mean:                   ", df['phone_accel_x_mps2'].mean())
