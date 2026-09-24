import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/synchronized/S1_sync.parquet')

# Check orientation columns
print("Orientation columns in df:")
for c in [c for c in df.columns if 'orient' in c]:
    print(" ", c)

# Compute correlation of phone_orient_azimuth_rad with ref_heading_rad
ref_heading = df['ref_heading_rad'].values
phone_azimuth = df['phone_orient_azimuth_rad'].values

# Wrap angle differences
diff = (phone_azimuth - ref_heading + np.pi) % (2.0 * np.pi) - np.pi
print("Mean heading diff (phone azimuth vs ref heading):", np.degrees(np.nanmean(diff)))
print("Std heading diff: ", np.degrees(np.nanstd(diff)))

# Sample values
sample = df[['time_s', 'ref_heading_deg', 'phone_orient_azimuth_rad', 'phone_orient_pitch_rad', 'phone_orient_roll_rad']].dropna().head(10)
sample['phone_azimuth_deg'] = np.degrees(sample['phone_orient_azimuth_rad'])
sample['phone_pitch_deg'] = np.degrees(sample['phone_orient_pitch_rad'])
sample['phone_roll_deg'] = np.degrees(sample['phone_orient_roll_rad'])
print("\nSample Orientation rows:")
print(sample[['time_s', 'ref_heading_deg', 'phone_azimuth_deg', 'phone_pitch_deg', 'phone_roll_deg']].to_string())
