import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/synchronized/S1_sync.parquet')
moving = (df['ref_speed_mps'] > 5.0)

ax = df['phone_accel_x_mps2'].values[moving]
ay = df['phone_accel_y_mps2'].values[moving]
a_long = df['ref_accel_long_mps2'].values[moving]
a_lat = df['ref_accel_lat_mps2'].values[moving]

# Center the signals
ax_c = ax - np.mean(ax)
ay_c = ay - np.mean(ay)
along_c = a_long - np.mean(a_long)
alat_c = a_lat - np.mean(a_lat)

# Sweep angle theta from 0 to 360 degrees to find the alignment between (ax, ay) and (along, alat)
# Let [a_fwd, a_right] = R(theta) @ [ax, ay]
best_corr = -1.0
best_theta = 0.0

angles = np.linspace(0, 360, 361)
results = []

for deg in angles:
    rad = np.radians(deg)
    # Rotation in plane
    # candidate forward = cos(rad)*ax + sin(rad)*ay
    # candidate right   = -sin(rad)*ax + cos(rad)*ay
    a_cand_fwd = np.cos(rad) * ax_c + np.sin(rad) * ay_c
    a_cand_lat = -np.sin(rad) * ax_c + np.cos(rad) * ay_c

    corr_fwd = np.corrcoef(a_cand_fwd, along_c)[0, 1]
    corr_lat = np.corrcoef(a_cand_lat, alat_c)[0, 1]
    score = corr_fwd + corr_lat
    results.append((deg, corr_fwd, corr_lat, score))

res_df = pd.DataFrame(results, columns=['deg', 'corr_fwd', 'corr_lat', 'score'])
res_df = res_df.sort_values('score', ascending=False)
print("Top 5 rotation angles aligning (ax, ay) to (VBOX_Long, VBOX_Lat):")
print(res_df.head(10).to_string(index=False))
