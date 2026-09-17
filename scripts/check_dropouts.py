import glob, os
import pandas as pd
import numpy as np

print('--- Checking V-files for GPS Satellite Drops / Dropouts ---')
v_files = glob.glob('data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/V-Dataset/*.csv')
print(f'Total V files in synchronised set: {len(v_files)}')

v_dropout_summary = []
for vf in v_files:
    try:
        df = pd.read_csv(vf, encoding='latin1', usecols=['No of GPS Satellites Available', ' Latitude (degrees)', ' Longitude (degrees)', ' Velocity (km/hr)'])
        df.columns = [c.strip() for c in df.columns]
        sat_col = 'No of GPS Satellites Available'
        sats = df[sat_col]
        zero_sats = (sats == 0).sum()
        low_sats = (sats < 4).sum()
        min_sats = sats.min()
        max_sats = sats.max()
        if zero_sats > 0 or low_sats > 0:
            v_dropout_summary.append({
                'file': os.path.basename(vf),
                'total_rows': len(df),
                'zero_sats': zero_sats,
                'low_sats_lt4': low_sats,
                'min_sats': min_sats,
                'max_sats': max_sats
            })
    except Exception as e:
        continue

print(f'V-files with 0 or <4 satellites: {len(v_dropout_summary)}')
for item in v_dropout_summary[:15]:
    print(item)
