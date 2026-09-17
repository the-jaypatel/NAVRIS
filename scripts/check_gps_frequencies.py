import glob, os
import pandas as pd
import numpy as np

s_files = [
    'data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset/S-S1.csv',
    'data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset/S-S2.csv',
    'data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset/S-M.csv',
    'data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset/S-Y1.csv',
    'data/raw/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/Vta (Driver E)/Vta01a/S-Vta1a.csv',
    'data/raw/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw01/S-Vw1.csv',
    'data/raw/IO-VNBD/Unsynchronised V and S Dataset/Uncategorised IOVNB Dataset/S-Dataset/S-T1.csv',
    'data/raw/IO-VNBD/Unsynchronised V and S Dataset/Uncategorised IOVNB Dataset/S-Dataset/S-A1.csv',
    'data/raw/IO-VNBD/Unsynchronised V and S Dataset/Uncategorised IOVNB Dataset/S-Dataset/S-I.csv'
]

for s_path in s_files:
    if not os.path.exists(s_path):
        print(f'{s_path}: NOT FOUND')
        continue
    df = pd.read_csv(s_path, encoding='latin1', nrows=10000)
    df.columns = [c.strip() for c in df.columns]
    
    time_col = [c for c in df.columns if 'time' in c.lower()][0]
    lat_col = [c for c in df.columns if 'lat' in c.lower()][0]
    lon_col = [c for c in df.columns if 'long' in c.lower()][0]
    speed_col = [c for c in df.columns if 'speed' in c.lower()][0]
    
    times = df[time_col].values
    sensor_dt = np.diff(times)
    
    # Filter only when vehicle is moving
    moving = df[df[speed_col] > 2.0]
    if len(moving) > 50:
        lat_diff = moving[lat_col].diff().values
        lon_diff = moving[lon_col].diff().values
        changes = np.where((lat_diff != 0) | (lon_diff != 0))[0]
        if len(changes) > 1:
            moving_times = moving[time_col].values
            change_times = moving_times[changes]
            gps_dt = np.diff(change_times)
            print(f'{os.path.basename(s_path)}: Sensor dt={np.median(sensor_dt):.0f}ms (~{1000/np.median(sensor_dt):.0f}Hz) | GPS Update dt={np.median(gps_dt):.0f}ms (~{1000/np.median(gps_dt):.2f}Hz) [min={np.min(gps_dt):.0f}, max={np.max(gps_dt):.0f}]')
        else:
            print(f'{os.path.basename(s_path)}: Not enough GPS changes while moving')
    else:
        print(f'{os.path.basename(s_path)}: Vehicle not moving in first 10000 rows')
