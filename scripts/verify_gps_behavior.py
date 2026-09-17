import pandas as pd
import numpy as np

def analyze_gps(fpath, is_phone=True):
    df = pd.read_csv(fpath, encoding='latin1')
    df.columns = [c.strip() for c in df.columns]
    
    print(f'=== File: {fpath} ===')
    print(f'Total rows: {len(df)}')
    
    if is_phone:
        time_col = 'TIME SINCE START (ms)'
        lat_col = 'GPS LATITUDE (degrees)'
        lon_col = 'GPS LONGITUDE (degrees)'
        acc_col = 'GPS ACCURACY (m)'
        sat_col = 'GPS SATELLITES IN RANGE'
        speed_col = 'GPS SPEED (Kmh)'
        
        # Time deltas
        times = df[time_col].values
        dt = np.diff(times)
        print(f'Phone Sampling dt (ms): median={np.median(dt)}, mean={np.mean(dt):.2f}, min={np.min(dt)}, max={np.max(dt)}')
        
        # GPS coordinate changes
        lat_diff = np.diff(df[lat_col].values)
        lon_diff = np.diff(df[lon_col].values)
        
        # When does GPS actually update?
        gps_updates = (lat_diff != 0) | (lon_diff != 0)
        update_indices = np.where(gps_updates)[0]
        if len(update_indices) > 0:
            gps_update_times = times[update_indices + 1]
            gps_dt = np.diff(gps_update_times)
            print(f'GPS Actual Position Change dt (ms): median={np.median(gps_dt):.1f}, mean={np.mean(gps_dt):.1f}, min={np.min(gps_dt)}, max={np.max(gps_dt)}')
            # Number of rows coordinate is held constant
            hold_counts = np.diff(np.insert(update_indices, 0, 0))
            print(f'Coordinate hold count between changes: median={np.median(hold_counts)}, min={np.min(hold_counts)}, max={np.max(hold_counts)}')
        
        # Inspect accuracy & satellites
        print(f'GPS Accuracy (m): min={df[acc_col].min()}, max={df[acc_col].max()}, median={df[acc_col].median()}')
        print(f'Unique Satellite values (sample): {df[sat_col].value_counts().head(5).to_dict()}')
        
        # Check first 25 consecutive rows of lat/lon
        print('\nFirst 15 consecutive rows of GPS Lat, Lon, Time (ms):')
        for i in range(15):
            print(f'Row {i:2d}: Time={df.loc[i, time_col]} ms, Lat={df.loc[i, lat_col]}, Lon={df.loc[i, lon_col]}, Acc={df.loc[i, acc_col]}')
            
    else: # Vehicle CAN/VBOX
        time_col = 'Time Since Start of Day (seconds)'
        lat_col = 'Latitude (degrees)'
        lon_col = 'Longitude (degrees)'
        sat_col = 'No of GPS Satellites Available'
        speed_col = 'Velocity (km/hr)'
        
        times = df[time_col].values
        dt = np.diff(times)
        print(f'VBOX Sampling dt (s): median={np.median(dt):.3f}, mean={np.mean(dt):.3f}, min={np.min(dt):.3f}, max={np.max(dt):.3f}')
        
        lat_diff = np.diff(df[lat_col].values)
        lon_diff = np.diff(df[lon_col].values)
        vbox_updates = (lat_diff != 0) | (lon_diff != 0)
        vbox_update_idx = np.where(vbox_updates)[0]
        if len(vbox_update_idx) > 0:
            vbox_dt = np.diff(times[vbox_update_idx + 1])
            print(f'VBOX Actual Position Change dt (s): median={np.median(vbox_dt):.3f}, mean={np.mean(vbox_dt):.3f}')
            
        print(f'VBOX Satellites: min={df[sat_col].min()}, max={df[sat_col].max()}, median={df[sat_col].median()}')
        print('\nFirst 10 consecutive rows of VBOX Lat, Lon, Time (s):')
        for i in range(10):
            print(f'Row {i:2d}: Time={df.loc[i, time_col]} s, Lat={df.loc[i, lat_col]}, Lon={df.loc[i, lon_col]}, Sats={df.loc[i, sat_col]}')

analyze_gps('data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset/S-S1.csv', is_phone=True)
print('\n' + '='*50 + '\n')
analyze_gps('data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/V-Dataset/V-S1.csv', is_phone=False)
