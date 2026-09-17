import pandas as pd
df = pd.read_csv('data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/V-Dataset/V-S1.csv', encoding='latin1')
df.columns = [c.strip() for c in df.columns]
sub = df.iloc[19365:19398]
for idx, r in sub.iterrows():
    t = r['Time Since Start of Day (seconds)']
    lat = r['Latitude (degrees)']
    lon = r['Longitude (degrees)']
    vel = r['Velocity (km/hr)']
    sats = r['No of GPS Satellites Available']
    print(f"{idx}: Time={t}, Lat={lat}, Lon={lon}, Vel={vel}, Sats={sats}")
