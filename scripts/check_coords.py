import glob, os
import pandas as pd

v_files = glob.glob('data/raw/IO-VNBD/**/*.csv', recursive=True)
print(f'Total CSV files found: {len(v_files)}')

def inspect_file(fpath):
    encodings = ['utf-8', 'latin-1', 'cp1252']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(fpath, nrows=50, encoding=enc)
            break
        except Exception:
            continue
    if df is None:
        return {'path': os.path.basename(fpath), 'error': 'Could not read'}
    
    # Strip column names
    df.columns = [c.strip() for c in df.columns]
    
    lat_col = [c for c in df.columns if 'lat' in c.lower()]
    lon_col = [c for c in df.columns if 'long' in c.lower()]
    
    lat_vals = df[lat_col[0]].dropna().values if lat_col else []
    lon_vals = df[lon_col[0]].dropna().values if lon_col else []
    
    lat_sample = lat_vals[0] if len(lat_vals) else None
    lon_sample = lon_vals[0] if len(lon_vals) else None
    
    return {
        'path': os.path.basename(fpath),
        'cols': len(df.columns),
        'lat_col': lat_col[0] if lat_col else None,
        'lon_col': lon_col[0] if lon_col else None,
        'lat_sample': lat_sample,
        'lon_sample': lon_sample
    }

samples = ['V-S1.csv', 'S-S1.csv', 'V-M.csv', 'S-M.csv', 'V-Y1.csv', 'S-Y1.csv', 'V-vta1a.csv', 'S-Vta1a.csv', 
           'V-vtb1.csv', 'S-Vtb1.csv', 'V-Vw1.csv', 'S-Vw1.csv', 'V-Vfa01.csv', 'S-Vfa01.csv', 
           'S-T1.csv', 'S-T10.csv', 'S-I.csv', 'S-A1.csv', 'V-St1.csv', 'V-St4.csv']

for target in samples:
    matches = [f for f in v_files if os.path.basename(f).lower() == target.lower()]
    if matches:
        res = inspect_file(matches[0])
        print(f"{res['path']}: lat_col={res.get('lat_col')}, lon_col={res.get('lon_col')}, lat={res.get('lat_sample')}, lon={res.get('lon_sample')}")
    else:
        print(f"{target}: NOT FOUND")
