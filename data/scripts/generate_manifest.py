import glob, os, re, json
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt

base_dir = 'data/raw/IO-VNBD'

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # km
    return c * r

def get_metadata(rec_id):
    rec = rec_id.upper()
    if rec.startswith('S') and not rec.startswith(('ST', 'S-')):
        return {
            'driver': 'Driver A',
            'driving_style': 'Defensive',
            'vehicle': 'Ford Fiesta Titanium (FWD)',
            'phone': 'Huawei P20 Pro',
            'country': 'United Kingdom',
            'region': 'Coventry / Rugby / Nuneaton',
            'tyre_pressure_code': 'E'
        }
    elif rec == 'M':
        return {
            'driver': 'Driver B',
            'driving_style': 'Defensive',
            'vehicle': 'Ford Fiesta Titanium (FWD)',
            'phone': 'Huawei P20 Pro',
            'country': 'United Kingdom',
            'region': 'Coventry',
            'tyre_pressure_code': 'E'
        }
    elif rec.startswith('ST'):
        return {
            'driver': 'Driver C',
            'driving_style': 'Defensive',
            'vehicle': 'Ford Fiesta Titanium (FWD)',
            'phone': 'None (CAN Only)',
            'country': 'United Kingdom',
            'region': 'Coventry / Warwick / Oxford / Stokenchurch',
            'tyre_pressure_code': 'E'
        }
    elif rec.startswith('Y'):
        return {
            'driver': 'Driver D',
            'driving_style': 'Defensive',
            'vehicle': 'Ford Fiesta Titanium (FWD)',
            'phone': 'Huawei P20 Pro',
            'country': 'United Kingdom',
            'region': 'Coventry',
            'tyre_pressure_code': 'E'
        }
    elif rec.startswith('VTA'):
        return {
            'driver': 'Driver E',
            'driving_style': 'Aggressive',
            'vehicle': 'Ford Fiesta Titanium (FWD)',
            'phone': 'Huawei P20 Pro',
            'country': 'United Kingdom',
            'region': 'Atherstone / Tamworth / Ashby / Leicestershire',
            'tyre_pressure_code': 'A'
        }
    elif rec.startswith('VTB'):
        return {
            'driver': 'Driver E',
            'driving_style': 'Aggressive',
            'vehicle': 'Ford Fiesta Titanium (FWD)',
            'phone': 'Huawei P20 Pro',
            'country': 'United Kingdom',
            'region': 'Buxton / Peak District / Derbyshire',
            'tyre_pressure_code': 'B'
        }
    elif rec.startswith('VW'):
        return {
            'driver': 'Driver E',
            'driving_style': 'Aggressive',
            'vehicle': 'Ford Fiesta Titanium (FWD)',
            'phone': 'Huawei P20 Pro',
            'country': 'United Kingdom',
            'region': 'Warwick / Leamington / Coventry',
            'tyre_pressure_code': 'C'
        }
    elif rec.startswith('VFA') or rec.startswith('VFB'):
        code = 'C' if rec.startswith('VFA') else 'D'
        return {
            'driver': 'Driver E',
            'driving_style': 'Aggressive',
            'vehicle': 'Ford Fiesta Titanium (FWD)',
            'phone': 'Huawei P20 Pro (fa) / None (fb)',
            'country': 'United Kingdom',
            'region': 'Nuthall / Nottinghamshire / Nuneaton',
            'tyre_pressure_code': code
        }
    elif rec.startswith('T'):
        return {
            'driver': 'Driver F',
            'driving_style': 'Defensive',
            'vehicle': 'Renault Megane',
            'phone': 'Motorola Moto G7 Power',
            'country': 'France',
            'region': 'Charente-Maritime / Paris',
            'tyre_pressure_code': 'Unknown'
        }
    elif rec == 'I':
        return {
            'driver': 'Driver G',
            'driving_style': 'Defensive',
            'vehicle': 'Toyota Corolla Verso',
            'phone': 'Huawei P20 Pro',
            'country': 'Nigeria',
            'region': 'Lagos (Victoria Island)',
            'tyre_pressure_code': 'Unknown'
        }
    elif rec.startswith('A'):
        return {
            'driver': 'Driver H',
            'driving_style': 'Defensive',
            'vehicle': 'Volvo XC70',
            'phone': 'Blackberry Priv',
            'country': 'United Kingdom',
            'region': 'Coventry / Warwickshire',
            'tyre_pressure_code': 'Unknown'
        }
    else:
        return {
            'driver': 'Unknown',
            'driving_style': 'Unknown',
            'vehicle': 'Unknown',
            'phone': 'Unknown',
            'country': 'Unknown',
            'region': 'Unknown',
            'tyre_pressure_code': 'Unknown'
        }

all_csvs = glob.glob('data/raw/IO-VNBD/**/*.csv', recursive=True)

def normalize_id(fname):
    base = os.path.splitext(fname)[0]
    if base.upper().startswith(('V-', 'S-', 'V_')):
        return base[2:].upper()
    elif base.upper().startswith(('S_')):
        return base[2:].upper()
    return base.upper()

v_files_map = {}
s_files_map = {}

for f in all_csvs:
    rel = os.path.relpath(f, base_dir)
    fname = os.path.basename(f)
    nid = normalize_id(fname)
    is_sync = 'Synchronised' in rel
    
    if fname.upper().startswith(('V-', 'V_')) or (not fname.upper().startswith(('S-', 'S_')) and fname.upper().startswith('V')):
        if nid not in v_files_map or (is_sync and not v_files_map[nid]['is_sync']):
            v_files_map[nid] = {'path': rel, 'fname': fname, 'is_sync': is_sync, 'full_path': f}
    elif fname.upper().startswith(('S-', 'S_')):
        if nid not in s_files_map or (is_sync and not s_files_map[nid]['is_sync']):
            s_files_map[nid] = {'path': rel, 'fname': fname, 'is_sync': is_sync, 'full_path': f}

all_ids = sorted(list(set(v_files_map.keys()) | set(s_files_map.keys())))
print(f'Building manifest for {len(all_ids)} unique recordings...')

manifest = []

for rec_id in all_ids:
    meta = get_metadata(rec_id)
    v_info = v_files_map.get(rec_id)
    s_info = s_files_map.get(rec_id)
    
    has_v = v_info is not None
    has_s = s_info is not None
    is_paired = has_v and has_s
    is_synchronized = (v_info['is_sync'] if has_v else False) or (s_info['is_sync'] if has_s else False)
    
    entry = {
        'recording_id': rec_id,
        'driver': meta['driver'],
        'driving_style': meta['driving_style'],
        'vehicle': meta['vehicle'],
        'phone_model': meta['phone'],
        'country': meta['country'],
        'region': meta['region'],
        'tyre_pressure_code': meta['tyre_pressure_code'],
        'has_v_data': has_v,
        'has_s_data': has_s,
        'is_paired': is_paired,
        'is_synchronized': is_synchronized,
        'v_file_path': v_info['path'] if has_v else None,
        's_file_path': s_info['path'] if has_s else None,
        'v_rows': None,
        's_rows': None,
        'v_cols': None,
        's_cols': None,
        'duration_seconds': None,
        'duration_minutes': None,
        'start_lat': None,
        'start_lon': None,
        'end_lat': None,
        'end_lon': None,
        'min_lat': None,
        'max_lat': None,
        'min_lon': None,
        'max_lon': None,
        'distance_km_approx': None,
        'max_speed_kmh': None,
        'v_gps_dropouts_count': None,
        's_sensor_names': None
    }
    
    # Analyze V file if present
    if has_v:
        try:
            df_v = pd.read_csv(v_info['full_path'], encoding='latin1')
            df_v.columns = [c.strip() for c in df_v.columns]
            entry['v_rows'] = len(df_v)
            entry['v_cols'] = len(df_v.columns)
            
            time_cols = [c for c in df_v.columns if 'time' in c.lower()]
            if time_cols:
                t = pd.to_numeric(df_v[time_cols[0]], errors='coerce').dropna()
                if len(t) > 1:
                    dt = t.iloc[-1] - t.iloc[0]
                    entry['duration_seconds'] = round(float(dt), 2)
                    entry['duration_minutes'] = round(float(dt / 60.0), 2)
                    
            lat_col = [c for c in df_v.columns if 'lat' in c.lower()][0]
            lon_col = [c for c in df_v.columns if 'long' in c.lower()][0]
            lats = pd.to_numeric(df_v[lat_col], errors='coerce').dropna()
            lons = pd.to_numeric(df_v[lon_col], errors='coerce').dropna()
            if len(lats) > 0 and len(lons) > 0:
                entry['start_lat'] = float(lats.iloc[0])
                entry['start_lon'] = float(lons.iloc[0])
                entry['end_lat'] = float(lats.iloc[-1])
                entry['end_lon'] = float(lons.iloc[-1])
                entry['min_lat'] = float(lats.min())
                entry['max_lat'] = float(lats.max())
                entry['min_lon'] = float(lons.min())
                entry['max_lon'] = float(lons.max())
                
                sub_lats = lats.iloc[::20].values
                sub_lons = lons.iloc[::20].values
                dist = sum(haversine(sub_lons[i], sub_lats[i], sub_lons[i+1], sub_lats[i+1]) for i in range(len(sub_lats)-1))
                entry['distance_km_approx'] = round(float(dist), 2)
                
            speed_col = [c for c in df_v.columns if 'velocity' in c.lower() or 'speed' in c.lower()]
            if speed_col:
                sp = pd.to_numeric(df_v[speed_col[0]], errors='coerce').dropna()
                if len(sp) > 0:
                    entry['max_speed_kmh'] = round(float(sp.max()), 1)
                
            sat_cols = [c for c in df_v.columns if 'sat' in c.lower()]
            if sat_cols:
                sats = pd.to_numeric(df_v[sat_cols[0]], errors='coerce').dropna()
                entry['v_gps_dropouts_count'] = int((sats == 0).sum())
        except Exception as e:
            print(f'Error analyzing V file for {rec_id}: {e}')
            
    # Analyze S file if present
    if has_s:
        try:
            df_s = pd.read_csv(s_info['full_path'], encoding='latin1')
            df_s.columns = [c.strip() for c in df_s.columns]
            entry['s_rows'] = len(df_s)
            entry['s_cols'] = len(df_s.columns)
            
            has_acc = any('acc' in c.lower() for c in df_s.columns)
            has_gyro = any('gyro' in c.lower() for c in df_s.columns)
            has_mag = any('mag' in c.lower() for c in df_s.columns)
            has_grav = any('grav' in c.lower() for c in df_s.columns)
            has_orient = any('orientation' in c.lower() and 'gps' not in c.lower() for c in df_s.columns)
            sensors = []
            if has_acc: sensors.append('accel')
            if has_gyro: sensors.append('gyro')
            if has_grav: sensors.append('gravity')
            if has_mag: sensors.append('mag')
            if has_orient: sensors.append('orientation')
            entry['s_sensor_names'] = '+'.join(sensors)
            
            if not has_v:
                # Handle possible misaligned A4
                if rec_id == 'A4':
                    # In A4, time is column 8 (0-indexed)
                    t = pd.to_numeric(df_s.iloc[:, 8], errors='coerce').dropna()
                else:
                    time_cols = [c for c in df_s.columns if 'time since start' in c.lower()]
                    t = pd.to_numeric(df_s[time_cols[0]], errors='coerce').dropna() if time_cols else pd.Series([])
                    
                if len(t) > 1:
                    dt = (t.iloc[-1] - t.iloc[0]) / 1000.0
                    entry['duration_seconds'] = round(float(dt), 2)
                    entry['duration_minutes'] = round(float(dt / 60.0), 2)
                    
                lat_cols = [c for c in df_s.columns if 'lat' in c.lower()]
                lon_cols = [c for c in df_s.columns if 'long' in c.lower()]
                if lat_cols and lon_cols:
                    lats = pd.to_numeric(df_s[lat_cols[0]], errors='coerce').dropna()
                    lons = pd.to_numeric(df_s[lon_cols[0]], errors='coerce').dropna()
                    if len(lats) > 0 and len(lons) > 0:
                        entry['start_lat'] = float(lats.iloc[0])
                        entry['start_lon'] = float(lons.iloc[0])
                        entry['end_lat'] = float(lats.iloc[-1])
                        entry['end_lon'] = float(lons.iloc[-1])
                        entry['min_lat'] = float(lats.min())
                        entry['max_lat'] = float(lats.max())
                        entry['min_lon'] = float(lons.min())
                        entry['max_lon'] = float(lons.max())
                        sub_lats = lats.iloc[::20].values
                        sub_lons = lons.iloc[::20].values
                        dist = sum(haversine(sub_lons[i], sub_lats[i], sub_lons[i+1], sub_lats[i+1]) for i in range(len(sub_lats)-1))
                        entry['distance_km_approx'] = round(float(dist), 2)
                speed_cols = [c for c in df_s.columns if 'speed' in c.lower()]
                if speed_cols:
                    sp = pd.to_numeric(df_s[speed_cols[0]], errors='coerce').dropna()
                    if len(sp) > 0:
                        entry['max_speed_kmh'] = round(float(sp.max()), 1)
        except Exception as e:
            print(f'Error analyzing S file for {rec_id}: {e}')
            
    manifest.append(entry)

df_manifest = pd.DataFrame(manifest)
print(f'Manifest successfully built with {len(df_manifest)} entries.')

json_path = 'data/manifest/recordings_manifest.json'
csv_path = 'data/manifest/recordings_manifest.csv'

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2)

df_manifest.to_csv(csv_path, index=False, encoding='utf-8')
print(f'Saved manifest to {json_path} and {csv_path}')
