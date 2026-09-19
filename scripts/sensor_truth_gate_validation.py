import os
import sys
import re
import numpy as np
import pandas as pd

def run_sensor_truth_gate():
    manifest_path = 'data/manifest/recordings_manifest.csv'
    if not os.path.exists(manifest_path):
        print(f"Manifest not found at {manifest_path}")
        return

    manifest = pd.read_csv(manifest_path)
    base_dir = 'data/raw/IO-VNBD'

    print("=" * 80)
    print("NAVRIS PHASE 2.1: SENSOR TRUTH GATE EXPERIMENTAL VALIDATION")
    print("=" * 80)

    sync_recs = manifest[manifest['is_synchronized']].copy()

    # 1. TIMESTAMP & TIMER RESET VALIDATION
    print("\n--- [1] TIMESTAMP & TIMER RESET VALIDATION ---")
    reset_report = []
    print(f"Total synchronized recordings to inspect: {len(sync_recs)}")

    for idx, row in sync_recs.iterrows():
        rid = row['recording_id']
        s_rel = row['s_file_path']
        v_rel = row['v_file_path']
        s_full = os.path.join(base_dir, s_rel)
        v_full = os.path.join(base_dir, v_rel)

        if not os.path.exists(s_full) or not os.path.exists(v_full):
            continue

        try:
            df_s = pd.read_csv(s_full, usecols=lambda c: 'time' in c.lower() or 'date' in c.lower(), encoding='latin1')
            df_s.columns = [c.strip() for c in df_s.columns]
            time_cols = [c for c in df_s.columns if 'time since start' in c.lower()]
            date_cols = [c for c in df_s.columns if 'date' in c.lower()]
            
            if not time_cols:
                continue

            ts_raw = pd.to_numeric(df_s[time_cols[0]], errors='coerce').values.astype(float)
            diffs = np.diff(ts_raw)
            neg_mask = diffs < -1000.0
            num_resets = np.sum(neg_mask)
            reset_indices = np.where(neg_mask)[0]
            has_date = len(date_cols) > 0
            
            df_v = pd.read_csv(v_full, usecols=lambda c: 'start of day' in c.lower(), encoding='latin1')
            df_v.columns = [c.strip() for c in df_v.columns]
            v_time_cols = [c for c in df_v.columns if 'start of day' in c.lower()]
            if v_time_cols:
                tv_raw = pd.to_numeric(df_v[v_time_cols[0]], errors='coerce').values
                v_span = tv_raw[-1] - tv_raw[0]
            else:
                v_span = np.nan

            s_raw_span = (ts_raw[-1] - ts_raw[0]) / 1000.0

            reset_report.append({
                'recording_id': rid,
                'driver': row['driver'],
                'phone': row['phone_model'],
                'num_samples': len(ts_raw),
                'num_resets': num_resets,
                'reset_indices': reset_indices.tolist(),
                's_raw_span_s': s_raw_span,
                'v_span_s': v_span,
                'has_date': has_date
            })
        except Exception as e:
            print(f"Error inspecting {rid}: {e}")

    df_resets = pd.DataFrame(reset_report)
    affected = df_resets[df_resets['num_resets'] > 0]
    print(f"\nRecordings with timer resets: {len(affected)} out of {len(df_resets)}")
    for _, r in affected.iterrows():
        print(f"  Recording {r['recording_id']:6s} (Driver: {r['driver']}, Phone: {r['phone']}): "
              f"{r['num_resets']} resets at indices {r['reset_indices']}. "
              f"Raw S span: {r['s_raw_span_s']:.1f}s vs V span: {r['v_span_s']:.1f}s")

    # 2. SMARTPHONE GPS UPDATE VALIDATION
    print("\n--- [2] SMARTPHONE GPS UPDATE VALIDATION ---")
    gps_stats = []
    sample_drivers = df_resets.groupby('driver').first().reset_index()
    for _, r in sample_drivers.iterrows():
        rid = r['recording_id']
        s_rel = manifest[manifest['recording_id'] == rid]['s_file_path'].values[0]
        s_full = os.path.join(base_dir, s_rel)
        df_s = pd.read_csv(s_full, encoding='latin1')
        df_s.columns = [c.strip() for c in df_s.columns]
        
        lat_cols = [c for c in df_s.columns if 'lat' in c.lower()]
        lon_cols = [c for c in df_s.columns if 'long' in c.lower()]
        acc_cols = [c for c in df_s.columns if 'accuracy' in c.lower()]

        if not lat_cols:
            continue

        lats = pd.to_numeric(df_s[lat_cols[0]], errors='coerce').values
        lons = pd.to_numeric(df_s[lon_cols[0]], errors='coerce').values
        
        valid = (np.abs(lats) > 0.1) & (~np.isnan(lats))
        is_new = np.zeros(len(lats), dtype=bool)
        if len(lats) > 0 and valid[0]:
            is_new[0] = True
        for i in range(1, len(lats)):
            if valid[i]:
                if not valid[i-1] or (lats[i] != lats[i-1]) or (lons[i] != lons[i-1]):
                    is_new[i] = True

        new_fix_indices = np.where(is_new)[0]
        if len(new_fix_indices) > 1:
            intervals_samples = np.diff(new_fix_indices)
            intervals_s = intervals_samples * 0.1
            mean_dt = np.mean(intervals_s)
            median_dt = np.median(intervals_s)
            p90_dt = np.percentile(intervals_s, 90)
            max_dt = np.max(intervals_s)
        else:
            mean_dt, median_dt, p90_dt, max_dt = np.nan, np.nan, np.nan, np.nan

        accuracies = pd.to_numeric(df_s[acc_cols[0]], errors='coerce').dropna().values if acc_cols else []
        mean_acc = np.mean(accuracies) if len(accuracies) > 0 else np.nan
        median_acc = np.median(accuracies) if len(accuracies) > 0 else np.nan

        gps_stats.append({
            'recording_id': rid,
            'driver': r['driver'],
            'phone': r['phone'],
            'total_samples': len(lats),
            'new_fixes': np.sum(is_new),
            'fix_ratio_pct': round((np.sum(is_new) / len(lats)) * 100.0, 2),
            'mean_update_interval_s': round(mean_dt, 2),
            'median_update_interval_s': round(median_dt, 2),
            'p90_update_interval_s': round(p90_dt, 2),
            'max_outage_s': round(max_dt, 2),
            'mean_accuracy_m': round(mean_acc, 2),
            'median_accuracy_m': round(median_acc, 2)
        })

    df_gps = pd.DataFrame(gps_stats)
    print(df_gps.to_string(index=False))

    # 3. ACCELEROMETER & GRAVITY SEMANTICS
    print("\n--- [3] ACCELEROMETER & GRAVITY SEMANTICS (STATIONARY ANALYSIS) ---")
    sensor_stats = []
    for idx, row in sync_recs.head(15).iterrows():
        rid = row['recording_id']
        s_rel = row['s_file_path']
        v_rel = row['v_file_path']
        s_full = os.path.join(base_dir, s_rel)
        v_full = os.path.join(base_dir, v_rel)

        try:
            df_v = pd.read_csv(v_full, nrows=500, encoding='latin1')
            df_v.columns = [c.strip() for c in df_v.columns]
            vel_col = [c for c in df_v.columns if c.lower().startswith('velocity')][0]
            v_speed = pd.to_numeric(df_v[vel_col], errors='coerce').values / 3.6

            df_s = pd.read_csv(s_full, nrows=500, encoding='latin1')
            df_s.columns = [c.strip() for c in df_s.columns]
            
            stat_v_mask = (v_speed[:min(len(v_speed), len(df_s))] < 0.1)
            stat_indices = np.where(stat_v_mask)[0]
            if len(stat_indices) < 20:
                continue
            stat_idx = stat_indices[:min(100, len(stat_indices))]

            ax_col = [c for c in df_s.columns if re.search(r'accel.*\bx\b', c, re.I)][0]
            ay_col = [c for c in df_s.columns if re.search(r'accel.*\by\b', c, re.I)][0]
            az_col = [c for c in df_s.columns if re.search(r'accel.*\bz\b', c, re.I)][0]

            gx_col = [c for c in df_s.columns if re.search(r'grav.*\bx\b', c, re.I)][0]
            gy_col = [c for c in df_s.columns if re.search(r'grav.*\by\b', c, re.I)][0]
            gz_col = [c for c in df_s.columns if re.search(r'grav.*\bz\b', c, re.I)][0]

            ax = pd.to_numeric(df_s[ax_col].iloc[stat_idx], errors='coerce').values
            ay = pd.to_numeric(df_s[ay_col].iloc[stat_idx], errors='coerce').values
            az = pd.to_numeric(df_s[az_col].iloc[stat_idx], errors='coerce').values

            gx = pd.to_numeric(df_s[gx_col].iloc[stat_idx], errors='coerce').values
            gy = pd.to_numeric(df_s[gy_col].iloc[stat_idx], errors='coerce').values
            gz = pd.to_numeric(df_s[gz_col].iloc[stat_idx], errors='coerce').values

            a_norm = np.sqrt(ax**2 + ay**2 + az**2)
            g_norm = np.sqrt(gx**2 + gy**2 + gz**2)
            diff_norm = np.sqrt((ax - gx)**2 + (ay - gy)**2 + (az - gz)**2)

            sensor_stats.append({
                'recording_id': rid,
                'phone': row['phone_model'],
                'ax_mean': round(float(np.mean(ax)), 3),
                'ay_mean': round(float(np.mean(ay)), 3),
                'az_mean': round(float(np.mean(az)), 3),
                'a_norm': round(float(np.mean(a_norm)), 3),
                'gx_mean': round(float(np.mean(gx)), 3),
                'gy_mean': round(float(np.mean(gy)), 3),
                'gz_mean': round(float(np.mean(gz)), 3),
                'g_norm': round(float(np.mean(g_norm)), 3),
                'diff_norm': round(float(np.mean(diff_norm)), 3)
            })
        except Exception as e:
            pass

    df_sensors = pd.DataFrame(sensor_stats)
    print(df_sensors.to_string(index=False))

    # 4. PHONE DYNAMICS COUPLING
    print("\n--- [4] PHONE DYNAMICS COUPLING (LONGITUDINAL & LATERAL) ---")
    coupling_results = []
    for rid in ['S1', 'S2', 'M', 'Y1', 'VTA1A', 'VFA01']:
        sub = manifest[manifest['recording_id'] == rid]
        if len(sub) == 0: continue
        row = sub.iloc[0]
        s_full = os.path.join(base_dir, row['s_file_path'])
        v_full = os.path.join(base_dir, row['v_file_path'])

        try:
            df_v = pd.read_csv(v_full, nrows=5000, encoding='latin1')
            df_v.columns = [c.strip() for c in df_v.columns]
            df_s = pd.read_csv(s_full, nrows=5000, encoding='latin1')
            df_s.columns = [c.strip() for c in df_s.columns]

            n_pts = min(len(df_v), len(df_s))
            df_v = df_v.iloc[:n_pts]
            df_s = df_s.iloc[:n_pts]

            v_spd_col = [c for c in df_v.columns if c.lower().startswith('velocity')][0]
            v_spd = pd.to_numeric(df_v[v_spd_col], errors='coerce').values / 3.6
            v_long_col = [c for c in df_v.columns if 'longitudinal' in c.lower()][0]
            v_long = pd.to_numeric(df_v[v_long_col], errors='coerce').values * 9.80665
            v_yaw_col = [c for c in df_v.columns if 'yaw rate' in c.lower()][0]
            v_yaw = np.radians(pd.to_numeric(df_v[v_yaw_col], errors='coerce').values)

            ax_col = [c for c in df_s.columns if re.search(r'accel.*\bx\b', c, re.I)][0]
            ay_col = [c for c in df_s.columns if re.search(r'accel.*\by\b', c, re.I)][0]
            az_col = [c for c in df_s.columns if re.search(r'accel.*\bz\b', c, re.I)][0]

            gx_col = [c for c in df_s.columns if re.search(r'grav.*\bx\b', c, re.I)][0]
            gy_col = [c for c in df_s.columns if re.search(r'grav.*\by\b', c, re.I)][0]
            gz_col = [c for c in df_s.columns if re.search(r'grav.*\bz\b', c, re.I)][0]

            lin_ax = pd.to_numeric(df_s[ax_col], errors='coerce').values - pd.to_numeric(df_s[gx_col], errors='coerce').values
            lin_ay = pd.to_numeric(df_s[ay_col], errors='coerce').values - pd.to_numeric(df_s[gy_col], errors='coerce').values
            lin_az = pd.to_numeric(df_s[az_col], errors='coerce').values - pd.to_numeric(df_s[gz_col], errors='coerce').values

            wx_col = [c for c in df_s.columns if re.search(r'gyroscope\s+(x|yaw)\b', c, re.I)][0]
            wy_col = [c for c in df_s.columns if re.search(r'gyroscope\s+(y|pitch)\b', c, re.I)][0]
            wz_col = [c for c in df_s.columns if re.search(r'gyroscope\s+(z|roll)\b', c, re.I)][0]

            wx = pd.to_numeric(df_s[wx_col], errors='coerce').values
            wy = pd.to_numeric(df_s[wy_col], errors='coerce').values
            wz = pd.to_numeric(df_s[wz_col], errors='coerce').values

            moving_mask = (v_spd > 5.0) & (~np.isnan(lin_ax)) & (~np.isnan(v_long)) & (~np.isnan(wx)) & (~np.isnan(v_yaw))
            if np.sum(moving_mask) > 100:
                corr_ax = np.corrcoef(lin_ax[moving_mask], v_long[moving_mask])[0, 1]
                corr_ay = np.corrcoef(lin_ay[moving_mask], v_long[moving_mask])[0, 1]
                corr_az = np.corrcoef(lin_az[moving_mask], v_long[moving_mask])[0, 1]

                corr_wx = np.corrcoef(wx[moving_mask], v_yaw[moving_mask])[0, 1]
                corr_wy = np.corrcoef(wy[moving_mask], v_yaw[moving_mask])[0, 1]
                corr_wz = np.corrcoef(wz[moving_mask], v_yaw[moving_mask])[0, 1]

                coupling_results.append({
                    'recording_id': rid,
                    'driver': row['driver'],
                    'corr_ax_vs_vlong': round(float(corr_ax), 3),
                    'corr_ay_vs_vlong': round(float(corr_ay), 3),
                    'corr_az_vs_vlong': round(float(corr_az), 3),
                    'corr_wx_vs_vyaw': round(float(corr_wx), 3),
                    'corr_wy_vs_vyaw': round(float(corr_wy), 3),
                    'corr_wz_vs_vyaw': round(float(corr_wz), 3)
                })
        except Exception as e:
            print(f"Coupling error on {rid}: {e}")

    df_coupling = pd.DataFrame(coupling_results)
    print(df_coupling.to_string(index=False))

    # 5. SENSOR BIAS AND NOISE CHARACTERIZATION PER DEVICE
    print("\n--- [5] SENSOR BIAS AND NOISE CHARACTERIZATION PER DEVICE ---")
    device_bias_noise = []
    for phone_model, group in manifest[manifest['is_synchronized']].groupby('phone_model'):
        accel_stds = []
        gyro_stds = []
        gyro_biases = []
        
        for _, row in group.head(10).iterrows():
            s_full = os.path.join(base_dir, row['s_file_path'])
            v_full = os.path.join(base_dir, row['v_file_path'])
            try:
                df_v = pd.read_csv(v_full, nrows=500, encoding='latin1')
                df_v.columns = [c.strip() for c in df_v.columns]
                vel_col = [c for c in df_v.columns if c.lower().startswith('velocity')][0]
                v_spd = pd.to_numeric(df_v[vel_col], errors='coerce').values / 3.6
                df_s = pd.read_csv(s_full, nrows=500, encoding='latin1')
                df_s.columns = [c.strip() for c in df_s.columns]

                stat_mask = (v_spd[:min(len(v_spd), len(df_s))] < 0.1)
                stat_idx = np.where(stat_mask)[0]
                if len(stat_idx) < 30:
                    continue
                stat_idx = stat_idx[:min(200, len(stat_idx))]

                ax_col = [c for c in df_s.columns if re.search(r'accel.*\bx\b', c, re.I)][0]
                ay_col = [c for c in df_s.columns if re.search(r'accel.*\by\b', c, re.I)][0]
                az_col = [c for c in df_s.columns if re.search(r'accel.*\bz\b', c, re.I)][0]

                wx_col = [c for c in df_s.columns if re.search(r'gyroscope\s+(x|yaw)\b', c, re.I)][0]
                wy_col = [c for c in df_s.columns if re.search(r'gyroscope\s+(y|pitch)\b', c, re.I)][0]
                wz_col = [c for c in df_s.columns if re.search(r'gyroscope\s+(z|roll)\b', c, re.I)][0]

                ax = pd.to_numeric(df_s[ax_col].iloc[stat_idx], errors='coerce').values
                ay = pd.to_numeric(df_s[ay_col].iloc[stat_idx], errors='coerce').values
                az = pd.to_numeric(df_s[az_col].iloc[stat_idx], errors='coerce').values

                wx = pd.to_numeric(df_s[wx_col].iloc[stat_idx], errors='coerce').values
                wy = pd.to_numeric(df_s[wy_col].iloc[stat_idx], errors='coerce').values
                wz = pd.to_numeric(df_s[wz_col].iloc[stat_idx], errors='coerce').values

                accel_stds.append([np.std(ax), np.std(ay), np.std(az)])
                gyro_stds.append([np.std(wx), np.std(wy), np.std(wz)])
                gyro_biases.append([np.mean(wx), np.mean(wy), np.mean(wz)])
            except Exception:
                pass

        if accel_stds:
            mean_a_std = np.mean(accel_stds, axis=0)
            mean_w_std = np.mean(gyro_stds, axis=0)
            mean_w_bias = np.mean(gyro_biases, axis=0)
            device_bias_noise.append({
                'phone_model': phone_model,
                'accel_noise_std_mps2': np.round(mean_a_std, 4).tolist(),
                'gyro_noise_std_radps': np.round(mean_w_std, 4).tolist(),
                'gyro_bias_radps': np.round(mean_w_bias, 5).tolist()
            })

    df_noise = pd.DataFrame(device_bias_noise)
    print(df_noise.to_string(index=False))

    print("\n" + "=" * 80)
    print("SENSOR TRUTH GATE VALIDATION COMPLETED")
    print("=" * 80)

if __name__ == '__main__':
    run_sensor_truth_gate()
