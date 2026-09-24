import sys, os
sys.path.insert(0, 'src')
import numpy as np
import pandas as pd
from scipy import stats
from navris.ingest import load_raw_csv_robust

raw_s_dir = 'data/raw/IO-VNBD/Unsynchronised V and S Dataset/Uncategorised IOVNB (V and S) Dataset/S-Dataset'
raw_v_dir = 'data/raw/IO-VNBD/Unsynchronised V and S Dataset/Uncategorised IOVNB (V and S) Dataset/V-Dataset'
sync_dir = 'data/processed/synchronized'

recordings = {
    'S1': ('S-S1.csv', 'V-S1.csv'),
    'S2': ('S-S2.csv', 'V-S2.csv'),
    'S3A': ('S-S3a.csv', 'V-S3a.csv'),
    'S4': ('S-S4.csv', 'V-S4.csv'),
    'M': ('S-M.csv', 'V-M.csv'),
    'Y1': ('S-Y1.csv', 'V-Y1.csv'),
    'VTA1A': ('S-Vta1a.csv', 'V-Vta1a.csv'),
    'VTA2': ('S-Vta2.csv', 'V-Vta2.csv')
}

speed_audit_rows = []
accel_audit_rows = []
gyro_audit_rows = []

print('======================================================================')
print('NAVRIS Phase 2.3B E3: MULTI-RECORDING RAW FRAME & UNIT FORENSICS AUDIT')
print('======================================================================')

for rec_id, (s_fname, v_fname) in recordings.items():
    s_path = os.path.join(raw_s_dir, s_fname)
    v_path = os.path.join(raw_v_dir, v_fname)
    sync_path = os.path.join(sync_dir, f'{rec_id}_sync.parquet')
    
    df_raw_s = load_raw_csv_robust(s_path)
    df_sync = pd.read_parquet(sync_path)
    
    print(f'>>> AUDITING RECORDING: {rec_id} ({s_fname})')
    print(f'    Raw rows: {len(df_raw_s)}, Synced rows: {len(df_sync)}')
    
    # 1. Timestamps
    time_col = [c for c in df_raw_s.columns if 'time since start' in c.lower()][0]
    raw_time_ms = pd.to_numeric(df_raw_s[time_col], errors='coerce').values
    dt_ms = np.diff(raw_time_ms)
    resets = np.where(dt_ms < -1000)[0]
    dt_valid = dt_ms[(dt_ms > 0) & (dt_ms < 500)]
    print(f'    Timestamps: start={raw_time_ms[0]} ms, end={raw_time_ms[-1]} ms, resets={len(resets)}')
    print(f'    dt ms: mean={np.mean(dt_valid):.1f}, median={np.median(dt_valid):.1f}, min={np.min(dt_ms)}, max={np.max(dt_ms)}')
    
    # 2. GNSS Speed
    speed_col = [c for c in df_raw_s.columns if 'speed' in c.lower()][0]
    phone_spd_mps = df_sync['phone_gps_speed_mps'].values
    vbox_spd_mps = df_sync['ref_speed_mps'].values
    raw_spd = phone_spd_mps * 3.6  # perfectly reconstructs raw GPS SPEED (Kmh)
    
    mask_moving = (vbox_spd_mps > 3.0) & (~np.isnan(phone_spd_mps)) & (phone_spd_mps > 0)
    raw_moving = raw_spd[mask_moving]
    vbox_moving = vbox_spd_mps[mask_moving]
    
    ratio_to_mps = raw_moving / vbox_moving
    ratio_to_kmh = raw_moving / (vbox_moving * 3.6)
    
    med_ratio_mps = float(np.median(ratio_to_mps))
    mean_ratio_mps = float(np.mean(ratio_to_mps))
    std_ratio_mps = float(np.std(ratio_to_mps))
    med_ratio_kmh = float(np.median(ratio_to_kmh))
    
    unit_implied = 'm/s (3.6x DEFECT)' if abs(med_ratio_mps - 1.0) < 0.2 else 'km/h (NORMAL)'
    print(f'    GNSS Speed: Moving samples (>3m/s) = {len(vbox_moving)}')
    print(f'      Ratio to VBOX m/s: median={med_ratio_mps:.3f}, mean={mean_ratio_mps:.3f}, std={std_ratio_mps:.3f}')
    print(f'      Ratio to VBOX km/h: median={med_ratio_kmh:.3f}')
    print(f'      Implied Unit: {unit_implied}')
    
    # Representative samples
    rep_indices = []
    for target_spd in [2.0, 5.0, 10.0, 15.0, 20.0, 25.0]:
        candidates = np.where(mask_moving & (np.abs(vbox_spd_mps - target_spd) < 0.5))[0]
        if len(candidates) > 0:
            rep_indices.append(candidates[len(candidates)//2])
            
    print('      Representative samples (Time s, Raw Speed, VBOX m/s, VBOX km/h, Raw/VBOX_mps):')
    for idx in rep_indices[:4]:
        t_s = df_sync['time_s'].iloc[idx]
        r_s = raw_spd[idx]
        v_m = vbox_spd_mps[idx]
        v_k = v_m * 3.6
        rat = r_s / v_m if v_m > 0 else 0
        print(f'        t={t_s:6.1f}s | Raw={r_s:6.2f} | VBOX_mps={v_m:6.2f} | VBOX_kmh={v_k:6.2f} | Ratio={rat:.3f}')
        
    speed_audit_rows.append({
        'recording': rec_id,
        's_file': s_fname,
        'speed_col': speed_col,
        'moving_samples': int(len(vbox_moving)),
        'median_ratio_raw_to_vbox_mps': med_ratio_mps,
        'mean_ratio_raw_to_vbox_mps': mean_ratio_mps,
        'std_ratio_raw_to_vbox_mps': std_ratio_mps,
        'median_ratio_raw_to_vbox_kmh': med_ratio_kmh,
        'implied_unit': unit_implied,
        'has_3_6x_defect': bool(abs(med_ratio_mps - 1.0) < 0.2)
    })
    
    # 3. Stationary Accelerometer Frame
    mask_stat = (vbox_spd_mps < 0.1) & (np.abs(df_sync['ref_yaw_rate_radps'].values) < 0.01)
    stat_indices = np.where(mask_stat)[0]
    
    # find contiguous runs >= 50 samples (5 sec)
    runs = []
    if len(stat_indices) > 0:
        split_pts = np.where(np.diff(stat_indices) > 1)[0]
        starts = np.insert(stat_indices[split_pts + 1], 0, stat_indices[0])
        ends = np.append(stat_indices[split_pts], stat_indices[-1])
        for s, e in zip(starts, ends):
            if (e - s + 1) >= 50:
                runs.append((s, e))
                
    if len(runs) > 0:
        # aggregate all valid stationary intervals
        all_stat_idx = np.concatenate([np.arange(s, e+1) for s, e in runs])
        ax = df_sync['phone_accel_x_mps2'].values[all_stat_idx]
        ay = df_sync['phone_accel_y_mps2'].values[all_stat_idx]
        az = df_sync['phone_accel_z_mps2'].values[all_stat_idx]
        
        mean_ax, std_ax = float(np.mean(ax)), float(np.std(ax))
        mean_ay, std_ay = float(np.mean(ay)), float(np.std(ay))
        mean_az, std_az = float(np.mean(az)), float(np.std(az))
        norm_f = float(np.sqrt(mean_ax**2 + mean_ay**2 + mean_az**2))
        
        dir_x = mean_ax / norm_f
        dir_y = mean_ay / norm_f
        dir_z = mean_az / norm_f
        
        # dominant axis
        axes = [('X', dir_x, mean_ax), ('Y', dir_y, mean_ay), ('Z', dir_z, mean_az)]
        axes_sorted = sorted(axes, key=lambda item: abs(item[1]), reverse=True)
        dom_axis = f'{axes_sorted[0][0]} (sign={np.sign(axes_sorted[0][1]):+.0f})'
        
        print(f'    Stationary Accel: {len(all_stat_idx)} samples ({len(runs)} intervals >=5s)')
        print(f'      Mean: ax={mean_ax:+.3f}, ay={mean_ay:+.3f}, az={mean_az:+.3f} m/s^2')
        print(f'      Std:  sx={std_ax:.3f}, sy={std_ay:.3f}, sz={std_az:.3f} m/s^2')
        print(f'      Norm: ||f||={norm_f:.3f} m/s^2 (~{norm_f/9.80665:.3f} g)')
        print(f'      Normalized Gravity Dir: [{dir_x:+.3f}, {dir_y:+.3f}, {dir_z:+.3f}]')
        print(f'      Dominant Vertical UP Axis: {dom_axis}')
        
        accel_audit_rows.append({
            'recording': rec_id,
            'stat_samples': int(len(all_stat_idx)),
            'mean_ax': mean_ax, 'std_ax': std_ax,
            'mean_ay': mean_ay, 'std_ay': std_ay,
            'mean_az': mean_az, 'std_az': std_az,
            'norm_f': norm_f,
            'gravity_dir_x': dir_x, 'gravity_dir_y': dir_y, 'gravity_dir_z': dir_z,
            'dominant_vertical_axis': dom_axis
        })
    else:
        print('    Stationary Accel: NO valid stationary interval >= 5s found!')
        
    # 4. Gyroscope Axis Audit
    mask_gyro_moving = (vbox_spd_mps > 2.0)
    vbox_yaw_rate = df_sync['ref_yaw_rate_radps'].values[mask_gyro_moving]
    
    gx = df_sync['phone_gyro_x_radps'].values[mask_gyro_moving]
    gy = df_sync['phone_gyro_y_radps'].values[mask_gyro_moving]
    gz = df_sync['phone_gyro_z_radps'].values[mask_gyro_moving]
    
    channels = [
        ('phone_gyro_x', 'GYROSCOPE Yaw', gx),
        ('phone_gyro_y', 'GYROSCOPE Pitch', gy),
        ('phone_gyro_z', 'GYROSCOPE Roll', gz)
    ]
    
    print(f'    Gyroscope Channels vs VBOX Yaw Rate (moving > 2m/s, N={len(vbox_yaw_rate)}):')
    dom_gyro = None
    max_corr = -1.0
    
    for col_name, raw_hdr, g_val in channels:
        valid_idx = (~np.isnan(g_val)) & (~np.isnan(vbox_yaw_rate))
        r, _ = stats.pearsonr(g_val[valid_idx], vbox_yaw_rate[valid_idx])
        slope, intercept, _, _, _ = stats.linregress(vbox_yaw_rate[valid_idx], g_val[valid_idx])
        
        is_dom = abs(r) > max_corr
        if is_dom:
            max_corr = abs(r)
            dom_gyro = (col_name, raw_hdr, r, slope)
            
        phys = 'Dominant Yaw' if abs(r) > 0.7 else ('Secondary / Tilt' if abs(r) > 0.2 else 'Uncorrelated / Horizontal')
        print(f'      {col_name:12s} ({raw_hdr:15s}): r={r:+.3f}, slope={slope:+.3f}, sign={np.sign(r):+.0f} | {phys}')
        
        gyro_audit_rows.append({
            'recording': rec_id,
            'channel': col_name,
            'raw_header': raw_hdr,
            'samples': int(np.sum(valid_idx)),
            'pearson_r': float(r),
            'regression_slope': float(slope),
            'sign': int(np.sign(r)),
            'physical_role': phys
        })
    print(f'    ==> Dominant Yaw Channel for {rec_id}: {dom_gyro[0]} ({dom_gyro[1]}) with r={dom_gyro[2]:+.3f}, slope={dom_gyro[3]:+.3f}\n')

print('Writing master summary CSV...')
os.makedirs('data/processed/phase2_3b/experiments/E3', exist_ok=True)
pd.DataFrame(speed_audit_rows).to_csv('data/processed/phase2_3b/experiments/E3/e3_speed_audit.csv', index=False)
pd.DataFrame(accel_audit_rows).to_csv('data/processed/phase2_3b/experiments/E3/e3_accel_audit.csv', index=False)
pd.DataFrame(gyro_audit_rows).to_csv('data/processed/phase2_3b/experiments/E3/e3_gyro_audit.csv', index=False)
print('Done!')
