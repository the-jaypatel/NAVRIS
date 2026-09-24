import sys, os
sys.path.insert(0, 'src')
import numpy as np
import pandas as pd
from scipy import stats, signal

sync_dir = 'data/processed/synchronized'
targets = ['S1', 'S2', 'S3A', 'S4', 'M', 'Y1', 'VTA1A', 'VTA2']

results = []

print('=== PART C: REPRODUCING LAG DIAGNOSTICS ACROSS 8 RECORDINGS ===')

for rec in targets:
    fpath = os.path.join(sync_dir, f'{rec}_sync.parquet')
    df = pd.read_parquet(fpath)
    
    # 1. Signals
    v_vbox = df['ref_speed_mps'].values
    v_phone = df['phone_gps_speed_mps'].values * 3.6 # raw speed in m/s
    
    r_vbox = df['ref_yaw_rate_radps'].values
    r_phone = df['phone_gyro_y_radps'].values # dominant yaw channel
    
    # Longitudinal acceleration
    a_vbox = df['ref_accel_long_mps2'].values if 'ref_accel_long_mps2' in df.columns else np.gradient(v_vbox, 0.1)
    
    # Moving mask for yaw rate (speed > 2 m/s)
    mask_moving = (v_vbox > 2.0) & (~np.isnan(r_vbox)) & (~np.isnan(r_phone))
    
    # TEST 1: Gyro Yaw Rate
    r_v_clean = np.nan_to_num(r_vbox)
    r_p_clean = np.nan_to_num(r_phone)
    
    # Search range: -20s to +20s (-200 to +200 steps of 0.1s)
    lags = np.arange(-250, 251)
    pad = 300
    
    # Zero-lag gyro correlation
    valid_zero = mask_moving
    r_zero_gyro, _ = stats.pearsonr(r_phone[valid_zero], r_vbox[valid_zero])
    slope_zero_gyro = stats.linregress(r_vbox[valid_zero], r_phone[valid_zero]).slope
    
    corrs_gyro = []
    for lag in lags:
        r_p_shifted = np.roll(r_p_clean, lag)[pad:-pad]
        r_v_sub = r_v_clean[pad:-pad]
        c = np.corrcoef(r_p_shifted, r_v_sub)[0, 1]
        corrs_gyro.append(c)
        
    best_idx_g = np.argmax(corrs_gyro)
    best_lag_g = lags[best_idx_g]
    best_lag_g_s = best_lag_g * 0.1
    best_corr_g = corrs_gyro[best_idx_g]
    
    # Regression slope at best lag
    r_p_best = np.roll(r_phone, best_lag_g)
    slope_best_g = stats.linregress(r_vbox[mask_moving], r_p_best[mask_moving]).slope
    
    # TEST 2: Speed Correlation across lags
    v_v_clean = np.nan_to_num(v_vbox)
    v_p_clean = np.nan_to_num(v_phone)
    
    valid_spd_zero = (v_vbox > 0.5) & (~np.isnan(v_phone))
    r_zero_spd, _ = stats.pearsonr(v_phone[valid_spd_zero], v_vbox[valid_spd_zero])
    
    corrs_spd = []
    for lag in lags:
        v_p_shifted = np.roll(v_p_clean, lag)[pad:-pad]
        v_v_sub = v_v_clean[pad:-pad]
        c = np.corrcoef(v_p_shifted, v_v_sub)[0, 1]
        corrs_spd.append(c)
        
    best_idx_v = np.argmax(corrs_spd)
    best_lag_v = lags[best_idx_v]
    best_lag_v_s = best_lag_v * 0.1
    best_corr_v = corrs_spd[best_idx_v]
    
    # TEST 3: Dynamic Acceleration
    # Angle in phone XY
    dv = np.gradient(v_vbox, 0.1)
    ax = df['phone_accel_x_mps2'].values
    ay = df['phone_accel_y_mps2'].values
    
    # Align by best gyro lag
    ax_roll = np.roll(ax, best_lag_g)
    ay_roll = np.roll(ay, best_lag_g)
    mask_acc = (v_vbox > 5.0) & (np.abs(r_vbox) < 0.02) & (np.abs(dv) > 0.5)
    if np.sum(mask_acc) > 50:
        corr_ax = np.corrcoef(dv[mask_acc], ax_roll[mask_acc])[0, 1]
        corr_ay = np.corrcoef(dv[mask_acc], ay_roll[mask_acc])[0, 1]
    else:
        corr_ax, corr_ay = np.nan, np.nan
        
    print(f'Recording: {rec:6s}')
    print(f'  Gyro Yaw Rate: Zero-lag r={r_zero_gyro:+.3f} (slope={slope_zero_gyro:+.3f}) | Best Lag={best_lag_g_s:+.2f}s -> r={best_corr_g:+.3f} (slope={slope_best_g:+.3f})')
    print(f'  GNSS Speed:    Zero-lag r={r_zero_spd:+.3f} | Best Lag={best_lag_v_s:+.2f}s -> r={best_corr_v:+.3f}')
    print(f'  Dynamic Accel: ax corr={corr_ax:+.3f}, ay corr={corr_ay:+.3f} at lag={best_lag_g_s:+.2f}s')
    print()
    
    results.append({
        'recording': rec,
        'zero_lag_gyro_r': r_zero_gyro,
        'zero_lag_gyro_slope': slope_zero_gyro,
        'best_lag_gyro_s': best_lag_g_s,
        'best_corr_gyro_r': best_corr_g,
        'best_slope_gyro': slope_best_g,
        'zero_lag_speed_r': r_zero_spd,
        'best_lag_speed_s': best_lag_v_s,
        'best_corr_speed_r': best_corr_v,
        'accel_ax_corr': corr_ax,
        'accel_ay_corr': corr_ay
    })

df_res = pd.DataFrame(results)
os.makedirs('data/processed/phase2_3b/experiments/Gate1_5A', exist_ok=True)
df_res.to_csv('data/processed/phase2_3b/experiments/Gate1_5A/reproduced_lags.csv', index=False)
print('Saved data/processed/phase2_3b/experiments/Gate1_5A/reproduced_lags.csv')
