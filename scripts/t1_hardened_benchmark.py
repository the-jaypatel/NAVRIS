"""
NAVRIS Phase 2.2.5: Hardened Multi-Instance T1 Outage Benchmark & Decomposition.

Executes:
1. Multi-instance outage sampling (every ~5 minutes of active driving) across all 8 recordings.
2. Controlled outage durations: 10s, 30s, 60s, 120s.
3. GNSS fix offset decomposition: Initial GNSS offset vs INS-induced displacement error vs Total error.
4. Cold-Start vs Warm-State outage evaluation.
5. Clustered statistical aggregation: Median, IQR, Mean, Min, Max, and Bootstrap 95% CI.
Generates:
- data/processed/phase2/phase22_t1_hardened.csv (all instance records)
- data/processed/phase2/phase22_t1_summary.csv (aggregated statistics)
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from navris.inertial.segments import split_into_contiguous_segments
from navris.inertial.init import align_attitude_causal
from navris.inertial.strapdown import propagate_strapdown_segment, run_strapdown_dr
from navris.inertial.metrics import compute_dr_metrics
from navris.inertial.gnss_fixes import filter_novel_gnss_fixes


def run_hardened_t1_benchmark(
    sync_dir: str = 'data/processed/synchronized',
    output_instance_csv: str = 'data/processed/phase2/phase22_t1_hardened.csv',
    output_summary_csv: str = 'data/processed/phase2/phase22_t1_summary.csv',
    sample_interval_s: float = 300.0, # every 5 minutes
    durations: List[int] = [10, 30, 60, 120]
) -> pd.DataFrame:
    manifest = pd.read_csv('data/manifest/recordings_manifest.csv')
    rec_ids = ['S1', 'S2', 'S3A', 'S4', 'M', 'Y1', 'VTA1A', 'VTA2']

    instance_records = []

    for rid in rec_ids:
        path = os.path.join(sync_dir, f"{rid}_sync.parquet")
        if not os.path.exists(path):
            continue

        df = pd.read_parquet(path)
        meta_row = manifest[manifest['recording_id'] == rid].iloc[0] if len(manifest[manifest['recording_id'] == rid]) > 0 else {}
        driver = meta_row.get('driver', 'Unknown')
        vehicle = meta_row.get('vehicle', 'Unknown')
        phone = meta_row.get('phone', 'Unknown')

        segs = split_into_contiguous_segments(df, max_dt=0.25, min_samples=100)
        if not segs:
            continue
        seg_df = segs[0][2]
        causal_history = seg_df.iloc[:min(len(seg_df), 3500)]
        align_res = align_attitude_causal(causal_history, min_stat_samples=30)
        eval_df = seg_df.iloc[align_res.init_sample_idx:].copy().reset_index(drop=True)

        # Run continuous warm-state strapdown baseline across eval_df
        p0_full = np.array([eval_df['phone_gps_east_m'].iloc[0], eval_df['phone_gps_north_m'].iloc[0], 0.0])
        v0_full = align_res.initial_velocity_enu
        q0_full = align_res.q_b_n
        b_g = align_res.gyro_bias_radps

        traj_warm_full = run_strapdown_dr(
            df=eval_df, p0=p0_full, v0=v0_full, q0=q0_full, gyro_bias=b_g
        )

        # Find valid candidate start times every sample_interval_s (300 s)
        total_time = eval_df['time_s'].iloc[-1] - eval_df['time_s'].iloc[0]
        max_duration = max(durations)

        # Sample epochs starting from t = 30s
        current_t = eval_df['time_s'].iloc[0] + 30.0
        candidate_epochs = []
        while current_t < eval_df['time_s'].iloc[-1] - max_duration - 30.0:
            candidate_epochs.append(current_t)
            current_t += sample_interval_s

        for t_target in candidate_epochs:
            # Find closest index
            idx_start = int(np.argmin(np.abs(eval_df['time_s'].values - t_target)))
            
            # Check motion validity (vehicle must be moving, ref_speed > 2.0 m/s)
            speed_at_start = float(eval_df['ref_speed_mps'].iloc[idx_start])
            if speed_at_start < 2.0:
                continue

            for T in durations:
                samples_T = int(T * 10)
                idx_end = idx_start + samples_T
                if idx_end >= len(eval_df):
                    continue

                sub_win = eval_df.iloc[idx_start:idx_end].copy().reset_index(drop=True)
                dt_checks = np.diff(sub_win['time_s'].values)
                if np.any(dt_checks > 0.25) or np.any(dt_checks <= 0.0):
                    continue

                t_win = sub_win['time_s'].values - sub_win['time_s'].iloc[0]
                dist_traveled = float(np.sum(np.sqrt(np.diff(sub_win['ref_east_m'])**2 + np.diff(sub_win['ref_north_m'])**2)))

                # Initial GNSS and Ref positions
                p_gnss_start = np.array([sub_win['phone_gps_east_m'].iloc[0], sub_win['phone_gps_north_m'].iloc[0], 0.0])
                p_ref_start = np.array([sub_win['ref_east_m'].iloc[0], sub_win['ref_north_m'].iloc[0], 0.0])
                initial_gnss_offset = float(np.linalg.norm(p_gnss_start[:2] - p_ref_start[:2]))

                # ---------------------------------------------------------
                # 1. Cold-Start T1 Outage (Attitude re-leveled from COG at outage epoch)
                # ---------------------------------------------------------
                head0_cold = float(sub_win['phone_gps_orientation_rad'].iloc[0]) if not np.isnan(sub_win['phone_gps_orientation_rad'].iloc[0]) else float(sub_win['ref_heading_rad'].iloc[0])
                theta_cold = np.pi / 2.0 - head0_cold
                q_yaw_cold = np.array([np.cos(0.5 * theta_cold), 0, 0, np.sin(0.5 * theta_cold)])
                q0_cold = align_attitude_causal(eval_df.iloc[:min(len(eval_df), 100)]).q_b_n # base leveling

                v_speed_start = float(sub_win['phone_gps_speed_mps'].iloc[0])
                v0_cold = np.array([v_speed_start * np.sin(head0_cold), v_speed_start * np.cos(head0_cold), 0.0])

                traj_cold = run_strapdown_dr(
                    df=sub_win, p0=p_gnss_start, v0=v0_cold, q0=q0_cold, gyro_bias=b_g
                )
                m_cold = compute_dr_metrics(t_win, traj_cold.pos_enu, traj_cold.vel_enu, traj_cold.heading_rad, sub_win)

                # INS-induced displacement error
                delta_p_dr = traj_cold.pos_enu[-1, :2] - traj_cold.pos_enu[0, :2]
                delta_p_ref = sub_win[['ref_east_m', 'ref_north_m']].iloc[-1].values - sub_win[['ref_east_m', 'ref_north_m']].iloc[0].values
                ins_displacement_err = float(np.linalg.norm(delta_p_dr - delta_p_ref))

                # ---------------------------------------------------------
                # 2. Warm-State T1 Outage (Attitude from running propagation)
                # ---------------------------------------------------------
                # In warm state, attitude is taken from ongoing strapdown at idx_start
                q0_warm = traj_warm_full.quat_b_n[idx_start]
                v0_warm = v0_cold # reset to latest GNSS fix speed/heading
                traj_warm = run_strapdown_dr(
                    df=sub_win, p0=p_gnss_start, v0=v0_warm, q0=q0_warm, gyro_bias=b_g
                )
                m_warm = compute_dr_metrics(t_win, traj_warm.pos_enu, traj_warm.vel_enu, traj_warm.heading_rad, sub_win)

                instance_records.append({
                    'recording_id': rid,
                    'driver': driver,
                    'vehicle': vehicle,
                    'phone': phone,
                    'outage_duration_s': T,
                    'outage_start_time_s': float(sub_win['time_s'].iloc[0]),
                    'outage_start_idx': idx_start,
                    'speed_at_start_mps': speed_at_start,
                    'distance_traveled_m': dist_traveled,
                    'initial_gnss_offset_m': initial_gnss_offset,
                    'ins_displacement_err_m': ins_displacement_err,
                    'cold_horiz_rmse_m': m_cold.horiz_rmse_m,
                    'cold_final_error_m': m_cold.final_error_m,
                    'cold_along_track_rmse_m': m_cold.along_track_rmse_m,
                    'cold_cross_track_rmse_m': m_cold.cross_track_rmse_m,
                    'warm_horiz_rmse_m': m_warm.horiz_rmse_m,
                    'warm_final_error_m': m_warm.final_error_m,
                    'is_warm_state': bool(sub_win['time_s'].iloc[0] >= 600.0)
                })

    df_inst = pd.DataFrame(instance_records)
    os.makedirs(os.path.dirname(output_instance_csv), exist_ok=True)
    df_inst.to_csv(output_instance_csv, index=False)
    print(f"Saved {len(df_inst)} T1 outage instances to {output_instance_csv}")

    # Compute Statistical Summary
    summary_rows = []
    for T in durations:
        sub_T = df_inst[df_inst['outage_duration_s'] == T]
        if len(sub_T) == 0:
            continue
        
        # Cold-Start stats
        c_rmse = sub_T['cold_horiz_rmse_m'].values
        c_final = sub_T['cold_final_error_m'].values
        c_ins_err = sub_T['ins_displacement_err_m'].values
        init_offset = sub_T['initial_gnss_offset_m'].values

        # Warm-State stats (for t >= 600s)
        sub_warm = sub_T[sub_T['is_warm_state']]
        w_rmse = sub_warm['warm_horiz_rmse_m'].values if len(sub_warm) > 0 else np.array([np.nan])

        # Clustered Bootstrap 95% CI on Cold RMSE (resampling by recording)
        unique_recs = sub_T['recording_id'].unique()
        boot_means = []
        np.random.seed(42)
        for _ in range(1000):
            boot_recs = np.random.choice(unique_recs, size=len(unique_recs), replace=True)
            boot_samples = sub_T[sub_T['recording_id'].isin(boot_recs)]['cold_horiz_rmse_m'].values
            if len(boot_samples) > 0:
                boot_means.append(np.mean(boot_samples))
        ci_low = float(np.percentile(boot_means, 2.5)) if boot_means else np.nan
        ci_high = float(np.percentile(boot_means, 97.5)) if boot_means else np.nan

        summary_rows.append({
            'outage_duration_s': T,
            'num_recordings': len(unique_recs),
            'total_instances': len(sub_T),
            'init_gnss_offset_mean_m': float(np.mean(init_offset)),
            'init_gnss_offset_median_m': float(np.median(init_offset)),
            'ins_displacement_err_mean_m': float(np.mean(c_ins_err)),
            'ins_displacement_err_median_m': float(np.median(c_ins_err)),
            'cold_rmse_mean_m': float(np.mean(c_rmse)),
            'cold_rmse_median_m': float(np.median(c_rmse)),
            'cold_rmse_iqr_m': float(np.percentile(c_rmse, 75) - np.percentile(c_rmse, 25)),
            'cold_rmse_min_m': float(np.min(c_rmse)),
            'cold_rmse_max_m': float(np.max(c_rmse)),
            'cold_rmse_boot_ci95_low': ci_low,
            'cold_rmse_boot_ci95_high': ci_high,
            'cold_final_err_mean_m': float(np.mean(c_final)),
            'cold_final_err_median_m': float(np.median(c_final)),
            'warm_rmse_mean_m': float(np.nanmean(w_rmse)),
            'warm_rmse_median_m': float(np.nanmedian(w_rmse)),
            'warm_instances_count': len(sub_warm)
        })

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(output_summary_csv, index=False)
    print(f"Saved statistical summary to {output_summary_csv}")
    return df_inst, df_sum


if __name__ == '__main__':
    run_hardened_t1_benchmark()
