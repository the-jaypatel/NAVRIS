"""
NAVRIS Phase 2.2 Benchmark Experiment Runner.

Executes:
1. T0: Deployable Cold-Start DR (causal phone-only initialization)
2. T1: Controlled Free-Run Outage Windows (10s, 30s, 60s, 120s, 300s)
Across Baselines:
- A0: Deployable 3D Quaternion Strapdown INS (Causal Initialization)
- A1: Oracle Heading Diagnostic Baseline
- A2: Oracle Heading & Stationary Gyro Bias Diagnostic Baseline
- A-planar: 2D Kinematic Planar Baseline
- Ablations: Unscaled vs Stationary Normalized Accelerometer
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from navris.inertial.segments import split_into_contiguous_segments
from navris.inertial.stationary import estimate_stationary_imu_stats
from navris.inertial.init import align_attitude_causal
from navris.inertial.strapdown import run_strapdown_dr
from navris.inertial.planar import run_planar_dr
from navris.inertial.oracle import run_oracle_heading_dr, run_oracle_heading_bias_dr
from navris.inertial.metrics import compute_dr_metrics
from navris.inertial.plots import (
    plot_trajectory_comparison,
    plot_error_breakdown,
    plot_baselines_comparison
)


def run_recording_evaluation(
    rec_id: str,
    parquet_path: str,
    output_dir: str = 'data/processed/phase2',
    outage_durations: List[int] = [10, 30, 60, 120]
) -> Dict[str, Any]:
    """
    Evaluates a single synchronized recording under Phase 2.2 protocol.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_parquet(parquet_path)
    print(f"\nEvaluating Recording {rec_id} (Samples: {len(df)})")

    # 1. Split into contiguous segments (dt <= 0.25 s)
    segments = split_into_contiguous_segments(df, time_col='time_s', max_dt=0.25, min_samples=100)
    if not segments:
        print(f"  [SKIPPED] No contiguous segments >= 100 samples found.")
        return {'status': 'SKIPPED', 'recording_id': rec_id}

    # Primary continuous segment
    start_idx, end_idx, seg_df = segments[0]
    print(f"  Primary segment: {len(seg_df)} samples ({seg_df['time_s'].iloc[-1] - seg_df['time_s'].iloc[0]:.1f} s)")

    # 2. Causal Alignment (deployable A0)
    # Search causal history up to 3500 samples (~350 s) for stationary + initial straight moving segment
    causal_history = seg_df.iloc[:min(len(seg_df), 3500)]
    align_res = align_attitude_causal(causal_history, min_stat_samples=30)
    print(f"  Alignment: is_valid={align_res.is_valid}, quality={align_res.quality}, heading={np.degrees(align_res.initial_heading_rad):.1f} deg")

    init_idx = align_res.init_sample_idx
    eval_df = seg_df.iloc[init_idx:].copy().reset_index(drop=True)
    
    if len(eval_df) < 50:
        print("  [SKIPPED] Insufficient samples remaining after initialization.")
        return {'status': 'SKIPPED', 'recording_id': rec_id}

    # Initial position from phone GNSS at init_idx
    p0 = np.array([
        eval_df['phone_gps_east_m'].iloc[0],
        eval_df['phone_gps_north_m'].iloc[0],
        eval_df['phone_gps_up_m'].iloc[0] if 'phone_gps_up_m' in eval_df.columns else 0.0
    ], dtype=np.float64)
    v0 = align_res.initial_velocity_enu
    q0 = align_res.q_b_n
    b_g = align_res.gyro_bias_radps

    # -------------------------------------------------------------
    # BASELINE EXECUTION
    # -------------------------------------------------------------
    # A0: Deployable 3D DR
    traj_a0 = run_strapdown_dr(
        df=eval_df, p0=p0, v0=v0, q0=q0, gyro_bias=b_g,
        accel_scale='none', f_stationary_norm=align_res.f_stationary_norm
    )
    metrics_a0 = compute_dr_metrics(traj_a0.time_s, traj_a0.pos_enu, traj_a0.vel_enu, traj_a0.heading_rad, eval_df)

    # A0-normalized ablation
    traj_a0_norm = run_strapdown_dr(
        df=eval_df, p0=p0, v0=v0, q0=q0, gyro_bias=b_g,
        accel_scale='normalize_stationary_norm', f_stationary_norm=align_res.f_stationary_norm
    )
    metrics_a0_norm = compute_dr_metrics(traj_a0_norm.time_s, traj_a0_norm.pos_enu, traj_a0_norm.vel_enu, traj_a0_norm.heading_rad, eval_df)

    # A1: Oracle Heading Diagnostic
    q_level_stat = np.array(align_res.details['q_level'], dtype=np.float64)
    traj_a1 = run_oracle_heading_dr(
        df=eval_df, p0=p0, v0=v0, q_level=q_level_stat, gyro_bias=b_g,
        f_stationary_norm=align_res.f_stationary_norm
    )
    metrics_a1 = compute_dr_metrics(traj_a1.time_s, traj_a1.pos_enu, traj_a1.vel_enu, traj_a1.heading_rad, eval_df)

    # A2: Oracle Heading & Bias Diagnostic
    traj_a2 = run_oracle_heading_bias_dr(
        df=eval_df, p0=p0, v0=v0, q_level=q_level_stat,
        stationary_df=causal_history,
        f_stationary_norm=align_res.f_stationary_norm
    )
    metrics_a2 = compute_dr_metrics(traj_a2.time_s, traj_a2.pos_enu, traj_a2.vel_enu, traj_a2.heading_rad, eval_df)

    # A-planar: 2D Kinematic Control
    v0_mag = float(np.linalg.norm(v0[:2]))
    traj_planar = run_planar_dr(
        df=eval_df, p0=p0, v0_mps=v0_mag, heading0_rad=align_res.initial_heading_rad,
        yaw_rate_bias=float(b_g[1])
    )
    metrics_planar = compute_dr_metrics(traj_planar.time_s, traj_planar.pos_enu, traj_planar.vel_enu, traj_planar.heading_rad, eval_df)

    # -------------------------------------------------------------
    # T1: CONTROLLED OUTAGE WINDOW SWEEPS
    # -------------------------------------------------------------
    outage_results = {}
    for T in outage_durations:
        samples_T = int(T * 10)
        if len(eval_df) > samples_T + 50:
            window_df = eval_df.iloc[:samples_T].copy().reset_index(drop=True)
            traj_win = run_strapdown_dr(
                df=window_df, p0=p0, v0=v0, q0=q0, gyro_bias=b_g
            )
            m_win = compute_dr_metrics(traj_win.time_s, traj_win.pos_enu, traj_win.vel_enu, traj_win.heading_rad, window_df)
            outage_results[f'outage_{T}s_horiz_rmse_m'] = m_win.horiz_rmse_m
            outage_results[f'outage_{T}s_final_err_m'] = m_win.final_error_m
            outage_results[f'outage_{T}s_drift_rate_mps'] = m_win.drift_rate_mps

    # Generate Plots
    p1 = plot_trajectory_comparison(traj_a0, eval_df, rec_id, output_dir)
    p2 = plot_error_breakdown(metrics_a0, rec_id, output_dir)
    p3 = plot_baselines_comparison({
        'A0_Deployable_3D_DR': metrics_a0,
        'A1_Oracle_Heading': metrics_a1,
        'A2_Oracle_Heading_and_Bias': metrics_a2,
        'A_Planar_Control': metrics_planar
    }, rec_id, output_dir)

    print(f"  A0 Results: RMSE={metrics_a0.horiz_rmse_m:.2f} m | Final={metrics_a0.final_error_m:.2f} m | DriftRate={metrics_a0.drift_rate_mps:.2f} m/s")
    print(f"  A1 Oracle Heading: RMSE={metrics_a1.horiz_rmse_m:.2f} m | Final={metrics_a1.final_error_m:.2f} m")
    print(f"  A2 Oracle H+Bias:  RMSE={metrics_a2.horiz_rmse_m:.2f} m | Final={metrics_a2.final_error_m:.2f} m")
    print(f"  A-Planar Control:  RMSE={metrics_planar.horiz_rmse_m:.2f} m | Final={metrics_planar.final_error_m:.2f} m")

    return {
        'status': 'SUCCESS',
        'recording_id': rec_id,
        'samples': len(eval_df),
        'duration_s': metrics_a0.duration_s,
        'dist_traveled_m': metrics_a0.distance_traveled_m,
        'a0_horiz_rmse_m': metrics_a0.horiz_rmse_m,
        'a0_horiz_mae_m': metrics_a0.horiz_mae_m,
        'a0_final_err_m': metrics_a0.final_error_m,
        'a0_drift_rate_mps': metrics_a0.drift_rate_mps,
        'a0_drift_pct_dist': metrics_a0.drift_pct_distance,
        'a0_along_track_rmse_m': metrics_a0.along_track_rmse_m,
        'a0_cross_track_rmse_m': metrics_a0.cross_track_rmse_m,
        'a0_heading_rmse_deg': metrics_a0.heading_rmse_deg,
        'a0_norm_horiz_rmse_m': metrics_a0_norm.horiz_rmse_m,
        'a1_oracle_head_rmse_m': metrics_a1.horiz_rmse_m,
        'a2_oracle_all_rmse_m': metrics_a2.horiz_rmse_m,
        'planar_horiz_rmse_m': metrics_planar.horiz_rmse_m,
        'time_to_50m_s': metrics_a0.time_to_50m_s,
        'time_to_100m_s': metrics_a0.time_to_100m_s,
        'plots': [p1, p2, p3],
        **outage_results
    }


def main():
    parser = argparse.ArgumentParser(description="NAVRIS Phase 2.2 Baseline Runner")
    parser.add_argument('--recordings', type=str, default='S1,S2,M,Y1,VTA1A', help='Comma-separated recording IDs')
    parser.add_argument('--sync_dir', type=str, default='data/processed/synchronized')
    parser.add_argument('--output_dir', type=str, default='data/processed/phase2')
    args = parser.parse_args()

    rec_ids = [r.strip() for r in args.recordings.split(',') if r.strip()]
    results = []

    for rid in rec_ids:
        path = os.path.join(args.sync_dir, f"{rid}_sync.parquet")
        if not os.path.exists(path):
            print(f"Sync parquet not found for {rid} at {path}")
            continue
        res = run_recording_evaluation(rid, path, output_dir=args.output_dir)
        if res['status'] == 'SUCCESS':
            results.append(res)

    if results:
        df_res = pd.DataFrame(results)
        csv_path = os.path.join(args.output_dir, "phase22_baseline_metrics.csv")
        df_res.to_csv(csv_path, index=False)
        print("\n" + "=" * 80)
        print("PHASE 2.2 BENCHMARK RESULTS SUMMARY:")
        print("=" * 80)
        print(df_res[[
            'recording_id', 'duration_s', 'a0_horiz_rmse_m', 'a0_final_err_m',
            'a0_drift_rate_mps', 'a1_oracle_head_rmse_m', 'a2_oracle_all_rmse_m',
            'planar_horiz_rmse_m', 'time_to_50m_s'
        ]].to_string(index=False))
        print(f"\nFull results saved to {csv_path}")


if __name__ == '__main__':
    main()
