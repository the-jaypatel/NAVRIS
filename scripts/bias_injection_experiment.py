"""
NAVRIS Phase 2.2.5: Real-Data Controlled Bias Injection Experiment.

Executes:
- Experiment A: Controlled accelerometer bias injection (+/-0.01, +/-0.05, +/-0.10 m/s^2)
- Experiment B: Controlled gyroscope bias injection (+/-0.0001, +/-0.0005, +/-0.001 rad/s)
- Experiment C: Combined sensitivity case (0.05 m/s^2, 0.0005 rad/s)
Across representative real-world recordings: S1, S2, VTA2.
Measures: RMSE, final error, heading error, along/cross-track error, and empirical growth exponent n in e(t) ~ c * t^n.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from navris.inertial.segments import split_into_contiguous_segments
from navris.inertial.init import align_attitude_causal
from navris.inertial.strapdown import propagate_strapdown_segment
from navris.inertial.metrics import compute_dr_metrics


def fit_growth_exponent(time_s: np.ndarray, error_m: np.ndarray) -> float:
    """Estimates empirical growth exponent n where error(t) ~ c * t^n via log-log regression."""
    # Filter valid positive time and error after initial 10 seconds
    mask = (time_s > 10.0) & (error_m > 1.0)
    if np.sum(mask) < 20:
        return 0.0
    log_t = np.log(time_s[mask])
    log_e = np.log(error_m[mask])
    # Linear fit: log(e) = n * log(t) + const
    poly = np.polyfit(log_t, log_e, 1)
    return float(poly[0])


def run_bias_injection_study(
    rec_ids: List[str] = ['S1', 'S2', 'VTA2'],
    output_csv: str = 'data/processed/phase2/bias_injection_results.csv'
) -> pd.DataFrame:
    results = []

    for rid in rec_ids:
        parquet_path = f"data/processed/synchronized/{rid}_sync.parquet"
        if not os.path.exists(parquet_path):
            print(f"Skipping {rid}: file not found")
            continue
        
        df = pd.read_parquet(parquet_path)
        segs = split_into_contiguous_segments(df, max_dt=0.25, min_samples=100)
        seg_df = segs[0][2]
        causal_history = seg_df.iloc[:min(len(seg_df), 3500)]
        align_res = align_attitude_causal(causal_history, min_stat_samples=30)
        eval_df = seg_df.iloc[align_res.init_sample_idx:].copy().reset_index(drop=True)

        t = eval_df['time_s'].values - eval_df['time_s'].iloc[0]
        acc_b = eval_df[['phone_accel_x_mps2', 'phone_accel_y_mps2', 'phone_accel_z_mps2']].values.copy()
        gyr_b = eval_df[['phone_gyro_x_radps', 'phone_gyro_y_radps', 'phone_gyro_z_radps']].values.copy()

        p0 = np.array([eval_df['phone_gps_east_m'].iloc[0], eval_df['phone_gps_north_m'].iloc[0], 0.0])
        v0 = align_res.initial_velocity_enu
        q0 = align_res.q_b_n
        b_g = align_res.gyro_bias_radps

        # Base A0 propagation
        traj_base = propagate_strapdown_segment(
            time_s=t, accel_body=acc_b, gyro_body=gyr_b,
            p0=p0, v0=v0, q0=q0, gyro_bias=b_g
        )
        m_base = compute_dr_metrics(t, traj_base.pos_enu, traj_base.vel_enu, traj_base.heading_rad, eval_df)
        base_err = np.linalg.norm(traj_base.pos_enu[:, :2] - eval_df[['ref_east_m', 'ref_north_m']].values, axis=1)
        exp_base = fit_growth_exponent(t, base_err)

        results.append({
            'recording_id': rid,
            'experiment': 'Baseline_A0',
            'axis': 'none',
            'bias_value': 0.0,
            'unit': 'none',
            'horiz_rmse_m': m_base.horiz_rmse_m,
            'final_error_m': m_base.final_error_m,
            'heading_rmse_deg': m_base.heading_rmse_deg,
            'along_track_rmse_m': m_base.along_track_rmse_m,
            'cross_track_rmse_m': m_base.cross_track_rmse_m,
            'growth_exponent_n': exp_base
        })

        # -------------------------------------------------------------
        # Experiment A: Accelerometer Bias Injection
        # -------------------------------------------------------------
        acc_biases = [-0.10, -0.05, -0.01, 0.01, 0.05, 0.10]
        for ax_idx, ax_name in enumerate(['body_X', 'body_Y', 'body_Z']):
            for delta_a in acc_biases:
                acc_mod = acc_b.copy()
                acc_mod[:, ax_idx] += delta_a
                
                traj_mod = propagate_strapdown_segment(
                    time_s=t, accel_body=acc_mod, gyro_body=gyr_b,
                    p0=p0, v0=v0, q0=q0, gyro_bias=b_g
                )
                m_mod = compute_dr_metrics(t, traj_mod.pos_enu, traj_mod.vel_enu, traj_mod.heading_rad, eval_df)
                err_mod = np.linalg.norm(traj_mod.pos_enu[:, :2] - eval_df[['ref_east_m', 'ref_north_m']].values, axis=1)
                exp_mod = fit_growth_exponent(t, err_mod)

                results.append({
                    'recording_id': rid,
                    'experiment': 'Exp_A_Accel_Bias',
                    'axis': ax_name,
                    'bias_value': delta_a,
                    'unit': 'm/s^2',
                    'horiz_rmse_m': m_mod.horiz_rmse_m,
                    'final_error_m': m_mod.final_error_m,
                    'heading_rmse_deg': m_mod.heading_rmse_deg,
                    'along_track_rmse_m': m_mod.along_track_rmse_m,
                    'cross_track_rmse_m': m_mod.cross_track_rmse_m,
                    'growth_exponent_n': exp_mod
                })

        # -------------------------------------------------------------
        # Experiment B: Gyroscope Bias Injection
        # -------------------------------------------------------------
        gyr_biases = [-0.001, -0.0005, -0.0001, 0.0001, 0.0005, 0.001]
        for ax_idx, ax_name in enumerate(['body_X', 'body_Y', 'body_Z']):
            for delta_w in gyr_biases:
                gyr_mod = gyr_b.copy()
                gyr_mod[:, ax_idx] += delta_w

                traj_mod = propagate_strapdown_segment(
                    time_s=t, accel_body=acc_b, gyro_body=gyr_mod,
                    p0=p0, v0=v0, q0=q0, gyro_bias=b_g
                )
                m_mod = compute_dr_metrics(t, traj_mod.pos_enu, traj_mod.vel_enu, traj_mod.heading_rad, eval_df)
                err_mod = np.linalg.norm(traj_mod.pos_enu[:, :2] - eval_df[['ref_east_m', 'ref_north_m']].values, axis=1)
                exp_mod = fit_growth_exponent(t, err_mod)

                results.append({
                    'recording_id': rid,
                    'experiment': 'Exp_B_Gyro_Bias',
                    'axis': ax_name,
                    'bias_value': delta_w,
                    'unit': 'rad/s',
                    'horiz_rmse_m': m_mod.horiz_rmse_m,
                    'final_error_m': m_mod.final_error_m,
                    'heading_rmse_deg': m_mod.heading_rmse_deg,
                    'along_track_rmse_m': m_mod.along_track_rmse_m,
                    'cross_track_rmse_m': m_mod.cross_track_rmse_m,
                    'growth_exponent_n': exp_mod
                })

        # -------------------------------------------------------------
        # Experiment C: Combined Reference Sensitivity Case
        # -------------------------------------------------------------
        acc_comb = acc_b.copy()
        acc_comb[:, 0] += 0.05
        gyr_comb = gyr_b.copy()
        gyr_comb[:, 1] += 0.0005

        traj_comb = propagate_strapdown_segment(
            time_s=t, accel_body=acc_comb, gyro_body=gyr_comb,
            p0=p0, v0=v0, q0=q0, gyro_bias=b_g
        )
        m_comb = compute_dr_metrics(t, traj_comb.pos_enu, traj_comb.vel_enu, traj_comb.heading_rad, eval_df)
        err_comb = np.linalg.norm(traj_comb.pos_enu[:, :2] - eval_df[['ref_east_m', 'ref_north_m']].values, axis=1)
        exp_comb = fit_growth_exponent(t, err_comb)

        results.append({
            'recording_id': rid,
            'experiment': 'Exp_C_Combined_Sensitivity',
            'axis': 'X_acc_Y_gyr',
            'bias_value': 0.05,
            'unit': '0.05 m/s^2 + 0.0005 rad/s',
            'horiz_rmse_m': m_comb.horiz_rmse_m,
            'final_error_m': m_comb.final_error_m,
            'heading_rmse_deg': m_comb.heading_rmse_deg,
            'along_track_rmse_m': m_comb.along_track_rmse_m,
            'cross_track_rmse_m': m_comb.cross_track_rmse_m,
            'growth_exponent_n': exp_comb
        })
        print(f"Completed bias injection study for {rid}")

    res_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    res_df.to_csv(output_csv, index=False)
    print(f"Saved {len(res_df)} experiment records to {output_csv}")
    return res_df


if __name__ == '__main__':
    run_bias_injection_study()
