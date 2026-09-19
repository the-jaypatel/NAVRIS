"""
NAVRIS Phase 2.2.5: Timestamp Sensitivity Experiment.

Tests the sensitivity of 3D strapdown dead reckoning to relative timing offsets
between the gyroscope stream and the accelerometer stream.
Tested relative offsets: -200 ms, -100 ms, -50 ms, 0 ms, +50 ms, +100 ms, +200 ms.
Evaluated on representative recordings: S1 and S2.
Measures: Horizontal RMSE, Final Error, Heading RMSE, and Velocity RMSE.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from navris.inertial.segments import split_into_contiguous_segments
from navris.inertial.init import align_attitude_causal
from navris.inertial.strapdown import propagate_strapdown_segment
from navris.inertial.metrics import compute_dr_metrics


def run_timestamp_sensitivity(
    rec_ids = ['S1', 'S2'],
    offsets_ms = [-200, -100, -50, 0, 50, 100, 200],
    output_csv = 'data/processed/phase2/timestamp_sensitivity_results.csv'
):
    results = []

    for rid in rec_ids:
        parquet_path = f"data/processed/synchronized/{rid}_sync.parquet"
        if not os.path.exists(parquet_path):
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

        # Base interpolator for gyro
        gyro_interp = interp1d(t, gyr_b, axis=0, bounds_error=False, fill_value='extrapolate')

        for offset_ms in offsets_ms:
            offset_s = offset_ms / 1000.0
            # Shift gyro time relative to accel: t_gyro = t + offset_s
            gyr_shifted = gyro_interp(t + offset_s)

            traj = propagate_strapdown_segment(
                time_s=t, accel_body=acc_b, gyro_body=gyr_shifted,
                p0=p0, v0=v0, q0=q0, gyro_bias=b_g
            )
            m = compute_dr_metrics(t, traj.pos_enu, traj.vel_enu, traj.heading_rad, eval_df)

            results.append({
                'recording_id': rid,
                'offset_ms': offset_ms,
                'offset_s': offset_s,
                'horiz_rmse_m': m.horiz_rmse_m,
                'final_error_m': m.final_error_m,
                'heading_rmse_deg': m.heading_rmse_deg,
                'along_track_rmse_m': m.along_track_rmse_m,
                'cross_track_rmse_m': m.cross_track_rmse_m
            })
            print(f"{rid} (offset {offset_ms:+4d} ms): RMSE = {m.horiz_rmse_m:12.1f} m | Final = {m.final_error_m:12.1f} m | Heading RMSE = {m.heading_rmse_deg:5.1f} deg")

    res_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    res_df.to_csv(output_csv, index=False)
    print(f"Saved timestamp sensitivity results to {output_csv}")
    return res_df


if __name__ == '__main__':
    run_timestamp_sensitivity()
