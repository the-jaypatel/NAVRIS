"""
NAVRIS Phase 2.3B Gate 1.5B: Controlled Offline Benchmark Synchronization Runner.

Processes the 8 benchmark recordings:
    S1, S2, S3A, S4, M, Y1, VTA1A, VTA2

For each recording:
1. Estimates offline benchmark lag using reference CAN/VBOX yaw rate and phone gyro Y
   under speed and yaw-rate excitation gating.
2. Validates 3-segment temporal stability.
3. For VTA1A, routes to untrimmed raw VBOX source (25,821 rows) to eliminate the 14.4s author truncation defect.
4. Generates time-aligned synchronized parquets in data/processed/synchronized/.
5. Evaluates pre- and post-alignment residual lags, correlation, and segment stability.
6. Saves detailed diagnostic CSV to data/processed/phase2_3b/experiments/Gate1_5B/gate1_5b_benchmark_sync_summary.csv.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from navris.pipeline import process_single_recording
from navris.ingest import ingest_reference_data, ingest_smartphone_data
from navris.sync import estimate_benchmark_lag


def main():
    target_ids = ['S1', 'S2', 'S3A', 'S4', 'M', 'Y1', 'VTA1A', 'VTA2']
    manifest_path = 'data/manifest/recordings_manifest.csv'
    out_dir = 'data/processed'
    exp_dir = os.path.join(out_dir, 'phase2_3b', 'experiments', 'Gate1_5B')
    os.makedirs(exp_dir, exist_ok=True)

    df_manifest = pd.read_csv(manifest_path)
    manifest_map = {row['recording_id']: row.to_dict() for _, row in df_manifest.iterrows()}

    records = []

    print("================================================================================")
    print("  NAVRIS Phase 2.3B Gate 1.5B — Controlled Offline Benchmark Synchronization   ")
    print("================================================================================")

    for rec_id in target_ids:
        if rec_id not in manifest_map:
            print(f"ERROR: {rec_id} not found in manifest!")
            continue

        entry = manifest_map[rec_id]
        print(f"\nProcessing Benchmark Recording: {rec_id} (Driver: {entry['driver']})")

        # 1. Run pipeline with auto_align_benchmark=True
        res = process_single_recording(
            rec_id=rec_id,
            manifest_entry=entry,
            output_dir=out_dir,
            auto_align_benchmark=True
        )

        sync_meta = res.get('sync_metadata', {})
        est_lag_s = sync_meta.get('estimated_lag_s', 0.0)
        zero_r = sync_meta.get('zero_lag_corr', np.nan)
        aligned_r = sync_meta.get('aligned_corr', np.nan)
        zero_slope = sync_meta.get('zero_lag_slope', np.nan)
        aligned_slope = sync_meta.get('aligned_slope', np.nan)
        seg_lags = sync_meta.get('segment_lags', [])
        seg_spread = sync_meta.get('segment_spread_s', 0.0)

        print(f"  Applied Offset:      {est_lag_s:+.2f} s")
        print(f"  Gyro Yaw Corr:       Zero-lag r={zero_r:+.4f} -> Aligned r={aligned_r:+.4f}")
        print(f"  Regression Slope:    Zero-lag={zero_slope:+.4f} -> Aligned={aligned_slope:+.4f}")
        print(f"  3-Segment Lags:      {seg_lags} (Spread: {seg_spread:.2f} s)")
        print(f"  Duration:            {res['duration_s']:.1f} s ({res['num_samples']} rows @ 10 Hz)")
        print(f"  Sparse GPS Fixes:    {res['num_gps_fixes']}")

        # 2. Post-alignment verification on newly saved parquet
        sync_parquet = res['sync_parquet_path']
        df_sync = pd.read_parquet(sync_parquet)
        
        # Verify residual lag on synchronized parquet
        v_vbox = df_sync['ref_speed_mps'].values
        r_vbox = df_sync['ref_yaw_rate_radps'].values
        r_phone = df_sync['phone_gyro_y_radps'].values
        
        mask_eval = (v_vbox > 3.0) & (np.abs(r_vbox) > 0.05) & (~np.isnan(r_vbox)) & (~np.isnan(r_phone))
        post_corr = float(np.corrcoef(r_phone[mask_eval], r_vbox[mask_eval])[0, 1]) if np.sum(mask_eval) > 30 else np.nan
        post_slope = float(np.polyfit(r_vbox[mask_eval], r_phone[mask_eval], 1)[0]) if np.sum(mask_eval) > 30 else np.nan

        # Test fine residual lag around 0: [-2.0s, +2.0s] in 0.1s steps
        fine_lags = np.round(np.arange(-2.0, 2.05, 0.1), 2)
        fine_corrs = []
        for fl in fine_lags:
            fl_steps = int(round(fl / 0.1))
            r_p_roll = np.roll(r_phone, fl_steps)
            # Exclude boundary wrap-around
            valid_b = mask_eval.copy()
            if fl_steps > 0:
                valid_b[:fl_steps] = False
            elif fl_steps < 0:
                valid_b[fl_steps:] = False
            if np.sum(valid_b) > 30:
                fine_corrs.append(np.corrcoef(r_p_roll[valid_b], r_vbox[valid_b])[0, 1])
            else:
                fine_corrs.append(-1.0)
        best_post_lag = fine_lags[np.argmax(fine_corrs)]
        print(f"  Post-Sync Check:     Residual Lag={best_post_lag:+.2f} s | In-parquet Corr={post_corr:+.4f} | Slope={post_slope:+.4f}")

        # GNSS speed check
        gnss_speeds = df_sync['phone_gps_obs_speed_mps'].dropna().values
        mean_gps_spd = float(np.mean(gnss_speeds)) if len(gnss_speeds) > 0 else np.nan
        ref_speeds = df_sync.loc[df_sync['phone_gps_obs_speed_mps'].notna(), 'ref_speed_mps'].values
        mean_ref_spd = float(np.mean(ref_speeds)) if len(ref_speeds) > 0 else np.nan
        gps_speed_ratio = float(mean_gps_spd / mean_ref_spd) if (mean_ref_spd > 0.1 and not np.isnan(mean_gps_spd)) else np.nan

        rec_dict = {
            'recording_id': rec_id,
            'driver': entry['driver'],
            'vehicle': entry['vehicle'],
            'applied_lag_s': est_lag_s,
            'pre_corr': zero_r,
            'pre_slope': zero_slope,
            'post_corr': aligned_r,
            'post_slope': aligned_slope,
            'seg1_lag_s': seg_lags[0] if len(seg_lags) > 0 else np.nan,
            'seg2_lag_s': seg_lags[1] if len(seg_lags) > 1 else np.nan,
            'seg3_lag_s': seg_lags[2] if len(seg_lags) > 2 else np.nan,
            'seg_spread_s': seg_spread,
            'post_residual_lag_s': best_post_lag,
            'duration_s': res['duration_s'],
            'num_samples': res['num_samples'],
            'num_gps_fixes': res['num_gps_fixes'],
            'mean_phone_gps_speed_mps': mean_gps_spd,
            'mean_ref_speed_mps': mean_ref_spd,
            'gps_speed_ratio': gps_speed_ratio,
            'parquet_path': sync_parquet
        }
        records.append(rec_dict)

    df_summary = pd.DataFrame(records)
    summary_path = os.path.join(exp_dir, 'gate1_5b_benchmark_sync_summary.csv')
    df_summary.to_csv(summary_path, index=False)
    print("\n================================================================================")
    print(f"Benchmark synchronization completed. Summary saved to:\n  {summary_path}")
    print("================================================================================")
    print(df_summary[['recording_id', 'applied_lag_s', 'pre_corr', 'post_corr', 'post_slope', 'seg_spread_s', 'post_residual_lag_s', 'gps_speed_ratio']].to_string(index=False))


if __name__ == '__main__':
    main()
