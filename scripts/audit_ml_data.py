"""
Phase 3.0 Step 1 — Audit Available ML Data.
Inspects synchronized IO-VNBD recordings and validated NAVRIS outputs.
"""

from pathlib import Path
import numpy as np
import pandas as pd

gate_dir = Path("data/processed/phase2_3b/gate2_3b")
sync_dir = Path("data/processed/synchronized")

recordings = ["S1", "S2", "S3A", "S4", "M", "Y1", "VTA1A", "VTA2"]

results = []

for rec in recordings:
    traj_p = gate_dir / f"{rec}_trajectory_comparison.csv"
    sync_p = sync_dir / f"{rec}_sync.parquet"

    print("=" * 60)
    print(f"Recording: {rec}")

    if traj_p.exists() and sync_p.exists():
        df_traj = pd.read_csv(traj_p)
        df_sync = pd.read_parquet(sync_p)

        t_traj = df_traj["time_s"].values
        t_sync = df_sync["time_s"].values

        dt_traj = np.diff(t_traj)
        dt_sync = np.diff(t_sync)

        t_start = max(t_traj[0], t_sync[0])
        t_end = min(t_traj[-1], t_sync[-1])
        overlap_dur = t_end - t_start

        # Filter to overlapping interval
        m_traj = (t_traj >= t_start) & (t_traj <= t_end)
        m_sync = (t_sync >= t_start) & (t_sync <= t_end)

        # Check missing values in key columns
        imu_cols = [
            "phone_accel_x_mps2", "phone_accel_y_mps2", "phone_accel_z_mps2",
            "phone_gyro_x_radps", "phone_gyro_y_radps", "phone_gyro_z_radps"
        ]
        gnss_cols = [
            "phone_gps_east_m", "phone_gps_north_m", "phone_gps_up_m",
            "phone_gps_is_valid", "phone_gps_is_new_fix"
        ]
        traj_cols = [
            "ref_east_m", "ref_north_m", "ref_up_m",
            "ref_speed_mps", "ref_heading_rad",
            "pos_B_east_m", "pos_B_north_m", "pos_B_up_m",
            "vel_B_east_mps", "vel_B_north_mps", "vel_B_up_mps",
            "head_B_rad", "horiz_err_B_m"
        ]

        imu_nan = df_sync.loc[m_sync, imu_cols].isna().sum().to_dict()
        gnss_nan = df_sync.loc[m_sync, gnss_cols].isna().sum().to_dict()
        traj_nan = df_traj.loc[m_traj, traj_cols].isna().sum().to_dict()

        row_info = {
            "rec": rec,
            "status": "USABLE",
            "traj_samples": len(df_traj),
            "sync_samples": len(df_sync),
            "overlap_samples": int(np.sum(m_traj)),
            "sample_rate_hz": round(1.0 / dt_traj.mean(), 2),
            "duration_s": round(overlap_dur, 2),
            "t_start": round(t_start, 2),
            "t_end": round(t_end, 2),
            "imu_nan_max": max(imu_nan.values()),
            "traj_nan_max": max(traj_nan.values()),
            "gnss_valid_pct": round(100.0 * df_sync.loc[m_sync, "phone_gps_is_valid"].mean(), 2),
            "max_horiz_err_B": round(df_traj["horiz_err_B_m"].max(), 2),
            "median_horiz_err_B": round(df_traj["horiz_err_B_m"].median(), 2),
        }
        results.append(row_info)

        print(f"  Status: USABLE")
        print(f"  Samples: {row_info['overlap_samples']} | Rate: {row_info['sample_rate_hz']} Hz | Duration: {row_info['duration_s']} s")
        print(f"  Time Span: [{row_info['t_start']}, {row_info['t_end']}] s")
        print(f"  GNSS Valid %: {row_info['gnss_valid_pct']}%")
        print(f"  Traj NaNs: {row_info['traj_nan_max']} | IMU NaNs: {row_info['imu_nan_max']}")
        print(f"  Horiz Err B: median={row_info['median_horiz_err_B']} m, max={row_info['max_horiz_err_B']} m")

    elif sync_p.exists():
        df_sync = pd.read_parquet(sync_p)
        t_sync = df_sync["time_s"].values
        results.append({
            "rec": rec,
            "status": "EXCLUDED (UNOBSERVABLE)",
            "traj_samples": 0,
            "sync_samples": len(df_sync),
            "overlap_samples": 0,
            "sample_rate_hz": round(1.0 / np.diff(t_sync).mean(), 2),
            "duration_s": round(t_sync[-1] - t_sync[0], 2),
            "t_start": round(t_sync[0], 2),
            "t_end": round(t_sync[-1], 2),
            "imu_nan_max": 0,
            "traj_nan_max": 0,
            "gnss_valid_pct": 0.0,
            "max_horiz_err_B": None,
            "median_horiz_err_B": None,
        })
        print(f"  Status: EXCLUDED (UNOBSERVABLE in Gate 2.3B baseline)")
        print(f"  Sync samples: {len(df_sync)}, Duration: {t_sync[-1] - t_sync[0]:.2f} s")
    else:
        print(f"  Status: MISSING")

df_results = pd.DataFrame(results)
print("\n" + "=" * 60)
print("AUDIT SUMMARY TABLE:")
print(df_results.to_string(index=False))
