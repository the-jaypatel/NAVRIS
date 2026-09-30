"""
NAVRIS Phase 3.1: ML Target Observability and Formulation Study.

Rigorous evaluation of candidate ML targets across 7 validated benchmark recordings:
- Candidate A: Body-Frame Forward Speed Residual
- Candidate B: Short-Horizon Velocity Error and Increment
- Candidate C: Local Velocity Error Growth Rate (Derivative)
- Candidate D: Body-Frame IMU Error / Bias Proxies
- Candidate E: Navigation Correction / Innovation Magnitude

Computes:
1. Summary distributions (min, max, mean, std, median, p95, p99, RMSE).
2. Driving condition breakdown (stationary, low-speed, normal, high-speed, turns, braking, accel).
3. Stationarity analysis across outage duration bins (0-5s, 5-30s, 30-120s, >120s).
4. Feature relationship study (Pearson & Spearman correlations with causal feature groups).
5. Cross-recording partition comparisons (Train vs Validation vs Held-Out Test).
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats


VALID_RECORDINGS = ["S1", "S2", "S4", "S3A", "VTA1A", "VTA2", "Y1"]
TRAIN_RECS = ["S1", "S2", "S4"]
VAL_RECS = ["S3A", "VTA1A"]
TEST_RECS = ["Y1", "VTA2"]


def compute_distribution_stats(arr: np.ndarray) -> Dict[str, float]:
    """Computes comprehensive distribution statistics for a 1D array."""
    clean = arr[~np.isnan(arr)]
    if len(clean) == 0:
        return {
            "count": 0, "min": float("nan"), "max": float("nan"), "mean": float("nan"),
            "std": float("nan"), "median": float("nan"), "p95": float("nan"),
            "p99": float("nan"), "rmse": float("nan")
        }
    return {
        "count": int(len(clean)),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean)),
        "median": float(np.median(clean)),
        "p95": float(np.percentile(clean, 95)),
        "p99": float(np.percentile(clean, 99)),
        "rmse": float(np.sqrt(np.mean(clean**2))),
    }


def load_and_preprocess_recording(
    rec_id: str,
    sync_dir: str = "data/processed/synchronized",
    gate_dir: str = "data/processed/phase2_3b/gate2_3b",
) -> pd.DataFrame:
    """
    Loads and aligns synchronized sensor data and classical trajectory data.
    Constructs all candidate targets and strictly causal features.
    """
    sync_p = Path(sync_dir) / f"{rec_id}_sync.parquet"
    traj_p = Path(gate_dir) / f"{rec_id}_trajectory_comparison.csv"

    df_sync = pd.read_parquet(sync_p)
    df_traj = pd.read_csv(traj_p)

    t_start = df_traj["time_s"].iloc[0]
    t_end = df_traj["time_s"].iloc[-1]
    mask = (df_sync["time_s"] >= t_start - 1e-4) & (df_sync["time_s"] <= t_end + 1e-4)
    df_sync = df_sync.loc[mask].reset_index(drop=True)
    df_traj = df_traj.reset_index(drop=True)

    assert len(df_sync) == len(df_traj), f"Length mismatch for {rec_id}"

    t = df_traj["time_s"].values
    dt = np.gradient(t)
    dt[dt <= 0] = 0.1

    # Reference kinematics
    ref_spd = df_traj["ref_speed_mps"].values
    ref_hdg = df_traj["ref_heading_rad"].values
    ref_v_u = df_sync["ref_vertical_speed_mps"].values
    ref_v_e = ref_spd * np.sin(ref_hdg)
    ref_v_n = ref_spd * np.cos(ref_hdg)

    # Reference accelerations (m/s^2)
    ref_acc_long = df_sync["ref_accel_long_mps2"].values if "ref_accel_long_mps2" in df_sync.columns else np.zeros_like(t)
    ref_acc_lat = df_sync["ref_accel_lat_mps2"].values if "ref_accel_lat_mps2" in df_sync.columns else np.zeros_like(t)
    ref_yaw_rate = df_sync["ref_yaw_rate_radps"].values if "ref_yaw_rate_radps" in df_sync.columns else np.zeros_like(t)

    # NAVRIS Baseline A estimates
    nav_v_e = df_traj["vel_A_east_mps"].values
    nav_v_n = df_traj["vel_A_north_mps"].values
    nav_v_u = df_traj["vel_A_up_mps"].values
    nav_spd = np.sqrt(nav_v_e**2 + nav_v_n**2)
    nav_hdg = df_traj["head_A_rad"].values if "head_A_rad" in df_traj.columns else df_traj["eskf_heading_rad"].values

    # Classical velocity residuals in ENU frame
    delta_v_e = ref_v_e - nav_v_e
    delta_v_n = ref_v_n - nav_v_n
    delta_v_u = ref_v_u - nav_v_u
    delta_v_horiz = np.sqrt(delta_v_e**2 + delta_v_n**2)
    delta_v_3d = np.sqrt(delta_v_e**2 + delta_v_n**2 + delta_v_u**2)

    # Causal forward speed projection along estimated heading
    nav_v_fwd = nav_v_e * np.sin(nav_hdg) + nav_v_n * np.cos(nav_hdg)

    # ==================== CANDIDATE TARGETS ====================

    # Candidate A: Body-Frame Forward Speed Residual
    target_A_fwd = ref_spd - nav_v_fwd

    # Candidate B & C: Increments and Growth Rates over horizons
    # H = 0.5s (5 steps), 1.0s (10 steps), 2.0s (20 steps), 5.0s (50 steps)
    horizons = {"0_5s": 5, "1_0s": 10, "2_0s": 20, "5_0s": 50}
    target_B_inc = {}
    target_B_fut = {}
    target_C_rate = {}

    n = len(t)
    for h_name, h_steps in horizons.items():
        h_sec = h_steps * 0.1
        # Increments: delta_v(t + H) - delta_v(t)
        inc_e = np.full(n, np.nan)
        inc_n = np.full(n, np.nan)
        inc_u = np.full(n, np.nan)
        inc_e[:-h_steps] = delta_v_e[h_steps:] - delta_v_e[:-h_steps]
        inc_n[:-h_steps] = delta_v_n[h_steps:] - delta_v_n[:-h_steps]
        inc_u[:-h_steps] = delta_v_u[h_steps:] - delta_v_u[:-h_steps]
        inc_horiz = np.sqrt(inc_e**2 + inc_n**2)
        inc_3d = np.sqrt(inc_e**2 + inc_n**2 + inc_u**2)

        target_B_inc[f"inc_horiz_{h_name}"] = inc_horiz
        target_B_inc[f"inc_3d_{h_name}"] = inc_3d

        # Future residual: delta_v(t + H)
        fut_3d = np.full(n, np.nan)
        fut_3d[:-h_steps] = delta_v_3d[h_steps:]
        target_B_fut[f"fut_3d_{h_name}"] = fut_3d

        # Candidate C: Rate of change: [delta_v(t + H) - delta_v(t)] / H
        rate_horiz = inc_horiz / h_sec
        rate_3d = inc_3d / h_sec
        target_C_rate[f"rate_horiz_{h_name}"] = rate_horiz
        target_C_rate[f"rate_3d_{h_name}"] = rate_3d

    # Candidate D: Body-Frame IMU Error / Bias Proxies
    # Proxy D1: Long accel error (ref_accel_long minus finite difference of forward speed)
    d_nav_spd = np.gradient(nav_v_fwd) / dt
    target_D_acc_long = ref_acc_long - d_nav_spd

    # Proxy D2: Lat accel error (ref_accel_lat minus navris turn acceleration nav_v_fwd * yaw_rate)
    phone_gz = df_sync["phone_gyro_z_radps"].values
    target_D_acc_lat = ref_acc_lat - (nav_v_fwd * phone_gz)

    # Proxy D3: Yaw rate error (ref_yaw_rate minus phone_gyro_z)
    target_D_yaw_rate = ref_yaw_rate - phone_gz

    # Proxy D4: Finite difference acceleration residual norm
    d_delta_v_horiz = np.gradient(delta_v_horiz) / dt
    target_D_acc_res_norm = np.abs(d_delta_v_horiz)

    # Candidate E: Navigation Correction & Innovation Magnitude
    target_E_corr_mag = delta_v_3d
    target_E_rel_err = delta_v_3d / (nav_spd + 1.0)
    target_E_local_scale = target_C_rate["rate_3d_1_0s"]

    # ==================== CAUSAL RUNTIME FEATURES ====================
    ax = df_sync["phone_accel_x_mps2"].values
    ay = df_sync["phone_accel_y_mps2"].values
    az = df_sync["phone_accel_z_mps2"].values
    gx = df_sync["phone_gyro_x_radps"].values
    gy = df_sync["phone_gyro_y_radps"].values
    gz = phone_gz

    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyr_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    grav_dev = np.abs(acc_mag - 9.80665)

    s_acc = pd.Series(acc_mag)
    s_gyr = pd.Series(gyr_mag)
    acc_mean_1s = s_acc.rolling(10, min_periods=1).mean().values
    acc_std_1s = s_acc.rolling(10, min_periods=1).std(ddof=0).fillna(0.0).values
    gyr_mean_1s = s_gyr.rolling(10, min_periods=1).mean().values
    gyr_std_1s = s_gyr.rolling(10, min_periods=1).std(ddof=0).fillna(0.0).values

    # Outage duration causally derived from valid new GPS fix
    is_new_fix = df_sync["phone_gps_is_new_fix"].values.astype(bool) if "phone_gps_is_new_fix" in df_sync.columns else np.zeros(n, dtype=bool)
    is_gps_valid = df_sync["phone_gps_is_valid"].values.astype(bool) if "phone_gps_is_valid" in df_sync.columns else np.zeros(n, dtype=bool)
    valid_fix_event = is_new_fix & is_gps_valid

    outage_dur = np.zeros(n)
    last_fix_t = t[0]
    for i in range(n):
        if valid_fix_event[i]:
            last_fix_t = t[i]
        outage_dur[i] = t[i] - last_fix_t

    filter_elapsed = t - t[0]

    # Driving condition masks
    cond_stationary = (ref_spd < 0.2)
    cond_low_speed = (ref_spd >= 0.2) & (ref_spd < 3.0)
    cond_normal = (ref_spd >= 3.0) & (ref_spd < 15.0)
    cond_high_speed = (ref_spd >= 15.0)
    cond_turning = (np.abs(ref_yaw_rate) > 0.1)
    cond_braking = (ref_acc_long < -1.0)
    cond_accel = (ref_acc_long > 1.0)

    # Outage duration bins
    bin_0_5s = (outage_dur < 5.0)
    bin_5_30s = (outage_dur >= 5.0) & (outage_dur < 30.0)
    bin_30_120s = (outage_dur >= 30.0) & (outage_dur < 120.0)
    bin_gt_120s = (outage_dur >= 120.0)

    df_out = pd.DataFrame({
        "time_s": t,
        "recording_id": rec_id,
        "ref_spd": ref_spd,
        "nav_v_fwd": nav_v_fwd,
        "delta_v_horiz": delta_v_horiz,
        "delta_v_3d": delta_v_3d,
        # Targets
        "target_A_fwd": target_A_fwd,
        "target_B_inc_1_0s": target_B_inc["inc_3d_1_0s"],
        "target_B_fut_1_0s": target_B_fut["fut_3d_1_0s"],
        "target_C_rate_1_0s": target_C_rate["rate_3d_1_0s"],
        "target_D_acc_long": target_D_acc_long,
        "target_D_acc_lat": target_D_acc_lat,
        "target_D_yaw_rate": target_D_yaw_rate,
        "target_D_acc_res_norm": target_D_acc_res_norm,
        "target_E_rel_err": target_E_rel_err,
        # Causal features
        "phone_accel_x_mps2": ax,
        "phone_accel_y_mps2": ay,
        "phone_accel_z_mps2": az,
        "phone_gyro_x_radps": gx,
        "phone_gyro_y_radps": gy,
        "phone_gyro_z_radps": gz,
        "accel_magnitude_mps2": acc_mag,
        "gravity_deviation_mps2": grav_dev,
        "gyro_magnitude_radps": gyr_mag,
        "acc_mean_1s": acc_mean_1s,
        "acc_std_1s": acc_std_1s,
        "gyr_mean_1s": gyr_mean_1s,
        "gyr_std_1s": gyr_std_1s,
        "nav_spd": nav_spd,
        "outage_duration_s": outage_dur,
        "filter_elapsed_s": filter_elapsed,
        # Masks
        "cond_stationary": cond_stationary,
        "cond_low_speed": cond_low_speed,
        "cond_normal": cond_normal,
        "cond_high_speed": cond_high_speed,
        "cond_turning": cond_turning,
        "cond_braking": cond_braking,
        "cond_accel": cond_accel,
        "bin_0_5s": bin_0_5s,
        "bin_5_30s": bin_5_30s,
        "bin_30_120s": bin_30_120s,
        "bin_gt_120s": bin_gt_120s,
    })

    # Attach all horizons
    for k, v in target_B_inc.items():
        df_out[f"target_B_{k}"] = v
    for k, v in target_B_fut.items():
        df_out[f"target_B_{k}"] = v
    for k, v in target_C_rate.items():
        df_out[f"target_C_{k}"] = v

    return df_out


def run_target_study(
    sync_dir: str = "data/processed/synchronized",
    gate_dir: str = "data/processed/phase2_3b/gate2_3b",
    out_dir: str = "data/processed/phase3_1",
) -> Dict[str, Any]:
    """
    Executes the full Phase 3.1 target observability study across all benchmark recordings.
    """
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("NAVRIS PHASE 3.1: ML TARGET OBSERVABILITY & FORMULATION STUDY")
    print("=" * 80)

    recs_data = {}
    for r in VALID_RECORDINGS:
        print(f"Loading and processing {r:6s} ...")
        df_r = load_and_preprocess_recording(r, sync_dir=sync_dir, gate_dir=gate_dir)
        recs_data[r] = df_r

    # 1. Target Distributions per recording
    target_keys = [
        "target_A_fwd",
        "target_B_inc_3d_0_5s", "target_B_inc_3d_1_0s", "target_B_inc_3d_2_0s", "target_B_inc_3d_5_0s",
        "target_B_fut_3d_1_0s",
        "target_C_rate_3d_0_5s", "target_C_rate_3d_1_0s", "target_C_rate_3d_2_0s",
        "target_D_acc_long", "target_D_acc_lat", "target_D_yaw_rate",
        "target_E_rel_err",
    ]

    per_rec_target_stats = {}
    flat_target_rows = []

    for r in VALID_RECORDINGS:
        df_r = recs_data[r]
        per_rec_target_stats[r] = {}
        for t_k in target_keys:
            st = compute_distribution_stats(df_r[t_k].values)
            per_rec_target_stats[r][t_k] = st
            flat_target_rows.append({
                "recording_id": r,
                "target_key": t_k,
                **st
            })

    df_target_stats = pd.DataFrame(flat_target_rows)
    df_target_stats.to_csv(out_p / "candidate_target_distributions.csv", index=False)

    # 2. Candidate A Driving Conditions Breakdown
    cond_keys = [
        "cond_stationary", "cond_low_speed", "cond_normal", "cond_high_speed",
        "cond_turning", "cond_braking", "cond_accel"
    ]
    cond_rows = []
    for r in VALID_RECORDINGS:
        df_r = recs_data[r]
        for c_k in cond_keys:
            mask = df_r[c_k].values
            sub_target_A = df_r.loc[mask, "target_A_fwd"].values
            st = compute_distribution_stats(sub_target_A)
            cond_rows.append({
                "recording_id": r,
                "condition": c_k.replace("cond_", ""),
                **st
            })
    df_cond_stats = pd.DataFrame(cond_rows)
    df_cond_stats.to_csv(out_p / "candidate_A_condition_distributions.csv", index=False)

    # 3. Stationarity Analysis across Outage Bins
    bin_keys = ["bin_0_5s", "bin_5_30s", "bin_30_120s", "bin_gt_120s"]
    bin_rows = []
    eval_targets_for_bins = [
        "delta_v_3d", "target_A_fwd", "target_B_inc_3d_1_0s",
        "target_C_rate_3d_1_0s", "target_D_acc_long", "target_D_yaw_rate"
    ]
    for r in VALID_RECORDINGS:
        df_r = recs_data[r]
        for b_k in bin_keys:
            mask = df_r[b_k].values
            if np.sum(mask) == 0:
                continue
            for t_k in eval_targets_for_bins:
                sub_arr = df_r.loc[mask, t_k].values
                st = compute_distribution_stats(sub_arr)
                bin_rows.append({
                    "recording_id": r,
                    "outage_bin": b_k.replace("bin_", ""),
                    "target_key": t_k,
                    **st
                })
    df_bin_stats = pd.DataFrame(bin_rows)
    df_bin_stats.to_csv(out_p / "outage_bin_stationarity.csv", index=False)

    # 4. Feature Relationship Study (Correlations)
    corr_features = [
        "phone_accel_x_mps2", "phone_accel_y_mps2", "phone_accel_z_mps2",
        "phone_gyro_x_radps", "phone_gyro_y_radps", "phone_gyro_z_radps",
        "accel_magnitude_mps2", "gravity_deviation_mps2", "gyro_magnitude_radps",
        "acc_mean_1s", "acc_std_1s", "gyr_mean_1s", "gyr_std_1s",
        "nav_spd", "outage_duration_s", "filter_elapsed_s"
    ]
    targets_for_corr = [
        "delta_v_3d", "target_A_fwd", "target_B_inc_3d_1_0s",
        "target_C_rate_3d_1_0s", "target_D_acc_long", "target_D_yaw_rate"
    ]

    corr_rows = []
    # Concatenate Train partition for correlation study
    df_train = pd.concat([recs_data[r] for r in TRAIN_RECS], ignore_index=True)
    df_val = pd.concat([recs_data[r] for r in VAL_RECS], ignore_index=True)
    df_test = pd.concat([recs_data[r] for r in TEST_RECS], ignore_index=True)

    for part_name, df_part in [("TRAIN", df_train), ("VAL", df_val), ("TEST", df_test)]:
        for t_k in targets_for_corr:
            y = df_part[t_k].values
            valid_y = ~np.isnan(y)
            for f_k in corr_features:
                x = df_part[f_k].values
                valid_mask = valid_y & (~np.isnan(x))
                if np.sum(valid_mask) < 100:
                    continue
                x_v = x[valid_mask]
                y_v = y[valid_mask]

                p_corr, _ = stats.pearsonr(x_v, y_v)
                s_corr, _ = stats.spearmanr(x_v, y_v)

                corr_rows.append({
                    "partition": part_name,
                    "target_key": t_k,
                    "feature_key": f_k,
                    "pearson_corr": float(p_corr),
                    "spearman_corr": float(s_corr)
                })

    df_corr = pd.DataFrame(corr_rows)
    df_corr.to_csv(out_p / "feature_target_correlations.csv", index=False)

    # 5. Partition Level Summary Comparison
    part_summary_rows = []
    for part_name, df_part in [("TRAIN", df_train), ("VAL", df_val), ("TEST", df_test)]:
        for t_k in target_keys:
            st = compute_distribution_stats(df_part[t_k].values)
            part_summary_rows.append({
                "partition": part_name,
                "target_key": t_k,
                **st
            })
    df_part_summary = pd.DataFrame(part_summary_rows)
    df_part_summary.to_csv(out_p / "partition_target_summary.csv", index=False)

    summary_bundle = {
        "per_recording_stats": per_rec_target_stats,
        "csv_outputs": [
            str(out_p / "candidate_target_distributions.csv"),
            str(out_p / "candidate_A_condition_distributions.csv"),
            str(out_p / "outage_bin_stationarity.csv"),
            str(out_p / "feature_target_correlations.csv"),
            str(out_p / "partition_target_summary.csv"),
        ]
    }

    with open(out_p / "study_summary.json", "w") as f:
        json.dump(summary_bundle, f, indent=2)

    print("\nStudy execution complete! Artifacts saved to:", out_p.resolve())
    return summary_bundle


if __name__ == "__main__":
    run_target_study()
