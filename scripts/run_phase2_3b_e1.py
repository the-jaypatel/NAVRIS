"""
NAVRIS Phase 2.3B: Controlled Experiment E1 Runner (E1-A, E1-B, E1-C).

Investigates the causality of the S1 divergence:
- E1-A: Velocity estimate only (strictly causal displacement-derived v0, sigma_v=2.0 m/s)
- E1-B: Covariance only (original v0, defensible sigma_v=6.0 m/s + sweep [4.0, 8.0])
- E1-C: Combined (displacement-derived v0, defensible sigma_v=6.0 m/s)

Outputs:
- data/processed/phase2_3b/experiments/E1_A/
- data/processed/phase2_3b/experiments/E1_B/
- data/processed/phase2_3b/experiments/E1_C/
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("src"))

from navris.eskf import (
    ESKF,
    NominalState,
    ESKFConfig,
    STATE_DIM,
)
from navris.eskf.update import compute_kalman_gain
from navris.inertial.init import align_attitude_causal
from navris.inertial.frames import quat_to_euler, wrap_angle_pi


def run_experiment(
    v0: np.ndarray,
    sigma_v: float,
    exp_id: str,
    exp_desc: str,
    out_dir: str,
    df: pd.DataFrame,
    align_res,
    eval_df: pd.DataFrame,
    v0_info: dict,
):
    print("=" * 80)
    print(f"RUNNING EXPERIMENT: {exp_id} — {exp_desc}")
    print(f"Initial Velocity: {v0.round(4)} m/s (norm: {np.linalg.norm(v0):.3f} m/s)")
    print(f"Initial Sigma_v:  {sigma_v:.1f} m/s (variance: {sigma_v**2:.1f})")
    print("=" * 80)

    os.makedirs(out_dir, exist_ok=True)
    n_samples = len(eval_df)

    p0 = np.array([
        eval_df['phone_gps_east_m'].iloc[0],
        eval_df['phone_gps_north_m'].iloc[0],
        eval_df['phone_gps_up_m'].iloc[0]
    ], dtype=np.float64)
    q0 = align_res.q_b_n.copy()
    b_g0 = align_res.gyro_bias_radps.copy()
    b_a0 = np.zeros(3, dtype=np.float64)

    init_state = NominalState(
        t=float(eval_df['time_s'].iloc[0]),
        p=p0,
        v=v0.copy(),
        q=q0,
        ba=b_a0,
        bg=b_g0
    )

    init_cov = np.diag([
        25.0, 25.0, 100.0,            # pos variance (5m horiz, 10m vert)
        sigma_v**2, sigma_v**2, 1.0,  # vel variance
        0.01, 0.01, 0.05,             # att variance
        0.04, 0.04, 0.04,             # acc bias variance
        1e-4, 1e-4, 1e-4              # gyr bias variance
    ])

    config = ESKFConfig(
        sigma_acc=0.20,
        sigma_gyr=0.02,
        sigma_acc_bias=1e-3,
        sigma_gyr_bias=1e-4,
        gnss_pos_std_horiz=3.0,
        gnss_pos_std_vert=10.0,
        chi2_threshold_pos=16.27
    )

    eskf = ESKF(init_state, init_cov, config, lat_deg=52.4, alt_m=100.0)

    # Logging arrays
    time_arr = np.zeros(n_samples, dtype=np.float64)
    ref_pos = np.zeros((n_samples, 3), dtype=np.float64)
    eskf_pos = np.zeros((n_samples, 3), dtype=np.float64)
    ref_speed = np.zeros(n_samples, dtype=np.float64)
    eskf_vel = np.zeros((n_samples, 3), dtype=np.float64)
    eskf_att_deg = np.zeros((n_samples, 3), dtype=np.float64)
    eskf_ba = np.zeros((n_samples, 3), dtype=np.float64)
    eskf_bg = np.zeros((n_samples, 3), dtype=np.float64)
    cov_diag = np.zeros((n_samples, STATE_DIM), dtype=np.float64)
    min_eigs = np.zeros(n_samples, dtype=np.float64)
    gnss_is_new = np.zeros(n_samples, dtype=bool)
    gnss_accepted = np.zeros(n_samples, dtype=bool)
    gnss_nis = np.zeros(n_samples, dtype=np.float64)
    gnss_reject_reason = [""] * n_samples

    # Diagnostics specific to early updates
    early_diagnostics = []
    total_gnss_attempts = 0
    novel_gnss_fixes = 0
    duplicates_gated = 0
    fixes_accepted = 0
    outliers_rejected = 0

    for i in range(n_samples):
        t = float(eval_df['time_s'].iloc[i])
        time_arr[i] = t
        ref_pos[i] = [eval_df['ref_east_m'].iloc[i], eval_df['ref_north_m'].iloc[i], eval_df['ref_up_m'].iloc[i]]
        ref_speed[i] = float(eval_df['ref_speed_mps'].iloc[i])

        if i > 0:
            dt = t - time_arr[i - 1]
            fb = np.array([
                eval_df['phone_accel_x_mps2'].iloc[i - 1],
                eval_df['phone_accel_y_mps2'].iloc[i - 1],
                eval_df['phone_accel_z_mps2'].iloc[i - 1]
            ], dtype=np.float64)
            wb = np.array([
                eval_df['phone_gyro_x_radps'].iloc[i - 1],
                eval_df['phone_gyro_y_radps'].iloc[i - 1],
                eval_df['phone_gyro_z_radps'].iloc[i - 1]
            ], dtype=np.float64)
            eskf.predict(fb, wb, dt)

        is_new = bool(eval_df['phone_gps_is_new_fix'].iloc[i])
        gnss_is_new[i] = is_new
        total_gnss_attempts += 1

        z_pos = np.array([
            eval_df['phone_gps_east_m'].iloc[i],
            eval_df['phone_gps_north_m'].iloc[i],
            eval_df['phone_gps_up_m'].iloc[i]
        ], dtype=np.float64)

        acc_m = float(eval_df['phone_gps_accuracy_m'].iloc[i]) if 'phone_gps_accuracy_m' in eval_df.columns else 3.0
        acc_m = max(1.0, acc_m)
        pos_cov = np.diag([acc_m**2, acc_m**2, (acc_m * 3.0)**2])

        if not is_new:
            duplicates_gated += 1
            gnss_accepted[i] = False
            gnss_reject_reason[i] = "duplicate_sample_and_hold"
            gnss_nis[i] = np.nan
        else:
            novel_gnss_fixes += 1
            # Capture pre-update state & Kalman gain
            p_pre = eskf.state.p.copy()
            v_pre = eskf.state.v.copy()
            q_pre = eskf.state.q.copy()
            P_pre = eskf.P.copy()

            H = np.zeros((3, STATE_DIM))
            H[0:3, 0:3] = np.eye(3)
            K, S = compute_kalman_gain(P_pre, H, pos_cov)

            accepted = eskf.update_gnss_pos(z_pos, pos_cov=pos_cov, is_new_fix=True, timestamp=t)
            gnss_accepted[i] = accepted
            last_inv = eskf.innovations[-1]
            gnss_nis[i] = last_inv.mahalanobis_sq

            p_post = eskf.state.p.copy()
            v_post = eskf.state.v.copy()
            q_post = eskf.state.q.copy()
            P_post = eskf.P.copy()

            r_pre, pitch_pre, y_pre = np.degrees(quat_to_euler(q_pre))
            r_post, pitch_post, y_post = np.degrees(quat_to_euler(q_post))

            diag_entry = {
                "fix_idx": novel_gnss_fixes - 1,
                "time_s": t,
                "step": i,
                "accepted": accepted,
                "nis": float(last_inv.mahalanobis_sq),
                "residual_enu": last_inv.residual.tolist(),
                "residual_norm": float(np.linalg.norm(last_inv.residual)),
                "p_pre": p_pre.tolist(),
                "p_post": p_post.tolist(),
                "v_pre": v_pre.tolist(),
                "v_post": v_post.tolist(),
                "delta_v": (v_post - v_pre).tolist(),
                "pitch_pre_deg": float(pitch_pre),
                "pitch_post_deg": float(pitch_post),
                "delta_pitch_deg": float(pitch_post - pitch_pre),
                "roll_pre_deg": float(r_pre),
                "roll_post_deg": float(r_post),
                "delta_roll_deg": float(r_post - r_pre),
                "yaw_pre_deg": float(y_pre),
                "yaw_post_deg": float(y_post),
                "delta_yaw_deg": float(y_post - y_pre),
                "horiz_err_ref": float(np.linalg.norm(p_post[:2] - ref_pos[i, :2])),
                "pos_std_3d": float(np.sqrt(np.trace(P_pre[:3, :3]))),
                "vel_std_3d": float(np.sqrt(np.trace(P_pre[3:6, 3:6]))),
                "att_std_3d_deg": float(np.degrees(np.sqrt(np.trace(P_pre[6:9, 6:9])))),
                "K_pos_diag": np.diag(K[0:3, :]).tolist(),
                "K_vel_diag": [float(K[3, 0]), float(K[4, 1]), float(K[5, 2])],
                "K_att": K[6:9, :].tolist(),
                "P_pre_diag": np.diag(P_pre).tolist(),
                "P_post_diag": np.diag(P_post).tolist(),
            }
            if novel_gnss_fixes <= 10 or t in [174.0, 214.0, 3491.0]:
                early_diagnostics.append(diag_entry)

            if accepted:
                fixes_accepted += 1
                gnss_reject_reason[i] = "none_accepted"
            else:
                outliers_rejected += 1
                gnss_reject_reason[i] = f"chi2_gated_nis_{last_inv.mahalanobis_sq:.1f}_gt_{config.chi2_threshold_pos}"

        eskf_pos[i] = eskf.state.p
        eskf_vel[i] = eskf.state.v
        roll, pitch, yaw = quat_to_euler(eskf.state.q)
        eskf_att_deg[i] = [np.degrees(roll), np.degrees(pitch), np.degrees(yaw)]
        eskf_ba[i] = eskf.state.ba
        eskf_bg[i] = eskf.state.bg
        cov_diag[i] = np.diag(eskf.P)
        min_eigs[i] = np.min(np.linalg.eigvalsh(eskf.P))

    # Performance metrics
    horiz_pos_err = np.linalg.norm(eskf_pos[:, :2] - ref_pos[:, :2], axis=1)
    pos_err_3d = np.linalg.norm(eskf_pos - ref_pos, axis=1)
    eskf_speed = np.linalg.norm(eskf_vel, axis=1)
    vel_err = np.abs(eskf_speed - ref_speed)

    ref_heading_rad = eval_df['ref_heading_rad'].values
    eskf_heading_rad = (np.pi / 2.0 - np.radians(eskf_att_deg[:, 2])) % (2.0 * np.pi)
    heading_diff_rad = wrap_angle_pi(eskf_heading_rad - ref_heading_rad)
    heading_err_deg = np.degrees(np.abs(heading_diff_rad))

    sin_psi = np.sin(ref_heading_rad)
    cos_psi = np.cos(ref_heading_rad)
    pos_diff = eskf_pos - ref_pos
    along_track_err = pos_diff[:, 0] * sin_psi + pos_diff[:, 1] * cos_psi
    cross_track_err = pos_diff[:, 0] * cos_psi - pos_diff[:, 1] * sin_psi

    idx_50 = np.where(horiz_pos_err >= 50.0)[0]
    time_to_50m = float(time_arr[idx_50[0]] - time_arr[0]) if len(idx_50) > 0 else float('nan')
    idx_100 = np.where(horiz_pos_err >= 100.0)[0]
    time_to_100m = float(time_arr[idx_100[0]] - time_arr[0]) if len(idx_100) > 0 else float('nan')

    idx_60s = np.where((time_arr - time_arr[0]) <= 60.0)[0]
    rmse_60s = float(np.sqrt(np.mean(horiz_pos_err[idx_60s]**2)))
    horiz_rmse = float(np.sqrt(np.mean(horiz_pos_err**2)))
    final_horiz_err = float(horiz_pos_err[-1])
    max_horiz_err = float(np.max(horiz_pos_err))
    along_track_rmse = float(np.sqrt(np.mean(along_track_err**2)))
    cross_track_rmse = float(np.sqrt(np.mean(cross_track_err**2)))
    vel_rmse = float(np.sqrt(np.mean(vel_err**2)))
    valid_heading_mask = (ref_speed > 2.0)
    heading_rmse = float(np.sqrt(np.mean(heading_err_deg[valid_heading_mask]**2))) if valid_heading_mask.any() else float('nan')

    # Save detailed parquet
    results_df = pd.DataFrame({
        "time_s": time_arr,
        "ref_east_m": ref_pos[:, 0],
        "ref_north_m": ref_pos[:, 1],
        "ref_up_m": ref_pos[:, 2],
        "eskf_east_m": eskf_pos[:, 0],
        "eskf_north_m": eskf_pos[:, 1],
        "eskf_up_m": eskf_pos[:, 2],
        "ref_speed_mps": ref_speed,
        "eskf_speed_mps": eskf_speed,
        "eskf_vel_east_mps": eskf_vel[:, 0],
        "eskf_vel_north_mps": eskf_vel[:, 1],
        "eskf_vel_up_mps": eskf_vel[:, 2],
        "eskf_roll_deg": eskf_att_deg[:, 0],
        "eskf_pitch_deg": eskf_att_deg[:, 1],
        "eskf_yaw_deg": eskf_att_deg[:, 2],
        "eskf_ba_x_mps2": eskf_ba[:, 0],
        "eskf_ba_y_mps2": eskf_ba[:, 1],
        "eskf_ba_z_mps2": eskf_ba[:, 2],
        "eskf_bg_x_radps": eskf_bg[:, 0],
        "eskf_bg_y_radps": eskf_bg[:, 1],
        "eskf_bg_z_radps": eskf_bg[:, 2],
        "phone_gps_is_new_fix": gnss_is_new,
        "gnss_accepted": gnss_accepted,
        "gnss_nis": gnss_nis,
        "gnss_reject_reason": gnss_reject_reason,
        "horiz_pos_err_m": horiz_pos_err,
        "pos_err_3d_m": pos_err_3d,
        "vel_err_mps": vel_err,
        "heading_err_deg": heading_err_deg,
        "along_track_err_m": along_track_err,
        "cross_track_err_m": cross_track_err,
        "pos_std_3d_m": np.sqrt(cov_diag[:, 0] + cov_diag[:, 1] + cov_diag[:, 2]),
        "min_eigenvalue": min_eigs,
    })
    res_parquet = os.path.join(out_dir, f"{exp_id}_results.parquet")
    results_df.to_parquet(res_parquet, index=False)

    # Save metrics summary
    metrics_summary = pd.DataFrame([{
        "experiment_id": exp_id,
        "description": exp_desc,
        "samples": n_samples,
        "duration_s": float(time_arr[-1] - time_arr[0]),
        "horiz_rmse_m": horiz_rmse,
        "first_60s_rmse_m": rmse_60s,
        "final_horiz_err_m": final_horiz_err,
        "max_horiz_err_m": max_horiz_err,
        "along_track_rmse_m": along_track_rmse,
        "cross_track_rmse_m": cross_track_rmse,
        "vel_rmse_mps": vel_rmse,
        "heading_rmse_deg": heading_rmse,
        "time_to_50m_s": time_to_50m,
        "time_to_100m_s": time_to_100m,
        "total_gnss_attempts": total_gnss_attempts,
        "duplicates_gated": duplicates_gated,
        "novel_gnss_fixes": novel_gnss_fixes,
        "fixes_accepted": fixes_accepted,
        "outliers_rejected": outliers_rejected,
        "acceptance_rate_pct": float(fixes_accepted / max(1, novel_gnss_fixes) * 100.0),
        "min_eigenvalue": float(np.min(min_eigs)),
        "v0_vector": v0.tolist(),
        "v0_norm": float(np.linalg.norm(v0)),
        "sigma_v": sigma_v,
    }])
    metrics_csv = os.path.join(out_dir, f"{exp_id}_metrics.csv")
    metrics_summary.to_csv(metrics_csv, index=False)

    # Save diagnostics JSON
    diag_file = os.path.join(out_dir, f"{exp_id}_diagnostics.json")
    with open(diag_file, "w", encoding="utf-8") as f:
        json.dump({
            "experiment_id": exp_id,
            "v0_info": v0_info,
            "sigma_v": sigma_v,
            "metrics": metrics_summary.iloc[0].to_dict(),
            "early_updates": early_diagnostics,
        }, f, indent=2)

    # Generate Plots
    plt.figure(figsize=(10, 8))
    plt.plot(ref_pos[:, 0], ref_pos[:, 1], 'k-', label="VBOX Reference", linewidth=2.0)
    plt.plot(eskf_pos[:, 0], eskf_pos[:, 1], 'r--', label=f"{exp_id} ESKF", linewidth=1.5, alpha=0.85)
    plt.plot(ref_pos[0, 0], ref_pos[0, 1], 'go', markersize=8, label="Start (t=156s)")
    plt.title(f"{exp_id}: VBOX Reference vs ESKF Trajectory", fontsize=14, fontweight='bold')
    plt.xlabel("East (m)", fontsize=12)
    plt.ylabel("North (m)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{exp_id}_trajectory.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(time_arr, horiz_pos_err, 'r-', linewidth=1.2, label=f"{exp_id} Error")
    plt.axhline(50.0, color='orange', linestyle='--', label="50m Threshold")
    plt.axhline(100.0, color='red', linestyle='--', label="100m Threshold")
    plt.title(f"{exp_id}: Horizontal Position Error vs Time", fontsize=14, fontweight='bold')
    plt.xlabel("Elapsed Time (s)", fontsize=12)
    plt.ylabel("Horizontal Error (m)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{exp_id}_horiz_error.png"), dpi=150)
    plt.close()

    # NIS plot
    plt.figure(figsize=(12, 5))
    novel_mask = gnss_is_new
    acc_mask = novel_mask & gnss_accepted
    rej_mask = novel_mask & (~gnss_accepted)
    plt.scatter(time_arr[acc_mask], gnss_nis[acc_mask], color='green', marker='o', s=25, label=f"Accepted ({fixes_accepted})")
    plt.scatter(time_arr[rej_mask], gnss_nis[rej_mask], color='red', marker='x', s=35, label=f"Rejected ({outliers_rejected})")
    plt.axhline(config.chi2_threshold_pos, color='black', linestyle='--', linewidth=1.5, label="Chi2 Gate (16.27)")
    plt.yscale('log')
    plt.title(f"{exp_id}: GNSS Innovation NIS vs Time", fontsize=14, fontweight='bold')
    plt.xlabel("Elapsed Time (s)", fontsize=12)
    plt.ylabel("NIS [log scale]", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{exp_id}_nis.png"), dpi=150)
    plt.close()

    print(f"Completed {exp_id}. RMSE: {horiz_rmse:.2f}m (60s RMSE: {rmse_60s:.2f}m), Accepted: {fixes_accepted}/{novel_gnss_fixes} ({fixes_accepted/max(1, novel_gnss_fixes)*100:.1f}%)")
    return metrics_summary.iloc[0].to_dict(), early_diagnostics


def main():
    print("=" * 80)
    print("NAVRIS Phase 2.3B: Controlled Experiment E1 Suite")
    print("=" * 80)

    parquet_path = "data/processed/synchronized/S1_sync.parquet"
    df = pd.read_parquet(parquet_path)

    causal_history = df.iloc[:min(len(df), 3500)]
    align_res = align_attitude_causal(causal_history, min_stat_samples=30)
    init_idx = align_res.init_sample_idx
    eval_df = df.iloc[init_idx:].copy().reset_index(drop=True)

    # 1. Base initial velocity (from phone_gps_speed_mps at t=156s)
    v0_base = align_res.initial_velocity_enu.copy()

    # 2. Causal displacement-derived velocity over initialization window
    # Using the last causal interval [147.0, 156.0]s:
    sub_gnss = df[(df['time_s'] >= 146.9) & (df['time_s'] <= 156.1) & df['phone_gps_is_new_fix']]
    t_prev = float(sub_gnss['time_s'].iloc[-2])
    t_curr = float(sub_gnss['time_s'].iloc[-1])
    dt_disp = t_curr - t_prev
    de_disp = float(sub_gnss['phone_gps_east_m'].iloc[-1] - sub_gnss['phone_gps_east_m'].iloc[-2])
    dn_disp = float(sub_gnss['phone_gps_north_m'].iloc[-1] - sub_gnss['phone_gps_north_m'].iloc[-2])
    v0_disp = np.array([de_disp / dt_disp, dn_disp / dt_disp, 0.0], dtype=np.float64)

    v0_info_base = {
        "method": "instantaneous_phone_gps_speed",
        "speed_source": "phone_gps_speed_mps at t=156.0s",
        "phone_gps_speed_mps": float(eval_df['phone_gps_speed_mps'].iloc[0]),
        "heading_rad": float(align_res.initial_heading_rad),
        "heading_deg": float(np.degrees(align_res.initial_heading_rad)),
        "v0": v0_base.tolist(),
        "v0_norm": float(np.linalg.norm(v0_base)),
    }

    v0_info_disp = {
        "method": "causal_gnss_displacement",
        "interval_s": [t_prev, t_curr],
        "dt_s": dt_disp,
        "delta_east_m": de_disp,
        "delta_north_m": dn_disp,
        "displacement_norm_m": float(np.sqrt(de_disp**2 + dn_disp**2)),
        "v0": v0_disp.tolist(),
        "v0_norm": float(np.linalg.norm(v0_disp)),
        "heading_from_displacement_deg": float(np.degrees(np.arctan2(de_disp, dn_disp) % (2.0 * np.pi))),
    }

    results = []

    # Run Baseline
    m_base, diag_base = run_experiment(
        v0=v0_base,
        sigma_v=2.0,
        exp_id="Baseline",
        exp_desc="Original S1 configuration (v0_base, sigma_v=2.0 m/s)",
        out_dir="data/processed/phase2_3b/experiments/Baseline",
        df=df,
        align_res=align_res,
        eval_df=eval_df,
        v0_info=v0_info_base,
    )
    results.append(m_base)

    # Run E1-A: Velocity Estimate Only
    m_e1a, diag_e1a = run_experiment(
        v0=v0_disp,
        sigma_v=2.0,
        exp_id="E1_A",
        exp_desc="Velocity estimate only (v0_disp, sigma_v=2.0 m/s)",
        out_dir="data/processed/phase2_3b/experiments/E1_A",
        df=df,
        align_res=align_res,
        eval_df=eval_df,
        v0_info=v0_info_disp,
    )
    results.append(m_e1a)

    # Run E1-B: Covariance Only (Primary sigma_v=6.0 m/s)
    m_e1b, diag_e1b = run_experiment(
        v0=v0_base,
        sigma_v=6.0,
        exp_id="E1_B",
        exp_desc="Covariance only (v0_base, sigma_v=6.0 m/s)",
        out_dir="data/processed/phase2_3b/experiments/E1_B",
        df=df,
        align_res=align_res,
        eval_df=eval_df,
        v0_info=v0_info_base,
    )
    results.append(m_e1b)

    # Run E1-B Sensitivity Sweep (sigma_v=4.0 and 8.0 m/s)
    for sig in [4.0, 8.0]:
        m_sweep, _ = run_experiment(
            v0=v0_base,
            sigma_v=sig,
            exp_id=f"E1_B_sigma_{int(sig)}",
            exp_desc=f"Covariance sensitivity sweep (v0_base, sigma_v={sig} m/s)",
            out_dir=f"data/processed/phase2_3b/experiments/E1_B_sigma_{int(sig)}",
            df=df,
            align_res=align_res,
            eval_df=eval_df,
            v0_info=v0_info_base,
        )
        results.append(m_sweep)

    # Run E1-C: Combined (v0_disp, sigma_v=6.0 m/s)
    m_e1c, diag_e1c = run_experiment(
        v0=v0_disp,
        sigma_v=6.0,
        exp_id="E1_C",
        exp_desc="Combined (v0_disp, sigma_v=6.0 m/s)",
        out_dir="data/processed/phase2_3b/experiments/E1_C",
        df=df,
        align_res=align_res,
        eval_df=eval_df,
        v0_info=v0_info_disp,
    )
    results.append(m_e1c)

    # Compile Comparison Table
    summary_df = pd.DataFrame(results)
    summary_csv = "data/processed/phase2_3b/experiments/e1_comparison_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print("\n" + "=" * 80)
    print("E1 EXPERIMENT COMPARISON SUMMARY")
    print("=" * 80)
    cols_show = [
        "experiment_id", "v0_norm", "sigma_v", "first_60s_rmse_m",
        "horiz_rmse_m", "final_horiz_err_m", "time_to_50m_s", "time_to_100m_s",
        "fixes_accepted", "acceptance_rate_pct"
    ]
    print(summary_df[cols_show].to_string(index=False))
    print(f"\nSaved comparison summary to: {summary_csv}")


if __name__ == "__main__":
    main()
