"""
NAVRIS Phase 2.3B Gate 1: Real IO-VNBD S1 Integration Runner.

Executes the 15-state ESKF on S1_sync.parquet using strictly causal information
and compares performance directly against Phase 2.2 Baseline A0.

Outputs:
- data/processed/phase2_3b/S1_eskf_results.parquet
- data/processed/phase2_3b/S1_eskf_metrics.csv
- Diagnostic plots in data/processed/phase2_3b/
"""

import os
import sys
sys.path.insert(0, os.path.abspath("src"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from navris.eskf import (
    ESKF,
    NominalState,
    ESKFConfig,
    STATE_DIM,
    IDX_POS,
    IDX_VEL,
    IDX_ATT,
    IDX_ACC_BIAS,
    IDX_GYR_BIAS,
)
from navris.inertial.init import align_attitude_causal
from navris.inertial.frames import (
    quat_to_euler,
    wrap_angle_pi,
)
from navris.inertial.metrics import compute_dr_metrics


def run_s1_eskf_gate1():
    print("=" * 80)
    print("NAVRIS Phase 2.3B Gate 1: S1 Real-Data ESKF Integration")
    print("=" * 80)

    parquet_path = "data/processed/synchronized/S1_sync.parquet"
    out_dir = "data/processed/phase2_3b"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"Total synchronized rows: {len(df)}")
    print(f"Timeline: {df['time_s'].iloc[0]:.2f}s to {df['time_s'].iloc[-1]:.2f}s (Duration: {df['time_s'].iloc[-1] - df['time_s'].iloc[0]:.2f}s)")

    # 1. Causal Initial Alignment (Matching Phase 2.2 Protocol)
    causal_history = df.iloc[:min(len(df), 3500)]
    align_res = align_attitude_causal(causal_history, min_stat_samples=30)
    print("\n--- CAUSAL INITIALIZATION REPORT ---")
    print(f"Initialization Succeeded: {align_res.is_valid} (Quality: {align_res.quality})")
    print(f"Init Sample Index:        {align_res.init_sample_idx}")
    print(f"Init Time:                {align_res.init_time_s:.2f} s")
    print(f"Init Heading (azimuth):   {np.degrees(align_res.initial_heading_rad):.2f}° ({align_res.initial_heading_rad:.4f} rad)")
    print(f"Init Velocity ENU:        {align_res.initial_velocity_enu.round(3)} m/s")
    print(f"Init Gyro Bias:           {align_res.gyro_bias_radps.round(5)} rad/s")
    print(f"Stationary Accel Norm:    {align_res.f_stationary_norm:.4f} m/s^2 (diff from g: {align_res.f_stationary_norm - 9.80665:+.4f})")
    print(f"Stationary Samples Used:  {align_res.stationary_samples_used} (0 to {align_res.stationary_samples_used * 0.1:.1f}s)")
    print(f"Heading Novel Fixes Used: {align_res.heading_novel_fixes_used}")

    init_idx = align_res.init_sample_idx
    eval_df = df.iloc[init_idx:].copy().reset_index(drop=True)
    n_samples = len(eval_df)
    print(f"\nEvaluation Timeline: {n_samples} samples ({eval_df['time_s'].iloc[0]:.1f}s to {eval_df['time_s'].iloc[-1]:.1f}s, duration: {eval_df['time_s'].iloc[-1] - eval_df['time_s'].iloc[0]:.1f}s)")

    # 2. Setup Initial State & Covariance
    p0 = np.array([
        eval_df['phone_gps_east_m'].iloc[0],
        eval_df['phone_gps_north_m'].iloc[0],
        eval_df['phone_gps_up_m'].iloc[0]
    ], dtype=np.float64)
    v0 = align_res.initial_velocity_enu.copy()
    q0 = align_res.q_b_n.copy()
    b_g0 = align_res.gyro_bias_radps.copy()
    b_a0 = np.zeros(3, dtype=np.float64)

    init_state = NominalState(
        t=float(eval_df['time_s'].iloc[0]),
        p=p0,
        v=v0,
        q=q0,
        ba=b_a0,
        bg=b_g0
    )

    init_cov = np.diag([
        25.0, 25.0, 100.0,            # pos variance (5m horiz, 10m vert std)
        4.0, 4.0, 1.0,                # vel variance (2m/s std)
        0.01, 0.01, 0.05,             # att variance (~5.7 deg tilt, ~12.8 deg yaw std)
        0.04, 0.04, 0.04,             # acc bias variance (0.2 m/s^2 std)
        1e-4, 1e-4, 1e-4              # gyr bias variance (0.01 rad/s std)
    ])

    config = ESKFConfig(
        sigma_acc=0.20,               # smartphone MEMS accel noise
        sigma_gyr=0.02,               # smartphone MEMS gyro noise
        sigma_acc_bias=1e-3,          # accel bias random walk
        sigma_gyr_bias=1e-4,          # gyro bias random walk
        gnss_pos_std_horiz=3.0,
        gnss_pos_std_vert=10.0,
        chi2_threshold_pos=16.27      # 99.9% confidence gate
    )

    eskf = ESKF(init_state, init_cov, config, lat_deg=52.4, alt_m=100.0)

    # 3. Preallocate Logging Arrays
    time_arr = np.zeros(n_samples, dtype=np.float64)
    ref_pos = np.zeros((n_samples, 3), dtype=np.float64)
    eskf_pos = np.zeros((n_samples, 3), dtype=np.float64)
    ref_speed = np.zeros(n_samples, dtype=np.float64)
    eskf_vel = np.zeros((n_samples, 3), dtype=np.float64)
    eskf_att_deg = np.zeros((n_samples, 3), dtype=np.float64)  # roll, pitch, yaw
    eskf_ba = np.zeros((n_samples, 3), dtype=np.float64)
    eskf_bg = np.zeros((n_samples, 3), dtype=np.float64)
    cov_diag = np.zeros((n_samples, STATE_DIM), dtype=np.float64)
    min_eigs = np.zeros(n_samples, dtype=np.float64)
    gnss_is_new = np.zeros(n_samples, dtype=bool)
    gnss_accepted = np.zeros(n_samples, dtype=bool)
    gnss_nis = np.zeros(n_samples, dtype=np.float64)
    gnss_reject_reason = [""] * n_samples

    total_gnss_attempts = 0
    novel_gnss_fixes = 0
    duplicates_gated = 0
    fixes_accepted = 0
    outliers_rejected = 0

    print("\nRunning ESKF prediction and measurement update loop...")
    for i in range(n_samples):
        t = float(eval_df['time_s'].iloc[i])
        time_arr[i] = t

        # Record Reference Trajectory
        ref_pos[i] = [eval_df['ref_east_m'].iloc[i], eval_df['ref_north_m'].iloc[i], eval_df['ref_up_m'].iloc[i]]
        ref_speed[i] = float(eval_df['ref_speed_mps'].iloc[i])

        # Prediction step (for i > 0)
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

        # Measurement update step
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
            accepted = eskf.update_gnss_pos(z_pos, pos_cov=pos_cov, is_new_fix=True, timestamp=t)
            gnss_accepted[i] = accepted
            last_inv = eskf.innovations[-1]
            gnss_nis[i] = last_inv.mahalanobis_sq
            if accepted:
                fixes_accepted += 1
                gnss_reject_reason[i] = "none_accepted"
            else:
                outliers_rejected += 1
                gnss_reject_reason[i] = f"chi2_gated_nis_{last_inv.mahalanobis_sq:.1f}_gt_{config.chi2_threshold_pos}"

        # Record Estimates
        eskf_pos[i] = eskf.state.p
        eskf_vel[i] = eskf.state.v
        roll, pitch, yaw = quat_to_euler(eskf.state.q)
        eskf_att_deg[i] = [np.degrees(roll), np.degrees(pitch), np.degrees(yaw)]
        eskf_ba[i] = eskf.state.ba
        eskf_bg[i] = eskf.state.bg
        cov_diag[i] = np.diag(eskf.P)

        eigs = np.linalg.eigvalsh(eskf.P)
        min_eigs[i] = np.min(eigs)

        if (i + 1) % 10000 == 0 or (i + 1) == n_samples:
            print(f"  Processed {i + 1}/{n_samples} samples ({t:.1f}s)... Min Eig(P): {min_eigs[i]:.2e}")

    # 4. Compute Performance Errors
    horiz_pos_err = np.linalg.norm(eskf_pos[:, :2] - ref_pos[:, :2], axis=1)
    pos_err_3d = np.linalg.norm(eskf_pos - ref_pos, axis=1)
    eskf_speed = np.linalg.norm(eskf_vel, axis=1)
    vel_err = np.abs(eskf_speed - ref_speed)

    # Heading error (navigational azimuth vs reference heading)
    # Polar angle in ENU to navigational azimuth: psi = (pi/2 - polar) % 2pi
    eskf_heading_rad = (np.pi / 2.0 - np.radians(eskf_att_deg[:, 2])) % (2.0 * np.pi)
    ref_heading_rad = eval_df['ref_heading_rad'].values
    heading_diff_rad = wrap_angle_pi(eskf_heading_rad - ref_heading_rad)
    heading_err_deg = np.degrees(np.abs(heading_diff_rad))

    # Along-track and cross-track errors
    # Unit vector along reference heading: u_along = [sin(psi), cos(psi), 0]
    # Unit vector cross-track (right): u_cross = [cos(psi), -sin(psi), 0]
    sin_psi = np.sin(ref_heading_rad)
    cos_psi = np.cos(ref_heading_rad)
    pos_diff = eskf_pos - ref_pos
    along_track_err = pos_diff[:, 0] * sin_psi + pos_diff[:, 1] * cos_psi
    cross_track_err = pos_diff[:, 0] * cos_psi - pos_diff[:, 1] * sin_psi

    # Time to 50m and 100m
    idx_50 = np.where(horiz_pos_err >= 50.0)[0]
    time_to_50m = float(time_arr[idx_50[0]] - time_arr[0]) if len(idx_50) > 0 else float('nan')
    idx_100 = np.where(horiz_pos_err >= 100.0)[0]
    time_to_100m = float(time_arr[idx_100[0]] - time_arr[0]) if len(idx_100) > 0 else float('nan')

    # Basic Sanity Checks
    print("\n--- BASIC SANITY CHECKS ---")
    has_nans = bool(np.isnan(eskf_pos).any() or np.isnan(eskf_vel).any() or np.isnan(cov_diag).any())
    has_infs = bool(np.isinf(eskf_pos).any() or np.isinf(eskf_vel).any() or np.isinf(cov_diag).any())
    is_monotonic = bool((np.diff(time_arr) > 0).all())
    cov_valid = bool((min_eigs > -1e-12).all())
    cov_symmetric = True  # enforced via Joseph & propagation
    print(f"NaNs Detected:             {has_nans}")
    print(f"Infinities Detected:       {has_infs}")
    print(f"Timestamps Monotonic:      {is_monotonic}")
    print(f"Covariance Validity (PSD): {cov_valid} (min eigenvalue across run: {np.min(min_eigs):.2e})")
    print(f"Max Position Coordinate:   {np.max(np.abs(eskf_pos)):.2f} m")
    print(f"Max Velocity Magnitude:    {np.max(eskf_speed):.2f} m/s")

    # Save detailed results dataframe
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

    res_parquet = os.path.join(out_dir, "S1_eskf_results.parquet")
    results_df.to_parquet(res_parquet, index=False)
    print(f"\nDetailed trajectory results saved to: {res_parquet}")

    # Summary Metrics
    horiz_rmse = float(np.sqrt(np.mean(horiz_pos_err**2)))
    final_horiz_err = float(horiz_pos_err[-1])
    max_horiz_err = float(np.max(horiz_pos_err))
    along_track_rmse = float(np.sqrt(np.mean(along_track_err**2)))
    cross_track_rmse = float(np.sqrt(np.mean(cross_track_err**2)))
    vel_rmse = float(np.sqrt(np.mean(vel_err**2)))
    valid_heading_mask = (ref_speed > 2.0)  # heading error only meaningful when vehicle is moving
    heading_rmse = float(np.sqrt(np.mean(heading_err_deg[valid_heading_mask]**2))) if valid_heading_mask.any() else float('nan')

    print("\n" + "=" * 80)
    print("PHASE 2.3B GATE 1 — S1 ESKF vs PHASE 2.2 BASELINE A0 COMPARISON")
    print("=" * 80)
    # Phase 2.2 A0 values on S1 from data/processed/phase2/phase22_baseline_metrics.csv
    a0_horiz_rmse = 6048530.76
    a0_final_err = 11334582.94
    a0_max_err = 11334582.94  # monotonic divergence
    a0_along_rmse = 4215398.84
    a0_cross_rmse = 4337641.86
    a0_heading_rmse = 102.82
    a0_t50 = 4.8
    a0_t100 = 9.0

    print(f"{'Metric':<28} {'Phase 2.2 A0 (Raw INS)':<24} {'Phase 2.3B ESKF':<20}")
    print("-" * 75)
    print(f"{'Horizontal RMSE (m)':<28} {a0_horiz_rmse:<24.2f} {horiz_rmse:<20.2f}")
    print(f"{'Final Horizontal Err (m)':<28} {a0_final_err:<24.2f} {final_horiz_err:<20.2f}")
    print(f"{'Max Horizontal Err (m)':<28} {a0_max_err:<24.2f} {max_horiz_err:<20.2f}")
    print(f"{'Along-Track RMSE (m)':<28} {a0_along_rmse:<24.2f} {along_track_rmse:<20.2f}")
    print(f"{'Cross-Track RMSE (m)':<28} {a0_cross_rmse:<24.2f} {cross_track_rmse:<20.2f}")
    print(f"{'Velocity RMSE (m/s)':<28} {'N/A (diverged)':<24} {vel_rmse:<20.2f}")
    print(f"{'Heading RMSE (deg)':<28} {a0_heading_rmse:<24.2f} {heading_rmse:<20.2f}")
    print(f"{'Time to 50m (s)':<28} {a0_t50:<24.1f} {time_to_50m:<20.1f}")
    print(f"{'Time to 100m (s)':<28} {a0_t100:<24.1f} {time_to_100m:<20.1f}")
    print("-" * 75)

    print("\n--- GNSS MEASUREMENT GATING AUDIT ---")
    print(f"Total GNSS Epochs:            {total_gnss_attempts}")
    print(f"Sample-and-Hold Duplicates:   {duplicates_gated} ({duplicates_gated / total_gnss_attempts * 100:.2f}%)")
    print(f"Novel GNSS Fixes:             {novel_gnss_fixes} ({novel_gnss_fixes / total_gnss_attempts * 100:.2f}%)")
    print(f"Novel Fixes Accepted:         {fixes_accepted} ({fixes_accepted / max(1, novel_gnss_fixes) * 100:.2f}%)")
    print(f"Novel Fixes Rejected (NIS):   {outliers_rejected} ({outliers_rejected / max(1, novel_gnss_fixes) * 100:.2f}%)")

    # Export metrics CSV
    metrics_csv = os.path.join(out_dir, "S1_eskf_metrics.csv")
    metrics_summary = pd.DataFrame([{
        "recording_id": "S1",
        "samples": n_samples,
        "duration_s": float(time_arr[-1] - time_arr[0]),
        "horiz_rmse_m": horiz_rmse,
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
        "min_eigenvalue": float(np.min(min_eigs)),
        "has_nans": has_nans,
        "has_infs": has_infs,
    }])
    metrics_summary.to_csv(metrics_csv, index=False)
    print(f"Metrics summary saved to: {metrics_csv}")

    # 5. Diagnostic Plots
    print("\nGenerating diagnostic plots...")

    # Plot 1: Reference vs ESKF Trajectory
    plt.figure(figsize=(10, 8))
    plt.plot(ref_pos[:, 0], ref_pos[:, 1], 'k-', label="VBOX Reference", linewidth=2.0)
    plt.plot(eskf_pos[:, 0], eskf_pos[:, 1], 'r--', label="Phase 2.3B ESKF", linewidth=1.5, alpha=0.85)
    plt.plot(ref_pos[0, 0], ref_pos[0, 1], 'go', markersize=8, label="Start (t=156s)")
    plt.title("S1 Trajectory: VBOX Reference vs Phase 2.3B ESKF", fontsize=14, fontweight='bold')
    plt.xlabel("East (m)", fontsize=12)
    plt.ylabel("North (m)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    p1_path = os.path.join(out_dir, "S1_trajectory_ref_vs_eskf.png")
    plt.savefig(p1_path, dpi=150)
    plt.close()

    # Plot 2: Horizontal Position Error vs Time
    plt.figure(figsize=(12, 5))
    plt.plot(time_arr, horiz_pos_err, 'r-', linewidth=1.2, label="ESKF Horizontal Error")
    plt.axhline(50.0, color='orange', linestyle='--', label="50m Threshold")
    plt.axhline(100.0, color='red', linestyle='--', label="100m Threshold")
    plt.title("S1: Horizontal Position Error vs Time", fontsize=14, fontweight='bold')
    plt.xlabel("Elapsed Time (s)", fontsize=12)
    plt.ylabel("Horizontal Error (m)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    p2_path = os.path.join(out_dir, "S1_horiz_error_vs_time.png")
    plt.savefig(p2_path, dpi=150)
    plt.close()

    # Plot 3: GNSS Updates and NIS vs Time
    novel_mask = gnss_is_new
    plt.figure(figsize=(12, 5))
    accepted_mask = novel_mask & gnss_accepted
    rejected_mask = novel_mask & (~gnss_accepted)
    plt.scatter(time_arr[accepted_mask], gnss_nis[accepted_mask], color='green', marker='o', s=25, label=f"Accepted Fixes ({fixes_accepted})")
    plt.scatter(time_arr[rejected_mask], gnss_nis[rejected_mask], color='red', marker='x', s=35, label=f"Rejected Outliers ({outliers_rejected})")
    plt.axhline(config.chi2_threshold_pos, color='black', linestyle='--', linewidth=1.5, label=f"Chi2 Gate ({config.chi2_threshold_pos})")
    plt.yscale('log')
    plt.title("S1: GNSS Innovation NIS vs Time with Chi-Square Gate", fontsize=14, fontweight='bold')
    plt.xlabel("Elapsed Time (s)", fontsize=12)
    plt.ylabel("NIS (Mahalanobis d^2) [log scale]", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    p3_path = os.path.join(out_dir, "S1_gnss_updates_and_nis.png")
    plt.savefig(p3_path, dpi=150)
    plt.close()

    # Plot 4: Estimated Biases vs Time
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axs[0].plot(time_arr, eskf_ba[:, 0], label="ba_x (m/s^2)")
    axs[0].plot(time_arr, eskf_ba[:, 1], label="ba_y (m/s^2)")
    axs[0].plot(time_arr, eskf_ba[:, 2], label="ba_z (m/s^2)")
    axs[0].set_title("S1: Estimated Accelerometer Biases vs Time", fontsize=13, fontweight='bold')
    axs[0].set_ylabel("Accel Bias (m/s^2)", fontsize=11)
    axs[0].grid(True, linestyle=':', alpha=0.6)
    axs[0].legend(fontsize=10)

    axs[1].plot(time_arr, np.degrees(eskf_bg[:, 0]), label="bg_x (deg/s)")
    axs[1].plot(time_arr, np.degrees(eskf_bg[:, 1]), label="bg_y (deg/s)")
    axs[1].plot(time_arr, np.degrees(eskf_bg[:, 2]), label="bg_z (deg/s)")
    axs[1].set_title("S1: Estimated Gyroscope Biases vs Time", fontsize=13, fontweight='bold')
    axs[1].set_xlabel("Elapsed Time (s)", fontsize=11)
    axs[1].set_ylabel("Gyro Bias (deg/s)", fontsize=11)
    axs[1].grid(True, linestyle=':', alpha=0.6)
    axs[1].legend(fontsize=10)
    plt.tight_layout()
    p4_path = os.path.join(out_dir, "S1_bias_estimates_vs_time.png")
    plt.savefig(p4_path, dpi=150)
    plt.close()

    # Plot 5: Position Uncertainty vs Actual Error
    plt.figure(figsize=(12, 5))
    pos_3d_std = np.sqrt(cov_diag[:, 0] + cov_diag[:, 1] + cov_diag[:, 2])
    plt.plot(time_arr, pos_err_3d, 'r-', linewidth=1.2, label="Actual 3D Error")
    plt.plot(time_arr, 3.0 * pos_3d_std, 'b--', linewidth=1.2, label="Filter 3-Sigma Position Uncertainty")
    plt.title("S1: Actual 3D Error vs Filter 3-Sigma Uncertainty Envelope", fontsize=14, fontweight='bold')
    plt.xlabel("Elapsed Time (s)", fontsize=12)
    plt.ylabel("Error / 3-Sigma (m)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    p5_path = os.path.join(out_dir, "S1_uncertainty_vs_actual_error.png")
    plt.savefig(p5_path, dpi=150)
    plt.close()

    print(f"Generated 5 diagnostic plots in {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    run_s1_eskf_gate1()
