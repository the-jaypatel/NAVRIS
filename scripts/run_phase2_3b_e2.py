"""
NAVRIS Phase 2.3B: Controlled Experiment E2 Runner.
DIAGNOSTIC ORACLE — NOT DEPLOYABLE.

Executes the controlled S1 replay with demonstrated input corrections:
1. GNSS Speed unit fix: raw GPS SPEED (Kmh) is in m/s (reverse the 1/3.6 factor).
2. Gyroscope-to-accelerometer frame alignment: omega_z = gyro_y.
3. Forward-axis alignment in horizontal plane.

Outputs:
- data/processed/phase2_3b/experiments/E2/
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
from navris.inertial.frames import quat_to_euler, quat_multiply, wrap_angle_pi


def run_e2_replay():
    print("=" * 80)
    print("NAVRIS Phase 2.3B: Controlled Experiment E2 Replay")
    print("DIAGNOSTIC ORACLE — NOT DEPLOYABLE")
    print("=" * 80)

    out_dir = "data/processed/phase2_3b/experiments/E2"
    os.makedirs(out_dir, exist_ok=True)

    parquet_path = "data/processed/synchronized/S1_sync.parquet"
    df = pd.read_parquet(parquet_path)

    init_idx = 1560
    eval_df = df.iloc[init_idx:].copy().reset_index(drop=True)
    n_samples = len(eval_df)

    # 1. Corrected Initial Velocity (using raw m/s, i.e. phone_gps_speed_mps * 3.6)
    v0_speed = float(eval_df['phone_gps_speed_mps'].iloc[0]) * 3.6
    init_heading = 5.5235368  # 316.48 deg (azimuth)
    v0 = np.array([v0_speed * np.sin(init_heading), v0_speed * np.cos(init_heading), 0.0], dtype=np.float64)

    # 2. Leveled Attitude with Forward-Axis Consistency
    # Leveling from stationary gravity:
    stat_sub = df.iloc[:30]
    f_mean = np.array([stat_sub['phone_accel_x_mps2'].mean(), stat_sub['phone_accel_y_mps2'].mean(), stat_sub['phone_accel_z_mps2'].mean()])
    up_b = f_mean / np.linalg.norm(f_mean)
    up_n = np.array([0.0, 0.0, 1.0])
    v = np.cross(up_b, up_n)
    s = np.dot(up_b, up_n)
    q_level = np.array([1.0 + s, v[0], v[1], v[2]])
    q_level /= np.linalg.norm(q_level)

    # Forward direction in body plane is [1, -1] / sqrt(2) (angle -pi/4)
    # Rotation angle around Up to match azimuth psi = 316.48 deg:
    rot = (3.0 * np.pi / 4.0) - init_heading
    q_yaw = np.array([np.cos(0.5 * rot), 0.0, 0.0, np.sin(0.5 * rot)], dtype=np.float64)
    q0 = quat_multiply(q_yaw, q_level)

    p0 = np.array([eval_df['phone_gps_east_m'].iloc[0], eval_df['phone_gps_north_m'].iloc[0], eval_df['phone_gps_up_m'].iloc[0]], dtype=np.float64)

    init_state = NominalState(
        t=float(eval_df['time_s'].iloc[0]),
        p=p0, v=v0.copy(), q=q0,
        ba=np.zeros(3), bg=np.zeros(3)
    )

    init_cov = np.diag([
        25.0, 25.0, 100.0,
        4.0, 4.0, 1.0,
        0.01, 0.01, 0.05,
        0.04, 0.04, 0.04,
        1e-4, 1e-4, 1e-4
    ])

    config = ESKFConfig(
        sigma_acc=0.20, sigma_gyr=0.02,
        sigma_acc_bias=1e-3, sigma_gyr_bias=1e-4,
        gnss_pos_std_horiz=3.0, gnss_pos_std_vert=10.0,
        chi2_threshold_pos=16.27
    )

    eskf = ESKF(init_state, init_cov, config, lat_deg=52.4, alt_m=100.0)

    time_arr = np.zeros(n_samples, dtype=np.float64)
    ref_pos = np.zeros((n_samples, 3), dtype=np.float64)
    eskf_pos = np.zeros((n_samples, 3), dtype=np.float64)
    ref_speed = np.zeros(n_samples, dtype=np.float64)
    eskf_vel = np.zeros((n_samples, 3), dtype=np.float64)
    eskf_att_deg = np.zeros((n_samples, 3), dtype=np.float64)
    cov_diag = np.zeros((n_samples, STATE_DIM), dtype=np.float64)
    min_eigs = np.zeros(n_samples, dtype=np.float64)
    gnss_is_new = np.zeros(n_samples, dtype=bool)
    gnss_accepted = np.zeros(n_samples, dtype=bool)
    gnss_nis = np.zeros(n_samples, dtype=np.float64)

    early_diagnostics = []
    total_gnss = 0
    novel_fixes = 0
    accepted_fixes = 0
    rejected_fixes = 0

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
            raw_wb = np.array([
                eval_df['phone_gyro_x_radps'].iloc[i - 1],
                eval_df['phone_gyro_y_radps'].iloc[i - 1],
                eval_df['phone_gyro_z_radps'].iloc[i - 1]
            ], dtype=np.float64)
            # Mapped gyro: omega_z is raw gyro_y (true vehicle yaw)
            wb = np.array([raw_wb[0], -raw_wb[2], raw_wb[1]], dtype=np.float64)
            eskf.predict(fb, wb, dt)

        is_new = bool(eval_df['phone_gps_is_new_fix'].iloc[i])
        gnss_is_new[i] = is_new
        total_gnss += 1

        z_pos = np.array([
            eval_df['phone_gps_east_m'].iloc[i],
            eval_df['phone_gps_north_m'].iloc[i],
            eval_df['phone_gps_up_m'].iloc[i]
        ], dtype=np.float64)

        acc_m = max(1.0, float(eval_df['phone_gps_accuracy_m'].iloc[i]))
        pos_cov = np.diag([acc_m**2, acc_m**2, (acc_m * 3.0)**2])

        if is_new:
            novel_fixes += 1
            p_pre = eskf.state.p.copy()
            v_pre = eskf.state.v.copy()
            q_pre = eskf.state.q.copy()
            P_pre = eskf.P.copy()

            acc = eskf.update_gnss_pos(z_pos, pos_cov=pos_cov, is_new_fix=True, timestamp=t)
            gnss_accepted[i] = acc
            last_inv = eskf.innovations[-1]
            gnss_nis[i] = last_inv.mahalanobis_sq

            p_post = eskf.state.p.copy()
            v_post = eskf.state.v.copy()
            q_post = eskf.state.q.copy()

            r_pre, pitch_pre, y_pre = np.degrees(quat_to_euler(q_pre))
            r_post, pitch_post, y_post = np.degrees(quat_to_euler(q_post))

            diag_entry = {
                "fix_idx": novel_fixes - 1,
                "time_s": t,
                "accepted": acc,
                "nis": float(last_inv.mahalanobis_sq),
                "residual_norm": float(np.linalg.norm(last_inv.residual)),
                "residual_enu": last_inv.residual.tolist(),
                "pitch_pre_deg": float(pitch_pre),
                "pitch_post_deg": float(pitch_post),
                "delta_pitch_deg": float(pitch_post - pitch_pre),
                "v_pre": v_pre.tolist(),
                "v_post": v_post.tolist(),
                "delta_v": (v_post - v_pre).tolist(),
                "horiz_err_ref": float(np.linalg.norm(p_post[:2] - ref_pos[i, :2])),
                "pos_std_3d": float(np.sqrt(np.trace(P_pre[:3, :3]))),
            }
            if novel_fixes <= 10 or t in [174.0, 183.0, 192.0, 205.0, 214.0]:
                early_diagnostics.append(diag_entry)

            if acc:
                accepted_fixes += 1
            else:
                rejected_fixes += 1
        else:
            gnss_accepted[i] = False
            gnss_nis[i] = np.nan

        eskf_pos[i] = eskf.state.p
        eskf_vel[i] = eskf.state.v
        roll, pitch, yaw = quat_to_euler(eskf.state.q)
        eskf_att_deg[i] = [np.degrees(roll), np.degrees(pitch), np.degrees(yaw)]
        cov_diag[i] = np.diag(eskf.P)
        min_eigs[i] = np.min(np.linalg.eigvalsh(eskf.P))

    horiz_err = np.linalg.norm(eskf_pos[:, :2] - ref_pos[:, :2], axis=1)
    eskf_speed = np.linalg.norm(eskf_vel, axis=1)
    vel_err = np.abs(eskf_speed - ref_speed)

    idx_60s = np.where((time_arr - time_arr[0]) <= 60.0)[0]
    rmse_60s = float(np.sqrt(np.mean(horiz_err[idx_60s]**2)))
    horiz_rmse = float(np.sqrt(np.mean(horiz_err**2)))

    # Save outputs
    res_df = pd.DataFrame({
        "time_s": time_arr,
        "ref_east_m": ref_pos[:, 0], "ref_north_m": ref_pos[:, 1], "ref_up_m": ref_pos[:, 2],
        "eskf_east_m": eskf_pos[:, 0], "eskf_north_m": eskf_pos[:, 1], "eskf_up_m": eskf_pos[:, 2],
        "ref_speed_mps": ref_speed, "eskf_speed_mps": eskf_speed,
        "eskf_roll_deg": eskf_att_deg[:, 0], "eskf_pitch_deg": eskf_att_deg[:, 1], "eskf_yaw_deg": eskf_att_deg[:, 2],
        "phone_gps_is_new_fix": gnss_is_new, "gnss_accepted": gnss_accepted, "gnss_nis": gnss_nis,
        "horiz_pos_err_m": horiz_err, "vel_err_mps": vel_err,
        "pos_std_3d_m": np.sqrt(cov_diag[:, 0] + cov_diag[:, 1] + cov_diag[:, 2]),
        "min_eigenvalue": min_eigs,
    })
    res_parquet = os.path.join(out_dir, "E2_results.parquet")
    res_df.to_parquet(res_parquet, index=False)

    # Save compact CSV for t in [150, 200]s
    c_sub = res_df[(res_df['time_s'] >= 150.0) & (res_df['time_s'] <= 200.0)].copy().reset_index(drop=True)
    c_csv = os.path.join(out_dir, "E2_150_200.csv")
    c_sub.to_csv(c_csv, index=False)

    diag_json = os.path.join(out_dir, "E2_diagnostics.json")
    with open(diag_json, "w", encoding="utf-8") as f:
        json.dump(early_diagnostics, f, indent=2)

    # Print summary of early fixes
    print("\nEARLY FIXES INVENTORY (Fix 0 to Fix 5):")
    print(f"{'Fix':<5} {'Time(s)':<8} {'Acc':<6} {'NIS':<8} {'ResNorm(m)':<12} {'HorizErr(m)':<12} {'PitchPre':<10} {'PitchPost':<10} {'dPitch':<8}")
    for d in early_diagnostics[:6]:
        print(f"{d['fix_idx']:<5} {d['time_s']:<8.1f} {str(d['accepted']):<6} {d['nis']:<8.2f} {d['residual_norm']:<12.2f} {d['horiz_err_ref']:<12.2f} {d['pitch_pre_deg']:<10.2f} {d['pitch_post_deg']:<10.2f} {d['delta_pitch_deg']:<+8.2f}")

    print(f"\nFull run results:")
    print(f"  First 60s RMSE:  {rmse_60s:.2f} m")
    print(f"  Full Run RMSE:   {horiz_rmse:.2f} m")
    print(f"  Fixes Accepted:  {accepted_fixes}/{novel_fixes} ({accepted_fixes/max(1, novel_fixes)*100:.1f}%)")
    print(f"  Min Eigenvalue:  {np.min(min_eigs):.2e}")


if __name__ == "__main__":
    run_e2_replay()
