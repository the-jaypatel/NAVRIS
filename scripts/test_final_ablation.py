"""
Diagnostic runner for Gate 2.1 Final Ablation across 4 modes.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("src"))

from navris.calibration import calibrate_causal_s1
from navris.eskf import (
    ESKF,
    NominalState,
    ESKFConfig,
    STATE_DIM
)
from navris.inertial.init import align_leveling_from_gravity
from navris.inertial.frames import (
    quat_multiply,
    quat_to_euler
)


def run_ablation():
    df = pd.read_parquet("data/processed/synchronized/S1_sync.parquet")
    calib_res = calibrate_causal_s1(df, t_decision=125.0)

    # Causal displacement velocity between 147.0 and 156.0
    p147 = df[df["time_s"] == 147.0].iloc[0]
    p156 = df[df["time_s"] == 156.0].iloc[0]
    de = float(p156["phone_gps_east_m"] - p147["phone_gps_east_m"])
    dn = float(p156["phone_gps_north_m"] - p147["phone_gps_north_m"])
    dt = float(p156["time_s"] - p147["time_s"])
    speed_disp = float(np.hypot(de, dn) / dt)
    course_disp = float(np.arctan2(de, dn) % (2 * np.pi))

    init_idx = 1560
    eval_df = df.iloc[init_idx:].copy().reset_index(drop=True)
    n_samples = len(eval_df)

    spd_ref = float(eval_df["ref_speed_mps"].iloc[0])
    hdg_ref = float(eval_df["ref_heading_rad"].iloc[0])
    ref_v0 = np.array([spd_ref * np.sin(hdg_ref), spd_ref * np.cos(hdg_ref), 0.0], dtype=np.float64)

    print(f"Displacement Speed Calculation:")
    print(f"  Interval: t = [147.0, 156.0] s (dt = {dt:.1f} s)")
    print(f"  Delta position: de = {de:+.2f} m, dn = {dn:+.2f} m")
    print(f"  Displacement speed: {speed_disp:.3f} m/s, course: {course_disp:.4f} rad ({np.degrees(course_disp):.2f} deg)")
    print(f"  Reference v0: {ref_v0.round(3)} (speed: {spd_ref:.3f} m/s, heading: {np.degrees(hdg_ref):.2f} deg)")

    modes = ["Mode_A", "Mode_B", "Mode_C", "Mode_D"]
    all_results = {}

    for mode in modes:
        if mode == "Mode_A":
            # Baseline (uncalibrated)
            stat_sub = df.iloc[:30]
            f_mean = np.array([
                stat_sub["phone_accel_x_mps2"].mean(),
                stat_sub["phone_accel_y_mps2"].mean(),
                stat_sub["phone_accel_z_mps2"].mean()
            ])
            q_level = align_leveling_from_gravity(f_mean)
            init_heading = float(eval_df["phone_gps_orientation_rad"].iloc[0])
            theta = np.pi / 2.0 - init_heading
            q_yaw = np.array([np.cos(0.5 * theta), 0.0, 0.0, np.sin(0.5 * theta)], dtype=np.float64)
            q0 = quat_multiply(q_yaw, q_level)
            M_gyro = np.eye(3)
            v0_speed = float(eval_df["phone_gps_speed_mps"].iloc[0])
            v0 = np.array([v0_speed * np.sin(init_heading), v0_speed * np.cos(init_heading), 0.0], dtype=np.float64)

        elif mode == "Mode_B":
            # Corrected Calibration + Existing phone_gps_speed_mps velocity
            init_heading = 5.5235368
            theta_nav = np.pi / 2.0 - init_heading
            q_v_n = np.array([np.cos(0.5 * theta_nav), 0.0, 0.0, np.sin(0.5 * theta_nav)], dtype=np.float64)
            q0 = quat_multiply(q_v_n, calib_res.q_body_vehicle)
            M_gyro = calib_res.gyro_mapping.mapping_matrix
            v0_speed = float(eval_df["phone_gps_speed_mps"].iloc[0])
            v0 = np.array([v0_speed * np.sin(init_heading), v0_speed * np.cos(init_heading), 0.0], dtype=np.float64)

        elif mode == "Mode_C":
            # Corrected Calibration + Causal Displacement Velocity
            init_heading = 5.5235368
            theta_nav = np.pi / 2.0 - init_heading
            q_v_n = np.array([np.cos(0.5 * theta_nav), 0.0, 0.0, np.sin(0.5 * theta_nav)], dtype=np.float64)
            q0 = quat_multiply(q_v_n, calib_res.q_body_vehicle)
            M_gyro = calib_res.gyro_mapping.mapping_matrix
            v0 = np.array([speed_disp * np.sin(init_heading), speed_disp * np.cos(init_heading), 0.0], dtype=np.float64)

        elif mode == "Mode_D":
            # E2 Oracle Diagnostic Reference
            stat_sub = df.iloc[:30]
            f_mean = np.array([
                stat_sub["phone_accel_x_mps2"].mean(),
                stat_sub["phone_accel_y_mps2"].mean(),
                stat_sub["phone_accel_z_mps2"].mean()
            ])
            q_level = align_leveling_from_gravity(f_mean)
            init_heading = 5.5235368
            rot = (3.0 * np.pi / 4.0) - init_heading
            q_yaw = np.array([np.cos(0.5 * rot), 0.0, 0.0, np.sin(0.5 * rot)], dtype=np.float64)
            q0 = quat_multiply(q_yaw, q_level)
            M_gyro = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
            v0_speed = float(eval_df["phone_gps_speed_mps"].iloc[0])
            v0 = np.array([v0_speed * np.sin(init_heading), v0_speed * np.cos(init_heading), 0.0], dtype=np.float64)

        p0 = np.array([
            eval_df["phone_gps_east_m"].iloc[0],
            eval_df["phone_gps_north_m"].iloc[0],
            eval_df["phone_gps_up_m"].iloc[0]
        ], dtype=np.float64)

        init_state = NominalState(
            t=float(eval_df["time_s"].iloc[0]),
            p=p0, v=v0.copy(), q=q0.copy(),
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

        time_arr = np.zeros(n_samples)
        ref_pos = np.zeros((n_samples, 3))
        eskf_pos = np.zeros((n_samples, 3))
        fix_history = []

        for i in range(n_samples):
            t = float(eval_df["time_s"].iloc[i])
            time_arr[i] = t
            ref_pos[i] = [
                eval_df["ref_east_m"].iloc[i],
                eval_df["ref_north_m"].iloc[i],
                eval_df["ref_up_m"].iloc[i]
            ]
            if i > 0:
                dt_step = t - time_arr[i - 1]
                fb = np.array([
                    eval_df["phone_accel_x_mps2"].iloc[i - 1],
                    eval_df["phone_accel_y_mps2"].iloc[i - 1],
                    eval_df["phone_accel_z_mps2"].iloc[i - 1]
                ])
                raw_wb = np.array([
                    eval_df["phone_gyro_x_radps"].iloc[i - 1],
                    eval_df["phone_gyro_y_radps"].iloc[i - 1],
                    eval_df["phone_gyro_z_radps"].iloc[i - 1]
                ])
                wb = M_gyro @ raw_wb
                eskf.predict(fb, wb, dt_step)

            is_new = bool(eval_df["phone_gps_is_new_fix"].iloc[i])
            if is_new:
                z_pos = np.array([
                    eval_df["phone_gps_east_m"].iloc[i],
                    eval_df["phone_gps_north_m"].iloc[i],
                    eval_df["phone_gps_up_m"].iloc[i]
                ])
                acc_m = max(1.0, float(eval_df["phone_gps_accuracy_m"].iloc[i]))
                pos_cov = np.diag([acc_m**2, acc_m**2, (acc_m * 3.0)**2])
                acc = eskf.update_gnss_pos(z_pos, pos_cov=pos_cov, is_new_fix=True, timestamp=t)
                nis = float(eskf.innovations[-1].mahalanobis_sq)
                horiz_err = float(np.hypot(eskf.state.p[0] - ref_pos[i, 0], eskf.state.p[1] - ref_pos[i, 1]))
                r, p, y = quat_to_euler(eskf.state.q)
                fix_history.append({
                    "time_s": t,
                    "accepted": acc,
                    "nis": nis,
                    "horiz_err_m": horiz_err,
                    "vel_norm": float(np.linalg.norm(eskf.state.v)),
                    "att_deg": [float(np.degrees(r)), float(np.degrees(p)), float(np.degrees(y))]
                })
            eskf_pos[i] = eskf.state.p

        mask_60 = (time_arr >= 156.0) & (time_arr <= 216.0)
        err_60 = np.hypot(eskf_pos[mask_60, 0] - ref_pos[mask_60, 0], eskf_pos[mask_60, 1] - ref_pos[mask_60, 1])
        rmse_60 = float(np.sqrt(np.mean(err_60**2)))
        max_60 = float(np.max(err_60))
        final_60 = float(err_60[-1])

        mask_51 = (time_arr >= 156.0) & (time_arr <= 624.0)
        err_51 = np.hypot(eskf_pos[mask_51, 0] - ref_pos[mask_51, 0], eskf_pos[mask_51, 1] - ref_pos[mask_51, 1])
        rmse_51 = float(np.sqrt(np.mean(err_51**2)))
        max_51 = float(np.max(err_51))

        fixes_51 = [f for f in fix_history if f["time_s"] <= 624.0]
        acc_51 = sum(1 for f in fixes_51 if f["accepted"])
        tot_51 = len(fixes_51)

        idx_50 = np.where(err_51 >= 50.0)[0]
        t_50 = float(time_arr[mask_51][idx_50[0]]) if len(idx_50) > 0 else None
        idx_100 = np.where(err_51 >= 100.0)[0]
        t_100 = float(time_arr[mask_51][idx_100[0]]) if len(idx_100) > 0 else None

        v0_err = float(np.linalg.norm(v0 - ref_v0))

        all_results[mode] = {
            "v0": v0.tolist(),
            "speed0": float(np.linalg.norm(v0)),
            "v0_err": v0_err,
            "rmse_60": rmse_60,
            "max_60": max_60,
            "final_60": final_60,
            "rmse_51": rmse_51,
            "max_51": max_51,
            "accepted_51": f"{acc_51}/{tot_51} ({acc_51/tot_51*100:.1f}%)",
            "t_50": t_50,
            "t_100": t_100,
            "early_fixes": fix_history[:5]
        }

        print(f"\n=== {mode} ===")
        print(f"  v0: {v0.round(3)} (speed: {np.linalg.norm(v0):.3f} m/s, err proxy: {v0_err:.3f} m/s)")
        print(f"  60s: RMSE = {rmse_60:.2f} m, Max = {max_60:.2f} m, Final = {final_60:.2f} m")
        print(f"  51-fix: RMSE = {rmse_51:.2f} m, Max = {max_51:.2f} m, Accepted = {acc_51}/{tot_51} ({acc_51/tot_51*100:.1f}%)")
        print(f"  Time to 50m: {t_50} s, Time to 100m: {t_100} s")
        print("  Early fixes (t in [156, 192 s]):")
        for ef in fix_history[:5]:
            print(f"    t = {ef['time_s']:.1f} s: acc = {ef['accepted']}, NIS = {ef['nis']:6.2f}, err = {ef['horiz_err_m']:6.2f} m, vel = {ef['vel_norm']:5.2f} m/s, att = {[round(x, 1) for x in ef['att_deg']]}")


if __name__ == "__main__":
    run_ablation()
