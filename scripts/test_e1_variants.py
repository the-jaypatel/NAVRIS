import os
import sys
sys.path.insert(0, os.path.abspath('src'))
import numpy as np
import pandas as pd

from navris.eskf import (
    ESKF,
    NominalState,
    ESKFConfig,
    STATE_DIM,
)
from navris.eskf.update import compute_kalman_gain
from navris.inertial.init import align_attitude_causal
from navris.inertial.frames import quat_to_euler, quat_multiply, quat_conjugate, wrap_angle_pi

df = pd.read_parquet('data/processed/synchronized/S1_sync.parquet')
causal_history = df.iloc[:min(len(df), 3500)]
align_res = align_attitude_causal(causal_history, min_stat_samples=30)
init_idx = align_res.init_sample_idx
eval_df = df.iloc[init_idx:init_idx + 2500].copy().reset_index(drop=True)

# Displacement velocity over [147.0, 156.0]s:
# Fix at 147s:
sub_gnss = df[(df['time_s'] >= 146.9) & (df['time_s'] <= 156.1) & df['phone_gps_is_new_fix']]
t1 = sub_gnss['time_s'].iloc[-2]
t2 = sub_gnss['time_s'].iloc[-1]
dt_disp = t2 - t1
v_disp_e = (sub_gnss['phone_gps_east_m'].iloc[-1] - sub_gnss['phone_gps_east_m'].iloc[-2]) / dt_disp
v_disp_n = (sub_gnss['phone_gps_north_m'].iloc[-1] - sub_gnss['phone_gps_north_m'].iloc[-2]) / dt_disp
v_disp_u = (sub_gnss['phone_gps_up_m'].iloc[-1] - sub_gnss['phone_gps_up_m'].iloc[-2]) / dt_disp
v0_disp = np.array([v_disp_e, v_disp_n, 0.0], dtype=np.float64)

v0_base = align_res.initial_velocity_enu.copy()

print(f"Base v0: {v0_base.round(3)} (norm: {np.linalg.norm(v0_base):.2f} m/s)")
print(f"Disp v0: {v0_disp.round(3)} (norm: {np.linalg.norm(v0_disp):.2f} m/s, dt: {dt_disp}s from t={t1} to {t2})")
ref_v0 = np.array([
    eval_df['ref_speed_mps'].iloc[0] * np.sin(eval_df['ref_heading_rad'].iloc[0]),
    eval_df['ref_speed_mps'].iloc[0] * np.cos(eval_df['ref_heading_rad'].iloc[0]),
    0.0
])
print(f"Ref  v0: {ref_v0.round(3)} (norm: {np.linalg.norm(ref_v0):.2f} m/s)")

def run_short(v0, sigma_v, label):
    print(f"\n========================================================")
    print(f"RUNNING: {label} (v0={v0.round(2)}, sigma_v={sigma_v} m/s)")
    print(f"========================================================")
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
        p=p0, v=v0.copy(), q=q0, ba=b_a0, bg=b_g0
    )
    init_cov = np.diag([
        25.0, 25.0, 100.0,
        sigma_v**2, sigma_v**2, 1.0,
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

    n = len(eval_df)
    novel_fixes_seen = 0
    accepted_fixes = 0

    for i in range(n):
        t = float(eval_df['time_s'].iloc[i])
        if i > 0:
            dt = t - float(eval_df['time_s'].iloc[i - 1])
            fb = np.array([eval_df['phone_accel_x_mps2'].iloc[i-1], eval_df['phone_accel_y_mps2'].iloc[i-1], eval_df['phone_accel_z_mps2'].iloc[i-1]])
            wb = np.array([eval_df['phone_gyro_x_radps'].iloc[i-1], eval_df['phone_gyro_y_radps'].iloc[i-1], eval_df['phone_gyro_z_radps'].iloc[i-1]])
            eskf.predict(fb, wb, dt)

        is_new = bool(eval_df['phone_gps_is_new_fix'].iloc[i])
        if is_new:
            novel_fixes_seen += 1
            z_pos = np.array([eval_df['phone_gps_east_m'].iloc[i], eval_df['phone_gps_north_m'].iloc[i], eval_df['phone_gps_up_m'].iloc[i]])
            acc_m = max(1.0, float(eval_df['phone_gps_accuracy_m'].iloc[i]))
            pos_cov = np.diag([acc_m**2, acc_m**2, (acc_m * 3.0)**2])

            p_pre = eskf.state.p.copy()
            v_pre = eskf.state.v.copy()
            q_pre = eskf.state.q.copy()
            P_pre = eskf.P.copy()

            H = np.zeros((3, STATE_DIM))
            H[0:3, 0:3] = np.eye(3)
            K, S = compute_kalman_gain(P_pre, H, pos_cov)

            accepted = eskf.update_gnss_pos(z_pos, pos_cov=pos_cov, is_new_fix=True, timestamp=t)
            if accepted:
                accepted_fixes += 1
            inv = eskf.innovations[-1]

            p_post = eskf.state.p.copy()
            v_post = eskf.state.v.copy()
            q_post = eskf.state.q.copy()

            r_pre, p_pre_deg, y_pre_deg = np.degrees(quat_to_euler(q_pre))
            r_post, p_post_deg, y_post_deg = np.degrees(quat_to_euler(q_post))
            d_pitch = p_post_deg - p_pre_deg
            d_v = v_post - v_pre
            ref_pos = np.array([eval_df['ref_east_m'].iloc[i], eval_df['ref_north_m'].iloc[i], eval_df['ref_up_m'].iloc[i]])
            horiz_err = np.linalg.norm(p_post[:2] - ref_pos[:2])

            if novel_fixes_seen <= 5:
                print(f"Fix {novel_fixes_seen - 1} at t={t:.1f}s:")
                print(f"  Accepted: {accepted}, NIS: {inv.mahalanobis_sq:.2f} (gate: 16.27)")
                print(f"  Residual: {inv.residual.round(2)} m (norm: {np.linalg.norm(inv.residual):.2f} m)")
                print(f"  v_pre: {v_pre.round(2)}, v_post: {v_post.round(2)} -> dv: {d_v.round(2)} m/s")
                print(f"  pitch_pre: {p_pre_deg:.2f}°, pitch_post: {p_post_deg:.2f}° -> dpitch: {d_pitch:+.2f}°")
                print(f"  Horiz Error vs Ref: {horiz_err:.2f} m, 3D Pos Std: {np.sqrt(np.trace(eskf.P[:3, :3])):.2f} m")
                print(f"  K_pos diag: {np.diag(K[0:3, :]).round(3)}")
                print(f"  K_vel: {K[3:6, :].round(3)}")
                print(f"  K_att: {K[6:9, :].round(4)}")

    print(f"Summary for {label}: Accepted {accepted_fixes}/{novel_fixes_seen} fixes.")

run_short(v0_base, 2.0, "Baseline (v0_base, sigma_v=2.0)")
run_short(v0_disp, 2.0, "E1-A (v0_disp, sigma_v=2.0)")
run_short(v0_base, 6.0, "E1-B (v0_base, sigma_v=6.0)")
run_short(v0_disp, 6.0, "E1-C (v0_disp, sigma_v=6.0)")
