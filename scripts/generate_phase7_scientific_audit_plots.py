"""
Generates the 9 scientific audit diagnostic plots for Gate 2.3A under:
data/processed/phase2_3b/gate2_3a/plots/

Plots:
1. plot1_error_A_vs_B_overview.png: A vs B horizontal position error over time
2. plot2_zupt_nis_over_time.png: ZUPT NIS over time across recordings
3. plot3_zupt_accepted_rejected_timeline.png: ZUPT accepted/rejected timeline
4. plot4_velocity_around_zupt.png: Velocity magnitude around ZUPT events
5. plot5_gnss_acceptance_timeline.png: GNSS acceptance timeline
6. plot6_representative_vta2_positive.png: Representative VTA2 positive case
7. plot7_representative_s2_negative.png: Representative S2 negative case
8. plot8_representative_s3a_divergence.png: Representative S3A divergence case
9. plot9_representative_s1_mixed.png: Representative S1 mixed case
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('.'))

from scripts.run_gate2_3a_zupt_benchmark import (
    calibrate_recording_causal,
    COMMON_CONFIG,
    COMMON_INIT_COV,
    quat_multiply,
    ESKF,
    NominalState,
)
from navris.inertial.gnss_fixes import filter_novel_gnss_fixes
from navris.zupt import CausalStationaryDetector, apply_zupt_update


def run_pipeline_for_plot(rec_id: str, df: pd.DataFrame, calib_res, t_eval_start: float, enable_zupt: bool):
    eval_df = df[df['time_s'] >= t_eval_start].copy().reset_index(drop=True)
    n_samples = len(eval_df)
    t_eval_end = float(eval_df['time_s'].iloc[-1])

    init_row = eval_df.iloc[0]
    p0 = np.array([init_row['ref_east_m'], init_row['ref_north_m'], init_row['ref_up_m']], dtype=np.float64)

    init_heading = float(init_row['ref_heading_rad'])
    init_spd = float(init_row['ref_speed_mps'])

    cand_novel = filter_novel_gnss_fixes(eval_df)
    if len(cand_novel) >= 2:
        g0 = cand_novel.iloc[0]
        g1 = cand_novel.iloc[1]
        dt_g = float(g1['time_s'] - g0['time_s'])
        if dt_g > 0:
            d_east = float(g1['phone_gps_east_m'] - g0['phone_gps_east_m'])
            d_north = float(g1['phone_gps_north_m'] - g0['phone_gps_north_m'])
            v_disp = float(np.sqrt(d_east**2 + d_north**2) / dt_g)
            v_gnss_speed = float(g0['phone_gps_speed_mps']) if 'phone_gps_speed_mps' in g0 else init_spd
            if v_disp > 2.0 and (abs(v_gnss_speed - v_disp) / max(v_disp, 1.0)) > 0.40:
                init_spd = v_disp

    v0 = np.array([init_spd * np.sin(init_heading), init_spd * np.cos(init_heading), 0.0], dtype=np.float64)
    theta_nav = np.pi / 2.0 - init_heading
    q_v_n = np.array([np.cos(0.5 * theta_nav), 0.0, 0.0, np.sin(0.5 * theta_nav)], dtype=np.float64)
    q0 = quat_multiply(q_v_n, calib_res.q_body_vehicle)

    init_state = NominalState(
        t=t_eval_start, p=p0.copy(), v=v0.copy(), q=q0.copy(),
        ba=np.zeros(3), bg=np.zeros(3)
    )
    eskf = ESKF(init_state, COMMON_INIT_COV.copy(), COMMON_CONFIG, lat_deg=52.4, alt_m=100.0)
    M_gyro = calib_res.gyro_mapping.mapping_matrix

    detector = CausalStationaryDetector() if enable_zupt else None

    time_arr = np.zeros(n_samples)
    ref_pos = np.zeros((n_samples, 3))
    ref_speed = np.zeros(n_samples)
    horiz_errs = np.zeros(n_samples)
    eskf_vel_norm = np.zeros(n_samples)
    p_cov_tr = np.zeros(n_samples)
    v_cov_tr = np.zeros(n_samples)

    gnss_time = []
    gnss_nis = []
    gnss_acc = []

    last_gps_fix_time = -1.0

    for i in range(n_samples):
        t = float(eval_df['time_s'].iloc[i])
        time_arr[i] = t
        ref_pos[i] = [eval_df['ref_east_m'].iloc[i], eval_df['ref_north_m'].iloc[i], eval_df['ref_up_m'].iloc[i]]
        ref_speed[i] = float(eval_df['ref_speed_mps'].iloc[i])

        raw_fb = np.array([
            eval_df['phone_accel_x_mps2'].iloc[i],
            eval_df['phone_accel_y_mps2'].iloc[i],
            eval_df['phone_accel_z_mps2'].iloc[i]
        ], dtype=np.float64)
        raw_wb = np.array([
            eval_df['phone_gyro_x_radps'].iloc[i],
            eval_df['phone_gyro_y_radps'].iloc[i],
            eval_df['phone_gyro_z_radps'].iloc[i]
        ], dtype=np.float64)

        if i > 0:
            dt = t - time_arr[i - 1]
            prev_fb = np.array([
                eval_df['phone_accel_x_mps2'].iloc[i - 1],
                eval_df['phone_accel_y_mps2'].iloc[i - 1],
                eval_df['phone_accel_z_mps2'].iloc[i - 1]
            ], dtype=np.float64)
            prev_wb = np.array([
                eval_df['phone_gyro_x_radps'].iloc[i - 1],
                eval_df['phone_gyro_y_radps'].iloc[i - 1],
                eval_df['phone_gyro_z_radps'].iloc[i - 1]
            ], dtype=np.float64)
            wb_corr = M_gyro @ prev_wb
            eskf.predict(prev_fb, wb_corr, dt)

        # GNSS fix
        is_gps_sample = bool(eval_df['phone_gps_is_new_fix'].iloc[i])
        if is_gps_sample:
            z_pos = np.array([
                eval_df['phone_gps_east_m'].iloc[i],
                eval_df['phone_gps_north_m'].iloc[i],
                eval_df['phone_gps_up_m'].iloc[i]
            ], dtype=np.float64)
            acc_m = max(1.0, float(eval_df['phone_gps_accuracy_m'].iloc[i]))
            pos_cov = np.diag([acc_m**2, acc_m**2, (acc_m * 3.0)**2])
            acc = eskf.update_gnss_pos(z_pos, pos_cov=pos_cov, is_new_fix=True, timestamp=t)
            nis = float(eskf.innovations[-1].mahalanobis_sq)
            gnss_time.append(t)
            gnss_nis.append(nis)
            gnss_acc.append(acc)

        # ZUPT update
        if enable_zupt and detector is not None:
            is_stat = detector.update(raw_fb, raw_wb)
            if is_stat:
                apply_zupt_update(eskf, timestamp=t)

        diff = eskf.state.p - ref_pos[i]
        horiz_errs[i] = np.sqrt(diff[0]**2 + diff[1]**2)
        eskf_vel_norm[i] = np.linalg.norm(eskf.state.v)
        p_cov_tr[i] = np.trace(eskf.P[0:3, 0:3])
        v_cov_tr[i] = np.trace(eskf.P[3:6, 3:6])

    res = {
        'time_s': time_arr,
        'horiz_err_m': horiz_errs,
        'vel_norm_mps': eskf_vel_norm,
        'ref_speed_mps': ref_speed,
        'p_cov_tr': p_cov_tr,
        'v_cov_tr': v_cov_tr,
        'gnss_time': np.array(gnss_time),
        'gnss_nis': np.array(gnss_nis),
        'gnss_acc': np.array(gnss_acc)
    }
    return res


def main():
    plots_dir = "data/processed/phase2_3b/gate2_3a/plots"
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Generating diagnostic plots into: {plots_dir}")

    # Load update and event logs
    df_up = pd.read_csv("data/processed/phase2_3b/gate2_3a/zupt_updates.csv")
    df_ev = pd.read_csv("data/processed/phase2_3b/gate2_3a/zupt_events.csv")

    # -------------------------------------------------------------------------
    # PLOT 2: ZUPT NIS OVER TIME
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=False)
    target_recs = ['VTA2', 'S1', 'S2', 'S3A', 'S4', 'Y1']
    for ax, rec in zip(axes.flatten(), target_recs):
        sub = df_up[df_up['recording_id'] == rec]
        if len(sub) == 0:
            continue
        acc = sub[sub['accepted'] == True]
        rej = sub[sub['accepted'] == False]
        if len(rej) > 0:
            ax.scatter(rej['time_s'], rej['nis'], color='crimson', alpha=0.4, s=8, label=f'Rejected (N={len(rej)})')
        if len(acc) > 0:
            ax.scatter(acc['time_s'], acc['nis'], color='forestgreen', alpha=0.8, s=12, label=f'Accepted (N={len(acc)})')
        ax.axhline(16.27, color='black', linestyle='--', linewidth=1.2, label=r'$\chi^2_{0.999}(3)=16.27$')
        ax.set_title(f"{rec} - ZUPT NIS over Time", fontsize=11, fontweight='bold')
        ax.set_ylabel("ZUPT NIS")
        ax.set_xlabel("Time (s)")
        ax.set_yscale('log')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot2_zupt_nis_over_time.png"), dpi=200)
    plt.close()
    print("Saved plot2_zupt_nis_over_time.png")

    # -------------------------------------------------------------------------
    # PLOT 3: ZUPT ACCEPTED / REJECTED TIMELINE
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    rec_order = ['VTA2', 'VTA1A', 'Y1', 'S4', 'S3A', 'S2', 'S1']
    y_ticks = []
    for idx, rec in enumerate(rec_order):
        sub = df_up[df_up['recording_id'] == rec]
        acc = sub[sub['accepted'] == True]
        rej = sub[sub['accepted'] == False]
        y_pos = idx * 2
        y_ticks.append(y_pos)
        if len(rej) > 0:
            ax.scatter(rej['time_s'], [y_pos] * len(rej), color='red', marker='x', s=12, alpha=0.5, label='Rejected' if idx==0 else "")
        if len(acc) > 0:
            ax.scatter(acc['time_s'], [y_pos] * len(acc), color='green', marker='o', s=18, alpha=0.9, label='Accepted' if idx==0 else "")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(rec_order, fontweight='bold')
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_title("ZUPT Update Timeline Across IO-VNBD Recordings (Green=Accepted, Red=Rejected)", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot3_zupt_accepted_rejected_timeline.png"), dpi=200)
    plt.close()
    print("Saved plot3_zupt_accepted_rejected_timeline.png")

    # -------------------------------------------------------------------------
    # PLOT 4: VELOCITY MAGNITUDE AROUND ZUPT EVENTS
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    acc_all = df_up[df_up['accepted'] == True]
    ax1.hist(acc_all['v_norm_prior_mps'], bins=np.linspace(0, 1.0, 50), color='royalblue', edgecolor='black', alpha=0.7)
    ax1.set_title("Prior Velocity Magnitude (Accepted ZUPT)", fontsize=11, fontweight='bold')
    ax1.set_xlabel(r"Prior $\|v\|_2$ (m/s)")
    ax1.set_ylabel("Count")
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.hist(acc_all['v_norm_post_mps'], bins=np.linspace(0, 0.10, 50), color='seagreen', edgecolor='black', alpha=0.7)
    ax2.axvline(0.05, color='darkred', linestyle='--', linewidth=1.5, label=r'Declared $\sigma_{zupt}=0.05$ m/s')
    ax2.set_title("Post-Update Velocity Magnitude (Accepted ZUPT)", fontsize=11, fontweight='bold')
    ax2.set_xlabel(r"Post-Update $\|v\|_2$ (m/s)")
    ax2.set_ylabel("Count")
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot4_velocity_around_zupt.png"), dpi=200)
    plt.close()
    print("Saved plot4_velocity_around_zupt.png")

    # -------------------------------------------------------------------------
    # RUN REPRESENTATIVE RECORDINGS FOR TRAJECTORY / ERROR PLOTS
    # -------------------------------------------------------------------------
    rep_recs = ['VTA2', 'S2', 'S3A', 'S1']
    tracks = {}

    for rec in rep_recs:
        print(f"Running trajectory simulation for: {rec}...")
        parquet_p = f"data/processed/synchronized/{rec}_sync.parquet"
        df = pd.read_parquet(parquet_p)
        _, _, _, calib_res, t_dec = calibrate_recording_causal(df, rec)
        if rec == 'S1':
            t_eval = 156.0
        else:
            cand = df[df['time_s'] >= t_dec]
            novel = filter_novel_gnss_fixes(cand)
            t_eval = float(novel.iloc[0]['time_s'])

        tr_A = run_pipeline_for_plot(rec, df, calib_res, t_eval, enable_zupt=False)
        tr_B = run_pipeline_for_plot(rec, df, calib_res, t_eval, enable_zupt=True)
        tracks[rec] = {'A': tr_A, 'B': tr_B}

    # -------------------------------------------------------------------------
    # PLOT 1: ERROR A VS B OVERVIEW
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, rec in zip(axes.flatten(), rep_recs):
        tr_A = tracks[rec]['A']
        tr_B = tracks[rec]['B']
        ax.plot(tr_A['time_s'], tr_A['horiz_err_m'], color='steelblue', label='Config A (Baseline, No ZUPT)', linewidth=1.5)
        ax.plot(tr_B['time_s'], tr_B['horiz_err_m'], color='darkorange', label='Config B (Causal ZUPT)', linewidth=1.5)
        ax.set_title(f"{rec} - Horizontal Error A vs B", fontsize=11, fontweight='bold')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Horizontal Error (m)")
        ax.set_yscale('log')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot1_error_A_vs_B_overview.png"), dpi=200)
    plt.close()
    print("Saved plot1_error_A_vs_B_overview.png")

    # -------------------------------------------------------------------------
    # PLOT 5: GNSS ACCEPTANCE TIMELINE (VTA2 vs S3A)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
    # VTA2
    v_A = tracks['VTA2']['A']
    v_B = tracks['VTA2']['B']
    ax1.scatter(v_A['gnss_time'][v_A['gnss_acc']], v_A['gnss_nis'][v_A['gnss_acc']], color='blue', alpha=0.4, s=10, label='Config A Accepted')
    ax1.scatter(v_A['gnss_time'][~v_A['gnss_acc']], v_A['gnss_nis'][~v_A['gnss_acc']], color='magenta', marker='x', s=15, label='Config A Rejected')
    ax1.scatter(v_B['gnss_time'][v_B['gnss_acc']], v_B['gnss_nis'][v_B['gnss_acc']], color='green', alpha=0.6, s=12, label='Config B Accepted')
    ax1.scatter(v_B['gnss_time'][~v_B['gnss_acc']], v_B['gnss_nis'][~v_B['gnss_acc']], color='red', marker='x', s=20, label='Config B Rejected')
    ax1.axhline(16.27, color='black', linestyle='--', label=r'$\chi^2_{0.999}(3)=16.27$')
    ax1.set_title("VTA2: GNSS Innovation NIS (ZUPT preserves 96.7% GNSS fixes)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("GNSS NIS")
    ax1.set_yscale('log')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=8)

    # S3A
    s_A = tracks['S3A']['A']
    s_B = tracks['S3A']['B']
    ax2.scatter(s_A['gnss_time'][s_A['gnss_acc']], s_A['gnss_nis'][s_A['gnss_acc']], color='blue', alpha=0.4, s=10, label='Config A Accepted')
    ax2.scatter(s_A['gnss_time'][~s_A['gnss_acc']], s_A['gnss_nis'][~s_A['gnss_acc']], color='magenta', marker='x', s=15, label='Config A Rejected')
    ax2.scatter(s_B['gnss_time'][s_B['gnss_acc']], s_B['gnss_nis'][s_B['gnss_acc']], color='green', alpha=0.6, s=12, label='Config B Accepted')
    ax2.scatter(s_B['gnss_time'][~s_B['gnss_acc']], s_B['gnss_nis'][~s_B['gnss_acc']], color='red', marker='x', s=20, label='Config B Rejected')
    ax2.axhline(16.27, color='black', linestyle='--', label=r'$\chi^2_{0.999}(3)=16.27$')
    ax2.set_title("S3A: GNSS Innovation NIS (Config B triggers early rejection at t=422s during 327s stop)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("GNSS NIS")
    ax2.set_xlabel("Time (s)")
    ax2.set_yscale('log')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot5_gnss_acceptance_timeline.png"), dpi=200)
    plt.close()
    print("Saved plot5_gnss_acceptance_timeline.png")

    # -------------------------------------------------------------------------
    # PLOT 6: REPRESENTATIVE VTA2 POSITIVE CASE
    # -------------------------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    v_A = tracks['VTA2']['A']
    v_B = tracks['VTA2']['B']
    ax1.plot(v_A['time_s'], v_A['horiz_err_m'], color='steelblue', label='Config A (Baseline: H-RMSE 12,544 m, Final 70,540 m)')
    ax1.plot(v_B['time_s'], v_B['horiz_err_m'], color='forestgreen', linewidth=1.8, label='Config B (ZUPT: H-RMSE 46.2 m, Final 5.1 m)')
    ax1.set_title("VTA2 Deep Dive: Horizontal Position Error (99.6% Reduction)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Horizontal Error (m)")
    ax1.set_yscale('log')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left')

    ax2.plot(v_B['time_s'], v_B['ref_speed_mps'], color='gray', linestyle=':', label='Ref Speed (m/s)')
    ax2.plot(v_B['time_s'], v_B['vel_norm_mps'], color='darkorange', label=r'Config B $\|v\|_2$')
    v2_up = df_up[df_up['recording_id'] == 'VTA2']
    v2_acc = v2_up[v2_up['accepted'] == True]
    ax2.scatter(v2_acc['time_s'], [0.0]*len(v2_acc), color='green', s=12, label='Accepted ZUPT (683 updates)')
    ax2.set_ylabel("Speed (m/s)")
    ax2.set_title("VTA2 Speed & ZUPT Engagement Intervals", fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right')

    ax3.plot(v_A['time_s'], np.sqrt(v_A['p_cov_tr']), color='steelblue', linestyle='--', label='Config A Pos Cov Std (m)')
    ax3.plot(v_B['time_s'], np.sqrt(v_B['p_cov_tr']), color='forestgreen', linewidth=1.5, label='Config B Pos Cov Std (m)')
    ax3.set_ylabel(r"$\sqrt{\mathrm{tr}(P_{pp})}$ (m)")
    ax3.set_xlabel("Time (s)")
    ax3.set_title("VTA2 Filter Uncertainty Evolution", fontsize=11, fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot6_representative_vta2_positive.png"), dpi=200)
    plt.close()
    print("Saved plot6_representative_vta2_positive.png")

    # -------------------------------------------------------------------------
    # PLOT 7: REPRESENTATIVE S2 NEGATIVE CASE
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    s2_A = tracks['S2']['A']
    s2_B = tracks['S2']['B']
    ax1.plot(s2_A['time_s'], s2_A['horiz_err_m'], color='steelblue', label='Config A (Baseline)')
    ax1.plot(s2_B['time_s'], s2_B['horiz_err_m'], color='darkred', linestyle='--', label='Config B (ZUPT)')
    ax1.set_title("S2 Deep Dive: Position Error Divergence (Filter Lockout Pattern B)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Horizontal Error (m)")
    ax1.set_yscale('log')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left')

    s2_up = df_up[df_up['recording_id'] == 'S2']
    s2_rej = s2_up[s2_up['accepted'] == False]
    ax2.scatter(s2_rej['time_s'], s2_rej['nis'], color='crimson', s=8, alpha=0.5, label=f'Rejected ZUPT Updates (N={len(s2_rej)}, Median NIS={s2_rej["nis"].median():.1f})')
    ax2.axhline(16.27, color='black', linestyle='--', label=r'$\chi^2_{0.999}(3)=16.27$')
    ax2.set_title("S2 ZUPT NIS: 13,652 / 13,653 Updates Gated Out (0.01% Acceptance)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("ZUPT NIS")
    ax2.set_xlabel("Time (s)")
    ax2.set_yscale('log')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot7_representative_s2_negative.png"), dpi=200)
    plt.close()
    print("Saved plot7_representative_s2_negative.png")

    # -------------------------------------------------------------------------
    # PLOT 8: REPRESENTATIVE S3A DIVERGENCE CASE
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    s3_A = tracks['S3A']['A']
    s3_B = tracks['S3A']['B']
    ax1.plot(s3_A['time_s'], s3_A['horiz_err_m'], color='steelblue', label='Config A (Baseline: H-RMSE 110 km)')
    ax1.plot(s3_B['time_s'], s3_B['horiz_err_m'], color='firebrick', linewidth=1.5, label='Config B (ZUPT: H-RMSE 2,347 km)')
    ax1.axvspan(285.0, 612.4, color='gold', alpha=0.2, label='327s Standstill (690 ZUPTs applied)')
    ax1.set_title("S3A Deep Dive: Covariance Starvation Precipitating Divergence", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Horizontal Error (m)")
    ax1.set_yscale('log')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left')

    ax2.plot(s3_A['time_s'], np.sqrt(s3_A['v_cov_tr']), color='steelblue', label=r'Config A $\sqrt{\mathrm{tr}(P_{vv})}$')
    ax2.plot(s3_B['time_s'], np.sqrt(s3_B['v_cov_tr']), color='firebrick', label=r'Config B $\sqrt{\mathrm{tr}(P_{vv})}$ (Starved)')
    ax2.axvspan(285.0, 612.4, color='gold', alpha=0.2)
    ax2.axvline(422.3, color='black', linestyle=':', linewidth=1.5, label='First GNSS Rejection in B (t=422.3s)')
    ax2.set_title("S3A Filter Velocity Covariance: Repeated ZUPT Updates Over-Constrain Uncertainty", fontsize=11, fontweight='bold')
    ax2.set_ylabel(r"Velocity Std $\sqrt{\mathrm{tr}(P_{vv})}$ (m/s)")
    ax2.set_xlabel("Time (s)")
    ax2.set_yscale('log')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot8_representative_s3a_divergence.png"), dpi=200)
    plt.close()
    print("Saved plot8_representative_s3a_divergence.png")

    # -------------------------------------------------------------------------
    # PLOT 9: REPRESENTATIVE S1 MIXED CASE
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
    s1_A = tracks['S1']['A']
    s1_B = tracks['S1']['B']

    # Full trajectory
    ax1.plot(s1_A['time_s'], s1_A['horiz_err_m'], color='steelblue', label='Config A (Baseline: H-RMSE 36.7M m, Peak 83.9M m)')
    ax1.plot(s1_B['time_s'], s1_B['horiz_err_m'], color='darkorange', label='Config B (ZUPT: H-RMSE 6.18M m, Peak 12.4M m)')
    ax1.set_title("S1 Full Route (83 min): Long-Term Error Bounding via Velocity Nulling (83% RMSE Reduction)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Horizontal Error (m)")
    ax1.set_yscale('log')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left')

    # Zoomed initial window [150s, 300s]
    mask_zoom_A = (s1_A['time_s'] >= 156.0) & (s1_A['time_s'] <= 300.0)
    mask_zoom_B = (s1_B['time_s'] >= 156.0) & (s1_B['time_s'] <= 300.0)
    ax2.plot(s1_A['time_s'][mask_zoom_A], s1_A['horiz_err_m'][mask_zoom_A], color='steelblue', linewidth=1.8, label='Config A (Smooth coasting)')
    ax2.plot(s1_B['time_s'][mask_zoom_B], s1_B['horiz_err_m'][mask_zoom_B], color='darkorange', linewidth=1.8, label='Config B (Jump at t=190.2s due to discrete v-correction)')
    ax2.axvline(190.2, color='red', linestyle='--', label='Accepted ZUPT at t=190.2s (delta_v = 11.7 m/s)')
    ax2.set_title("S1 Initial Window [156s - 300s]: Discrete State Correction Induces Transient Position Jump", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Horizontal Error (m)")
    ax2.set_xlabel("Time (s)")
    ax2.set_yscale('log')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot9_representative_s1_mixed.png"), dpi=200)
    plt.close()
    print("Saved plot9_representative_s1_mixed.png")
    print("\nALL 9 AUDIT PLOTS GENERATED SUCCESSFULLY.")

if __name__ == '__main__':
    main()
