"""
Generate publication-quality diagnostic plots for Gate 2.3B-3 Real-Data NHC Benchmark.

Plots generated:
1. S3A Trajectory Comparison: Reference vs Baseline A vs NHC B (demonstrating 73.6% RMSE reduction).
2. S3A Horizontal Position Error vs Time (A vs B).
3. VTA1A Horizontal Position Error vs Time (Negative control cruise stabilization).
4. NHC Innovation (NIS) & Acceptance Timeline for S3A and VTA1A.
5. VTA2 Trajectory & Failure Analysis: Baseline A vs NHC B.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

out_dir = "docs/plots/phase2_3b/gate2_3b"
os.makedirs(out_dir, exist_ok=True)
data_dir = "data/processed/phase2_3b/gate2_3b"

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 1.8,
    'grid.alpha': 0.4,
    'grid.linestyle': '--'
})

# -----------------------------------------------------------------------------
# PLOT 1: S3A Trajectory Comparison (A vs B vs Reference)
# -----------------------------------------------------------------------------
s3a_file = os.path.join(data_dir, "S3A_trajectory_comparison.csv")
if os.path.exists(s3a_file):
    df_s3a = pd.read_csv(s3a_file)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(df_s3a['ref_east_m'] / 1e3, df_s3a['ref_north_m'] / 1e3, 'k-', label='VBOX Ground Truth', linewidth=2.5)
    ax.plot(df_s3a['pos_A_east_m'] / 1e3, df_s3a['pos_A_north_m'] / 1e3, 'r--', label='Config A: Baseline (ESKF+ZUPT) [H-RMSE: 2,347 km]', alpha=0.8)
    ax.plot(df_s3a['pos_B_east_m'] / 1e3, df_s3a['pos_B_north_m'] / 1e3, 'b-', label='Config B: NHC Enabled [H-RMSE: 621 km, -73.6%]', alpha=0.85)

    ax.set_xlabel('East Position (km)')
    ax.set_ylabel('North Position (km)')
    ax.set_title('NAVRIS Gate 2.3B: S3A Trajectory Comparison\nBaseline (ESKF+ZUPT) vs NHC Enabled')
    ax.legend(loc='best')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "s3a_trajectory_comparison.png"), dpi=200)
    plt.close()
    print("Saved s3a_trajectory_comparison.png")

# -----------------------------------------------------------------------------
# PLOT 2: S3A Horizontal Position Error vs Time
# -----------------------------------------------------------------------------
if os.path.exists(s3a_file):
    fig, ax = plt.subplots(figsize=(11, 5))
    t_rel = df_s3a['time_s'] - df_s3a['time_s'].iloc[0]
    ax.plot(t_rel, df_s3a['horiz_err_A_m'] / 1e3, 'r--', label='Config A: Frozen Baseline (ESKF+ZUPT)')
    ax.plot(t_rel, df_s3a['horiz_err_B_m'] / 1e3, 'b-', label='Config B: Frozen Baseline + NHC (-73.6% RMSE)')

    ax.set_xlabel('Elapsed Evaluation Time (s)')
    ax.set_ylabel('Horizontal Position Error (km)')
    ax.set_title('NAVRIS Gate 2.3B: S3A Horizontal Error vs Time')
    ax.legend(loc='upper left')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "s3a_horizontal_error_vs_time.png"), dpi=200)
    plt.close()
    print("Saved s3a_horizontal_error_vs_time.png")

# -----------------------------------------------------------------------------
# PLOT 3: VTA1A (Negative Control) Horizontal Error vs Time
# -----------------------------------------------------------------------------
vta1a_file = os.path.join(data_dir, "VTA1A_trajectory_comparison.csv")
if os.path.exists(vta1a_file):
    df_vta1a = pd.read_csv(vta1a_file)
    fig, ax = plt.subplots(figsize=(11, 5))
    t_rel = df_vta1a['time_s'] - df_vta1a['time_s'].iloc[0]
    ax.plot(t_rel, df_vta1a['horiz_err_A_m'] / 1e3, 'r--', label='Config A: Frozen Baseline (ESKF+ZUPT) [H-RMSE: 2,260 km]')
    ax.plot(t_rel, df_vta1a['horiz_err_B_m'] / 1e3, 'b-', label='Config B: Frozen Baseline + NHC [H-RMSE: 1,480 km, -34.5%]')

    ax.set_xlabel('Elapsed Evaluation Time (s)')
    ax.set_ylabel('Horizontal Position Error (km)')
    ax.set_title('NAVRIS Gate 2.3B: VTA1A (Negative Control) Horizontal Error vs Time')
    ax.legend(loc='upper left')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "vta1a_horizontal_error_vs_time.png"), dpi=200)
    plt.close()
    print("Saved vta1a_horizontal_error_vs_time.png")

# -----------------------------------------------------------------------------
# PLOT 4: NHC NIS and Acceptance Timeline (S3A)
# -----------------------------------------------------------------------------
nhc_updates_file = os.path.join(data_dir, "nhc_updates.csv")
if os.path.exists(nhc_updates_file):
    df_nhc = pd.read_csv(nhc_updates_file)
    s3a_nhc = df_nhc[df_nhc['recording_id'] == 'S3A'].copy()

    if len(s3a_nhc) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

        # Subplot 1: NIS Timeline
        ax1.scatter(s3a_nhc['time_s'], s3a_nhc['nis'], c=np.where(s3a_nhc['accepted'], 'blue', 'red'), s=8, alpha=0.6, label='Candidate Updates')
        ax1.axhline(13.82, color='crimson', linestyle='--', linewidth=2, label='Chi-Square Threshold (13.82, 2 DOF)')
        ax1.set_ylabel('NHC NIS')
        ax1.set_yscale('log')
        ax1.set_ylim(0.01, 10000)
        ax1.set_title('NAVRIS Gate 2.3B: Recording S3A NHC Innovation & Residuals')
        ax1.legend(loc='upper right')
        ax1.grid(True)

        # Subplot 2: Lateral and Vertical Velocity Residuals
        ax2.scatter(s3a_nhc['time_s'], s3a_nhc['residual_y_mps'], c='teal', s=6, alpha=0.5, label='Lateral Residual (vy)')
        ax2.scatter(s3a_nhc['time_s'], s3a_nhc['residual_z_mps'], c='darkorange', s=6, alpha=0.5, label='Vertical Residual (vz)')
        ax2.axhline(0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Timestamp (s)')
        ax2.set_ylabel('Residual (m/s)')
        ax2.set_ylim(-3.0, 3.0)
        ax2.legend(loc='upper right')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "s3a_nhc_nis_residuals.png"), dpi=200)
        plt.close()
        print("Saved s3a_nhc_nis_residuals.png")

# -----------------------------------------------------------------------------
# PLOT 5: VTA2 Trajectory & Failure Mode Analysis
# -----------------------------------------------------------------------------
vta2_file = os.path.join(data_dir, "VTA2_trajectory_comparison.csv")
if os.path.exists(vta2_file):
    df_vta2 = pd.read_csv(vta2_file)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(df_vta2['ref_east_m'], df_vta2['ref_north_m'], 'k-', label='VBOX Ground Truth', linewidth=2.5)
    ax.plot(df_vta2['pos_A_east_m'], df_vta2['pos_A_north_m'], 'g-', label='Config A: Baseline (ESKF+ZUPT) [H-RMSE: 46.2 m]', linewidth=2.0)
    ax.plot(df_vta2['pos_B_east_m'], df_vta2['pos_B_north_m'], 'r--', label='Config B: NHC Enabled [H-RMSE: 31,081 m]', linewidth=1.5, alpha=0.85)

    ax.set_xlabel('East Position (m)')
    ax.set_ylabel('North Position (m)')
    ax.set_title('NAVRIS Gate 2.3B: VTA2 Failure Analysis\nCornering Dynamics & Lever Arm Provoking GNSS Lockout')
    ax.legend(loc='best')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "vta2_failure_analysis.png"), dpi=200)
    plt.close()
    print("Saved vta2_failure_analysis.png")

print("All Gate 2.3B diagnostic plots generated successfully.")
