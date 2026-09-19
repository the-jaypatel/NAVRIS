"""
NAVRIS Publication-Quality Plotting Suite for Phase 2.2.

Generates comprehensive evaluation and failure analysis plots:
1. Trajectory (East vs North) comparison: Reference vs DR
2. Horizontal error vs time
3. Error vs distance traveled
4. Along-track vs cross-track error breakdown
5. Heading error vs time
6. Speed and velocity error
7. Vertical position and velocity drift
8. Baselines comparison (A0 deployable vs A1/A2 oracle vs A-planar)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List

from navris.inertial.metrics import TrajectoryMetrics
from navris.inertial.strapdown import StrapdownTrajectory


def setup_style():
    plt.rcParams.update({
        'font.sans-serif': 'Arial',
        'font.family': 'sans-serif',
        'figure.autolayout': True,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9
    })


def plot_trajectory_comparison(
    traj_dr: StrapdownTrajectory,
    df_ref: pd.DataFrame,
    rec_id: str,
    output_dir: str = 'data/processed/phase2'
) -> str:
    """Plots 2D Horizontal Trajectory: Reference vs DR Baseline."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

    n = len(traj_dr.time_s)
    ref_e = df_ref['ref_east_m'].values[:n]
    ref_n = df_ref['ref_north_m'].values[:n]

    dr_e = traj_dr.pos_enu[:, 0]
    dr_n = traj_dr.pos_enu[:, 1]

    ax.plot(ref_e, ref_n, 'k-', linewidth=2.0, label='VBOX Reference')
    ax.plot(dr_e, dr_n, 'r--', linewidth=1.8, label='A0: Deployable 3D DR (Raw IMU)')

    ax.scatter([ref_e[0]], [ref_n[0]], color='green', s=70, zorder=5, label='Start')
    ax.scatter([ref_e[-1]], [ref_n[-1]], color='black', marker='x', s=70, zorder=5, label='Ref End')
    ax.scatter([dr_e[-1]], [dr_n[-1]], color='red', marker='x', s=70, zorder=5, label='DR End')

    ax.set_title(f"NAVRIS Phase 2.2: Trajectory Comparison — {rec_id}")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.legend(loc='best')
    ax.axis('equal')

    out_path = os.path.join(output_dir, f"{rec_id}_trajectory_dr_vs_ref.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_error_breakdown(
    metrics: TrajectoryMetrics,
    rec_id: str,
    output_dir: str = 'data/processed/phase2'
) -> str:
    """Plots Error vs Time, Along-Track vs Cross-Track, and Vertical Drift."""
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)

    t_rel = metrics.details['t_rel_s']
    horiz_err = metrics.details['horiz_errors']
    along = metrics.details['along_track']
    cross = metrics.details['cross_track']

    # 1. Total Horizontal Error vs Time
    axes[0, 0].plot(t_rel, horiz_err, 'r-', linewidth=1.5, label='Total Horizontal Error')
    axes[0, 0].set_title("Horizontal Position Error vs Time")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Error (m)")
    axes[0, 0].legend()

    # 2. Along-track vs Cross-track
    axes[0, 1].plot(t_rel, along, 'b-', linewidth=1.2, label='Along-Track (Forward)')
    axes[0, 1].plot(t_rel, cross, 'g--', linewidth=1.2, label='Cross-Track (Lateral)')
    axes[0, 1].axhline(0, color='k', linestyle=':', alpha=0.5)
    axes[0, 1].set_title("Along-Track vs Cross-Track Error Breakdown")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Error (m)")
    axes[0, 1].legend()

    # 3. Log-scale Horizontal Error vs Time (shows quadratic growth)
    axes[1, 0].semilogy(t_rel, np.maximum(horiz_err, 0.01), 'm-', linewidth=1.5)
    axes[1, 0].set_title("Log-Scale Error Growth (Inertial Double Integration)")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Horizontal Error (m, log scale)")

    # 4. Error Statistics Summary Card
    axes[1, 1].axis('off')
    summary_text = (
        f"PERFORMANCE SUMMARY ({rec_id})\n"
        f"----------------------------------------\n"
        f"Duration:                {metrics.duration_s:.1f} s\n"
        f"Distance Traveled:       {metrics.distance_traveled_m:.1f} m\n"
        f"Horizontal RMSE:         {metrics.horiz_rmse_m:.2f} m\n"
        f"Horizontal MAE:          {metrics.horiz_mae_m:.2f} m\n"
        f"Final Error:             {metrics.final_error_m:.2f} m\n"
        f"Drift Rate:              {metrics.drift_rate_mps:.2f} m/s\n"
        f"Relative Drift:          {metrics.drift_pct_distance:.1f} % of dist\n"
        f"Along-Track RMSE:        {metrics.along_track_rmse_m:.2f} m\n"
        f"Cross-Track RMSE:        {metrics.cross_track_rmse_m:.2f} m\n"
        f"Heading RMSE:            {metrics.heading_rmse_deg:.2f} deg\n"
        f"Final Vertical Drift:    {metrics.vertical_drift_final_m:.2f} m\n"
        f"Time to 50m drift:       {metrics.time_to_50m_s if metrics.time_to_50m_s else 'N/A'}\n"
        f"Time to 100m drift:      {metrics.time_to_100m_s if metrics.time_to_100m_s else 'N/A'}"
    )
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                    fontfamily='monospace', fontsize=10, verticalalignment='top')

    out_path = os.path.join(output_dir, f"{rec_id}_error_breakdown.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_baselines_comparison(
    results_dict: Dict[str, TrajectoryMetrics],
    rec_id: str,
    output_dir: str = 'data/processed/phase2'
) -> str:
    """Plots A0 vs A1 vs A2 vs Planar error progression comparison."""
    setup_style()
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    styles = {
        'A0_Deployable_3D_DR': ('r-', 2.0, 'A0: Deployable 3D DR (Causal Init)'),
        'A1_Oracle_Heading': ('b--', 1.8, 'A1: Oracle Heading (Diagnostic)'),
        'A2_Oracle_Heading_and_Bias': ('g-.', 1.8, 'A2: Oracle Heading & Bias (Diagnostic)'),
        'A_Planar_Control': ('m:', 1.8, 'A-planar: 2D Kinematic Control')
    }

    for key, metrics in results_dict.items():
        if key in styles:
            color, lw, label = styles[key]
            t_rel = metrics.details['t_rel_s']
            err = metrics.details['horiz_errors']
            ax.plot(t_rel, err, color, linewidth=lw, label=label)

    ax.set_title(f"NAVRIS Baselines Error Progression Comparison — {rec_id}")
    ax.set_xlabel("Time since initialization (s)")
    ax.set_ylabel("Horizontal Position Error (m)")
    ax.legend(loc='best')

    out_path = os.path.join(output_dir, f"{rec_id}_baselines_comparison.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
