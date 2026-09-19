"""
NAVRIS Diagnostic Oracle Baselines (A1 and A2).

NON-DEPLOYABLE ORACLE DIAGNOSTIC CONTROLS:
- Baseline A1 (Oracle Heading): Uses reference trajectory VBOX initial heading
  to isolate the impact of heading initialization error on subsequent DR drift.
- Baseline A2 (Oracle Heading + Bias): Uses reference initial heading and
  ideal stationary gyro bias to isolate sensor noise/drift from initialization error.

These baselines exist strictly for offline error attribution.
They MUST NOT be used in deployable navigation mode (A0).
"""

import numpy as np
import pandas as pd
from typing import Optional

from navris.inertial.strapdown import StrapdownTrajectory, run_strapdown_dr
from navris.inertial.init import align_leveling_from_gravity
from navris.inertial.frames import quat_multiply


def run_oracle_heading_dr(
    df: pd.DataFrame,
    p0: np.ndarray,
    v0: np.ndarray,
    q_level: Optional[np.ndarray] = None,
    gyro_bias: Optional[np.ndarray] = None,
    f_stationary_norm: Optional[float] = None
) -> StrapdownTrajectory:
    """
    Baseline A1: Diagnostic oracle heading.
    Injects exact initial reference heading from ref_heading_rad at t0.
    Leveling uses the stationary leveling quaternion q_level.
    """
    if 'ref_heading_rad' not in df.columns:
        raise ValueError("Oracle heading baseline requires 'ref_heading_rad' column")

    ref_head0 = float(df['ref_heading_rad'].iloc[0])

    if q_level is None:
        ax = df['phone_accel_x_mps2'].iloc[:30].mean()
        ay = df['phone_accel_y_mps2'].iloc[:30].mean()
        az = df['phone_accel_z_mps2'].iloc[:30].mean()
        f_mean = np.array([ax, ay, az], dtype=np.float64)
        q_level = align_leveling_from_gravity(f_mean)

    # Convert reference heading (azimuth) to level-to-nav yaw quaternion
    theta = np.pi / 2.0 - ref_head0
    q_yaw = np.array([np.cos(0.5 * theta), 0.0, 0.0, np.sin(0.5 * theta)], dtype=np.float64)
    q0_oracle = quat_multiply(q_yaw, q_level)

    traj = run_strapdown_dr(
        df=df,
        p0=p0,
        v0=v0,
        q0=q0_oracle,
        gyro_bias=gyro_bias,
        f_stationary_norm=f_stationary_norm
    )
    traj.meta['baseline_type'] = 'A1_Oracle_Heading (NON-DEPLOYABLE)'
    return traj


def run_oracle_heading_bias_dr(
    df: pd.DataFrame,
    p0: np.ndarray,
    v0: np.ndarray,
    q_level: Optional[np.ndarray] = None,
    stationary_df: Optional[pd.DataFrame] = None,
    f_stationary_norm: Optional[float] = None
) -> StrapdownTrajectory:
    """
    Baseline A2: Diagnostic oracle heading + oracle stationary gyro bias.
    Injects exact initial reference heading and stationary gyro bias derived from
    the stationary period at t0.
    """
    source_df = stationary_df if stationary_df is not None else df
    stat_mask = source_df['ref_speed_mps'] < 0.1 if 'ref_speed_mps' in source_df.columns else np.ones(len(source_df), dtype=bool)
    stat_df = source_df[stat_mask].iloc[:50]
    if len(stat_df) < 10:
        stat_df = source_df.iloc[:30]

    b_oracle = np.array([
        stat_df['phone_gyro_x_radps'].mean(),
        stat_df['phone_gyro_y_radps'].mean(),
        stat_df['phone_gyro_z_radps'].mean()
    ], dtype=np.float64)

    traj = run_oracle_heading_dr(
        df=df,
        p0=p0,
        v0=v0,
        q_level=q_level,
        gyro_bias=b_oracle,
        f_stationary_norm=f_stationary_norm
    )
    traj.meta['baseline_type'] = 'A2_Oracle_Heading_and_Bias (NON-DEPLOYABLE)'
    return traj
