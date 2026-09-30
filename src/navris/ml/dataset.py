"""
NAVRIS Phase 3.0: ML Dataset Construction and Recording Splitting.

Enforces:
1. Strict recording-level separation (zero row-level split leakage).
2. Explicit missing-data policy (no forward-filling reference targets, no synthetic sensor data).
3. Exact column alignment between synchronized sensor data and classical filter trajectory solutions.
4. Target definition: delta_v^n = v_reference^n - v_NAVRIS^n in local navigation frame [East, North, Up].
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class RecordingSplit:
    """
    Specifies recording IDs partitioned across train, validation, and held-out test splits.
    Guarantees no recording overlap across splits.
    """
    train_recordings: List[str] = field(default_factory=lambda: ["S1", "S2", "S4"])
    val_recordings: List[str] = field(default_factory=lambda: ["S3A", "VTA1A"])
    test_recordings: List[str] = field(default_factory=lambda: ["Y1", "VTA2"])

    def __post_init__(self):
        s_train = set(self.train_recordings)
        s_val = set(self.val_recordings)
        s_test = set(self.test_recordings)

        overlap_tv = s_train & s_val
        overlap_tt = s_train & s_test
        overlap_vt = s_val & s_test

        if overlap_tv or overlap_tt or overlap_vt:
            raise ValueError(
                f"Data leakage detected in RecordingSplit! "
                f"Train/Val overlap: {overlap_tv}, "
                f"Train/Test overlap: {overlap_tt}, "
                f"Val/Test overlap: {overlap_vt}"
            )

    def all_recordings(self) -> List[str]:
        return self.train_recordings + self.val_recordings + self.test_recordings


def load_recording_ml_data(
    rec_id: str,
    sync_dir: Union[str, Path] = "data/processed/synchronized",
    gate_dir: Union[str, Path] = "data/processed/phase2_3b/gate2_3b",
    baseline_col_prefix: str = "vel_A",
) -> pd.DataFrame:
    """
    Loads synchronized sensor observations and classical NAVRIS filter solutions
    for a single recording, verifying alignment and constructing ground truth velocity
    and velocity residual targets.

    Args:
        rec_id: Recording identifier (e.g., 'S1', 'VTA2').
        sync_dir: Path to data/processed/synchronized directory.
        gate_dir: Path to data/processed/phase2_3b/gate2_3b directory.
        baseline_col_prefix: Column prefix for NAVRIS velocity ('vel_A' for Baseline A, 'vel_B' for Baseline B).

    Returns:
        Aligned pandas DataFrame with columns:
        - time_s
        - phone IMU columns
        - NAVRIS classical estimate columns
        - v_ref_east_mps, v_ref_north_mps, v_ref_up_mps
        - target_delta_v_east_mps, target_delta_v_north_mps, target_delta_v_up_mps
        - recording_id
    """
    sync_p = Path(sync_dir) / f"{rec_id}_sync.parquet"
    traj_p = Path(gate_dir) / f"{rec_id}_trajectory_comparison.csv"

    if not sync_p.exists():
        raise FileNotFoundError(f"Synchronized parquet not found for {rec_id} at {sync_p}")
    if not traj_p.exists():
        raise FileNotFoundError(f"Trajectory comparison CSV not found for {rec_id} at {traj_p}")

    df_sync = pd.read_parquet(sync_p)
    df_traj = pd.read_csv(traj_p)

    # Validate required columns in df_sync
    req_sync_cols = [
        "time_s",
        "phone_accel_x_mps2", "phone_accel_y_mps2", "phone_accel_z_mps2",
        "phone_gyro_x_radps", "phone_gyro_y_radps", "phone_gyro_z_radps",
        "ref_vertical_speed_mps",
    ]
    for c in req_sync_cols:
        if c not in df_sync.columns:
            raise KeyError(f"Missing required sync column '{c}' in {sync_p}")

    # Validate required columns in df_traj
    req_traj_cols = [
        "time_s",
        "ref_speed_mps",
        "ref_heading_rad",
        f"{baseline_col_prefix}_east_mps",
        f"{baseline_col_prefix}_north_mps",
        f"{baseline_col_prefix}_up_mps",
        "head_A_rad" if "head_A_rad" in df_traj.columns else "eskf_heading_rad",
    ]
    for c in req_traj_cols:
        if c not in df_traj.columns:
            raise KeyError(f"Missing required trajectory column '{c}' in {traj_p}")

    # Align timestamps: df_traj covers the post-calibration evaluation interval
    t_start = df_traj["time_s"].iloc[0]
    t_end = df_traj["time_s"].iloc[-1]

    # Filter df_sync to the exact evaluation window with 1e-4 tolerance
    sync_mask = (df_sync["time_s"] >= t_start - 1e-4) & (df_sync["time_s"] <= t_end + 1e-4)
    df_sync_aligned = df_sync.loc[sync_mask].copy().reset_index(drop=True)
    df_traj_aligned = df_traj.copy().reset_index(drop=True)

    if len(df_sync_aligned) != len(df_traj_aligned):
        raise ValueError(
            f"Row count mismatch after alignment for {rec_id}: "
            f"sync={len(df_sync_aligned)}, traj={len(df_traj_aligned)}"
        )

    max_dt = np.max(np.abs(df_sync_aligned["time_s"].values - df_traj_aligned["time_s"].values))
    if max_dt > 1e-3:
        raise ValueError(
            f"Timestamp desynchronization detected for {rec_id}: max dt={max_dt:.6e} s"
        )

    # Construct Reference Velocity in local navigation frame [East, North, Up]
    # v_ref_E = ref_speed * sin(ref_heading)
    # v_ref_N = ref_speed * cos(ref_heading)
    # v_ref_U = ref_vertical_speed
    ref_spd = df_traj_aligned["ref_speed_mps"].values
    ref_hdg = df_traj_aligned["ref_heading_rad"].values
    ref_v_u = df_sync_aligned["ref_vertical_speed_mps"].values

    v_ref_e = ref_spd * np.sin(ref_hdg)
    v_ref_n = ref_spd * np.cos(ref_hdg)

    # Classical NAVRIS filter velocity
    v_nav_e = df_traj_aligned[f"{baseline_col_prefix}_east_mps"].values
    v_nav_n = df_traj_aligned[f"{baseline_col_prefix}_north_mps"].values
    v_nav_u = df_traj_aligned[f"{baseline_col_prefix}_up_mps"].values

    # Navigation velocity residuals: delta_v^n = v_ref^n - v_NAVRIS^n
    delta_v_e = v_ref_e - v_nav_e
    delta_v_n = v_ref_n - v_nav_n
    delta_v_u = ref_v_u - v_nav_u

    # Audit missing data: strictly drop rows with NaNs in targets or critical features
    nan_mask = (
        np.isnan(delta_v_e) | np.isnan(delta_v_n) | np.isnan(delta_v_u) |
        np.isnan(df_sync_aligned["phone_accel_x_mps2"].values) |
        np.isnan(df_sync_aligned["phone_accel_y_mps2"].values) |
        np.isnan(df_sync_aligned["phone_accel_z_mps2"].values) |
        np.isnan(df_sync_aligned["phone_gyro_x_radps"].values) |
        np.isnan(df_sync_aligned["phone_gyro_y_radps"].values) |
        np.isnan(df_sync_aligned["phone_gyro_z_radps"].values)
    )
    if np.any(nan_mask):
        raise ValueError(
            f"Recording {rec_id} contains {np.sum(nan_mask)} NaN rows in target or critical sensor columns! "
            f"Forward filling reference targets is strictly prohibited."
        )

    # Assemble unified recording DataFrame
    heading_col = "head_A_rad" if "head_A_rad" in df_traj_aligned.columns else "eskf_heading_rad"
    df_out = pd.DataFrame({
        "time_s": df_traj_aligned["time_s"].values,
        "recording_id": rec_id,
        # IMU features
        "phone_accel_x_mps2": df_sync_aligned["phone_accel_x_mps2"].values,
        "phone_accel_y_mps2": df_sync_aligned["phone_accel_y_mps2"].values,
        "phone_accel_z_mps2": df_sync_aligned["phone_accel_z_mps2"].values,
        "phone_gyro_x_radps": df_sync_aligned["phone_gyro_x_radps"].values,
        "phone_gyro_y_radps": df_sync_aligned["phone_gyro_y_radps"].values,
        "phone_gyro_z_radps": df_sync_aligned["phone_gyro_z_radps"].values,
        # NAVRIS filter features
        "navris_vel_east_mps": v_nav_e,
        "navris_vel_north_mps": v_nav_n,
        "navris_vel_up_mps": v_nav_u,
        "navris_heading_rad": df_traj_aligned[heading_col].values,
        # Reference ground truth velocity (for evaluation only — NEVER input features)
        "v_ref_east_mps": v_ref_e,
        "v_ref_north_mps": v_ref_n,
        "v_ref_up_mps": ref_v_u,
        "ref_speed_mps": ref_spd,
        "ref_heading_rad": ref_hdg,
        # Target residuals to predict
        "target_delta_v_east_mps": delta_v_e,
        "target_delta_v_north_mps": delta_v_n,
        "target_delta_v_up_mps": delta_v_u,
    })

    # Optional Baseline B columns if present
    if "vel_B_east_mps" in df_traj_aligned.columns:
        df_out["navris_B_vel_east_mps"] = df_traj_aligned["vel_B_east_mps"].values
        df_out["navris_B_vel_north_mps"] = df_traj_aligned["vel_B_north_mps"].values
        df_out["navris_B_vel_up_mps"] = df_traj_aligned["vel_B_up_mps"].values

    return df_out


def build_ml_dataset(
    recordings: List[str],
    sync_dir: Union[str, Path] = "data/processed/synchronized",
    gate_dir: Union[str, Path] = "data/processed/phase2_3b/gate2_3b",
    baseline_col_prefix: str = "vel_A",
) -> pd.DataFrame:
    """
    Loads and concatenates raw aligned ML data for a list of recordings.
    """
    dfs = []
    for rec in recordings:
        df_rec = load_recording_ml_data(
            rec_id=rec,
            sync_dir=sync_dir,
            gate_dir=gate_dir,
            baseline_col_prefix=baseline_col_prefix,
        )
        dfs.append(df_rec)
    return pd.concat(dfs, ignore_index=True)

