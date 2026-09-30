"""
NAVRIS Phase 3.2A: Vehicle-Domain Robustness Diagnostic.

Core Scientific Question:
Determine whether the forward-speed signal identified in Phase 3.2 is:
1. primarily a vehicle/road-specific vibration signature, OR
2. a more generalizable relationship between IMU dynamics and vehicle forward speed.

Strict Boundaries:
- Diagnostic study only. Zero model tuning for leaderboard accuracy.
- Frozen partition: Train (S1, S2, S4), Val (S3A, VTA1A), Held-Out Test (Y1, VTA2).
- Target: v_fwd_ref(t) (OFFLINE LABEL ONLY).
- Model architecture: Identical fixed lightweight XGBoost from Phase 3.2.
- Evaluates:
  * Experiment A: Raw IMU representation (Phase 3.2 Reference)
  * Experiment B: Scale-Invariant Magnitude Features (Dimensionless ratios)
  * Experiment C: Dynamic-Only Features (Deltas, jerk, lag-1 autocorrelation)
  * Experiment D: Vehicle-Specific Holdout Diagnostic (Fiesta transfer vs Golf transfer)
  * Experiment E: Causal Running Normalization Representation
  * Negative Control: OFFLINE Non-Causal Normalization (NOT DEPLOYABLE)
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure src and project root are on sys.path
root_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from navris.ml.dataset import RecordingSplit, load_recording_ml_data
from scripts.phase3.run_phase3_2_speed_baseline import (
    ABLATION_C_FEATURES,
    STANDARD_GRAVITY_MPS2,
    WINDOW_SAMPLES,
    compute_distribution_stats,
    compute_regression_metrics,
    extract_causal_imu_features,
)


EPSILON = 1e-6

# ---------------------------------------------------------------------------
# Feature Column Definitions for Phase 3.2A
# ---------------------------------------------------------------------------

SCALE_INVARIANT_FEATURES: List[str] = [
    "accel_cv_1s",               # sigma_a / (mu_a + eps)
    "gyro_cv_1s",                # sigma_omega / (mu_omega + eps)
    "accel_std_over_g0",         # sigma_a / g0
    "gravity_dev_ratio",         # | ||a|| - g0 | / g0
    "gyro_mag_over_norm",        # ||omega|| / (mu_omega + eps)
    "accel_mag_over_norm",       # ||a|| / (mu_a + eps)
    "component_energy_ratio_x",  # ax^2 / (||a||^2 + eps)
    "component_energy_ratio_y",  # ay^2 / (||a||^2 + eps)
    "component_energy_ratio_z",  # az^2 / (||a||^2 + eps)
    "gyro_energy_ratio_x",       # gx^2 / (||omega||^2 + eps)
    "gyro_energy_ratio_y",       # gy^2 / (||omega||^2 + eps)
    "gyro_energy_ratio_z",       # gz^2 / (||omega||^2 + eps)
    "accel_axis_std_ratio_x",    # sigma_ax / (sigma_a + eps)
    "accel_axis_std_ratio_y",    # sigma_ay / (sigma_a + eps)
    "accel_axis_std_ratio_z",    # sigma_az / (sigma_a + eps)
]

DYNAMIC_ONLY_FEATURES: List[str] = [
    "norm_delta_ax_1s",          # (ax[k] - ax[k-9]) / (mu_a + eps)
    "norm_delta_ay_1s",          # (ay[k] - ay[k-9]) / (mu_a + eps)
    "norm_delta_az_1s",          # (az[k] - az[k-9]) / (mu_a + eps)
    "norm_delta_gx_1s",          # (gx[k] - gx[k-9]) / (mu_omega + eps)
    "norm_delta_gy_1s",          # (gy[k] - gy[k-9]) / (mu_omega + eps)
    "norm_delta_gz_1s",          # (gz[k] - gz[k-9]) / (mu_omega + eps)
    "norm_delta_accel_mag",      # (||a[k]|| - ||a[k-9]||) / g0
    "norm_delta_gyro_mag",       # (||omega[k]|| - ||omega[k-9]||) / (mu_omega + eps)
    "accel_jerk_proxy_1s",       # mean(|diff(||a||)|) / g0
    "gyro_accel_ratio_proxy",    # ||omega|| / (||delta_a|| + eps)
    "accel_autocorr_lag1",       # 1-sample lag correlation of ||a||
    "gyro_autocorr_lag1",        # 1-sample lag correlation of ||omega||
]

CAUSAL_RUNNING_FEATURES: List[str] = ABLATION_C_FEATURES + [
    "causal_running_z_accel_std",
    "causal_running_z_gyro_std",
]

OFFLINE_NORMALIZED_FEATURES: List[str] = ABLATION_C_FEATURES + [
    "offline_z_accel_std",
    "offline_z_gyro_std",
]


# ---------------------------------------------------------------------------
# Feature Extraction: Scale-Invariant and Dynamic-Only
# ---------------------------------------------------------------------------

def extract_scale_invariant_features(df_rec: pd.DataFrame, window_size: int = WINDOW_SAMPLES) -> pd.DataFrame:
    """
    Extracts strictly causal dimensionless and scale-invariant features over [k-9 ... k].
    """
    n_samples = len(df_rec)
    w = window_size
    k_indices = np.arange(w - 1, n_samples)
    m = len(k_indices)

    ax = df_rec["phone_accel_x_mps2"].values.astype(np.float64)
    ay = df_rec["phone_accel_y_mps2"].values.astype(np.float64)
    az = df_rec["phone_accel_z_mps2"].values.astype(np.float64)

    gx = df_rec["phone_gyro_x_radps"].values.astype(np.float64)
    gy = df_rec["phone_gyro_y_radps"].values.astype(np.float64)
    gz = df_rec["phone_gyro_z_radps"].values.astype(np.float64)

    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    g0 = STANDARD_GRAVITY_MPS2

    s_amag = pd.Series(accel_mag)
    s_gmag = pd.Series(gyro_mag)
    s_ax = pd.Series(ax)
    s_ay = pd.Series(ay)
    s_az = pd.Series(az)

    mu_a = s_amag.rolling(w).mean().values[k_indices]
    sig_a = s_amag.rolling(w).std(ddof=0).values[k_indices]
    mu_w = s_gmag.rolling(w).mean().values[k_indices]
    sig_w = s_gmag.rolling(w).std(ddof=0).values[k_indices]

    sig_ax = s_ax.rolling(w).std(ddof=0).values[k_indices]
    sig_ay = s_ay.rolling(w).std(ddof=0).values[k_indices]
    sig_az = s_az.rolling(w).std(ddof=0).values[k_indices]

    a_curr = accel_mag[k_indices]
    w_curr = gyro_mag[k_indices]

    feats = {
        "accel_cv_1s": sig_a / (mu_a + EPSILON),
        "gyro_cv_1s": sig_w / (mu_w + EPSILON),
        "accel_std_over_g0": sig_a / g0,
        "gravity_dev_ratio": np.abs(a_curr - g0) / g0,
        "gyro_mag_over_norm": w_curr / (mu_w + EPSILON),
        "accel_mag_over_norm": a_curr / (mu_a + EPSILON),
        "component_energy_ratio_x": (ax[k_indices]**2) / (a_curr**2 + EPSILON),
        "component_energy_ratio_y": (ay[k_indices]**2) / (a_curr**2 + EPSILON),
        "component_energy_ratio_z": (az[k_indices]**2) / (a_curr**2 + EPSILON),
        "gyro_energy_ratio_x": (gx[k_indices]**2) / (w_curr**2 + EPSILON),
        "gyro_energy_ratio_y": (gy[k_indices]**2) / (w_curr**2 + EPSILON),
        "gyro_energy_ratio_z": (gz[k_indices]**2) / (w_curr**2 + EPSILON),
        "accel_axis_std_ratio_x": sig_ax / (sig_a + EPSILON),
        "accel_axis_std_ratio_y": sig_ay / (sig_a + EPSILON),
        "accel_axis_std_ratio_z": sig_az / (sig_a + EPSILON),
    }

    df_out = pd.DataFrame(feats, index=range(m))
    assert list(df_out.columns) == SCALE_INVARIANT_FEATURES
    return df_out


def extract_dynamic_only_features(df_rec: pd.DataFrame, window_size: int = WINDOW_SAMPLES) -> pd.DataFrame:
    """
    Extracts strictly causal dynamic/temporal delta features over [k-9 ... k].
    """
    n_samples = len(df_rec)
    w = window_size
    k_indices = np.arange(w - 1, n_samples)
    m = len(k_indices)

    ax = df_rec["phone_accel_x_mps2"].values.astype(np.float64)
    ay = df_rec["phone_accel_y_mps2"].values.astype(np.float64)
    az = df_rec["phone_accel_z_mps2"].values.astype(np.float64)

    gx = df_rec["phone_gyro_x_radps"].values.astype(np.float64)
    gy = df_rec["phone_gyro_y_radps"].values.astype(np.float64)
    gz = df_rec["phone_gyro_z_radps"].values.astype(np.float64)

    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    g0 = STANDARD_GRAVITY_MPS2

    s_amag = pd.Series(accel_mag)
    s_gmag = pd.Series(gyro_mag)

    mu_a = s_amag.rolling(w).mean().values[k_indices]
    sig_a = s_amag.rolling(w).std(ddof=0).values[k_indices]
    mu_w = s_gmag.rolling(w).mean().values[k_indices]
    sig_w = s_gmag.rolling(w).std(ddof=0).values[k_indices]

    # Deltas over 1 second: [k] - [k - (w - 1)]
    d_ax = ax[k_indices] - ax[k_indices - (w - 1)]
    d_ay = ay[k_indices] - ay[k_indices - (w - 1)]
    d_az = az[k_indices] - az[k_indices - (w - 1)]

    d_gx = gx[k_indices] - gx[k_indices - (w - 1)]
    d_gy = gy[k_indices] - gy[k_indices - (w - 1)]
    d_gz = gz[k_indices] - gz[k_indices - (w - 1)]

    d_amag = accel_mag[k_indices] - accel_mag[k_indices - (w - 1)]
    d_gmag = gyro_mag[k_indices] - gyro_mag[k_indices - (w - 1)]
    d_a_norm = np.sqrt(d_ax**2 + d_ay**2 + d_az**2)

    # Mean absolute jerk proxy over window: mean of |amag[i] - amag[i-1]|
    diff_amag = pd.Series(np.abs(np.diff(accel_mag, prepend=accel_mag[0])))
    jerk_proxy = diff_amag.rolling(w).mean().values[k_indices] / g0

    # 1-sample lag correlation over rolling window:
    # Autocorrelation r1 = sum((x_t - mu)(x_{t-1} - mu)) / (w * var)
    # Computed causally across the w samples
    autocorr_a = np.zeros(m)
    autocorr_w = np.zeros(m)
    for idx, k in enumerate(k_indices):
        win_a = accel_mag[k - w + 1 : k + 1]
        win_w = gyro_mag[k - w + 1 : k + 1]
        var_a = np.var(win_a)
        var_w = np.var(win_w)
        if var_a > EPSILON:
            autocorr_a[idx] = np.mean((win_a[1:] - np.mean(win_a)) * (win_a[:-1] - np.mean(win_a))) / (var_a + EPSILON)
        if var_w > EPSILON:
            autocorr_w[idx] = np.mean((win_w[1:] - np.mean(win_w)) * (win_w[:-1] - np.mean(win_w))) / (var_w + EPSILON)

    feats = {
        "norm_delta_ax_1s": d_ax / (mu_a + EPSILON),
        "norm_delta_ay_1s": d_ay / (mu_a + EPSILON),
        "norm_delta_az_1s": d_az / (mu_a + EPSILON),
        "norm_delta_gx_1s": d_gx / (mu_w + EPSILON),
        "norm_delta_gy_1s": d_gy / (mu_w + EPSILON),
        "norm_delta_gz_1s": d_gz / (mu_w + EPSILON),
        "norm_delta_accel_mag": d_amag / g0,
        "norm_delta_gyro_mag": d_gmag / (mu_w + EPSILON),
        "accel_jerk_proxy_1s": jerk_proxy,
        "gyro_accel_ratio_proxy": gyro_mag[k_indices] / (d_a_norm + EPSILON),
        "accel_autocorr_lag1": autocorr_a,
        "gyro_autocorr_lag1": autocorr_w,
    }

    df_out = pd.DataFrame(feats, index=range(m))
    assert list(df_out.columns) == DYNAMIC_ONLY_FEATURES
    return df_out


def compute_causal_running_normalization(
    X_raw: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes strictly causal running z-score of rolling std features up to timestamp t.
    Uses cumulative running mean and std over past samples [0 ... k]:
        mu_run[k] = mean(v[0:k+1])
        std_run[k] = std(v[0:k+1])
        z[k] = (v[k] - mu_run[k]) / (std_run[k] + eps)
    Zero future information is used.
    """
    sig_a = X_raw["accel_mag_rolling_std_1s"].values.astype(np.float64)
    sig_w = X_raw["gyro_mag_rolling_std_1s"].values.astype(np.float64)
    m = len(sig_a)

    # Cumulative running mean and std
    cum_sum_a = np.cumsum(sig_a)
    cum_sum2_a = np.cumsum(sig_a**2)

    cum_sum_w = np.cumsum(sig_w)
    cum_sum2_w = np.cumsum(sig_w**2)

    counts = np.arange(1, m + 1, dtype=np.float64)

    run_mu_a = cum_sum_a / counts
    run_var_a = np.maximum(0.0, (cum_sum2_a / counts) - (run_mu_a**2))
    run_std_a = np.sqrt(run_var_a)

    run_mu_w = cum_sum_w / counts
    run_var_w = np.maximum(0.0, (cum_sum2_w / counts) - (run_mu_w**2))
    run_std_w = np.sqrt(run_var_w)

    z_a = (sig_a - run_mu_a) / (run_std_a + EPSILON)
    z_w = (sig_w - run_mu_w) / (run_std_w + EPSILON)

    return z_a, z_w


def compute_offline_full_recording_normalization(
    X_raw: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    NEGATIVE CONTROL: OFFLINE NON-CAUSAL DIAGNOSTIC — NOT DEPLOYABLE.
    Computes full-recording global z-score of rolling vibration statistics in hindsight.
    """
    sig_a = X_raw["accel_mag_rolling_std_1s"].values.astype(np.float64)
    sig_w = X_raw["gyro_mag_rolling_std_1s"].values.astype(np.float64)

    z_a = (sig_a - np.mean(sig_a)) / (np.std(sig_a) + EPSILON)
    z_w = (sig_w - np.mean(sig_w)) / (np.std(sig_w) + EPSILON)

    return z_a, z_w


# ---------------------------------------------------------------------------
# Vibration-Speed Relationship and Domain Shift Analytics
# ---------------------------------------------------------------------------

def compute_vibration_speed_correlations(
    y_true: np.ndarray,
    X_raw: pd.DataFrame,
) -> Dict[str, float]:
    """
    Computes descriptive Pearson and Spearman correlations between
    reference vehicle speed and rolling vibration features.
    """
    sig_a = X_raw["accel_mag_rolling_std_1s"].values
    sig_w = X_raw["gyro_mag_rolling_std_1s"].values

    p_corr_a, _ = stats.pearsonr(sig_a, y_true)
    s_corr_a, _ = stats.spearmanr(sig_a, y_true)

    p_corr_w, _ = stats.pearsonr(sig_w, y_true)
    s_corr_w, _ = stats.spearmanr(sig_w, y_true)

    return {
        "pearson_accel_std_speed": float(p_corr_a),
        "spearman_accel_std_speed": float(s_corr_a),
        "pearson_gyro_std_speed": float(p_corr_w),
        "spearman_gyro_std_speed": float(s_corr_w),
    }


def compute_speed_conditioned_vibration(
    y_true: np.ndarray,
    X_raw: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """
    Computes speed-conditioned vibration statistics across regimes.
    """
    sig_a = X_raw["accel_mag_rolling_std_1s"].values
    sig_w = X_raw["gyro_mag_rolling_std_1s"].values

    regimes = {
        "stationary (< 0.5 m/s)": y_true < 0.5,
        "low_speed (0.5 - 5 m/s)": (y_true >= 0.5) & (y_true < 5.0),
        "medium_speed (5 - 15 m/s)": (y_true >= 5.0) & (y_true < 15.0),
        "high_speed (>= 15 m/s)": y_true >= 15.0,
    }

    out = {}
    for r_name, mask in regimes.items():
        if np.sum(mask) > 0:
            out[r_name] = {
                "mean_accel_std": float(np.mean(sig_a[mask])),
                "std_accel_std": float(np.std(sig_a[mask])),
                "mean_gyro_std": float(np.mean(sig_w[mask])),
                "std_gyro_std": float(np.std(sig_w[mask])),
                "sample_count": int(np.sum(mask)),
            }
    return out


# ---------------------------------------------------------------------------
# Strict Causal and Leakage Audit
# ---------------------------------------------------------------------------

def run_phase3_2a_causality_audit(
    split: RecordingSplit,
    rep_matrices: Dict[str, Dict[str, pd.DataFrame]],
) -> Dict[str, Any]:
    """
    Verifies:
    1. Zero forbidden substrings across all representation feature names.
    2. Zero target leakage.
    3. Partition recording isolation.
    4. Causal future-perturbation invariance:
       Corrupting future IMU samples (t' > t) produces identical feature vectors
       for Scale-Invariant, Dynamic-Only, and Causal Running representations.
    """
    audit = {}
    all_passed = True

    # 1. Column name check
    forbidden = ["ref_", "vbox", "navris", "gnss", "target", "truth", "eskf", "zupt", "nhc"]
    violations = []
    for rep_name, rec_dict in rep_matrices.items():
        cols = list(rec_dict["S1"].columns)
        for c in cols:
            c_low = c.lower()
            for f in forbidden:
                if f in c_low:
                    violations.append((rep_name, c, f))

    check_forbidden = len(violations) == 0
    audit["check_no_forbidden_fields"] = {"passed": check_forbidden, "violations": violations}
    if not check_forbidden:
        all_passed = False

    # 2. Partition isolation
    s_tr = set(split.train_recordings)
    s_va = set(split.val_recordings)
    s_te = set(split.test_recordings)
    sep_check = len(s_tr & s_va) == 0 and len(s_tr & s_te) == 0 and len(s_va & s_te) == 0
    audit["check_partition_isolation"] = {"passed": sep_check}
    if not sep_check:
        all_passed = False

    # 3. Causal future-perturbation test
    df_raw = load_recording_ml_data("VTA2").iloc[:250].copy()
    k_eval = 150

    # Extract original
    X_raw_orig, _, _ = extract_causal_imu_features(df_raw)
    X_si_orig = extract_scale_invariant_features(df_raw)
    X_dyn_orig = extract_dynamic_only_features(df_raw)
    z_a_orig, z_w_orig = compute_causal_running_normalization(X_raw_orig)

    row_si_orig = X_si_orig.iloc[k_eval - WINDOW_SAMPLES + 1].values
    row_dyn_orig = X_dyn_orig.iloc[k_eval - WINDOW_SAMPLES + 1].values
    z_at_eval_orig = np.array([z_a_orig[k_eval - WINDOW_SAMPLES + 1], z_w_orig[k_eval - WINDOW_SAMPLES + 1]])

    # Corrupt future samples strictly AFTER k_eval
    df_corrupt = df_raw.copy()
    corrupt_cols = [
        "phone_accel_x_mps2", "phone_accel_y_mps2", "phone_accel_z_mps2",
        "phone_gyro_x_radps", "phone_gyro_y_radps", "phone_gyro_z_radps",
    ]
    for c in corrupt_cols:
        df_corrupt.loc[k_eval + 1:, c] = df_corrupt.loc[k_eval + 1:, c] * -88.0 + 777.7

    X_raw_corrupt, _, _ = extract_causal_imu_features(df_corrupt)
    X_si_corrupt = extract_scale_invariant_features(df_corrupt)
    X_dyn_corrupt = extract_dynamic_only_features(df_corrupt)
    z_a_corrupt, z_w_corrupt = compute_causal_running_normalization(X_raw_corrupt)

    row_si_corrupt = X_si_corrupt.iloc[k_eval - WINDOW_SAMPLES + 1].values
    row_dyn_corrupt = X_dyn_corrupt.iloc[k_eval - WINDOW_SAMPLES + 1].values
    z_at_eval_corrupt = np.array([z_a_corrupt[k_eval - WINDOW_SAMPLES + 1], z_w_corrupt[k_eval - WINDOW_SAMPLES + 1]])

    diff_si = float(np.max(np.abs(row_si_orig - row_si_corrupt)))
    diff_dyn = float(np.max(np.abs(row_dyn_orig - row_dyn_corrupt)))
    diff_z = float(np.max(np.abs(z_at_eval_orig - z_at_eval_corrupt)))

    causal_invariance = (diff_si == 0.0) and (diff_dyn == 0.0) and (diff_z == 0.0)
    audit["check_causal_future_perturbation_invariance"] = {
        "passed": causal_invariance,
        "diff_scale_invariant": diff_si,
        "diff_dynamic_only": diff_dyn,
        "diff_causal_running_z": diff_z,
    }
    if not causal_invariance:
        all_passed = False

    audit["overall_causal_audit_passed"] = all_passed
    return audit


# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------

def run_phase3_2a_study(output_dir: str = "data/processed/phase3_2a") -> Dict[str, Any]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    split = RecordingSplit(
        train_recordings=["S1", "S2", "S4"],
        val_recordings=["S3A", "VTA1A"],
        test_recordings=["Y1", "VTA2"],
    )

    print("==================================================")
    print("NAVRIS PHASE 3.2A: VEHICLE-DOMAIN ROBUSTNESS DIAGNOSTIC")
    print("==================================================")
    print(f"TRAIN: {split.train_recordings} (Ford Fiesta, UK)")
    print(f"VAL:   {split.val_recordings} (Fiesta UK, Golf France)")
    print(f"TEST:  {split.test_recordings} (Fiesta UK, Golf France)")

    all_recs = split.train_recordings + split.val_recordings + split.test_recordings

    # 1. Load data and extract all feature representations per recording
    raw_data: Dict[str, Tuple[pd.DataFrame, pd.Series, pd.DataFrame]] = {}
    rep_si: Dict[str, pd.DataFrame] = {}
    rep_dyn: Dict[str, pd.DataFrame] = {}
    rep_causal_run: Dict[str, pd.DataFrame] = {}
    rep_offline_norm: Dict[str, pd.DataFrame] = {}

    vibration_corr_list = []
    speed_cond_vib_list = []
    feat_dist_list = []

    for rec_id in all_recs:
        print(f"Extracting diagnostic representations for {rec_id}...")
        df_raw = load_recording_ml_data(rec_id)

        X_raw, y_rec, meta_rec = extract_causal_imu_features(df_raw)
        raw_data[rec_id] = (X_raw, y_rec, meta_rec)

        X_si = extract_scale_invariant_features(df_raw)
        rep_si[rec_id] = X_si

        X_dyn = extract_dynamic_only_features(df_raw)
        rep_dyn[rec_id] = X_dyn

        # Causal running z-score
        z_a_causal, z_w_causal = compute_causal_running_normalization(X_raw)
        df_causal_run = X_raw.copy()
        df_causal_run["causal_running_z_accel_std"] = z_a_causal
        df_causal_run["causal_running_z_gyro_std"] = z_w_causal
        rep_causal_run[rec_id] = df_causal_run

        # OFFLINE non-causal z-score (negative control)
        z_a_off, z_w_off = compute_offline_full_recording_normalization(X_raw)
        df_off_norm = X_raw.copy()
        df_off_norm["offline_z_accel_std"] = z_a_off
        df_off_norm["offline_z_gyro_std"] = z_w_off
        rep_offline_norm[rec_id] = df_off_norm

        # Analytics: Vibration-Speed correlation
        corr_dict = compute_vibration_speed_correlations(y_rec.values, X_raw)
        corr_dict["recording_id"] = rec_id
        corr_dict["platform"] = "Ford Fiesta" if rec_id in ["S1", "S2", "S4", "S3A", "Y1"] else "VW Golf"
        corr_dict["country"] = "UK" if rec_id in ["S1", "S2", "S4", "S3A", "Y1"] else "France"
        vibration_corr_list.append(corr_dict)

        # Analytics: Speed-conditioned vibration
        sc_vib = compute_speed_conditioned_vibration(y_rec.values, X_raw)
        for r_name, v_stats in sc_vib.items():
            row = {"recording_id": rec_id, "speed_regime": r_name}
            row.update(v_stats)
            speed_cond_vib_list.append(row)

        # Analytics: Distribution metrics of primary features
        for col in ["accel_mag_rolling_std_1s", "gyro_mag_rolling_std_1s"]:
            dist = compute_distribution_stats(X_raw[col].values)
            row = {
                "recording_id": rec_id,
                "feature": col,
                "platform": "Ford Fiesta" if rec_id in ["S1", "S2", "S4", "S3A", "Y1"] else "VW Golf",
            }
            row.update(dist)
            feat_dist_list.append(row)

    # Save feature distribution and vibration-speed CSVs
    df_vib_corr = pd.DataFrame(vibration_corr_list)
    df_vib_corr.to_csv(out_path / "vibration_speed_relationship.csv", index=False)
    print("\nVibration vs Reference Speed Correlations:")
    print(df_vib_corr[["recording_id", "platform", "pearson_accel_std_speed", "spearman_accel_std_speed", "pearson_gyro_std_speed"]])

    df_feat_dist = pd.DataFrame(feat_dist_list)
    df_feat_dist.to_csv(out_path / "feature_distribution_metrics.csv", index=False)

    # 2. Causality and Leakage Audit
    all_reps_dict = {
        "Raw_IMU": {r: raw_data[r][0] for r in all_recs},
        "Scale_Invariant": rep_si,
        "Dynamic_Only": rep_dyn,
        "Causal_Running": rep_causal_run,
        "Offline_Normalized": rep_offline_norm,
    }
    causal_audit = run_phase3_2a_causality_audit(split, all_reps_dict)
    with open(out_path / "causal_audit_summary.json", "w") as f:
        json.dump(causal_audit, f, indent=2)
    print(f"\nCausality Audit Passed: {causal_audit['overall_causal_audit_passed']}")

    # 3. Assemble Train Targets
    y_train = np.concatenate([raw_data[r][1].values for r in split.train_recordings])
    y_val = np.concatenate([raw_data[r][1].values for r in split.val_recordings])
    y_test = np.concatenate([raw_data[r][1].values for r in split.test_recordings])

    # 4. Train and Evaluate Representations
    # Representations to evaluate
    representations_spec = {
        "1. Raw IMU (Phase 3.2 Reference)": (
            ABLATION_C_FEATURES,
            {r: raw_data[r][0][ABLATION_C_FEATURES] for r in all_recs},
        ),
        "2. Scale-Invariant (Dimensionless)": (
            SCALE_INVARIANT_FEATURES,
            rep_si,
        ),
        "3. Dynamic-Only (Deltas & Jerk)": (
            DYNAMIC_ONLY_FEATURES,
            rep_dyn,
        ),
        "4. Causal Running Normalization": (
            CAUSAL_RUNNING_FEATURES,
            rep_causal_run,
        ),
        "5. OFFLINE Non-Causal (Negative Control)": (
            OFFLINE_NORMALIZED_FEATURES,
            rep_offline_norm,
        ),
    }

    rep_metrics_list = []
    domain_metrics_list = []
    speed_regime_list = []

    print("\n--- Training and Evaluating Diagnostic Representations ---")
    for rep_name, (feat_cols, rec_dfs) in representations_spec.items():
        print(f"Training XGBoost on {rep_name}...")

        # Concat Train
        X_train = pd.concat([rec_dfs[r][feat_cols] for r in split.train_recordings], ignore_index=True)
        X_val = pd.concat([rec_dfs[r][feat_cols] for r in split.val_recordings], ignore_index=True)
        X_test = pd.concat([rec_dfs[r][feat_cols] for r in split.test_recordings], ignore_index=True)

        # Fixed XGBoost architecture
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # Evaluate across partitions
        preds = {
            "TRAIN": model.predict(X_train),
            "VAL": model.predict(X_val),
            "TEST": model.predict(X_test),
        }
        for r in all_recs:
            preds[r] = model.predict(rec_dfs[r][feat_cols])

        # Record representation metrics
        for p_name, y_p in preds.items():
            if p_name == "TRAIN":
                y_t = y_train
            elif p_name == "VAL":
                y_t = y_val
            elif p_name == "TEST":
                y_t = y_test
            else:
                y_t = raw_data[p_name][1].values

            m = compute_regression_metrics(y_t, y_p)
            m["representation"] = rep_name
            m["partition_or_recording"] = p_name
            rep_metrics_list.append(m)

        # Domain shift tracking: Same-Platform vs Cross-Platform
        # Same-Platform: Train -> S3A (Val), Train -> Y1 (Test)
        # Cross-Platform: Train -> VTA1A (Val), Train -> VTA2 (Test)
        domain_metrics_list.append({
            "representation": rep_name,
            "fiesta_val_S3A_mae": float(compute_regression_metrics(raw_data["S3A"][1].values, preds["S3A"])["mae"]),
            "golf_val_VTA1A_mae": float(compute_regression_metrics(raw_data["VTA1A"][1].values, preds["VTA1A"])["mae"]),
            "fiesta_val_S3A_r2": float(compute_regression_metrics(raw_data["S3A"][1].values, preds["S3A"])["r2"]),
            "golf_val_VTA1A_r2": float(compute_regression_metrics(raw_data["VTA1A"][1].values, preds["VTA1A"])["r2"]),
            "fiesta_test_Y1_mae": float(compute_regression_metrics(raw_data["Y1"][1].values, preds["Y1"])["mae"]),
            "golf_test_VTA2_mae": float(compute_regression_metrics(raw_data["VTA2"][1].values, preds["VTA2"])["mae"]),
            "fiesta_test_Y1_r2": float(compute_regression_metrics(raw_data["Y1"][1].values, preds["Y1"])["r2"]),
            "golf_test_VTA2_r2": float(compute_regression_metrics(raw_data["VTA2"][1].values, preds["VTA2"])["r2"]),
        })

        # Speed-regime analysis on Held-Out Test (Y1 and VTA2)
        for rec_id in ["Y1", "VTA2"]:
            y_t = raw_data[rec_id][1].values
            y_p = preds[rec_id]
            regimes = {
                "stationary (< 0.5 m/s)": y_t < 0.5,
                "low_speed (0.5 - 5 m/s)": (y_t >= 0.5) & (y_t < 5.0),
                "medium_speed (5 - 15 m/s)": (y_t >= 5.0) & (y_t < 15.0),
                "high_speed (>= 15 m/s)": y_t >= 15.0,
            }
            for reg_name, mask in regimes.items():
                if np.sum(mask) > 0:
                    rm = compute_regression_metrics(y_t[mask], y_p[mask])
                    speed_regime_list.append({
                        "representation": rep_name,
                        "recording_id": rec_id,
                        "speed_regime": reg_name,
                        "mae": rm["mae"],
                        "rmse": rm["rmse"],
                        "sample_count": int(np.sum(mask)),
                    })

    # Save output CSVs
    df_rep_metrics = pd.DataFrame(rep_metrics_list)
    df_rep_metrics.to_csv(out_path / "representation_metrics.csv", index=False)

    df_domain = pd.DataFrame(domain_metrics_list)
    df_domain.to_csv(out_path / "domain_metrics.csv", index=False)
    print("\nDomain Transfer Metrics (Same-Platform vs Cross-Platform):")
    print(df_domain[["representation", "fiesta_test_Y1_mae", "golf_test_VTA2_mae", "fiesta_test_Y1_r2", "golf_test_VTA2_r2"]])

    df_speed_reg = pd.DataFrame(speed_regime_list)
    df_speed_reg.to_csv(out_path / "speed_regime_metrics.csv", index=False)

    # 5. Save Study Summary JSON
    summary = {
        "status": "PHASE 3.2A COMPLETE",
        "causal_audit_passed": bool(causal_audit["overall_causal_audit_passed"]),
        "domain_transfer_summary": df_domain.to_dict(orient="records"),
        "representation_summary": df_rep_metrics.to_dict(orient="records"),
    }
    with open(out_path / "phase3_2a_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nPhase 3.2A Diagnostics Completed Successfully.")
    return summary


if __name__ == "__main__":
    run_phase3_2a_study()
