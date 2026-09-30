"""
NAVRIS Phase 3.2: Direct Forward-Speed ML Baseline.

Hypothesis:
Smartphone IMU measurements contain sufficient causal information to estimate
instantaneous vehicle forward speed v_fwd_ref(t) without relying on accumulated
NAVRIS velocity state or GNSS fixes.

Strict Constraints:
1. Inputs: Causal phone-frame IMU signals ONLY over a 1.0-second backward window [k-9 ... k].
2. Target: v_fwd_ref(t) from validated reference data (OFFLINE LABEL ONLY).
3. No NAVRIS filter states, no GNSS features, no reference features, no future lookahead.
4. Partitions: Train (S1, S2, S4), Validation (S3A, VTA1A), Held-Out Test (Y1, VTA2).
5. Preprocessing (StandardScaler) fitted strictly on TRAIN.
6. Baseline Models:
   - Baseline 0: Constant/Mean Predictor
   - Baseline 1: Regularized Linear Regression (Ridge)
   - Baseline 2: Small Tree Model (XGBoost, max_depth=4, n_estimators=100, lr=0.05)
7. Ablations:
   - Ablation A: Instantaneous IMU (9 features)
   - Ablation B: IMU + 1s causal rolling stats (13 features)
   - Ablation C: IMU + 1s causal rolling stats + 1s causal deltas (19 features)
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from navris.ml.dataset import RecordingSplit, load_recording_ml_data


# ---------------------------------------------------------------------------
# Feature Column Definitions
# ---------------------------------------------------------------------------

ABLATION_A_FEATURES: List[str] = [
    "phone_accel_x_mps2",
    "phone_accel_y_mps2",
    "phone_accel_z_mps2",
    "phone_gyro_x_radps",
    "phone_gyro_y_radps",
    "phone_gyro_z_radps",
    "accel_magnitude_mps2",
    "gyro_magnitude_radps",
    "gravity_deviation_mps2",
]

ABLATION_B_EXTRA: List[str] = [
    "accel_mag_rolling_mean_1s",
    "accel_mag_rolling_std_1s",
    "gyro_mag_rolling_mean_1s",
    "gyro_mag_rolling_std_1s",
]

ABLATION_C_EXTRA: List[str] = [
    "delta_accel_x_1s",
    "delta_accel_y_1s",
    "delta_accel_z_1s",
    "delta_gyro_x_1s",
    "delta_gyro_y_1s",
    "delta_gyro_z_1s",
]

ABLATION_B_FEATURES = ABLATION_A_FEATURES + ABLATION_B_EXTRA
ABLATION_C_FEATURES = ABLATION_B_FEATURES + ABLATION_C_EXTRA

FORBIDDEN_SUBSTRINGS = [
    "ref_",
    "vbox",
    "target",
    "truth",
    "navris",
    "gnss",
    "eskf",
    "zupt",
    "nhc",
    "oracle",
]

STANDARD_GRAVITY_MPS2 = 9.80665
WINDOW_SAMPLES = 10  # 1.0 s at 10 Hz


# ---------------------------------------------------------------------------
# Feature Extraction (Strictly Causal, Phone IMU Only)
# ---------------------------------------------------------------------------

def extract_causal_imu_features(
    df_rec: pd.DataFrame,
    window_size: int = WINDOW_SAMPLES,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Extracts strictly causal IMU features, reference forward speed target,
    and metadata for a single recording.

    Args:
        df_rec: Aligned recording DataFrame from load_recording_ml_data.
        window_size: Number of samples in causal history window (default 10 = 1.0s).

    Returns:
        X: DataFrame of 19 strictly causal features over [k-9 ... k].
        y: Series of true reference forward speed (m/s) at time k.
        meta: Metadata DataFrame (time_s, recording_id, ref_accel, etc.).
    """
    n_samples = len(df_rec)
    if n_samples < window_size:
        raise ValueError(
            f"Recording has {n_samples} samples, less than window size {window_size}"
        )

    ax = df_rec["phone_accel_x_mps2"].values.astype(np.float64)
    ay = df_rec["phone_accel_y_mps2"].values.astype(np.float64)
    az = df_rec["phone_accel_z_mps2"].values.astype(np.float64)

    gx = df_rec["phone_gyro_x_radps"].values.astype(np.float64)
    gy = df_rec["phone_gyro_y_radps"].values.astype(np.float64)
    gz = df_rec["phone_gyro_z_radps"].values.astype(np.float64)

    # Derived instantaneous norms
    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    grav_dev = np.abs(accel_mag - STANDARD_GRAVITY_MPS2)

    # Rolling causal statistics strictly over [k - window_size + 1 ... k]
    w = window_size
    k_indices = np.arange(w - 1, n_samples)
    m = len(k_indices)

    s_accel_mag = pd.Series(accel_mag)
    s_gyro_mag = pd.Series(gyro_mag)

    accel_mag_mean = s_accel_mag.rolling(w).mean().values[k_indices]
    accel_mag_std = s_accel_mag.rolling(w).std(ddof=0).values[k_indices]
    gyro_mag_mean = s_gyro_mag.rolling(w).mean().values[k_indices]
    gyro_mag_std = s_gyro_mag.rolling(w).std(ddof=0).values[k_indices]

    # 1.0s deltas: val[k] - val[k - 9]
    delta_ax = ax[k_indices] - ax[k_indices - (w - 1)]
    delta_ay = ay[k_indices] - ay[k_indices - (w - 1)]
    delta_az = az[k_indices] - az[k_indices - (w - 1)]

    delta_gx = gx[k_indices] - gx[k_indices - (w - 1)]
    delta_gy = gy[k_indices] - gy[k_indices - (w - 1)]
    delta_gz = gz[k_indices] - gz[k_indices - (w - 1)]

    feat_dict = {
        # Instantaneous IMU (Ablation A)
        "phone_accel_x_mps2": ax[k_indices],
        "phone_accel_y_mps2": ay[k_indices],
        "phone_accel_z_mps2": az[k_indices],
        "phone_gyro_x_radps": gx[k_indices],
        "phone_gyro_y_radps": gy[k_indices],
        "phone_gyro_z_radps": gz[k_indices],
        "accel_magnitude_mps2": accel_mag[k_indices],
        "gyro_magnitude_radps": gyro_mag[k_indices],
        "gravity_deviation_mps2": grav_dev[k_indices],
        # 1-second rolling statistics (Ablation B extra)
        "accel_mag_rolling_mean_1s": accel_mag_mean,
        "accel_mag_rolling_std_1s": accel_mag_std,
        "gyro_mag_rolling_mean_1s": gyro_mag_mean,
        "gyro_mag_rolling_std_1s": gyro_mag_std,
        # 1-second causal temporal deltas (Ablation C extra)
        "delta_accel_x_1s": delta_ax,
        "delta_accel_y_1s": delta_ay,
        "delta_accel_z_1s": delta_az,
        "delta_gyro_x_1s": delta_gx,
        "delta_gyro_y_1s": delta_gy,
        "delta_gyro_z_1s": delta_gz,
    }

    df_X = pd.DataFrame(feat_dict, index=range(m))
    y = pd.Series(df_rec["ref_speed_mps"].values[k_indices], name="target_v_fwd_mps", index=range(m))

    meta_dict = {
        "time_s": df_rec["time_s"].values[k_indices],
        "recording_id": [df_rec["recording_id"].iloc[0]] * m,
        "ref_speed_mps": df_rec["ref_speed_mps"].values[k_indices],
    }
    df_meta = pd.DataFrame(meta_dict, index=range(m))

    return df_X, y, df_meta


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes comprehensive regression metrics."""
    err = y_pred - y_true
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err**2)))
    med_ae = float(np.median(abs_err))
    p95_ae = float(np.percentile(abs_err, 95))
    max_ae = float(np.max(abs_err))

    # R^2
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    ss_res = np.sum(err**2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-9 else float("nan")

    # Pearson correlation
    if np.std(y_pred) > 1e-9 and np.std(y_true) > 1e-9:
        corr = float(np.corrcoef(y_pred, y_true)[0, 1])
    else:
        corr = 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "median_ae": med_ae,
        "p95_ae": p95_ae,
        "max_ae": max_ae,
        "r2": r2,
        "correlation": corr,
    }


def compute_distribution_stats(arr: np.ndarray) -> Dict[str, float]:
    """Computes empirical distribution percentiles and bounds."""
    clean = arr[~np.isnan(arr)]
    if len(clean) == 0:
        return {}
    return {
        "count": int(len(clean)),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean)),
        "median": float(np.median(clean)),
        "p95": float(np.percentile(clean, 95)),
        "p99": float(np.percentile(clean, 99)),
        "p99.9": float(np.percentile(clean, 99.9)),
    }


# ---------------------------------------------------------------------------
# Temporal Dynamics & Regime Analysis
# ---------------------------------------------------------------------------

def analyze_speed_regimes(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Evaluates performance across clearly defined speed regimes."""
    regimes = {
        "stationary (< 0.5 m/s)": y_true < 0.5,
        "low_speed (0.5 - 5 m/s)": (y_true >= 0.5) & (y_true < 5.0),
        "medium_speed (5 - 15 m/s)": (y_true >= 5.0) & (y_true < 15.0),
        "high_speed (>= 15 m/s)": y_true >= 15.0,
    }
    out = {}
    for name, mask in regimes.items():
        if np.sum(mask) > 0:
            m = compute_regression_metrics(y_true[mask], y_pred[mask])
            m["sample_count"] = int(np.sum(mask))
            m["sample_pct"] = float(np.mean(mask) * 100.0)
            out[name] = m
        else:
            out[name] = {"sample_count": 0}
    return out


def analyze_dynamic_behaviors(
    time_s: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates speed estimation under distinct dynamic conditions:
    - Diagnostic reference categories: acceleration, braking, steady cruising, low speed maneuvering.
    - Causal IMU categories: stationary IMU, high rotation/turning.
    """
    dt = np.gradient(time_s)
    dt[dt <= 0] = 0.1
    accel_ref = np.gradient(y_true, time_s)

    # Categories
    categories = {
        "stationary_ref (< 0.5 m/s)": y_true < 0.5,
        "acceleration_ref (> 0.5 m/s2)": accel_ref > 0.5,
        "braking_ref (< -0.5 m/s2)": accel_ref < -0.5,
        "steady_cruising_ref (|a| <= 0.5 m/s2, v >= 5 m/s)": (np.abs(accel_ref) <= 0.5) & (y_true >= 5.0),
        "low_speed_maneuver_ref (0.5 <= v < 5 m/s)": (y_true >= 0.5) & (y_true < 5.0),
        "turning_causal (gyro_mag >= 0.15 rad/s)": X["gyro_magnitude_radps"].values >= 0.15,
        "stationary_causal (accel_std < 0.10 m/s2, gyro < 0.05 rad/s)": (
            (X["accel_mag_rolling_std_1s"].values < 0.10) & (X["gyro_magnitude_radps"].values < 0.05)
        ),
    }

    out = {}
    for name, mask in categories.items():
        if np.sum(mask) > 5:
            m = compute_regression_metrics(y_true[mask], y_pred[mask])
            m["sample_count"] = int(np.sum(mask))
            m["sample_pct"] = float(np.mean(mask) * 100.0)
            out[name] = m
        else:
            out[name] = {"sample_count": int(np.sum(mask))}
    return out


def compute_temporal_lag_and_derivative(
    time_s: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_lag_samples: int = 30,  # 3.0 seconds at 10 Hz
) -> Dict[str, float]:
    """
    Calculates empirical prediction lag via cross-correlation peak,
    and speed derivative correlation.
    """
    # Cross correlation on zero-mean signals
    y_t_zm = y_true - np.mean(y_true)
    y_p_zm = y_pred - np.mean(y_pred)

    if np.std(y_t_zm) < 1e-6 or np.std(y_p_zm) < 1e-6:
        return {"lag_seconds": 0.0, "derivative_corr": 0.0}

    # Cross-correlation: mode='full'
    xcorr = np.correlate(y_t_zm, y_p_zm, mode="full")
    lags = np.arange(-len(y_true) + 1, len(y_true))

    # Restrict to [-max_lag_samples, max_lag_samples]
    valid_mask = (lags >= -max_lag_samples) & (lags <= max_lag_samples)
    restricted_lags = lags[valid_mask]
    restricted_xcorr = xcorr[valid_mask]

    peak_idx = np.argmax(restricted_xcorr)
    best_lag_samples = restricted_lags[peak_idx]
    # Note: correlate(y_t, y_p) at lag k sums y_t[i] * y_p[i-k].
    # If peak is at k > 0, y_t[i] matches y_p[i-k], meaning prediction y_p is shifted to the left (leads reference).
    # If peak is at k < 0, prediction lags reference.
    # Convert to physical lag: positive tau means prediction lags reference by tau seconds.
    lag_seconds = float(-best_lag_samples * 0.1)

    # Speed derivative correlation (finite difference)
    dy_t = np.diff(y_true)
    dy_p = np.diff(y_pred)
    if np.std(dy_t) > 1e-6 and np.std(dy_p) > 1e-6:
        deriv_corr = float(np.corrcoef(dy_t, dy_p)[0, 1])
    else:
        deriv_corr = 0.0

    return {
        "best_lag_samples": int(-best_lag_samples),
        "lag_seconds": lag_seconds,
        "derivative_correlation": deriv_corr,
    }


# ---------------------------------------------------------------------------
# Strict Leakage Audit & Causal Perturbation Test
# ---------------------------------------------------------------------------

def run_causal_leakage_audit(
    split: RecordingSplit,
    train_recordings: Dict[str, Tuple[pd.DataFrame, pd.Series, pd.DataFrame]],
    val_recordings: Dict[str, Tuple[pd.DataFrame, pd.Series, pd.DataFrame]],
    test_recordings: Dict[str, Tuple[pd.DataFrame, pd.Series, pd.DataFrame]],
    scaler: StandardScaler,
    model: Any,
) -> Dict[str, Any]:
    """
    Automated leakage audit verifying:
    1. Zero forbidden substrings in feature columns.
    2. Zero target columns in feature matrix.
    3. Strict recording partition separation.
    4. Feature scaler isolation (fitted on TRAIN only).
    5. Causal future-perturbation test:
       Perturbing future IMU samples (t' > t) does NOT alter the feature vector at t
       and produces an identical prediction at t (discrepancy = 0.0).
    """
    audit: Dict[str, Any] = {}
    all_passed = True

    # 1. Forbidden column names in X
    feat_cols = list(train_recordings["S1"][0].columns)
    forbidden_matches = []
    for c in feat_cols:
        c_low = c.lower()
        for f in FORBIDDEN_SUBSTRINGS:
            if f in c_low:
                forbidden_matches.append((c, f))
    check_forbidden = len(forbidden_matches) == 0
    audit["check_no_forbidden_fields_in_features"] = {
        "passed": check_forbidden,
        "violations": forbidden_matches,
    }
    if not check_forbidden:
        all_passed = False

    # 2. Target in features
    target_in_x = "target_v_fwd_mps" in feat_cols or "ref_speed_mps" in feat_cols
    audit["check_no_target_in_features"] = {
        "passed": not target_in_x,
        "violations": ["target_v_fwd_mps"] if target_in_x else [],
    }
    if target_in_x:
        all_passed = False

    # 3. Partition separation
    s_tr = set(split.train_recordings)
    s_va = set(split.val_recordings)
    s_te = set(split.test_recordings)
    sep_check = len(s_tr & s_va) == 0 and len(s_tr & s_te) == 0 and len(s_va & s_te) == 0
    audit["check_partition_separation"] = {
        "passed": sep_check,
        "train": list(s_tr),
        "val": list(s_va),
        "test": list(s_te),
    }
    if not sep_check:
        all_passed = False

    # 4. Scaler fitted strictly on TRAIN
    X_train_concat = pd.concat([train_recordings[r][0] for r in split.train_recordings], ignore_index=True)
    ref_scaler = StandardScaler().fit(X_train_concat[ABLATION_C_FEATURES])
    diff_mean = float(np.max(np.abs(scaler.mean_ - ref_scaler.mean_)))
    diff_scale = float(np.max(np.abs(scaler.scale_ - ref_scaler.scale_)))
    check_scaler = max(diff_mean, diff_scale) < 1e-7
    audit["check_scaler_isolation"] = {
        "passed": check_scaler,
        "max_mean_diff": diff_mean,
        "max_scale_diff": diff_scale,
    }
    if not check_scaler:
        all_passed = False

    # 5. Causal future-perturbation test
    # Load unwindowed raw recording for VTA2
    df_raw_vta2 = load_recording_ml_data("VTA2")
    # Choose arbitrary evaluation timestamp t at index k=150
    k_eval = 150
    # Original extraction up to k_eval + 50
    X_orig, y_orig, meta_orig = extract_causal_imu_features(df_raw_vta2.iloc[:k_eval + 50].copy())
    feat_at_eval_orig = X_orig.iloc[k_eval - WINDOW_SAMPLES + 1].values

    # Corrupt all future samples strictly AFTER k_eval with massive noise and inverted signs
    df_corrupted = df_raw_vta2.iloc[:k_eval + 50].copy()
    corrupt_cols = [
        "phone_accel_x_mps2", "phone_accel_y_mps2", "phone_accel_z_mps2",
        "phone_gyro_x_radps", "phone_gyro_y_radps", "phone_gyro_z_radps",
    ]
    for c in corrupt_cols:
        df_corrupted.loc[k_eval + 1:, c] = df_corrupted.loc[k_eval + 1:, c] * 100.0 + 999.0

    X_corrupt, y_corrupt, meta_corrupt = extract_causal_imu_features(df_corrupted)
    feat_at_eval_corrupt = X_corrupt.iloc[k_eval - WINDOW_SAMPLES + 1].values

    feat_diff = float(np.max(np.abs(feat_at_eval_orig - feat_at_eval_corrupt)))

    # Predict at eval time
    pred_orig = float(model.predict(X_orig.iloc[[k_eval - WINDOW_SAMPLES + 1]][ABLATION_C_FEATURES])[0])
    pred_corrupt = float(model.predict(X_corrupt.iloc[[k_eval - WINDOW_SAMPLES + 1]][ABLATION_C_FEATURES])[0])
    pred_diff = float(np.abs(pred_orig - pred_corrupt))

    check_causal = (feat_diff == 0.0) and (pred_diff == 0.0)
    audit["check_causal_future_perturbation_invariance"] = {
        "passed": check_causal,
        "max_feature_difference": feat_diff,
        "prediction_difference": pred_diff,
        "eval_timestamp_s": float(meta_orig["time_s"].iloc[k_eval - WINDOW_SAMPLES + 1]),
    }
    if not check_causal:
        all_passed = False

    audit["overall_leakage_audit_passed"] = all_passed
    return audit


# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------

def run_phase3_2_experiment(output_dir: str = "data/processed/phase3_2") -> Dict[str, Any]:
    """
    Executes the full Phase 3.2 baseline study.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    split = RecordingSplit(
        train_recordings=["S1", "S2", "S4"],
        val_recordings=["S3A", "VTA1A"],
        test_recordings=["Y1", "VTA2"],
    )

    print("==================================================")
    print("NAVRIS PHASE 3.2: DIRECT FORWARD-SPEED ML BASELINE")
    print("==================================================")
    print(f"TRAIN: {split.train_recordings}")
    print(f"VAL:   {split.val_recordings}")
    print(f"TEST:  {split.test_recordings}")

    # 1. Load data and extract strictly causal features per recording
    rec_data: Dict[str, Tuple[pd.DataFrame, pd.Series, pd.DataFrame]] = {}
    target_stats: Dict[str, Dict[str, float]] = {}

    all_recs = split.train_recordings + split.val_recordings + split.test_recordings
    for rec_id in all_recs:
        print(f"Loading and extracting causal features for {rec_id}...")
        df_raw = load_recording_ml_data(rec_id)
        X_rec, y_rec, meta_rec = extract_causal_imu_features(df_raw)
        rec_data[rec_id] = (X_rec, y_rec, meta_rec)
        target_stats[rec_id] = compute_distribution_stats(y_rec.values)

    # Target stats by partition
    y_train_all = np.concatenate([rec_data[r][1].values for r in split.train_recordings])
    y_val_all = np.concatenate([rec_data[r][1].values for r in split.val_recordings])
    y_test_all = np.concatenate([rec_data[r][1].values for r in split.test_recordings])
    y_global_all = np.concatenate([y_train_all, y_val_all, y_test_all])

    target_stats["TRAIN_POOL"] = compute_distribution_stats(y_train_all)
    target_stats["VAL_POOL"] = compute_distribution_stats(y_val_all)
    target_stats["TEST_POOL"] = compute_distribution_stats(y_test_all)
    target_stats["GLOBAL_POOL"] = compute_distribution_stats(y_global_all)

    # Save target distribution summary
    df_target_dist = pd.DataFrame(target_stats).T
    df_target_dist.to_csv(out_path / "target_distributions.csv", index_label="recording_or_partition")
    print("\nTarget Distributions Summary (m/s):")
    print(df_target_dist[["count", "min", "max", "mean", "median", "p95", "p99", "p99.9"]])

    # 2. Assemble training matrices
    X_train_full = pd.concat([rec_data[r][0] for r in split.train_recordings], ignore_index=True)
    y_train = y_train_all

    X_val_full = pd.concat([rec_data[r][0] for r in split.val_recordings], ignore_index=True)
    y_val = y_val_all

    X_test_full = pd.concat([rec_data[r][0] for r in split.test_recordings], ignore_index=True)
    y_test = y_test_all

    # 3. Fit scaler strictly on TRAIN (full feature set C)
    scaler_C = StandardScaler().fit(X_train_full[ABLATION_C_FEATURES])
    scaler_A = StandardScaler().fit(X_train_full[ABLATION_A_FEATURES])
    scaler_B = StandardScaler().fit(X_train_full[ABLATION_B_FEATURES])

    # 4. Train Baseline Models
    print("\n--- Training Baseline Models ---")

    # Baseline 0: Constant Mean Predictor
    mean_speed_train = float(np.mean(y_train))
    print(f"Baseline 0 (Mean Predictor): constant speed = {mean_speed_train:.4f} m/s")

    def predict_b0(n: int) -> np.ndarray:
        return np.full(n, mean_speed_train)

    # Baseline 1: Regularized Linear Regression (Ridge) on Feature Set C
    X_train_C_scaled = scaler_C.transform(X_train_full[ABLATION_C_FEATURES])
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_C_scaled, y_train)

    # Baseline 2: Lightweight XGBoost on Feature Set C
    # Config: n_estimators=100, max_depth=4, lr=0.05, subsample=0.8, colsample_bytree=0.8
    xgb_c = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    xgb_c.fit(X_train_full[ABLATION_C_FEATURES], y_train)

    # 5. Run Controlled Ablations with Baseline 2 (XGBoost)
    print("\n--- Running Controlled Feature Ablations (XGBoost) ---")
    xgb_a = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    xgb_a.fit(X_train_full[ABLATION_A_FEATURES], y_train)

    xgb_b = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    xgb_b.fit(X_train_full[ABLATION_B_FEATURES], y_train)

    # 6. Run Strict Leakage Audit
    print("\n--- Running Causal & Partition Leakage Audit ---")
    leakage_audit = run_causal_leakage_audit(
        split=split,
        train_recordings=rec_data,
        val_recordings=rec_data,
        test_recordings=rec_data,
        scaler=scaler_C,
        model=xgb_c,
    )
    with open(out_path / "leakage_audit_summary.json", "w") as f:
        json.dump(leakage_audit, f, indent=2)
    print(f"Leakage Audit Overall Passed: {leakage_audit['overall_leakage_audit_passed']}")
    print(f"Causal Perturbation Invariance: {leakage_audit['check_causal_future_perturbation_invariance']['passed']}")

    # 7. Evaluate Baseline Models across Partitions
    model_preds = {
        "Baseline 0 (Mean Predictor)": {
            "TRAIN": predict_b0(len(y_train)),
            "VAL": predict_b0(len(y_val)),
            "TEST": predict_b0(len(y_test)),
            "Y1": predict_b0(len(rec_data["Y1"][1])),
            "VTA2": predict_b0(len(rec_data["VTA2"][1])),
            "S3A": predict_b0(len(rec_data["S3A"][1])),
            "VTA1A": predict_b0(len(rec_data["VTA1A"][1])),
        },
        "Baseline 1 (Ridge Linear)": {
            "TRAIN": ridge_model.predict(X_train_C_scaled),
            "VAL": ridge_model.predict(scaler_C.transform(X_val_full[ABLATION_C_FEATURES])),
            "TEST": ridge_model.predict(scaler_C.transform(X_test_full[ABLATION_C_FEATURES])),
            "Y1": ridge_model.predict(scaler_C.transform(rec_data["Y1"][0][ABLATION_C_FEATURES])),
            "VTA2": ridge_model.predict(scaler_C.transform(rec_data["VTA2"][0][ABLATION_C_FEATURES])),
            "S3A": ridge_model.predict(scaler_C.transform(rec_data["S3A"][0][ABLATION_C_FEATURES])),
            "VTA1A": ridge_model.predict(scaler_C.transform(rec_data["VTA1A"][0][ABLATION_C_FEATURES])),
        },
        "Baseline 2 (XGBoost C)": {
            "TRAIN": xgb_c.predict(X_train_full[ABLATION_C_FEATURES]),
            "VAL": xgb_c.predict(X_val_full[ABLATION_C_FEATURES]),
            "TEST": xgb_c.predict(X_test_full[ABLATION_C_FEATURES]),
            "Y1": xgb_c.predict(rec_data["Y1"][0][ABLATION_C_FEATURES]),
            "VTA2": xgb_c.predict(rec_data["VTA2"][0][ABLATION_C_FEATURES]),
            "S3A": xgb_c.predict(rec_data["S3A"][0][ABLATION_C_FEATURES]),
            "VTA1A": xgb_c.predict(rec_data["VTA1A"][0][ABLATION_C_FEATURES]),
        },
    }

    baseline_metrics_list = []
    for model_name, preds_dict in model_preds.items():
        for part_name, y_p in preds_dict.items():
            if part_name == "TRAIN":
                y_t = y_train
            elif part_name == "VAL":
                y_t = y_val
            elif part_name == "TEST":
                y_t = y_test
            else:
                y_t = rec_data[part_name][1].values

            m = compute_regression_metrics(y_t, y_p)
            m["model"] = model_name
            m["partition_or_recording"] = part_name
            baseline_metrics_list.append(m)

    df_baseline_metrics = pd.DataFrame(baseline_metrics_list)
    df_baseline_metrics.to_csv(out_path / "baseline_model_metrics.csv", index=False)
    print("\nBaseline Model Partition Metrics:")
    print(df_baseline_metrics.pivot(index="model", columns="partition_or_recording", values=["mae", "rmse", "r2"]))

    # 8. Ablation Metrics Comparison
    ablation_preds = {
        "Ablation A (Instantaneous IMU, 9 feats)": {
            "TRAIN": xgb_a.predict(X_train_full[ABLATION_A_FEATURES]),
            "VAL": xgb_a.predict(X_val_full[ABLATION_A_FEATURES]),
            "TEST": xgb_a.predict(X_test_full[ABLATION_A_FEATURES]),
            "Y1": xgb_a.predict(rec_data["Y1"][0][ABLATION_A_FEATURES]),
            "VTA2": xgb_a.predict(rec_data["VTA2"][0][ABLATION_A_FEATURES]),
        },
        "Ablation B (IMU + 1s Rolling Stats, 13 feats)": {
            "TRAIN": xgb_b.predict(X_train_full[ABLATION_B_FEATURES]),
            "VAL": xgb_b.predict(X_val_full[ABLATION_B_FEATURES]),
            "TEST": xgb_b.predict(X_test_full[ABLATION_B_FEATURES]),
            "Y1": xgb_b.predict(rec_data["Y1"][0][ABLATION_B_FEATURES]),
            "VTA2": xgb_b.predict(rec_data["VTA2"][0][ABLATION_B_FEATURES]),
        },
        "Ablation C (IMU + Rolling + Deltas, 19 feats)": {
            "TRAIN": xgb_c.predict(X_train_full[ABLATION_C_FEATURES]),
            "VAL": xgb_c.predict(X_val_full[ABLATION_C_FEATURES]),
            "TEST": xgb_c.predict(X_test_full[ABLATION_C_FEATURES]),
            "Y1": xgb_c.predict(rec_data["Y1"][0][ABLATION_C_FEATURES]),
            "VTA2": xgb_c.predict(rec_data["VTA2"][0][ABLATION_C_FEATURES]),
        },
    }

    ablation_metrics_list = []
    for abl_name, preds_dict in ablation_preds.items():
        for part_name, y_p in preds_dict.items():
            if part_name == "TRAIN":
                y_t = y_train
            elif part_name == "VAL":
                y_t = y_val
            elif part_name == "TEST":
                y_t = y_test
            else:
                y_t = rec_data[part_name][1].values

            m = compute_regression_metrics(y_t, y_p)
            m["ablation"] = abl_name
            m["partition_or_recording"] = part_name
            ablation_metrics_list.append(m)

    df_ablation_metrics = pd.DataFrame(ablation_metrics_list)
    df_ablation_metrics.to_csv(out_path / "ablation_metrics.csv", index=False)
    print("\nAblation Comparison (XGBoost):")
    print(df_ablation_metrics.pivot(index="ablation", columns="partition_or_recording", values=["mae", "rmse", "r2"]))

    # 9. Detailed Per-Recording Performance (Baseline 2 - XGBoost C)
    rec_metrics_list = []
    for rec_id in all_recs:
        X_r = rec_data[rec_id][0][ABLATION_C_FEATURES]
        y_r = rec_data[rec_id][1].values
        y_p = xgb_c.predict(X_r)
        m = compute_regression_metrics(y_r, y_p)
        m["recording_id"] = rec_id
        if rec_id in split.train_recordings:
            m["partition"] = "TRAIN"
        elif rec_id in split.val_recordings:
            m["partition"] = "VAL"
        else:
            m["partition"] = "TEST"
        rec_metrics_list.append(m)

    df_rec_metrics = pd.DataFrame(rec_metrics_list)
    df_rec_metrics.to_csv(out_path / "recording_metrics.csv", index=False)
    print("\nPer-Recording Metrics (XGBoost C):")
    print(df_rec_metrics[["recording_id", "partition", "mae", "rmse", "median_ae", "p95_ae", "r2", "correlation"]])

    # 10. Speed-Regime Analysis (Baseline 2 - XGBoost C)
    regime_results = {}
    for part_name, (y_t, y_p) in [
        ("TRAIN", (y_train, xgb_c.predict(X_train_full[ABLATION_C_FEATURES]))),
        ("VAL", (y_val, xgb_c.predict(X_val_full[ABLATION_C_FEATURES]))),
        ("TEST_Y1", (rec_data["Y1"][1].values, xgb_c.predict(rec_data["Y1"][0][ABLATION_C_FEATURES]))),
        ("TEST_VTA2", (rec_data["VTA2"][1].values, xgb_c.predict(rec_data["VTA2"][0][ABLATION_C_FEATURES]))),
    ]:
        regime_results[part_name] = analyze_speed_regimes(y_t, y_p)

    # Flatten for CSV
    regime_rows = []
    for part_name, reg_dict in regime_results.items():
        for reg_name, m in reg_dict.items():
            row = {"partition": part_name, "speed_regime": reg_name}
            row.update(m)
            regime_rows.append(row)
    df_regimes = pd.DataFrame(regime_rows)
    df_regimes.to_csv(out_path / "speed_regime_metrics.csv", index=False)
    print("\nSpeed Regime Analysis (XGBoost C):")
    print(df_regimes[["partition", "speed_regime", "sample_count", "mae", "rmse"]])

    # 11. Dynamic Behaviors Analysis (Baseline 2 - XGBoost C)
    dynamic_results = {}
    for rec_id in ["S1", "S3A", "VTA1A", "Y1", "VTA2"]:
        t_r = rec_data[rec_id][2]["time_s"].values
        y_t = rec_data[rec_id][1].values
        y_p = xgb_c.predict(rec_data[rec_id][0][ABLATION_C_FEATURES])
        X_r = rec_data[rec_id][0]
        dyn = analyze_dynamic_behaviors(t_r, y_t, y_p, X_r)
        dynamic_results[rec_id] = dyn

    dynamic_rows = []
    for rec_id, d_dict in dynamic_results.items():
        for dyn_name, m in d_dict.items():
            row = {"recording_id": rec_id, "dynamic_category": dyn_name}
            row.update(m)
            dynamic_rows.append(row)
    df_dynamic = pd.DataFrame(dynamic_rows)
    df_dynamic.to_csv(out_path / "dynamic_behavior_metrics.csv", index=False)

    # 12. Temporal Lag and Derivative Correlation
    temporal_rows = []
    for rec_id in all_recs:
        t_r = rec_data[rec_id][2]["time_s"].values
        y_t = rec_data[rec_id][1].values
        y_p = xgb_c.predict(rec_data[rec_id][0][ABLATION_C_FEATURES])
        t_stats = compute_temporal_lag_and_derivative(t_r, y_t, y_p)
        t_stats["recording_id"] = rec_id
        t_stats["speed_correlation"] = float(np.corrcoef(y_t, y_p)[0, 1])
        temporal_rows.append(t_stats)
    df_temporal = pd.DataFrame(temporal_rows)
    df_temporal.to_csv(out_path / "temporal_metrics.csv", index=False)
    print("\nTemporal Lag & Correlation Metrics:")
    print(df_temporal[["recording_id", "lag_seconds", "derivative_correlation", "speed_correlation"]])

    # 13. Feature Importance (Baseline 2 - XGBoost C)
    # Extract feature importance (gain, weight, cover)
    booster = xgb_c.get_booster()
    score_gain = booster.get_score(importance_type="gain")
    score_weight = booster.get_score(importance_type="weight")

    feat_imp_list = []
    total_gain = sum(score_gain.values()) if score_gain else 1.0
    for feat in ABLATION_C_FEATURES:
        gain = score_gain.get(feat, 0.0)
        weight = score_weight.get(feat, 0.0)
        feat_imp_list.append({
            "feature": feat,
            "gain": gain,
            "normalized_gain_pct": (gain / total_gain) * 100.0 if total_gain > 0 else 0.0,
            "split_weight": int(weight),
        })
    df_feat_imp = pd.DataFrame(feat_imp_list).sort_values(by="gain", ascending=False).reset_index(drop=True)
    df_feat_imp.to_csv(out_path / "feature_importance.csv", index=False)
    print("\nTop 10 Feature Importances (XGBoost C by Gain):")
    print(df_feat_imp.head(10)[["feature", "gain", "normalized_gain_pct", "split_weight"]])

    # 14. Save Complete Study Summary JSON
    summary = {
        "status": "PHASE 3.2 COMPLETE",
        "partitions": {
            "train": split.train_recordings,
            "val": split.val_recordings,
            "test": split.test_recordings,
        },
        "sample_counts": {
            "train": int(len(y_train)),
            "val": int(len(y_val)),
            "test": int(len(y_test)),
            "global": int(len(y_global_all)),
        },
        "leakage_audit_passed": bool(leakage_audit["overall_leakage_audit_passed"]),
        "baseline_summary": df_baseline_metrics.to_dict(orient="records"),
        "ablation_summary": df_ablation_metrics.to_dict(orient="records"),
        "per_recording_summary": df_rec_metrics.to_dict(orient="records"),
    }
    with open(out_path / "phase3_2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nPhase 3.2 Experiment Completed Successfully.")
    return summary


if __name__ == "__main__":
    run_phase3_2_experiment()
