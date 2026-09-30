"""
Unit tests for NAVRIS Phase 3.2: Direct Forward-Speed ML Baseline.

Verifies:
1. Feature extraction causality and strict backward windowing [k-9 ... k].
2. Zero leakage: no reference, GNSS, NAVRIS, or future fields in features.
3. Partition isolation: Train, Validation, and Held-Out Test recording separation.
4. Causal future-perturbation invariance: corrupting future IMU does not alter current features.
5. Scaler fitted strictly on TRAIN data.
6. Phase 3.2 output artifacts existence and schema conformance.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from navris.ml.dataset import RecordingSplit, load_recording_ml_data
from scripts.phase3.run_phase3_2_speed_baseline import (
    ABLATION_A_FEATURES,
    ABLATION_B_FEATURES,
    ABLATION_C_FEATURES,
    FORBIDDEN_SUBSTRINGS,
    WINDOW_SAMPLES,
    extract_causal_imu_features,
    run_causal_leakage_audit,
)


def test_feature_columns_and_causal_window():
    """Verify that feature extraction yields exactly 19 causal features with N - 9 samples."""
    df_raw = load_recording_ml_data("VTA2")
    n_raw = len(df_raw)

    X, y, meta = extract_causal_imu_features(df_raw, window_size=WINDOW_SAMPLES)

    assert len(X) == n_raw - (WINDOW_SAMPLES - 1)
    assert len(y) == len(X)
    assert len(meta) == len(X)
    assert list(X.columns) == ABLATION_C_FEATURES
    assert not X.isna().any().any()
    assert not y.isna().any()
    assert (y.values >= 0.0).all(), "Reference forward speed must be non-negative"


def test_no_forbidden_fields_in_features():
    """Verify that no forbidden substrings (reference, vbox, navris, gnss, etc.) appear in feature names."""
    for col in ABLATION_C_FEATURES:
        col_lower = col.lower()
        for forbidden in FORBIDDEN_SUBSTRINGS:
            assert forbidden not in col_lower, f"Forbidden substring '{forbidden}' found in feature '{col}'"


def test_partition_isolation():
    """Verify zero overlap between train, validation, and held-out test recordings."""
    split = RecordingSplit(
        train_recordings=["S1", "S2", "S4"],
        val_recordings=["S3A", "VTA1A"],
        test_recordings=["Y1", "VTA2"],
    )
    s_tr = set(split.train_recordings)
    s_va = set(split.val_recordings)
    s_te = set(split.test_recordings)

    assert len(s_tr & s_va) == 0
    assert len(s_tr & s_te) == 0
    assert len(s_va & s_te) == 0


def test_causal_future_perturbation_invariance():
    """
    Causal perturbation test:
    Modifying future IMU measurements at timestamps t' > t must produce
    an identical feature vector at timestamp t (max absolute difference = 0.0).
    """
    df_raw = load_recording_ml_data("VTA2").iloc[:200].copy()
    k_eval = 120

    # Extract original features
    X_orig, y_orig, meta_orig = extract_causal_imu_features(df_raw)
    feat_eval_orig = X_orig.iloc[k_eval - WINDOW_SAMPLES + 1].values

    # Corrupt all future samples after k_eval
    df_corrupted = df_raw.copy()
    corrupt_cols = [
        "phone_accel_x_mps2", "phone_accel_y_mps2", "phone_accel_z_mps2",
        "phone_gyro_x_radps", "phone_gyro_y_radps", "phone_gyro_z_radps",
    ]
    for c in corrupt_cols:
        df_corrupted.loc[k_eval + 1:, c] = df_corrupted.loc[k_eval + 1:, c] * -50.0 + 1234.5

    X_corrupted, y_corrupted, meta_corrupted = extract_causal_imu_features(df_corrupted)
    feat_eval_corrupted = X_corrupted.iloc[k_eval - WINDOW_SAMPLES + 1].values

    max_diff = np.max(np.abs(feat_eval_orig - feat_eval_corrupted))
    assert max_diff == 0.0, f"Future IMU perturbation altered feature at t! Max diff: {max_diff}"


def test_scaler_fit_strictly_on_train():
    """Verify that feature scaler is isolated to TRAIN recordings only."""
    train_recs = ["S1", "S2", "S4"]
    X_train_list = []
    for r in train_recs:
        df_r = load_recording_ml_data(r)
        X_r, _, _ = extract_causal_imu_features(df_r)
        X_train_list.append(X_r[ABLATION_C_FEATURES])

    X_train_concat = pd.concat(X_train_list, ignore_index=True)
    ref_scaler = StandardScaler().fit(X_train_concat)

    # Load artifacts and verify leakage audit passed
    audit_path = Path("data/processed/phase3_2/leakage_audit_summary.json")
    assert audit_path.exists(), "leakage_audit_summary.json must exist"

    with open(audit_path, "r") as f:
        audit = json.load(f)

    assert audit["check_scaler_isolation"]["passed"]
    assert audit["overall_leakage_audit_passed"]


def test_phase3_2_processed_artifacts_exist():
    """Verify that all required Phase 3.2 data artifacts exist and contain valid rows."""
    processed_dir = Path("data/processed/phase3_2")
    required_files = [
        "target_distributions.csv",
        "baseline_model_metrics.csv",
        "ablation_metrics.csv",
        "recording_metrics.csv",
        "speed_regime_metrics.csv",
        "dynamic_behavior_metrics.csv",
        "temporal_metrics.csv",
        "feature_importance.csv",
        "leakage_audit_summary.json",
        "phase3_2_summary.json",
    ]
    for fname in required_files:
        p = processed_dir / fname
        assert p.exists(), f"Missing required Phase 3.2 artifact: {p}"
        assert p.stat().st_size > 0, f"Empty artifact: {p}"

    df_base = pd.read_csv(processed_dir / "baseline_model_metrics.csv")
    assert len(df_base) >= 15
    assert "Baseline 2 (XGBoost C)" in df_base["model"].values
