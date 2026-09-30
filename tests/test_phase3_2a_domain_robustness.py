"""
Unit tests for NAVRIS Phase 3.2A: Vehicle-Domain Robustness Diagnostic.

Verifies:
1. Feature extraction correctness for Scale-Invariant and Dynamic-Only representations.
2. Strict causal windowing [k-9 ... k] and zero future contamination.
3. Dedicated causality audit for Causal Running Normalization: feature(t) depends strictly on x[0:t].
4. Causal future-perturbation invariance across all representations.
5. Zero forbidden substrings (reference, vbox, navris, gnss, etc.).
6. Output artifact completeness in data/processed/phase3_2a/.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from navris.ml.dataset import RecordingSplit, load_recording_ml_data
from scripts.phase3.run_phase3_2_speed_baseline import extract_causal_imu_features, WINDOW_SAMPLES
from scripts.phase3.run_phase3_2a_domain_robustness import (
    DYNAMIC_ONLY_FEATURES,
    SCALE_INVARIANT_FEATURES,
    compute_causal_running_normalization,
    extract_dynamic_only_features,
    extract_scale_invariant_features,
)


def test_scale_invariant_and_dynamic_feature_columns():
    """Verify exact column counts and names for new diagnostic representations."""
    df_raw = load_recording_ml_data("VTA2")
    n_raw = len(df_raw)

    X_si = extract_scale_invariant_features(df_raw)
    X_dyn = extract_dynamic_only_features(df_raw)

    assert len(X_si) == n_raw - (WINDOW_SAMPLES - 1)
    assert len(X_dyn) == n_raw - (WINDOW_SAMPLES - 1)

    assert list(X_si.columns) == SCALE_INVARIANT_FEATURES
    assert list(X_dyn.columns) == DYNAMIC_ONLY_FEATURES

    assert not X_si.isna().any().any(), "Scale-invariant features must contain no NaNs"
    assert not X_dyn.isna().any().any(), "Dynamic-only features must contain no NaNs"


def test_causal_future_perturbation_invariance():
    """
    Verifies that corrupting future IMU observations strictly after timestamp t
    produces identical feature vectors at timestamp t across all representations.
    """
    df_raw = load_recording_ml_data("VTA2").iloc[:180].copy()
    k_eval = 110

    # Extract original
    X_si_orig = extract_scale_invariant_features(df_raw)
    X_dyn_orig = extract_dynamic_only_features(df_raw)

    row_si_orig = X_si_orig.iloc[k_eval - WINDOW_SAMPLES + 1].values
    row_dyn_orig = X_dyn_orig.iloc[k_eval - WINDOW_SAMPLES + 1].values

    # Corrupt future samples strictly AFTER k_eval
    df_corrupt = df_raw.copy()
    corrupt_cols = [
        "phone_accel_x_mps2", "phone_accel_y_mps2", "phone_accel_z_mps2",
        "phone_gyro_x_radps", "phone_gyro_y_radps", "phone_gyro_z_radps",
    ]
    for c in corrupt_cols:
        df_corrupt.loc[k_eval + 1:, c] = df_corrupt.loc[k_eval + 1:, c] * -99.0 + 8888.8

    X_si_corrupt = extract_scale_invariant_features(df_corrupt)
    X_dyn_corrupt = extract_dynamic_only_features(df_corrupt)

    row_si_corrupt = X_si_corrupt.iloc[k_eval - WINDOW_SAMPLES + 1].values
    row_dyn_corrupt = X_dyn_corrupt.iloc[k_eval - WINDOW_SAMPLES + 1].values

    diff_si = np.max(np.abs(row_si_orig - row_si_corrupt))
    diff_dyn = np.max(np.abs(row_dyn_orig - row_dyn_corrupt))

    assert diff_si == 0.0, f"Scale-invariant feature altered by future data! Diff: {diff_si}"
    assert diff_dyn == 0.0, f"Dynamic-only feature altered by future data! Diff: {diff_dyn}"


def test_causal_running_normalization_strict_causality():
    """
    Dedicated audit: verify mathematically and programmatically that
    causal running z-score at timestamp t depends ONLY on observations x[0:t]
    and is strictly independent of future observations x[t+1:].
    """
    df_raw = load_recording_ml_data("VTA2").iloc[:150].copy()
    X_raw, _, _ = extract_causal_imu_features(df_raw)
    k_eval = 80

    z_a_orig, z_w_orig = compute_causal_running_normalization(X_raw)
    val_at_t_orig = (z_a_orig[k_eval], z_w_orig[k_eval])

    # Modify future values in X_raw after k_eval
    X_raw_corrupted = X_raw.copy()
    X_raw_corrupted.loc[k_eval + 1:, "accel_mag_rolling_std_1s"] = 999.0
    X_raw_corrupted.loc[k_eval + 1:, "gyro_mag_rolling_std_1s"] = 888.0

    z_a_corrupt, z_w_corrupt = compute_causal_running_normalization(X_raw_corrupted)
    val_at_t_corrupt = (z_a_corrupt[k_eval], z_w_corrupt[k_eval])

    diff_a = abs(val_at_t_orig[0] - val_at_t_corrupt[0])
    diff_w = abs(val_at_t_orig[1] - val_at_t_corrupt[1])

    assert diff_a == 0.0, f"Causal running accel z-score at t altered by future values! Diff: {diff_a}"
    assert diff_w == 0.0, f"Causal running gyro z-score at t altered by future values! Diff: {diff_w}"


def test_no_forbidden_fields_in_new_representations():
    """Verify that no forbidden substrings appear in new feature column names."""
    forbidden = ["ref_", "vbox", "navris", "gnss", "target", "truth", "eskf", "zupt", "nhc"]
    for col in SCALE_INVARIANT_FEATURES + DYNAMIC_ONLY_FEATURES:
        c_low = col.lower()
        for f in forbidden:
            assert f not in c_low, f"Forbidden substring '{f}' found in feature '{col}'"


def test_phase3_2a_artifacts_exist():
    """Verify that all required Phase 3.2A artifacts exist and contain non-empty tables."""
    out_dir = Path("data/processed/phase3_2a")
    required_files = [
        "domain_metrics.csv",
        "representation_metrics.csv",
        "speed_regime_metrics.csv",
        "feature_distribution_metrics.csv",
        "vibration_speed_relationship.csv",
        "causal_audit_summary.json",
        "phase3_2a_summary.json",
    ]
    for fname in required_files:
        p = out_dir / fname
        assert p.exists(), f"Missing required Phase 3.2A artifact: {p}"
        assert p.stat().st_size > 0, f"Empty artifact file: {p}"

    df_dom = pd.read_csv(out_dir / "domain_metrics.csv")
    assert len(df_dom) == 5
    assert "2. Scale-Invariant (Dimensionless)" in df_dom["representation"].values
    assert "3. Dynamic-Only (Deltas & Jerk)" in df_dom["representation"].values
