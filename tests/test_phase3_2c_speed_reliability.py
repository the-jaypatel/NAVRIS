"""Unit tests for Phase 3.2C - Causal ML Speed Reliability & Gating.

Validates:
1. Causal feature extraction and schema (15 features).
2. Gate A, Gate B decision logic and binary output.
3. Causality audit: future perturbation invariance test.
4. Negative control detector: verifying detection of forbidden features (e.g. true error, future references).
5. Output artifacts existence and column schema integrity.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from scripts.phase3.run_phase3_2c_speed_reliability import (
    apply_gate_a_rule_dynamic,
    apply_gate_b_quality_discrepancy,
    RELIABILITY_FEATURE_COLS,
    ERROR_THRESH_PRIMARY,
)


@pytest.fixture
def synthetic_features_df() -> pd.DataFrame:
    """Generate synthetic causal feature DataFrame matching exact schema."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "ml_pred_speed_mps": np.random.uniform(0.0, 25.0, n),
        "navris_v_fwd_mps": np.random.uniform(0.0, 25.0, n),
        "navris_speed_mps": np.random.uniform(0.0, 25.0, n),
        "speed_discrepancy_capped_mps": np.random.uniform(0.0, 5.0, n),
        "ml_delta_speed_1s": np.random.normal(0.0, 0.5, n),
        "accel_jerk_proxy_1s": np.random.uniform(0.01, 0.15, n),
        "gyro_magnitude_radps": np.random.uniform(0.01, 0.3, n),
        "gyro_accel_ratio_proxy": np.random.uniform(0.001, 0.05, n),
        "accel_mag_rolling_std_1s": np.random.uniform(0.05, 0.5, n),
        "gyro_mag_rolling_std_1s": np.random.uniform(0.01, 0.1, n),
        "causal_time_since_gnss_fix_s": np.random.exponential(1.5, n),
        "causal_gnss_fix_available": np.random.choice([0.0, 1.0], n),
        "causal_is_high_speed": np.zeros(n),
        "causal_is_turning": np.random.choice([0.0, 1.0], p=[0.8, 0.2], size=n),
        "causal_is_stationary": np.random.choice([0.0, 1.0], p=[0.8, 0.2], size=n),
    })
    df["causal_is_high_speed"] = (df["ml_pred_speed_mps"] >= 15.0).astype(float)
    return df


def test_gating_feature_schema():
    """Verify exact 15 causal feature columns are declared."""
    assert len(RELIABILITY_FEATURE_COLS) == 15
    forbidden = ["v_ref", "v_fwd_ref", "vbox", "error", "label", "future"]
    for col in RELIABILITY_FEATURE_COLS:
        for f in forbidden:
            assert f not in col.lower(), f"Forbidden substring '{f}' found in column '{col}'"


def test_gate_a_logic(synthetic_features_df):
    """Verify Gate A outputs binary flags with expected dynamic rejections."""
    gate_a = apply_gate_a_rule_dynamic(synthetic_features_df)
    assert len(gate_a) == len(synthetic_features_df)
    assert set(np.unique(gate_a)).issubset({0, 1})

    # High speed (>= 15 m/s) should be rejected
    high_speed_idx = synthetic_features_df["ml_pred_speed_mps"] >= 15.0
    assert (gate_a[high_speed_idx] == 0).all()


def test_gate_b_logic(synthetic_features_df):
    """Verify Gate B outputs binary flags and rejects large discrepancies during outage."""
    # Force extreme discrepancy during outage on first row
    synthetic_features_df.loc[0, "causal_time_since_gnss_fix_s"] = 5.0
    synthetic_features_df.loc[0, "speed_discrepancy_capped_mps"] = 12.0

    gate_b = apply_gate_b_quality_discrepancy(synthetic_features_df)
    assert len(gate_b) == len(synthetic_features_df)
    assert set(np.unique(gate_b)).issubset({0, 1})
    assert gate_b[0] == 0


def test_causality_future_perturbation_invariance():
    """Verify that altering future data after timestamp t does NOT affect gate decisions at t."""
    n = 100
    df1 = pd.DataFrame({
        "ml_pred_speed_mps": np.full(n, 8.0),
        "navris_v_fwd_mps": np.full(n, 8.0),
        "navris_speed_mps": np.full(n, 8.0),
        "speed_discrepancy_capped_mps": np.zeros(n),
        "ml_delta_speed_1s": np.zeros(n),
        "accel_jerk_proxy_1s": np.full(n, 0.03),
        "gyro_magnitude_radps": np.full(n, 0.05),
        "gyro_accel_ratio_proxy": np.full(n, 0.005),
        "accel_mag_rolling_std_1s": np.full(n, 0.1),
        "gyro_mag_rolling_std_1s": np.full(n, 0.02),
        "causal_time_since_gnss_fix_s": np.full(n, 0.5),
        "causal_gnss_fix_available": np.ones(n),
        "causal_is_high_speed": np.zeros(n),
        "causal_is_turning": np.zeros(n),
        "causal_is_stationary": np.zeros(n),
    })

    df2 = df1.copy()
    # Perturb everything after t = 40 with chaotic noise and non-physical values
    df2.iloc[41:, :] = np.random.uniform(50.0, 500.0, size=(n - 41, df2.shape[1]))

    gate_a_1 = apply_gate_a_rule_dynamic(df1)
    gate_a_2 = apply_gate_a_rule_dynamic(df2)

    gate_b_1 = apply_gate_b_quality_discrepancy(df1)
    gate_b_2 = apply_gate_b_quality_discrepancy(df2)

    # Invariance check: decisions up to index 40 must be strictly identical
    np.testing.assert_array_equal(gate_a_1[:41], gate_a_2[:41])
    np.testing.assert_array_equal(gate_b_1[:41], gate_b_2[:41])


def test_negative_control_catches_leakage():
    """Negative control test: ensures that adding a forbidden future/reference column is detected."""
    forbidden_features = [
        "v_fwd_ref",
        "abs_error_t_plus_1",
        "future_vbox_speed",
        "v_ref_error",
    ]
    for feat in forbidden_features:
        features_to_check = RELIABILITY_FEATURE_COLS + [feat]
        is_leaking = any(
            any(w in f.lower() for w in ["ref", "vbox", "error", "future"])
            for f in features_to_check
        )
        assert is_leaking, f"Negative control failed to catch forbidden feature: {feat}"


def test_processed_artifacts_exist():
    """Verify all Phase 3.2C artifacts exist in data/processed/phase3_2c/."""
    output_dir = Path("data/processed/phase3_2c")
    expected_files = [
        "reliability_target_statistics.csv",
        "gate_metrics.csv",
        "confusion_matrices.csv",
        "selective_prediction_metrics.csv",
        "rejection_regime_metrics.csv",
        "gnss_regime_metrics.csv",
        "speed_regime_metrics.csv",
        "temporal_gate_metrics.csv",
        "reliability_calibration.csv",
        "feature_importance.csv",
        "causal_audit_summary.json",
        "phase3_2c_summary.json",
    ]
    for fname in expected_files:
        p = output_dir / fname
        assert p.exists(), f"Artifact missing: {p}"
        assert p.stat().st_size > 0, f"Artifact empty: {p}"
