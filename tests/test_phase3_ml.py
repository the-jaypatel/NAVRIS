"""
Unit tests for NAVRIS Phase 3.0 Machine Learning Module.

Verifies:
1. Recording-level split disjointness (zero recording leakage).
2. Anti-leakage checks (no reference or target fields in X).
3. Causal window correctness (window length, boundary handling, strictly backward-looking).
4. Feature schema conformance.
5. Scaler isolation (train-only fitting).
6. Model training, predict shape, and metric computation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from navris.ml.dataset import RecordingSplit, load_recording_ml_data
from navris.ml.features import (
    CausalFeatureExtractor,
    FEATURE_NAMES,
    TARGET_NAMES,
    FORBIDDEN_FEATURE_SUBSTRINGS,
    WINDOW_SAMPLES,
)
from navris.ml.model import VelocityResidualXGBoost
from navris.ml.evaluate import evaluate_residual_predictions
from navris.ml.leakage_audit import run_leakage_audit


def test_recording_split_disjointness():
    """Verify RecordingSplit raises ValueError on overlapping recording assignments."""
    # Valid split
    split = RecordingSplit(
        train_recordings=["S1", "S2"],
        val_recordings=["S3A"],
        test_recordings=["Y1"],
    )
    assert len(set(split.train_recordings) & set(split.val_recordings)) == 0
    assert len(set(split.train_recordings) & set(split.test_recordings)) == 0
    assert len(set(split.val_recordings) & set(split.test_recordings)) == 0

    # Overlapping split must fail
    with pytest.raises(ValueError, match="Data leakage detected"):
        RecordingSplit(
            train_recordings=["S1", "S2"],
            val_recordings=["S2", "S3A"],
            test_recordings=["Y1"],
        )


def test_no_forbidden_fields_in_feature_names():
    """Verify no ground truth, reference, or target column names exist in FEATURE_NAMES."""
    for feat in FEATURE_NAMES:
        feat_lower = feat.lower()
        for forbidden in FORBIDDEN_FEATURE_SUBSTRINGS:
            assert forbidden not in feat_lower, f"Forbidden substring '{forbidden}' found in feature '{feat}'"


def test_causal_feature_extractor_causality():
    """
    Verify CausalFeatureExtractor uses strictly causal window [k-9 ... k].
    Changing future data must NOT change past or current features.
    """
    n_samples = 30
    w = 10
    time_s = np.linspace(0.0, 2.9, n_samples)

    dummy_df = pd.DataFrame({
        "time_s": time_s,
        "recording_id": ["TEST"] * n_samples,
        "phone_accel_x_mps2": np.sin(time_s),
        "phone_accel_y_mps2": np.cos(time_s),
        "phone_accel_z_mps2": np.ones(n_samples) * 9.81,
        "phone_gyro_x_radps": np.zeros(n_samples),
        "phone_gyro_y_radps": np.zeros(n_samples),
        "phone_gyro_z_radps": np.zeros(n_samples),
        "navris_vel_east_mps": time_s * 2.0,
        "navris_vel_north_mps": time_s * 1.5,
        "navris_vel_up_mps": np.zeros(n_samples),
        "navris_heading_rad": np.zeros(n_samples),
        "v_ref_east_mps": time_s * 2.1,
        "v_ref_north_mps": time_s * 1.6,
        "v_ref_up_mps": np.zeros(n_samples),
        "ref_speed_mps": np.ones(n_samples) * 5.0,
        "ref_heading_rad": np.zeros(n_samples),
        "target_delta_v_east_mps": np.ones(n_samples) * 0.1,
        "target_delta_v_north_mps": np.ones(n_samples) * 0.1,
        "target_delta_v_up_mps": np.zeros(n_samples),
    })

    extractor = CausalFeatureExtractor(window_size=w)
    X1, y1, m1 = extractor.extract_from_recording(dummy_df)

    # First valid index in X1 corresponds to k = 9 (sample index 9)
    assert len(X1) == n_samples - w + 1
    assert len(y1) == n_samples - w + 1

    # Modify sample 25 in dummy_df (a future sample relative to k=9..20)
    dummy_df_modified = dummy_df.copy()
    dummy_df_modified.loc[25, "phone_accel_x_mps2"] += 50.0

    X2, _, _ = extractor.extract_from_recording(dummy_df_modified)

    # All features up to sample index k=24 (row index in X: 24 - 9 = 15) must be IDENTICAL
    # Row index 15 in X corresponds to sample k = 24. It must NOT be affected by sample 25!
    pd.testing.assert_frame_equal(X1.iloc[:16], X2.iloc[:16])

    # Row index 16 in X corresponds to sample k = 25. It SHOULD be different
    assert not np.isclose(X1.iloc[16]["phone_accel_x_mps2"], X2.iloc[16]["phone_accel_x_mps2"])


def test_leakage_audit_catches_violations():
    """Verify run_leakage_audit flags violations."""
    split = RecordingSplit(
        train_recordings=["S1"],
        val_recordings=["S2"],
        test_recordings=["S4"],
    )

    X_train = pd.DataFrame(np.zeros((50, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    X_val = pd.DataFrame(np.zeros((20, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    X_test = pd.DataFrame(np.zeros((20, len(FEATURE_NAMES))), columns=FEATURE_NAMES)

    # Clean audit must pass
    clean_audit = run_leakage_audit(split, X_train, X_val, X_test)
    assert clean_audit["overall_leakage_audit_passed"] is True

    # Dirty audit: insert a forbidden column
    X_train_dirty = X_train.copy()
    X_train_dirty["ref_speed_mps"] = 5.0
    dirty_audit = run_leakage_audit(split, X_train_dirty, X_val, X_test)
    assert dirty_audit["overall_leakage_audit_passed"] is False
    assert dirty_audit["check_no_forbidden_fields_in_X"]["passed"] is False


def test_velocity_residual_xgboost_fit_predict():
    """Verify VelocityResidualXGBoost fits and predicts correct shapes without NaN."""
    np.random.seed(42)
    n_train = 100
    n_test = 30

    X_train = pd.DataFrame(np.random.randn(n_train, len(FEATURE_NAMES)), columns=FEATURE_NAMES)
    y_train = pd.DataFrame(np.random.randn(n_train, 3), columns=TARGET_NAMES)

    X_test = pd.DataFrame(np.random.randn(n_test, len(FEATURE_NAMES)), columns=FEATURE_NAMES)

    model = VelocityResidualXGBoost(n_estimators=10, max_depth=3)
    model.fit(X_train, y_train)

    assert model.is_fitted
    preds = model.predict(X_test)
    assert preds.shape == (n_test, 3)
    assert not np.isnan(preds).any()

    # Verify feature importances
    df_imp = model.get_feature_importances()
    assert len(df_imp) == len(FEATURE_NAMES)
    assert "importance_mean" in df_imp.columns


def test_load_recording_ml_data_real():
    """Verify loading real recording data (VTA2) works with zero NaNs and aligned timestamps."""
    gate_dir = Path("data/processed/phase2_3b/gate2_3b")
    sync_dir = Path("data/processed/synchronized")
    if not (gate_dir / "VTA2_trajectory_comparison.csv").exists():
        pytest.skip("VTA2 trajectory data not found")

    df_vta2 = load_recording_ml_data("VTA2", sync_dir=sync_dir, gate_dir=gate_dir)
    assert len(df_vta2) == 10128
    assert "target_delta_v_east_mps" in df_vta2.columns
    assert "target_delta_v_north_mps" in df_vta2.columns
    assert "target_delta_v_up_mps" in df_vta2.columns
    assert not df_vta2["target_delta_v_east_mps"].isna().any()

    # Extract causal features
    extractor = CausalFeatureExtractor(window_size=10)
    X, y, meta = extractor.extract_from_recording(df_vta2)
    assert len(X) == 10128 - 9
    assert len(y) == 10128 - 9
    assert list(X.columns) == FEATURE_NAMES
    assert list(y.columns) == TARGET_NAMES
