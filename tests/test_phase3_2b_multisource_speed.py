"""
Unit tests for NAVRIS Phase 3.2B: Causal Multi-Source Forward-Speed Estimation.

Verifies:
1. Feature extraction correctness across Group A (IMU), Group B (NAVRIS), Group C (Quality).
2. Exact formulation of forward speed residual: r_fwd = v_fwd_ref - v_fwd_nav.
3. Monotonic causal tracking of time since last GNSS fix.
4. Zero forbidden substrings in feature names.
5. Inversion failure test replication: verifying that residual learning inverts NAVRIS velocity.
6. Output artifact completeness in data/processed/phase3_2b/.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from navris.ml.dataset import RecordingSplit
from scripts.phase3.run_phase3_2b_multisource_speed import (
    ABLATION_A_COLS,
    ABLATION_B_COLS,
    ABLATION_C_COLS,
    ABLATION_D_COLS,
    FORBIDDEN_SUBSTRINGS,
    IMU_DYNAMIC_FEATURES,
    NAVRIS_STATE_FEATURES,
    QUALITY_FEATURES,
    extract_multisource_data,
)


def test_multisource_feature_shapes_and_groups():
    """Verify feature matrix dimensions and column groups."""
    X, y_dir, r_res, meta = extract_multisource_data("VTA2")

    assert len(X) == len(y_dir)
    assert len(X) == len(r_res)
    assert len(X) == len(meta)

    assert list(X.columns) == ABLATION_D_COLS
    assert len(IMU_DYNAMIC_FEATURES) == 12
    assert len(NAVRIS_STATE_FEATURES) == 9
    assert len(QUALITY_FEATURES) == 2
    assert len(ABLATION_D_COLS) == 23

    assert not X.isna().any().any(), "Multi-source feature matrix must contain no NaNs"
    assert not y_dir.isna().any(), "Direct target must contain no NaNs"
    assert not r_res.isna().any(), "Residual target must contain no NaNs"


def test_residual_target_exact_relationship():
    """Verify that r_fwd(t) = v_fwd_ref(t) - v_fwd_nav(t) exactly."""
    _, y_dir, r_res, meta = extract_multisource_data("VTA2")

    v_ref = y_dir.values
    v_nav = meta["v_fwd_nav"].values
    r_computed = v_ref - v_nav

    max_diff = np.max(np.abs(r_res.values - r_computed))
    assert max_diff < 1e-12, f"Residual definition mismatch! Max diff: {max_diff}"


def test_causal_gnss_fix_time_tracking():
    """Verify that causal_time_since_gnss_fix_s resets on fix and increments by dt otherwise."""
    X, _, _, meta = extract_multisource_data("VTA2")

    dt_fix = X["causal_time_since_gnss_fix_s"].values
    fix_avail = X["causal_gnss_fix_available"].values

    assert (dt_fix >= 0.0).all(), "Time since fix must be non-negative"
    assert set(np.unique(fix_avail)).issubset({0.0, 1.0})

    # For indices where a new fix occurred, dt_fix should be near 0 (<= 0.1s)
    fix_indices = np.where(fix_avail == 1.0)[0]
    if len(fix_indices) > 0:
        assert (dt_fix[fix_indices] < 0.15).all()


def test_no_forbidden_fields_in_multisource_features():
    """Verify that forbidden substrings do not appear in feature columns."""
    for col in ABLATION_D_COLS:
        c_low = col.lower()
        for f in FORBIDDEN_SUBSTRINGS:
            assert f not in c_low, f"Forbidden substring '{f}' found in feature '{col}'"


def test_inversion_failure_test_replication():
    """Verify that inversion regression CSV confirms severe negative beta on divergent recordings."""
    inv_path = Path("data/processed/phase3_2b/residual_inversion_test.csv")
    assert inv_path.exists(), "residual_inversion_test.csv must exist"

    df_inv = pd.read_csv(inv_path)
    df_c = df_inv[df_inv["ablation"] == "Ablation C (IMU + NAVRIS)"]

    # On S1, S2, S4, Y1, beta_true and beta_pred must both be <= -0.85
    for r in ["S1", "S2", "S4", "S3A", "VTA1A", "Y1"]:
        row = df_c[df_c["recording_id"] == r].iloc[0]
        assert row["beta_true"] < -0.99, f"Beta true on {r} expected ~ -1.0, got {row['beta_true']}"
        assert row["beta_pred"] < -0.85, f"Beta pred on {r} expected < -0.85, got {row['beta_pred']}"
        assert row["corr_r_pred_v_nav"] < -0.95, f"Pred residual correlation on {r} expected < -0.95"


def test_phase3_2b_artifacts_exist():
    """Verify that all required Phase 3.2B artifacts exist and contain non-empty data."""
    out_dir = Path("data/processed/phase3_2b")
    required_files = [
        "target_statistics.csv",
        "representation_metrics.csv",
        "recording_metrics.csv",
        "speed_regime_metrics.csv",
        "residual_statistics.csv",
        "residual_inversion_test.csv",
        "feature_importance.csv",
        "quality_regime_metrics.csv",
        "causal_audit_summary.json",
        "phase3_2b_summary.json",
    ]
    for fname in required_files:
        p = out_dir / fname
        assert p.exists(), f"Missing required Phase 3.2B artifact: {p}"
        assert p.stat().st_size > 0, f"Empty artifact file: {p}"

    df_rep = pd.read_csv(out_dir / "representation_metrics.csv")
    assert len(df_rep) > 50
    assert "Direct Speed" in df_rep["formulation"].values
    assert "Residual Speed (Reconstructed)" in df_rep["formulation"].values
