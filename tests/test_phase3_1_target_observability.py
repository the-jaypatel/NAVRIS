"""
Unit tests for NAVRIS Phase 3.1 Target Observability and Formulation Study.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from scripts.phase3.target_observability import (
    compute_distribution_stats,
    load_and_preprocess_recording,
    VALID_RECORDINGS,
)


def test_compute_distribution_stats():
    """Verify statistical computation on known distributions."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    st = compute_distribution_stats(arr)
    assert st["count"] == 5
    assert np.isclose(st["mean"], 3.0)
    assert np.isclose(st["median"], 3.0)
    assert np.isclose(st["min"], 1.0)
    assert np.isclose(st["max"], 5.0)
    assert np.isclose(st["rmse"], np.sqrt(11.0))

    # Test handling of NaNs
    arr_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    st_nan = compute_distribution_stats(arr_nan)
    assert st_nan["count"] == 4
    assert np.isclose(st_nan["mean"], 3.0)


def test_load_and_preprocess_vta2():
    """Verify loading and pre-processing of VTA2 produces all required candidate targets."""
    df_vta2 = load_and_preprocess_recording("VTA2")
    assert len(df_vta2) == 10128

    # Verify candidate targets exist
    assert "target_A_fwd" in df_vta2.columns
    assert "target_B_inc_3d_1_0s" in df_vta2.columns
    assert "target_B_fut_3d_1_0s" in df_vta2.columns
    assert "target_C_rate_3d_1_0s" in df_vta2.columns
    assert "target_D_acc_long" in df_vta2.columns
    assert "target_D_yaw_rate" in df_vta2.columns
    assert "target_E_rel_err" in df_vta2.columns

    # Verify causal features exist
    assert "phone_accel_x_mps2" in df_vta2.columns
    assert "phone_gyro_z_radps" in df_vta2.columns
    assert "outage_duration_s" in df_vta2.columns

    # Verify VTA2 target A forward residual is physically bounded under nominal GNSS
    vta2_target_A = df_vta2["target_A_fwd"].values
    assert np.mean(vta2_target_A) < 10.0
    assert np.std(vta2_target_A) < 15.0


def test_causal_outage_duration_properties():
    """Verify outage duration is non-negative and resets on valid GPS fixes."""
    df_vta2 = load_and_preprocess_recording("VTA2")
    outage = df_vta2["outage_duration_s"].values
    assert np.all(outage >= 0.0)
    assert np.min(outage) == 0.0
    # On VTA2, GNSS is available 96.7% of the time, so max outage must be moderate
    assert np.max(outage) < 100.0


def test_short_horizon_window_boundaries():
    """Verify short-horizon targets handle future window edges with valid padding."""
    df_vta2 = load_and_preprocess_recording("VTA2")
    # For H = 1.0s (10 steps), the last 10 samples must be NaN in target_B_inc
    inc_1s = df_vta2["target_B_inc_3d_1_0s"].values
    assert np.all(~np.isnan(inc_1s[:-10]))
    assert np.all(np.isnan(inc_1s[-10:]))

    # For H = 5.0s (50 steps), the last 50 samples must be NaN
    inc_5s = df_vta2["target_B_inc_3d_5_0s"].values
    assert np.all(~np.isnan(inc_5s[:-50]))
    assert np.all(np.isnan(inc_5s[-50:]))


def test_candidate_a_unbounded_during_divergence():
    """Verify that Candidate A reflects classical filter divergence in S3A."""
    df_s3a = load_and_preprocess_recording("S3A")
    target_A = df_s3a["target_A_fwd"].values
    # In S3A, classical filter diverged, so target_A must exhibit massive extremes
    assert np.max(np.abs(target_A)) > 5000.0
    assert np.std(target_A) > 1000.0


def test_candidate_d_yaw_rate_stability():
    """Verify that Candidate D yaw rate residual is stable and physically bounded."""
    df_s3a = load_and_preprocess_recording("S3A")
    yaw_res = df_s3a["target_D_yaw_rate"].values
    st = compute_distribution_stats(yaw_res)
    # Yaw rate residual should be on the order of rad/s (< 2 rad/s)
    assert st["rmse"] < 1.0
    assert np.abs(st["mean"]) < 0.1
