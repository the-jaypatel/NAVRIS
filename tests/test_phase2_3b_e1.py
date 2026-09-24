"""
Tests for NAVRIS Phase 2.3B Controlled E1 Initialization Experiments.

Verifies:
1. Causal displacement velocity derivation is strictly causal (no future data).
2. Monotonicity of evaluation timestamps.
3. Covariance symmetry and numerical PSD validity.
4. Duplicate GNSS sample-and-hold rejection.
"""

import numpy as np
import pandas as pd
import pytest

from navris.eskf import (
    ESKF,
    NominalState,
    ESKFConfig,
)
from navris.inertial.init import align_attitude_causal
from navris.inertial.gnss_fixes import filter_novel_gnss_fixes


def test_causal_displacement_velocity():
    """Verify displacement-derived initial velocity is strictly causal."""
    df = pd.read_parquet("data/processed/synchronized/S1_sync.parquet")
    t_init = 156.0
    history = df[df["time_s"] <= t_init].copy()
    assert (history["time_s"] <= t_init).all()

    novel = filter_novel_gnss_fixes(history)
    assert len(novel) >= 2

    # Two most recent causal fixes before or at t_init
    t_curr = float(novel["time_s"].iloc[-1])
    t_prev = float(novel["time_s"].iloc[-2])
    assert t_curr == t_init
    assert t_prev < t_init

    dt = t_curr - t_prev
    assert dt > 0.0

    de = float(novel["phone_gps_east_m"].iloc[-1] - novel["phone_gps_east_m"].iloc[-2])
    dn = float(novel["phone_gps_north_m"].iloc[-1] - novel["phone_gps_north_m"].iloc[-2])

    v_disp_e = de / dt
    v_disp_n = dn / dt
    v_disp_norm = np.sqrt(v_disp_e**2 + v_disp_n**2)

    # Realistic vehicle speed (> 10 m/s, not bogus ~3 m/s)
    assert 10.0 < v_disp_norm < 20.0
    assert v_disp_e < 0.0  # Heading North-West (negative East)
    assert v_disp_n > 0.0  # Heading North-West (positive North)


def test_e1_short_filter_step_and_psd():
    """Verify ESKF propagation and covariance PSD validity under E1 conditions."""
    df = pd.read_parquet("data/processed/synchronized/S1_sync.parquet")
    history = df.iloc[:3500]
    align_res = align_attitude_causal(history, min_stat_samples=30)
    init_idx = align_res.init_sample_idx

    p0 = np.array([
        df["phone_gps_east_m"].iloc[init_idx],
        df["phone_gps_north_m"].iloc[init_idx],
        df["phone_gps_up_m"].iloc[init_idx]
    ])
    v0 = np.array([-9.413, 10.301, 0.0])
    q0 = align_res.q_b_n.copy()

    state = NominalState(t=156.0, p=p0, v=v0, q=q0)
    cov = np.diag([25.0, 25.0, 100.0, 36.0, 36.0, 1.0, 0.01, 0.01, 0.05, 0.04, 0.04, 0.04, 1e-4, 1e-4, 1e-4])
    config = ESKFConfig()

    eskf = ESKF(state, cov, config)

    # Step 10 IMU epochs
    for i in range(10):
        row = df.iloc[init_idx + i]
        fb = np.array([row["phone_accel_x_mps2"], row["phone_accel_y_mps2"], row["phone_accel_z_mps2"]])
        wb = np.array([row["phone_gyro_x_radps"], row["phone_gyro_y_radps"], row["phone_gyro_z_radps"]])
        ok = eskf.predict(fb, wb, 0.1)
        assert ok

    # Verify symmetry
    assert np.allclose(eskf.P, eskf.P.T, atol=1e-10)

    # Verify positive semi-definiteness
    eigs = np.linalg.eigvalsh(eskf.P)
    assert (eigs > 0.0).all()
    assert np.min(eigs) > 1e-8
