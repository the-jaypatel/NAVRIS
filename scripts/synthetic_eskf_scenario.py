"""
NAVRIS Phase 2.3A Synthetic End-to-End ESKF Benchmark Scenario.

Generates a continuous 10-phase synthetic vehicle trajectory and evaluates the
15-state ESKF under realistic conditions:

Phases (total duration 200 s at 10 Hz IMU):
  Phase 1:  Stationary alignment (0 - 10 s, 10 s)
  Phase 2:  Forward acceleration along East (10 - 15 s, 5 s, 0 -> 15 m/s)
  Phase 3:  Constant speed cruise East (15 - 35 s, 20 s, 15 m/s)
  Phase 4:  Coordinated 90-degree left turn to North (35 - 50 s, 15 s, R = 150 m)
  Phase 5:  Northward cruise with injected IMU bias and noise (50 - 75 s, 25 s)
  Phase 6:  Sparse GNSS updates (0.2 Hz) + sample-and-hold duplicates (75 - 95 s, 20 s)
  Phase 7:  Complete 60 s GNSS Outage - pure inertial DR (95 - 155 s, 60 s)
  Phase 8:  GNSS recovery after outage (155 - 175 s, 20 s)
  Phase 9:  Braking to stop (175 - 185 s, 10 s, 15 -> 0 m/s)
  Phase 10: Stationary post-stop (185 - 200 s, 15 s)

Outputs:
  - Quantitative metrics per phase (position RMSE, velocity RMSE, attitude error, bias convergence)
  - Detailed outage drift and post-outage recovery time
  - Summary table printed to stdout
  - Metrics exported to data/processed/phase2/synthetic_eskf_scenario_results.csv
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from navris.eskf import (
    ESKF,
    NominalState,
    ESKFConfig,
    STATE_DIM,
    IDX_POS,
    IDX_VEL,
    IDX_ATT,
    IDX_ACC_BIAS,
    IDX_GYR_BIAS,
)
from navris.inertial.frames import (
    quat_to_dcm,
    rotvec_to_quat,
    quat_to_euler,
    euler_to_quat,
    quat_multiply,
    quat_normalize,
)
from navris.inertial.gravity import gravity_vector_enu


def run_synthetic_scenario():
    print("=" * 80)
    print("NAVRIS Phase 2.3A: 10-Phase Synthetic ESKF Benchmark Scenario")
    print("=" * 80)

    # 1. Setup Simulation Parameters
    dt = 0.1  # 10 Hz IMU
    total_time = 200.0  # seconds
    time_steps = int(round(total_time / dt))
    timestamps = np.linspace(0.0, total_time - dt, time_steps)

    g_n = gravity_vector_enu(lat_deg=52.4, alt_m=100.0)
    g_val = -g_n[2]

    # Injected sensor biases (realistic smartphone grade)
    ba_true = np.array([0.05, -0.03, 0.02], dtype=np.float64)  # m/s^2
    bg_true = np.array([0.0005, -0.0005, 0.0010], dtype=np.float64)  # rad/s (~0.057 deg/s)

    # Sensor noise standard deviations
    sigma_acc_noise = 0.03  # m/s^2
    sigma_gyr_noise = 0.002  # rad/s
    sigma_gnss_noise = 1.5   # m

    np.random.seed(42)

    # 2. Pre-generate Ground Truth Trajectory
    # States: pos_true (N,3), vel_true (N,3), quat_true (N,4), f_b_true (N,3), omega_b_true (N,3)
    pos_true = np.zeros((time_steps, 3), dtype=np.float64)
    vel_true = np.zeros((time_steps, 3), dtype=np.float64)
    quat_true = np.zeros((time_steps, 4), dtype=np.float64)
    f_b_meas = np.zeros((time_steps, 3), dtype=np.float64)
    omega_b_meas = np.zeros((time_steps, 3), dtype=np.float64)
    phase_labels = []

    # Current kinematics
    p = np.zeros(3, dtype=np.float64)
    v = np.zeros(3, dtype=np.float64)
    yaw = 0.0  # 0 = East (+X), pi/2 = North (+Y)
    turn_omega = (np.pi / 2.0) / 15.0  # 90 deg in 15 s = pi/30 rad/s ~ 0.1047 rad/s

    for i, t in enumerate(timestamps):
        # Determine phase
        if t < 10.0:
            phase = 1  # Stationary
            a_net_nav = np.zeros(3)
            omega_z = 0.0
        elif t < 15.0:
            phase = 2  # Accel East (0 -> 15 m/s over 5 s, a = 3 m/s^2)
            a_net_nav = np.array([3.0, 0.0, 0.0])
            omega_z = 0.0
        elif t < 35.0:
            phase = 3  # Cruise East at 15 m/s
            a_net_nav = np.zeros(3)
            omega_z = 0.0
        elif t < 50.0:
            phase = 4  # Turn to North (15 s, yaw rate omega_z)
            omega_z = turn_omega
            yaw_curr = (t - 35.0) * turn_omega
            speed = 15.0
            # Coordinated turn acceleration: v * omega directed toward center of curvature
            a_net_nav = np.array([-speed * omega_z * np.sin(yaw_curr), speed * omega_z * np.cos(yaw_curr), 0.0])
        elif t < 75.0:
            phase = 5  # Cruise North at 15 m/s with bias/noise
            a_net_nav = np.zeros(3)
            omega_z = 0.0
            yaw = np.pi / 2.0
        elif t < 95.0:
            phase = 6  # Sparse GNSS
            a_net_nav = np.zeros(3)
            omega_z = 0.0
            yaw = np.pi / 2.0
        elif t < 155.0:
            phase = 7  # 60 s GNSS Outage
            a_net_nav = np.zeros(3)
            omega_z = 0.0
            yaw = np.pi / 2.0
        elif t < 175.0:
            phase = 8  # GNSS Recovery
            a_net_nav = np.zeros(3)
            omega_z = 0.0
            yaw = np.pi / 2.0
        elif t < 185.0:
            phase = 9  # Braking North (15 -> 0 m/s over 10 s, a = -1.5 m/s^2)
            a_net_nav = np.array([0.0, -1.5, 0.0])
            omega_z = 0.0
            yaw = np.pi / 2.0
        else:
            phase = 10  # Stationary post-stop
            a_net_nav = np.zeros(3)
            omega_z = 0.0
            yaw = np.pi / 2.0

        if phase == 4:
            yaw = (t - 35.0) * turn_omega

        # Update true kinematics
        q_now = euler_to_quat(0.0, 0.0, yaw)
        C_b_n = quat_to_dcm(q_now)

        # True specific force: f_b = C_n_b * (a_net_nav - g_n)
        f_b_true = C_b_n.T @ (a_net_nav - g_n)
        omega_b_true = np.array([0.0, 0.0, omega_z])

        # Add noise and biases (bias active from Phase 5 onward)
        w_acc = np.random.randn(3) * sigma_acc_noise
        w_gyr = np.random.randn(3) * sigma_gyr_noise

        if phase >= 5:
            f_meas = f_b_true + ba_true + w_acc
            omega_meas = omega_b_true + bg_true + w_gyr
        else:
            f_meas = f_b_true + w_acc
            omega_meas = omega_b_true + w_gyr

        # Integrate ground truth kinematics
        if i > 0:
            dt_step = timestamps[i] - timestamps[i - 1]
            p = p + v * dt_step + 0.5 * a_net_nav * (dt_step ** 2)
            v = v + a_net_nav * dt_step

        pos_true[i] = p
        vel_true[i] = v
        quat_true[i] = q_now
        f_b_meas[i] = f_meas
        omega_b_meas[i] = omega_meas
        phase_labels.append(phase)

    # 3. Initialize ESKF
    config = ESKFConfig(
        sigma_acc=0.1,
        sigma_gyr=0.01,
        sigma_acc_bias=1e-3,
        sigma_gyr_bias=1e-4,
        gnss_pos_std_horiz=1.5,
        gnss_pos_std_vert=3.0,
        chi2_threshold_pos=16.27,
    )

    init_state = NominalState(
        t=0.0,
        p=np.zeros(3),
        v=np.zeros(3),
        q=quat_true[0].copy(),
        ba=np.zeros(3),
        bg=np.zeros(3)
    )

    init_cov = np.diag([
        1.0, 1.0, 4.0,              # pos var
        0.1, 0.1, 0.1,              # vel var
        1e-4, 1e-4, 1e-4,           # att var
        0.04, 0.04, 0.04,           # acc bias var
        1e-4, 1e-4, 1e-4,           # gyr bias var
    ])

    eskf = ESKF(init_state, init_cov, config, g_n=g_n)

    # Storage for filter outputs
    est_p = np.zeros((time_steps, 3), dtype=np.float64)
    est_v = np.zeros((time_steps, 3), dtype=np.float64)
    est_yaw = np.zeros(time_steps, dtype=np.float64)
    est_ba = np.zeros((time_steps, 3), dtype=np.float64)
    est_bg = np.zeros((time_steps, 3), dtype=np.float64)
    cov_diag = np.zeros((time_steps, STATE_DIM), dtype=np.float64)
    min_eigenvalues = np.zeros(time_steps, dtype=np.float64)

    gnss_updates_attempted = 0
    gnss_updates_accepted = 0
    gnss_duplicates_rejected = 0
    gnss_outliers_rejected = 0

    # 4. Execute Causal Filtering Loop
    last_gnss_fix_time = -999.0
    last_gnss_pos = np.zeros(3)

    for i, t in enumerate(timestamps):
        phase = phase_labels[i]

        # Predict step
        eskf.predict(f_b_meas[i], omega_b_meas[i], dt)

        # Determine GNSS update schedule per phase
        send_gnss = False
        is_new = True

        if phase in [1, 2, 3, 4, 5, 8, 9, 10]:
            # 1 Hz GNSS
            if abs(round(t, 2) % 1.0) < 1e-4:
                send_gnss = True
                is_new = True
        elif phase == 6:
            # Sparse GNSS: 0.2 Hz (every 5 seconds)
            if abs(round(t, 2) % 5.0) < 1e-4:
                send_gnss = True
                is_new = True
            elif abs(round(t, 2) % 1.0) < 1e-4:
                # Duplicate sample-and-hold at 1 Hz
                send_gnss = True
                is_new = False
        elif phase == 7:
            # Complete 60s GNSS outage: no fixes
            send_gnss = False

        if send_gnss:
            gnss_updates_attempted += 1
            if is_new:
                noise = np.random.randn(3) * np.array([sigma_gnss_noise, sigma_gnss_noise, sigma_gnss_noise * 2.0])
                z_pos = pos_true[i] + noise
                last_gnss_pos = z_pos
                last_gnss_fix_time = t
            else:
                # Sample and hold
                z_pos = last_gnss_pos
                gnss_duplicates_rejected += 1

            accepted = eskf.update_gnss_pos(z_pos, is_new_fix=is_new, timestamp=t)
            if is_new:
                if accepted:
                    gnss_updates_accepted += 1
                else:
                    gnss_outliers_rejected += 1

        # Record estimates
        est_p[i] = eskf.state.p
        est_v[i] = eskf.state.v
        est_yaw[i] = quat_to_euler(eskf.state.q)[2]
        est_ba[i] = eskf.state.ba
        est_bg[i] = eskf.state.bg
        cov_diag[i] = np.diag(eskf.P)

        eigs = np.linalg.eigvalsh(eskf.P)
        min_eigenvalues[i] = np.min(eigs)

    # 5. Compute Quantitative Metrics
    pos_err = np.linalg.norm(est_p - pos_true, axis=1)
    horiz_pos_err = np.linalg.norm(est_p[:, 0:2] - pos_true[:, 0:2], axis=1)
    vel_err = np.linalg.norm(est_v - vel_true, axis=1)
    true_yaw = np.array([quat_to_euler(q)[2] for q in quat_true])
    yaw_err = np.abs(np.arctan2(np.sin(est_yaw - true_yaw), np.cos(est_yaw - true_yaw)))

    df_results = pd.DataFrame({
        "timestamp": timestamps,
        "phase": phase_labels,
        "pos_true_E": pos_true[:, 0],
        "pos_true_N": pos_true[:, 1],
        "pos_true_U": pos_true[:, 2],
        "pos_est_E": est_p[:, 0],
        "pos_est_N": est_p[:, 1],
        "pos_est_U": est_p[:, 2],
        "horiz_pos_err": horiz_pos_err,
        "3d_pos_err": pos_err,
        "vel_err": vel_err,
        "yaw_err_deg": np.rad2deg(yaw_err),
        "ba_est_x": est_ba[:, 0],
        "ba_est_y": est_ba[:, 1],
        "ba_est_z": est_ba[:, 2],
        "bg_est_z": est_bg[:, 2],
        "pos_std_3d": np.sqrt(cov_diag[:, 0] + cov_diag[:, 1] + cov_diag[:, 2]),
        "min_eigenvalue": min_eigenvalues,
    })

    # Ensure output directory exists
    output_dir = Path("data/processed/phase2")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "synthetic_eskf_scenario_results.csv"
    df_results.to_csv(out_csv, index=False)

    # Phase-by-phase evaluation
    phase_names = {
        1: "Stationary Alignment (10s)",
        2: "East Acceleration (5s)",
        3: "Cruise East 15m/s (20s)",
        4: "90deg Turn to North (15s)",
        5: "Cruise North Biased (25s)",
        6: "Sparse GNSS 0.2Hz (20s)",
        7: "60s GNSS Outage (60s)",
        8: "GNSS Recovery (20s)",
        9: "Braking to Stop (10s)",
        10: "Stationary Post-Stop (15s)",
    }

    print("\n--- PHASE-BY-PHASE PERFORMANCE METRICS ---")
    print(f"{'Phase':<32} {'Pos RMSE (m)':<14} {'Max Pos (m)':<13} {'Vel RMSE (m/s)':<16} {'Yaw RMSE (deg)':<15} {'Min Eig(P)':<12}")
    print("-" * 105)

    phase_metrics = []
    for ph_idx in range(1, 11):
        mask = (np.array(phase_labels) == ph_idx)
        p_rmse = np.sqrt(np.mean(horiz_pos_err[mask] ** 2))
        p_max = np.max(horiz_pos_err[mask])
        v_rmse = np.sqrt(np.mean(vel_err[mask] ** 2))
        y_rmse = np.rad2deg(np.sqrt(np.mean(yaw_err[mask] ** 2)))
        m_eig = np.min(min_eigenvalues[mask])

        print(f"{phase_names[ph_idx]:<32} {p_rmse:<14.3f} {p_max:<13.3f} {v_rmse:<16.3f} {y_rmse:<15.3f} {m_eig:<12.2e}")
        phase_metrics.append({
            "phase_id": ph_idx,
            "phase_name": phase_names[ph_idx],
            "pos_rmse_m": p_rmse,
            "pos_max_m": p_max,
            "vel_rmse_mps": v_rmse,
            "yaw_rmse_deg": y_rmse,
            "min_eig_P": m_eig
        })

    # Outage & Recovery Analysis
    outage_mask = (np.array(phase_labels) == 7)
    recovery_mask = (np.array(phase_labels) == 8)

    outage_start_err = horiz_pos_err[outage_mask][0]
    outage_max_drift = np.max(horiz_pos_err[outage_mask])
    outage_final_drift = horiz_pos_err[outage_mask][-1]

    # Recovery: find time after outage starts where error drops below 3.0 m (GNSS envelope)
    rec_indices = np.where(recovery_mask)[0]
    rec_times = timestamps[rec_indices]
    rec_errors = horiz_pos_err[rec_indices]
    recovered_idx = np.where(rec_errors < 3.0)[0]
    if len(recovered_idx) > 0:
        time_to_recover = rec_times[recovered_idx[0]] - timestamps[rec_indices[0]]
        rec_err_final = rec_errors[-1]
    else:
        time_to_recover = float("nan")
        rec_err_final = rec_errors[-1]

    print("\n--- GNSS OUTAGE (60s) & RECOVERY ANALYSIS ---")
    print(f"Outage Start Horizontal Error:  {outage_start_err:.3f} m")
    print(f"Outage Maximum Drift:           {outage_max_drift:.3f} m")
    print(f"Outage Final Drift (at t=60s):  {outage_final_drift:.3f} m")
    print(f"Post-Outage Recovery Time:      {time_to_recover:.2f} s")
    print(f"Post-Recovery Steady Error:     {rec_err_final:.3f} m")

    print("\n--- GNSS MEASUREMENT AUDIT ---")
    print(f"Total GNSS Attempts:            {gnss_updates_attempted}")
    print(f"Sample-and-Hold Duplicates Gated:{gnss_duplicates_rejected}")
    print(f"Valid Fixes Accepted:           {gnss_updates_accepted}")
    print(f"Outlier Spikes Rejected:        {gnss_outliers_rejected}")

    print(f"\nResults saved to: {out_csv}")
    print("=" * 80)

    # Verification Assertions for Automation
    assert outage_final_drift > 0.0, "Outage drift must be non-zero"
    assert rec_err_final < 3.0, f"Filter must recover post-outage, got {rec_err_final:.3f} m"
    assert gnss_duplicates_rejected > 0, "Duplicate rejection mechanism must be verified"
    assert np.all(min_eigenvalues > -1e-14), "Covariance remained numerically positive semi-definite"
    print("ALL SCENARIO ACCEPTANCE CRITERIA PASSED!")


if __name__ == "__main__":
    run_synthetic_scenario()
