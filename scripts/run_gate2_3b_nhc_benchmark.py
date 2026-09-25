"""
NAVRIS Phase 2.3B Gate 2.3B-3: Real-Data Non-Holonomic Constraints (NHC) A/B Benchmark.

Compares:
- Configuration A: Frozen Gate 2.3A Baseline (Control: Calibrated ESKF + GNSS + Causal ZUPT)
- Configuration B: Experiment (Identical pipeline to A + Frozen Causal NHC Module enabled)

The ONLY experimental difference between Configuration A and Configuration B is the presence of NHC.

Evaluates all 8 IO-VNBD benchmark recordings:
S1, S2, S3A, S4, M, Y1, VTA1A, VTA2.

STRICT PROTOCOLS ENFORCED:
1. Frozen ESKF core: src/navris/eskf/* 100% untouched.
2. Frozen ZUPT module: src/navris/zupt.py 100% untouched.
3. Frozen NHC module: src/navris/nhc.py with pre-declared frozen thresholds:
     sigma_lat = 0.25 m/s, sigma_vert = 0.15 m/s, chi2_thresh = 13.82
     v_min = 1.5 m/s, omega_max = 0.087 rad/s, delta_a_max = 1.0 m/s^2
4. Zero VBOX / future data in runtime detector or filter.
5. Configuration A reproduces Gate 2.3A Configuration B baseline bit-for-bit.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("."))

from navris.calibration import (
    detect_standstill_intervals,
    get_settled_standstill_window,
    estimate_gravity_leveling,
    estimate_mounting_yaw_from_motion,
    identify_gyro_mapping,
    dcm_to_quat,
    CausalCalibrationResult,
)
from navris.eskf import (
    ESKF,
    NominalState,
    ESKFConfig,
    STATE_DIM,
)
from navris.inertial.gnss_fixes import filter_novel_gnss_fixes
from navris.inertial.frames import (
    quat_multiply,
    quat_to_euler,
    quat_to_dcm,
    wrap_angle_pi,
)
from navris.inertial.metrics import compute_along_cross_track_errors
from navris.zupt import (
    ZUPTDetectorConfig,
    CausalStationaryDetector,
    apply_zupt_update,
    DEFAULT_R_ZUPT,
    ZUPT_CHI2_THRESHOLD,
)
from navris.nhc import (
    NHCConfig,
    NHCDetectorConfig,
    CausalNHCDetector,
    compute_nhc_jacobian,
    apply_nhc_update,
    DEFAULT_R_NHC,
    DEFAULT_SIGMA_LAT,
    DEFAULT_SIGMA_VERT,
    NHC_CHI2_THRESHOLD,
    M_NHC,
)

# Common Predeclared ESKF Configuration (Identical to Gate 2.1, Gate 2.2, Gate 2.3A)
COMMON_CONFIG = ESKFConfig(
    sigma_acc=0.20,             # m/s^2
    sigma_gyr=0.02,             # rad/s
    sigma_acc_bias=1e-3,        # m/s^2 / sqrt(s)
    sigma_gyr_bias=1e-4,        # rad/s / sqrt(s)
    gnss_pos_std_horiz=3.0,     # m
    gnss_pos_std_vert=10.0,     # m
    chi2_threshold_pos=16.27    # 3-DOF 99.9% gating
)

COMMON_INIT_COV = np.diag([
    25.0, 25.0, 100.0,          # Position var (m^2)
    4.0, 4.0, 1.0,              # Velocity var (m/s)^2
    0.01, 0.01, 0.05,           # Attitude var (rad^2)
    0.04, 0.04, 0.04,           # Accel bias var (m/s^2)^2
    1e-4, 1e-4, 1e-4            # Gyro bias var (rad/s)^2
])


def calibrate_recording_causal(
    df: pd.DataFrame,
    rec_id: str,
    max_search_time_s: float = 250.0,
    motion_window_s: float = 60.0
) -> Tuple[bool, str, Dict[str, Any], Optional[CausalCalibrationResult], float]:
    """Causal calibration identical to Gate 2.2 and Gate 2.3A."""
    df_search = df[df['time_s'] <= max_search_time_s]
    standstills = detect_standstill_intervals(df_search)
    robust_s = [s for s in standstills if s.duration_s >= 3.0]

    if not robust_s:
        calib_dict = {
            'recording_id': rec_id,
            'calibration_status': 'UNOBSERVABLE',
            'observability_classification': 'Class F — Insufficient excitation',
            'leveling_status': 'FAILED',
            'mounting_status': 'FAILED',
            'gyro_status': 'FAILED',
            'standstills_found': 0,
            'settled_window_s': [0.0, 0.0],
            'roll_deg': np.nan,
            'pitch_deg': np.nan,
            'roll_std_deg': np.nan,
            'pitch_std_deg': np.nan,
            'residual_g_mps2': np.nan,
            'forward_angle_deg': np.nan,
            'mounting_yaw_deg': np.nan,
            'gyro_dominant_axis': 'NONE',
            'gyro_correlation': np.nan,
            't_decision_s': np.nan,
            'notes': 'Zero standstill intervals >= 3.0s detected in initial search window [0, 250s].'
        }
        return False, 'UNOBSERVABLE', calib_dict, None, 0.0

    if rec_id == 'S1':
        t_lev_start, t_lev_end = 15.0, 35.0
        leveling = estimate_gravity_leveling(df, start_time_s=t_lev_start, end_time_s=t_lev_end)
        t_after = 65.0
        t_dec = 125.0
        mounting_yaw = estimate_mounting_yaw_from_motion(
            df[df['time_s'] <= t_dec],
            q_level=leveling.q_level,
            start_time_s=t_after,
            end_time_s=t_dec,
            min_accel_mps2=0.4
        )
    else:
        if len(robust_s) > 1 and robust_s[1].duration_s > 2.0 * robust_s[0].duration_s and robust_s[1].start_time_s < 150.0:
            s_selected = robust_s[1]
        else:
            s_selected = robust_s[0]

        if s_selected.duration_s >= 8.0:
            t_lev_start, t_lev_end = get_settled_standstill_window(s_selected, settle_trim_s=2.0, pre_motion_trim_s=2.0)
        else:
            t_lev_start, t_lev_end = s_selected.start_time_s, s_selected.end_time_s

        leveling = estimate_gravity_leveling(df, start_time_s=t_lev_start, end_time_s=t_lev_end)
        t_after = s_selected.end_time_s
        t_dec = min(df['time_s'].iloc[-1], t_after + motion_window_s)

        mounting_yaw = estimate_mounting_yaw_from_motion(
            df[df['time_s'] <= t_dec],
            q_level=leveling.q_level,
            start_time_s=t_after,
            end_time_s=t_dec,
            min_accel_mps2=0.3
        )

    gyro_map = identify_gyro_mapping(df[df['time_s'] <= t_dec], q_level=leveling.q_level)

    u_up_s = leveling.f_mean / leveling.f_norm
    u_up_s /= np.linalg.norm(u_up_s)

    if mounting_yaw.is_valid and np.linalg.norm(mounting_yaw.u_fwd_body) > 1e-3:
        u_fwd_s = mounting_yaw.u_fwd_body.copy()
        u_fwd_s = u_fwd_s - np.dot(u_fwd_s, u_up_s) * u_up_s
        u_fwd_s /= np.linalg.norm(u_fwd_s)

        u_left_s = np.cross(u_up_s, u_fwd_s)
        u_left_s /= np.linalg.norm(u_left_s)

        u_fwd_s = np.cross(u_left_s, u_up_s)
        u_fwd_s /= np.linalg.norm(u_fwd_s)

        R_body_vehicle = np.vstack([u_fwd_s, u_left_s, u_up_s])
        q_body_vehicle = dcm_to_quat(R_body_vehicle)
    else:
        R_body_vehicle = np.eye(3)
        q_body_vehicle = np.array([1.0, 0.0, 0.0, 0.0])

    is_calibrated = leveling.is_valid and mounting_yaw.is_valid
    status_str = 'PASS' if is_calibrated else 'CONDITIONAL'

    calib_dict = {
        'recording_id': rec_id,
        'calibration_status': status_str,
        'observability_classification': 'Class A — Stable calibrated tracking' if is_calibrated else 'Class B — Partial Observability',
        'leveling_status': leveling.quality,
        'mounting_status': mounting_yaw.quality,
        'gyro_status': 'PASS' if gyro_map.is_valid else 'DEGRADED',
        'standstills_found': len(robust_s),
        'settled_window_s': [t_lev_start, t_lev_end],
        'roll_deg': float(leveling.roll_deg),
        'pitch_deg': float(leveling.pitch_deg),
        'roll_std_deg': float(leveling.roll_std_deg),
        'pitch_std_deg': float(leveling.pitch_std_deg),
        'residual_g_mps2': float(leveling.residual_g_mps2),
        'forward_angle_deg': float(mounting_yaw.forward_angle_phone_frame_deg),
        'mounting_yaw_deg': float(mounting_yaw.mounting_yaw_vehicle_from_phone_deg),
        'gyro_dominant_axis': gyro_map.dominant_yaw_channel,
        'gyro_correlation': float(gyro_map.correlation_with_heading_rate),
        't_decision_s': float(t_dec),
        'notes': f'Standstill at [{t_lev_start:.1f}, {t_lev_end:.1f}s], motion evaluated in [{t_after:.1f}, {t_dec:.1f}s].'
    }

    calib_res = CausalCalibrationResult(
        is_calibrated=is_calibrated,
        calibration_time_s=t_dec,
        method="Method D (Triad: Leveling + Forward Alignment + Gyro Mapping)",
        leveling=leveling,
        mounting_yaw=mounting_yaw,
        gyro_mapping=gyro_map,
        q_body_vehicle=q_body_vehicle,
        R_body_vehicle=R_body_vehicle,
        R_vehicle_body=R_body_vehicle.T,
        forward_angle_phone_frame_deg=mounting_yaw.forward_angle_phone_frame_deg,
        mounting_yaw_vehicle_from_phone_deg=mounting_yaw.mounting_yaw_vehicle_from_phone_deg,
        initial_attitude_q=np.array([1.0, 0.0, 0.0, 0.0]),
        diagnostics={'settled_window_s': [t_lev_start, t_lev_end]}
    )

    return True, status_str, calib_dict, calib_res, t_dec


def run_pipeline_configuration(
    rec_id: str,
    df: pd.DataFrame,
    calib_res: CausalCalibrationResult,
    t_eval_start: float,
    enable_zupt: bool = True,
    enable_nhc: bool = False
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], pd.DataFrame]:
    """
    Executes the real-data ESKF pipeline either as Control (enable_nhc=False)
    or Experiment (enable_nhc=True).
    Both configurations have ZUPT enabled (enable_zupt=True).

    Returns:
        (metrics_dict, gnss_update_logs, zupt_update_logs, nhc_update_logs, trajectory_df)
    """
    if rec_id == 'S1':
        init_idx = int(df.index[df['time_s'] == 156.0][0])
    else:
        init_idx = int(df.index[df['time_s'] >= t_eval_start][0])

    eval_df = df.iloc[init_idx:].copy().reset_index(drop=True)
    n_samples = len(eval_df)
    t_eval_end = float(eval_df['time_s'].iloc[-1])
    eval_duration_s = t_eval_end - t_eval_start

    # Initial Position
    p0 = np.array([
        eval_df['phone_gps_east_m'].iloc[0],
        eval_df['phone_gps_north_m'].iloc[0],
        eval_df['phone_gps_up_m'].iloc[0] if 'phone_gps_up_m' in eval_df.columns else 0.0
    ], dtype=np.float64)

    # Initial Heading
    c_fixes = df[(df['time_s'] <= t_eval_start) & (df['phone_gps_is_new_fix'] == True)]
    c_moving = c_fixes[c_fixes['phone_gps_speed_mps'] > 1.0]
    if len(c_moving) > 0:
        init_heading = float(c_moving['phone_gps_orientation_rad'].iloc[-1])
    elif len(c_fixes) > 0:
        init_heading = float(c_fixes['phone_gps_orientation_rad'].iloc[-1])
    else:
        init_heading = 0.0

    # Initial Speed
    v_gnss_speed = float(eval_df['phone_gps_speed_mps'].iloc[0])
    vel_init_method = 'phone_gps_speed_mps'
    init_spd = v_gnss_speed
    if len(c_fixes) >= 2:
        f_last = c_fixes.iloc[-1]
        f_prev = c_fixes.iloc[-2]
        de_d = float(f_last['phone_gps_east_m'] - f_prev['phone_gps_east_m'])
        dn_d = float(f_last['phone_gps_north_m'] - f_prev['phone_gps_north_m'])
        dt_d = float(f_last['time_s'] - f_prev['time_s'])
        if dt_d > 0.1:
            v_disp = float(np.hypot(de_d, dn_d) / dt_d)
            if v_disp > 2.0 and (abs(v_gnss_speed - v_disp) / max(v_disp, 1.0)) > 0.40:
                init_spd = v_disp
                vel_init_method = 'causal_displacement_velocity'

    v0 = np.array([init_spd * np.sin(init_heading), init_spd * np.cos(init_heading), 0.0], dtype=np.float64)

    # Initial Attitude
    theta_nav = np.pi / 2.0 - init_heading
    q_v_n = np.array([np.cos(0.5 * theta_nav), 0.0, 0.0, np.sin(0.5 * theta_nav)], dtype=np.float64)
    q0 = quat_multiply(q_v_n, calib_res.q_body_vehicle)

    # Initialize ESKF
    init_state = NominalState(
        t=t_eval_start, p=p0.copy(), v=v0.copy(), q=q0.copy(),
        ba=np.zeros(3), bg=np.zeros(3)
    )
    eskf = ESKF(init_state, COMMON_INIT_COV.copy(), COMMON_CONFIG, lat_deg=52.4, alt_m=100.0)
    M_gyro = calib_res.gyro_mapping.mapping_matrix
    C_b_v = calib_res.R_body_vehicle

    # Detectors
    zupt_detector = CausalStationaryDetector() if enable_zupt else None
    nhc_detector = CausalNHCDetector() if enable_nhc else None

    # State arrays
    time_arr = np.zeros(n_samples)
    ref_pos = np.zeros((n_samples, 3))
    ref_vel = np.zeros((n_samples, 3))
    ref_head = np.zeros(n_samples)
    eskf_pos = np.zeros((n_samples, 3))
    eskf_vel = np.zeros((n_samples, 3))
    eskf_head = np.zeros(n_samples)

    min_cov_eig = float('inf')
    cov_symmetric = True
    nan_detected = False

    gnss_update_logs = []
    zupt_update_logs = []
    nhc_update_logs = []

    acc_count = 0
    rej_count = 0
    nises = []

    curr_rej_streak_s = 0.0
    max_rej_streak_s = 0.0
    first_rejection_time_s = None

    for i in range(n_samples):
        t = float(eval_df['time_s'].iloc[i])
        time_arr[i] = t
        ref_pos[i] = [eval_df['ref_east_m'].iloc[i], eval_df['ref_north_m'].iloc[i], eval_df['ref_up_m'].iloc[i]]

        r_spd = float(eval_df['ref_speed_mps'].iloc[i])
        r_hdg = float(eval_df['ref_heading_rad'].iloc[i])
        ref_vel[i] = [r_spd * np.sin(r_hdg), r_spd * np.cos(r_hdg), 0.0]
        ref_head[i] = r_hdg

        # Raw IMU
        raw_fb = np.array([
            eval_df['phone_accel_x_mps2'].iloc[i],
            eval_df['phone_accel_y_mps2'].iloc[i],
            eval_df['phone_accel_z_mps2'].iloc[i]
        ], dtype=np.float64)
        raw_wb = np.array([
            eval_df['phone_gyro_x_radps'].iloc[i],
            eval_df['phone_gyro_y_radps'].iloc[i],
            eval_df['phone_gyro_z_radps'].iloc[i]
        ], dtype=np.float64)

        if i > 0:
            dt = t - time_arr[i - 1]
            prev_fb = np.array([
                eval_df['phone_accel_x_mps2'].iloc[i - 1],
                eval_df['phone_accel_y_mps2'].iloc[i - 1],
                eval_df['phone_accel_z_mps2'].iloc[i - 1]
            ], dtype=np.float64)
            prev_wb = np.array([
                eval_df['phone_gyro_x_radps'].iloc[i - 1],
                eval_df['phone_gyro_y_radps'].iloc[i - 1],
                eval_df['phone_gyro_z_radps'].iloc[i - 1]
            ], dtype=np.float64)
            wb_corr = M_gyro @ prev_wb
            eskf.predict(prev_fb, wb_corr, dt)

        # 1. GNSS Measurement Update
        is_new_gnss = bool(eval_df['phone_gps_is_new_fix'].iloc[i])
        if is_new_gnss:
            z_pos = np.array([
                eval_df['phone_gps_east_m'].iloc[i],
                eval_df['phone_gps_north_m'].iloc[i],
                eval_df['phone_gps_up_m'].iloc[i]
            ], dtype=np.float64)
            acc_m = max(1.0, float(eval_df['phone_gps_accuracy_m'].iloc[i]))
            pos_cov = np.diag([acc_m**2, acc_m**2, (acc_m * 3.0)**2])
            acc = eskf.update_gnss_pos(z_pos, pos_cov=pos_cov, is_new_fix=True, timestamp=t)
            nis = float(eskf.innovations[-1].mahalanobis_sq)
            nises.append(nis)

            gnss_update_logs.append({
                'time_s': t,
                'accepted': acc,
                'nis': nis,
                'accuracy_m': acc_m,
                'pos_error_prior_m': float(np.linalg.norm(eskf_pos[i-1][:2] - ref_pos[i-1][:2])) if i > 0 else 0.0
            })

            if acc:
                acc_count += 1
                curr_rej_streak_s = 0.0
            else:
                rej_count += 1
                if first_rejection_time_s is None:
                    first_rejection_time_s = t
                if i > 0:
                    curr_rej_streak_s += (t - time_arr[i-1])
                if curr_rej_streak_s > max_rej_streak_s:
                    max_rej_streak_s = curr_rej_streak_s

        # 2. Causal ZUPT Update
        is_stat = False
        if enable_zupt and zupt_detector is not None:
            is_stat = zupt_detector.update(raw_fb, raw_wb)
            if is_stat:
                v_prior = eskf.state.v.copy()
                z_acc, z_record = apply_zupt_update(eskf, timestamp=t)
                v_post = eskf.state.v.copy()

                zupt_update_logs.append({
                    'time_s': t,
                    'accepted': z_acc,
                    'nis': float(z_record.mahalanobis_sq),
                    'v_norm_prior_mps': float(np.linalg.norm(v_prior)),
                    'v_norm_post_mps': float(np.linalg.norm(v_post)),
                    'v_error_ref_prior_mps': float(np.linalg.norm(v_prior - ref_vel[i])),
                    'v_error_ref_post_mps': float(np.linalg.norm(v_post - ref_vel[i])),
                })

        # 3. Causal NHC Update (Configuration B only)
        if enable_nhc and nhc_detector is not None and not is_stat:
            wb_corr = M_gyro @ raw_wb
            should_apply_nhc = nhc_detector.update(
                nominal_state=eskf.state,
                C_b_v=C_b_v,
                f_meas_b=raw_fb,
                omega_meas_b=wb_corr,
                is_stationary=is_stat
            )

            if should_apply_nhc:
                # Predict vehicle velocity for logging
                C_bn = quat_to_dcm(eskf.state.q)
                C_nv = C_b_v @ C_bn.T
                v_veh = C_nv @ eskf.state.v

                # Apply update
                nhc_acc, nhc_record = apply_nhc_update(eskf, C_b_v, timestamp=t)

                nhc_update_logs.append({
                    'time_s': t,
                    'accepted': nhc_acc,
                    'nis': float(nhc_record.mahalanobis_sq),
                    'forward_speed_mps': float(v_veh[0]),
                    'pred_vy_mps': float(v_veh[1]),
                    'pred_vz_mps': float(v_veh[2]),
                    'residual_y_mps': float(nhc_record.residual[0]),
                    'residual_z_mps': float(nhc_record.residual[1]),
                })

        eskf_pos[i] = eskf.state.p
        eskf_vel[i] = eskf.state.v

        # Yaw in navigation frame (azimuth clockwise from North)
        r, p, y = quat_to_euler(eskf.state.q)
        eskf_head[i] = (np.pi / 2.0 - y) % (2.0 * np.pi)

        # Numerical sanity check
        if i % 100 == 0:
            eig = float(np.min(np.linalg.eigvalsh(eskf.P)))
            if eig < min_cov_eig:
                min_cov_eig = eig
            if not np.allclose(eskf.P, eskf.P.T, atol=1e-6):
                cov_symmetric = False
            if np.isnan(eskf.P).any() or np.isnan(eskf.state.p).any():
                nan_detected = True

    # Trajectory dataframe for plotting
    horiz_errs = np.linalg.norm(eskf_pos[:, :2] - ref_pos[:, :2], axis=1)
    traj_dict = {
        'time_s': time_arr,
        'ref_east_m': ref_pos[:, 0],
        'ref_north_m': ref_pos[:, 1],
        'ref_up_m': ref_pos[:, 2],
        'ref_speed_mps': eval_df['ref_speed_mps'].values,
        'ref_heading_rad': ref_head,
        'eskf_east_m': eskf_pos[:, 0],
        'eskf_north_m': eskf_pos[:, 1],
        'eskf_up_m': eskf_pos[:, 2],
        'eskf_vel_east_mps': eskf_vel[:, 0],
        'eskf_vel_north_mps': eskf_vel[:, 1],
        'eskf_vel_up_mps': eskf_vel[:, 2],
        'eskf_heading_rad': eskf_head,
        'horiz_err_m': horiz_errs,
    }
    traj_df = pd.DataFrame(traj_dict)

    # Metrics computation
    dist_traveled_m = float(np.sum(np.linalg.norm(np.diff(ref_pos[:, :2], axis=0), axis=1)))
    rmse_horiz = float(np.sqrt(np.mean(horiz_errs**2)))
    mae_horiz = float(np.mean(horiz_errs))
    max_horiz = float(np.max(horiz_errs))
    final_horiz = float(horiz_errs[-1])

    # Along-track / cross-track errors
    along_err, cross_err = compute_along_cross_track_errors(eskf_pos, ref_pos, ref_head)
    rmse_along = float(np.sqrt(np.mean(along_err**2)))
    rmse_cross = float(np.sqrt(np.mean(cross_err**2)))

    # Velocity metrics
    vel_errs = np.linalg.norm(eskf_vel - ref_vel, axis=1)
    rmse_vel = float(np.sqrt(np.mean(vel_errs**2)))

    eskf_speed = np.linalg.norm(eskf_vel[:, :2], axis=1)
    ref_spd = eval_df['ref_speed_mps'].values
    rmse_speed = float(np.sqrt(np.mean((eskf_speed - ref_spd)**2)))

    # Heading error (evaluated only when speed > 2.0 m/s)
    speed_mask = ref_spd > 2.0
    if np.any(speed_mask):
        head_diff = wrap_angle_pi(eskf_head[speed_mask] - ref_head[speed_mask])
        rmse_head_deg = float(np.degrees(np.sqrt(np.mean(head_diff**2))))
    else:
        rmse_head_deg = np.nan

    total_fixes = acc_count + rej_count
    acc_pct = float((acc_count / total_fixes * 100.0)) if total_fixes > 0 else 0.0
    med_nis = float(np.median(nises)) if len(nises) > 0 else np.nan
    max_nis = float(np.max(nises)) if len(nises) > 0 else np.nan

    # Tracking thresholds
    t_to_50m = np.nan
    t_to_100m = np.nan
    idx_50 = np.where(horiz_errs > 50.0)[0]
    if len(idx_50) > 0:
        t_to_50m = float(time_arr[idx_50[0]] - time_arr[0])
    idx_100 = np.where(horiz_errs > 100.0)[0]
    if len(idx_100) > 0:
        t_to_100m = float(time_arr[idx_100[0]] - time_arr[0])

    # Error at time milestones
    err_at_time = {}
    for mark in [10.0, 30.0, 60.0, 120.0]:
        idx_mark = np.searchsorted(time_arr - time_arr[0], mark)
        if idx_mark < n_samples:
            err_at_time[f'error_{int(mark)}s_m'] = float(horiz_errs[idx_mark])
        else:
            err_at_time[f'error_{int(mark)}s_m'] = np.nan

    metrics = {
        'status': 'SUCCESS',
        'recording_id': rec_id,
        'configuration': 'B_EXPERIMENT_NHC' if enable_nhc else 'A_CONTROL_BASELINE',
        'enable_zupt': enable_zupt,
        'enable_nhc': enable_nhc,
        'evaluation_start_time_s': float(t_eval_start),
        'evaluation_end_time_s': float(t_eval_end),
        'duration_s': float(eval_duration_s),
        'samples': int(n_samples),
        'distance_traveled_m': dist_traveled_m,
        'horiz_rmse_m': rmse_horiz,
        'horiz_mae_m': mae_horiz,
        'horiz_max_m': max_horiz,
        'final_horiz_m': final_horiz,
        'along_track_rmse_m': rmse_along,
        'cross_track_rmse_m': rmse_cross,
        'vel_rmse_mps': rmse_vel,
        'speed_rmse_mps': rmse_speed,
        'heading_rmse_deg': rmse_head_deg,
        'gnss_total_novel_fixes': total_fixes,
        'gnss_accepted_fixes': acc_count,
        'gnss_rejected_fixes': rej_count,
        'gnss_acceptance_pct': acc_pct,
        'gnss_median_nis': med_nis,
        'gnss_max_nis': max_nis,
        'gnss_first_rejection_time_s': first_rejection_time_s,
        'gnss_max_consecutive_rejection_s': max_rej_streak_s,
        'time_to_50m_s': t_to_50m,
        'time_to_100m_s': t_to_100m,
        'error_10s_m': err_at_time['error_10s_m'],
        'error_30s_m': err_at_time['error_30s_m'],
        'error_60s_m': err_at_time['error_60s_m'],
        'error_120s_m': err_at_time['error_120s_m'],
        'min_cov_eig': min_cov_eig,
        'cov_symmetric': cov_symmetric,
        'nan_detected': nan_detected,
    }

    # ZUPT metrics
    z_attempts = len(zupt_update_logs)
    z_accs = sum(1 for u in zupt_update_logs if u['accepted'])
    z_rejs = z_attempts - z_accs
    z_nises = [u['nis'] for u in zupt_update_logs]

    metrics.update({
        'zupt_updates_attempted': z_attempts,
        'zupt_updates_accepted': z_accs,
        'zupt_updates_rejected': z_rejs,
        'zupt_acceptance_pct': (z_accs / max(1, z_attempts)) * 100.0 if z_attempts > 0 else 0.0,
        'zupt_median_nis': float(np.median(z_nises)) if z_nises else np.nan,
        'zupt_max_nis': float(np.max(z_nises)) if z_nises else np.nan,
    })

    # NHC metrics
    if enable_nhc:
        nhc_candidates = len(nhc_update_logs)
        nhc_accs = sum(1 for u in nhc_update_logs if u['accepted'])
        nhc_rejs = nhc_candidates - nhc_accs
        nhc_nises = [u['nis'] for u in nhc_update_logs]

        metrics.update({
            'nhc_candidate_count': nhc_candidates,
            'nhc_accepted_count': nhc_accs,
            'nhc_rejected_count': nhc_rejs,
            'nhc_acceptance_pct': (nhc_accs / max(1, nhc_candidates)) * 100.0 if nhc_candidates > 0 else 0.0,
            'nhc_median_nis': float(np.median(nhc_nises)) if nhc_nises else np.nan,
            'nhc_max_nis': float(np.max(nhc_nises)) if nhc_nises else np.nan,
            'nhc_p95_nis': float(np.percentile(nhc_nises, 95)) if nhc_nises else np.nan,
        })
    else:
        metrics.update({
            'nhc_candidate_count': 0,
            'nhc_accepted_count': 0,
            'nhc_rejected_count': 0,
            'nhc_acceptance_pct': 0.0,
            'nhc_median_nis': np.nan,
            'nhc_max_nis': np.nan,
            'nhc_p95_nis': np.nan,
        })

    return metrics, gnss_update_logs, zupt_update_logs, nhc_update_logs, traj_df


def main():
    print("================================================================================")
    print("NAVRIS Phase 2.3B Gate 2.3B-3: Real-Data Non-Holonomic Constraints (NHC) A/B Benchmark")
    print("================================================================================")

    out_dir = "data/processed/phase2_3b/gate2_3b"
    os.makedirs(out_dir, exist_ok=True)

    recordings = ['S1', 'S2', 'S3A', 'S4', 'M', 'Y1', 'VTA1A', 'VTA2']
    summary_rows = []
    all_nhc_updates = []

    for rec in recordings:
        parquet_p = f"data/processed/synchronized/{rec}_sync.parquet"
        print(f"\n=======================================================")
        print(f"PROCESSING RECORDING: {rec}")
        print(f"=======================================================")

        df = pd.read_parquet(parquet_p)
        is_obs, status_str, calib_dict, calib_res, t_dec = calibrate_recording_causal(df, rec)

        if not is_obs:
            print(f"{rec}: UNOBSERVABLE (Class F). Preserving baseline classification.")
            for cfg_name in ['A_CONTROL_BASELINE', 'B_EXPERIMENT_NHC']:
                summary_rows.append({
                    'recording_id': rec,
                    'configuration': cfg_name,
                    'status': 'UNOBSERVABLE',
                    'duration_s': float(df['time_s'].iloc[-1] - df['time_s'].iloc[0]),
                    'samples': len(df)
                })
            continue

        # Find evaluation start time
        if rec == 'S1':
            t_eval_start = 156.0
        else:
            cand = df[df['time_s'] >= t_dec]
            novel = filter_novel_gnss_fixes(cand)
            t_eval_start = float(novel.iloc[0]['time_s'])

        print(f"Causal calibration PASS. Evaluation start: {t_eval_start:.1f} s")

        # -------------------------------------------------------------
        # 1. RUN CONFIGURATION A (CONTROL: Frozen Gate 2.3A Baseline)
        # -------------------------------------------------------------
        print(f"Running Configuration A (Control: Frozen Gate 2.3A Baseline with ZUPT)...")
        m_A, gnss_A, zupt_A, _, traj_A = run_pipeline_configuration(
            rec, df, calib_res, t_eval_start, enable_zupt=True, enable_nhc=False
        )
        summary_rows.append(m_A)
        print(f"  Config A -> H-RMSE: {m_A['horiz_rmse_m']:.2f} m | Final: {m_A['final_horiz_m']:.2f} m | GNSS Acc: {m_A['gnss_accepted_fixes']}/{m_A['gnss_total_novel_fixes']} ({m_A['gnss_acceptance_pct']:.1f}%) | ZUPT Acc: {m_A['zupt_updates_accepted']}")

        # -------------------------------------------------------------
        # 2. RUN CONFIGURATION B (EXPERIMENT: Causal NHC Added)
        # -------------------------------------------------------------
        print(f"Running Configuration B (Experiment: Causal NHC Enabled)...")
        m_B, gnss_B, zupt_B, nhc_B, traj_B = run_pipeline_configuration(
            rec, df, calib_res, t_eval_start, enable_zupt=True, enable_nhc=True
        )
        summary_rows.append(m_B)
        print(f"  Config B -> H-RMSE: {m_B['horiz_rmse_m']:.2f} m | Final: {m_B['final_horiz_m']:.2f} m | GNSS Acc: {m_B['gnss_accepted_fixes']}/{m_B['gnss_total_novel_fixes']} ({m_B['gnss_acceptance_pct']:.1f}%) | ZUPT Acc: {m_B['zupt_updates_accepted']}")
        print(f"  NHC Stats: {m_B['nhc_accepted_count']}/{m_B['nhc_candidate_count']} accepted ({m_B['nhc_acceptance_pct']:.1f}%) | Median NIS: {m_B['nhc_median_nis']:.2f} | Max NIS: {m_B['nhc_max_nis']:.2f}")

        # Attach recording ID to NHC logs
        for u in nhc_B:
            u['recording_id'] = rec
            all_nhc_updates.append(u)

        # Save individual per-recording JSONs
        with open(os.path.join(out_dir, f"{rec}_A_metrics.json"), "w") as f:
            json.dump(m_A, f, indent=2)
        with open(os.path.join(out_dir, f"{rec}_B_metrics.json"), "w") as f:
            json.dump(m_B, f, indent=2)

        # Save comparison trajectory CSV for plot generation
        comp_traj = pd.DataFrame({
            'time_s': traj_A['time_s'],
            'ref_east_m': traj_A['ref_east_m'],
            'ref_north_m': traj_A['ref_north_m'],
            'ref_up_m': traj_A['ref_up_m'],
            'ref_speed_mps': traj_A['ref_speed_mps'],
            'ref_heading_rad': traj_A['ref_heading_rad'],
            'pos_A_east_m': traj_A['eskf_east_m'],
            'pos_A_north_m': traj_A['eskf_north_m'],
            'pos_A_up_m': traj_A['eskf_up_m'],
            'pos_B_east_m': traj_B['eskf_east_m'],
            'pos_B_north_m': traj_B['eskf_north_m'],
            'pos_B_up_m': traj_B['eskf_up_m'],
            'vel_A_east_mps': traj_A['eskf_vel_east_mps'],
            'vel_A_north_mps': traj_A['eskf_vel_north_mps'],
            'vel_A_up_mps': traj_A['eskf_vel_up_mps'],
            'vel_B_east_mps': traj_B['eskf_vel_east_mps'],
            'vel_B_north_mps': traj_B['eskf_vel_north_mps'],
            'vel_B_up_mps': traj_B['eskf_vel_up_mps'],
            'head_A_rad': traj_A['eskf_heading_rad'],
            'head_B_rad': traj_B['eskf_heading_rad'],
            'horiz_err_A_m': traj_A['horiz_err_m'],
            'horiz_err_B_m': traj_B['horiz_err_m'],
        })
        comp_traj.to_csv(os.path.join(out_dir, f"{rec}_trajectory_comparison.csv"), index=False)

    # Save summary tables
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(out_dir, "gate2_3b_summary.csv")
    summary_json = os.path.join(out_dir, "gate2_3b_summary.json")
    summary_df.to_csv(summary_csv, index=False)
    summary_df.to_json(summary_json, orient="records", indent=2)

    # Save NHC updates
    if all_nhc_updates:
        nhc_df = pd.DataFrame(all_nhc_updates)
        nhc_df.to_csv(os.path.join(out_dir, "nhc_updates.csv"), index=False)

    print("\n================================================================================")
    print("GATE 2.3B-3 BENCHMARK COMPLETE. Summary saved to:")
    print(f"  {summary_csv}")
    print("================================================================================")


if __name__ == '__main__':
    main()
