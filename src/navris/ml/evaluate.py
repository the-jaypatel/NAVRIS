"""
NAVRIS Phase 3.0: Evaluation Metrics and Baseline Comparison.

Computes:
1. ML Target Metrics:
   - MAE, RMSE, Bias per axis (East, North, Up)
   - Horizontal and 3D residual RMSE
2. Navigation Velocity Metrics:
   - Baseline A (Existing NAVRIS classical solution)
   - Baseline B (Existing NAVRIS + ZUPT + NHC)
   - Baseline C (Preliminary ML residual prediction: v_corr = v_NAVRIS + pred_delta_v)
3. Open-Loop Position Trajectory Drift (integrating velocity to evaluate navigation impact):
   - Horizontal RMSE (m)
   - Final horizontal error (m)
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def compute_axis_metrics(true_val: np.ndarray, pred_val: np.ndarray) -> Dict[str, float]:
    """
    Computes MAE, RMSE, and bias for a 1D residual/error signal.
    error = true - pred
    """
    err = true_val - pred_val
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))
    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "max_err": float(np.max(np.abs(err))),
    }


def evaluate_residual_predictions(
    df_meta: pd.DataFrame,
    y_true: pd.DataFrame,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluates ML target predictions and compares Navigation Baselines A, B, and C.

    Args:
        df_meta: Metadata DataFrame containing time_s, v_ref, navris_vel, etc.
        y_true: True delta_v target DataFrame [delta_v_E, delta_v_N, delta_v_U].
        y_pred: Predicted delta_v array of shape (N, 3).

    Returns:
        Structured dictionary of ML and Navigation metrics.
    """
    t_delta_e = y_true.iloc[:, 0].values
    t_delta_n = y_true.iloc[:, 1].values
    t_delta_u = y_true.iloc[:, 2].values

    p_delta_e = y_pred[:, 0]
    p_delta_n = y_pred[:, 1]
    p_delta_u = y_pred[:, 2]

    # --- 1. ML Target Metrics (Prediction of delta_v) ---
    m_e = compute_axis_metrics(t_delta_e, p_delta_e)
    m_n = compute_axis_metrics(t_delta_n, p_delta_n)
    m_u = compute_axis_metrics(t_delta_u, p_delta_u)

    # Combined target metrics
    err_e = t_delta_e - p_delta_e
    err_n = t_delta_n - p_delta_n
    err_u = t_delta_u - p_delta_u

    target_horiz_rmse = float(np.sqrt(np.mean(err_e**2 + err_n**2)))
    target_3d_rmse = float(np.sqrt(np.mean(err_e**2 + err_n**2 + err_u**2)))
    target_3d_mae = float(np.mean(np.sqrt(err_e**2 + err_n**2 + err_u**2)))

    ml_metrics = {
        "east": m_e,
        "north": m_n,
        "up": m_u,
        "horizontal_rmse_mps": target_horiz_rmse,
        "rmse_3d_mps": target_3d_rmse,
        "mae_3d_mps": target_3d_mae,
    }

    # --- 2. Navigation Velocity Comparisons (Baselines A, B, C) ---
    v_ref_e = df_meta["v_ref_east_mps"].values
    v_ref_n = df_meta["v_ref_north_mps"].values
    v_ref_u = df_meta["v_ref_up_mps"].values

    # Baseline A (Classical NAVRIS)
    v_A_e = df_meta["navris_vel_east_mps"].values
    v_A_n = df_meta["navris_vel_north_mps"].values
    v_A_u = df_meta["navris_vel_up_mps"].values

    err_A_e = v_ref_e - v_A_e
    err_A_n = v_ref_n - v_A_n
    err_A_u = v_ref_u - v_A_u

    nav_A = {
        "horiz_vel_rmse_mps": float(np.sqrt(np.mean(err_A_e**2 + err_A_n**2))),
        "vel_3d_rmse_mps": float(np.sqrt(np.mean(err_A_e**2 + err_A_n**2 + err_A_u**2))),
        "vel_3d_mae_mps": float(np.mean(np.sqrt(err_A_e**2 + err_A_n**2 + err_A_u**2))),
        "final_vel_err_mps": float(np.sqrt(err_A_e[-1]**2 + err_A_n[-1]**2 + err_A_u[-1]**2)),
    }

    # Baseline B (Classical + NHC) if present
    nav_B = None
    if "navris_B_vel_east_mps" in df_meta.columns:
        v_B_e = df_meta["navris_B_vel_east_mps"].values
        v_B_n = df_meta["navris_B_vel_north_mps"].values
        v_B_u = df_meta["navris_B_vel_up_mps"].values

        err_B_e = v_ref_e - v_B_e
        err_B_n = v_ref_n - v_B_n
        err_B_u = v_ref_u - v_B_u

        nav_B = {
            "horiz_vel_rmse_mps": float(np.sqrt(np.mean(err_B_e**2 + err_B_n**2))),
            "vel_3d_rmse_mps": float(np.sqrt(np.mean(err_B_e**2 + err_B_n**2 + err_B_u**2))),
            "vel_3d_mae_mps": float(np.mean(np.sqrt(err_B_e**2 + err_B_n**2 + err_B_u**2))),
            "final_vel_err_mps": float(np.sqrt(err_B_e[-1]**2 + err_B_n[-1]**2 + err_B_u[-1]**2)),
        }

    # Baseline C (Preliminary ML Corrected: v_C = v_A + pred_delta_v)
    v_C_e = v_A_e + p_delta_e
    v_C_n = v_A_n + p_delta_n
    v_C_u = v_A_u + p_delta_u

    err_C_e = v_ref_e - v_C_e
    err_C_n = v_ref_n - v_C_n
    err_C_u = v_ref_u - v_C_u

    nav_C = {
        "horiz_vel_rmse_mps": float(np.sqrt(np.mean(err_C_e**2 + err_C_n**2))),
        "vel_3d_rmse_mps": float(np.sqrt(np.mean(err_C_e**2 + err_C_n**2 + err_C_u**2))),
        "vel_3d_mae_mps": float(np.mean(np.sqrt(err_C_e**2 + err_C_n**2 + err_C_u**2))),
        "final_vel_err_mps": float(np.sqrt(err_C_e[-1]**2 + err_C_n[-1]**2 + err_C_u[-1]**2)),
    }

    # --- 3. Open-Loop Velocity Integration (Position Drift Comparison) ---
    # Integrate velocities from t0 to evaluate navigation horizontal position drift
    t_vals = df_meta["time_s"].values
    dt = np.diff(t_vals)
    # Ensure dt is valid
    if len(dt) > 0 and np.all(dt > 0):
        # Trapezoidal or forward-Euler integration
        pos_ref_e = np.concatenate([[0.0], np.cumsum(0.5 * (v_ref_e[:-1] + v_ref_e[1:]) * dt)])
        pos_ref_n = np.concatenate([[0.0], np.cumsum(0.5 * (v_ref_n[:-1] + v_ref_n[1:]) * dt)])

        pos_A_e = np.concatenate([[0.0], np.cumsum(0.5 * (v_A_e[:-1] + v_A_e[1:]) * dt)])
        pos_A_n = np.concatenate([[0.0], np.cumsum(0.5 * (v_A_n[:-1] + v_A_n[1:]) * dt)])

        pos_C_e = np.concatenate([[0.0], np.cumsum(0.5 * (v_C_e[:-1] + v_C_e[1:]) * dt)])
        pos_C_n = np.concatenate([[0.0], np.cumsum(0.5 * (v_C_n[:-1] + v_C_n[1:]) * dt)])

        p_err_A = np.sqrt((pos_A_e - pos_ref_e)**2 + (pos_A_n - pos_ref_n)**2)
        p_err_C = np.sqrt((pos_C_e - pos_ref_e)**2 + (pos_C_n - pos_ref_n)**2)

        pos_drift = {
            "baseline_A_horiz_pos_rmse_m": float(np.sqrt(np.mean(p_err_A**2))),
            "baseline_A_final_horiz_err_m": float(p_err_A[-1]),
            "baseline_C_horiz_pos_rmse_m": float(np.sqrt(np.mean(p_err_C**2))),
            "baseline_C_final_horiz_err_m": float(p_err_C[-1]),
        }

        if nav_B is not None:
            pos_B_e = np.concatenate([[0.0], np.cumsum(0.5 * (v_B_e[:-1] + v_B_e[1:]) * dt)])
            pos_B_n = np.concatenate([[0.0], np.cumsum(0.5 * (v_B_n[:-1] + v_B_n[1:]) * dt)])
            p_err_B = np.sqrt((pos_B_e - pos_ref_e)**2 + (pos_B_n - pos_ref_n)**2)
            pos_drift["baseline_B_horiz_pos_rmse_m"] = float(np.sqrt(np.mean(p_err_B**2)))
            pos_drift["baseline_B_final_horiz_err_m"] = float(p_err_B[-1])
    else:
        pos_drift = {}

    return {
        "ml_target_metrics": ml_metrics,
        "navigation_metrics": {
            "baseline_A": nav_A,
            "baseline_B": nav_B,
            "baseline_C_ml": nav_C,
            "open_loop_position_drift": pos_drift,
        },
    }


def format_metrics_table(results: Dict[str, Any]) -> str:
    """
    Formats the evaluation results as a markdown table for reporting.
    """
    ml = results["ml_target_metrics"]
    nav = results["navigation_metrics"]

    lines = [
        "| Metric | Axis E | Axis N | Axis U | Combined / Horiz |",
        "| :--- | :--- | :--- | :--- | :--- |",
        f"| ML Target MAE (m/s) | {ml['east']['mae']:.2f} | {ml['north']['mae']:.2f} | {ml['up']['mae']:.2f} | 3D: {ml['mae_3d_mps']:.2f} |",
        f"| ML Target RMSE (m/s) | {ml['east']['rmse']:.2f} | {ml['north']['rmse']:.2f} | {ml['up']['rmse']:.2f} | Horiz: {ml['horizontal_rmse_mps']:.2f} (3D: {ml['rmse_3d_mps']:.2f}) |",
        f"| ML Target Bias (m/s) | {ml['east']['bias']:.2f} | {ml['north']['bias']:.2f} | {ml['up']['bias']:.2f} | — |",
        "",
        "### Navigation Velocity RMSE Comparison:",
        "| Solution | Horiz Vel RMSE (m/s) | 3D Vel RMSE (m/s) | Final Vel Err (m/s) |",
        "| :--- | :--- | :--- | :--- |",
        f"| Baseline A (Classical NAVRIS) | {nav['baseline_A']['horiz_vel_rmse_mps']:.2f} | {nav['baseline_A']['vel_3d_rmse_mps']:.2f} | {nav['baseline_A']['final_vel_err_mps']:.2f} |",
    ]
    if nav["baseline_B"] is not None:
        lines.append(
            f"| Baseline B (NAVRIS + ZUPT + NHC) | {nav['baseline_B']['horiz_vel_rmse_mps']:.2f} | {nav['baseline_B']['vel_3d_rmse_mps']:.2f} | {nav['baseline_B']['final_vel_err_mps']:.2f} |"
        )
    lines.append(
        f"| Baseline C (Preliminary ML Residual) | {nav['baseline_C_ml']['horiz_vel_rmse_mps']:.2f} | {nav['baseline_C_ml']['vel_3d_rmse_mps']:.2f} | {nav['baseline_C_ml']['final_vel_err_mps']:.2f} |"
    )

    p_drift = nav.get("open_loop_position_drift", {})
    if p_drift:
        lines.extend([
            "",
            "### Open-Loop Horizontal Position Drift Comparison:",
            "| Solution | Horiz Pos RMSE (m) | Final Horiz Err (m) |",
            "| :--- | :--- | :--- |",
            f"| Baseline A (Classical) | {p_drift['baseline_A_horiz_pos_rmse_m']:.1f} | {p_drift['baseline_A_final_horiz_err_m']:.1f} |",
        ])
        if "baseline_B_horiz_pos_rmse_m" in p_drift:
            lines.append(
                f"| Baseline B (Classical + NHC) | {p_drift['baseline_B_horiz_pos_rmse_m']:.1f} | {p_drift['baseline_B_final_horiz_err_m']:.1f} |"
            )
        lines.append(
            f"| Baseline C (ML Residual Corrected) | {p_drift['baseline_C_horiz_pos_rmse_m']:.1f} | {p_drift['baseline_C_final_horiz_err_m']:.1f} |"
        )

    return "\n".join(lines)
