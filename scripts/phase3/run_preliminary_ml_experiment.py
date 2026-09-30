"""
NAVRIS Phase 3.0: Preliminary Machine Learning Experiment Runner.

Executes the auditable supervised ML pipeline to predict local navigation velocity residuals:
delta_v^n = v_reference^n - v_NAVRIS^n
using strictly causal IMU kinematics and classical NAVRIS filter states.

Saves experiment outputs and evaluations under data/processed/phase3/.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd


from navris.ml.dataset import RecordingSplit, load_recording_ml_data
from navris.ml.features import CausalFeatureExtractor
from navris.ml.model import VelocityResidualXGBoost
from navris.ml.evaluate import evaluate_residual_predictions, format_metrics_table
from navris.ml.leakage_audit import run_leakage_audit


def run_experiment(
    sync_dir: str = "data/processed/synchronized",
    gate_dir: str = "data/processed/phase2_3b/gate2_3b",
    out_dir: str = "data/processed/phase3",
    train_recordings: List[str] = None,
    val_recordings: List[str] = None,
    test_recordings: List[str] = None,
) -> Dict[str, Any]:
    """
    Runs the complete Phase 3.0 preliminary ML experiment.
    """
    if train_recordings is None:
        train_recordings = ["S1", "S2", "S4"]
    if val_recordings is None:
        val_recordings = ["S3A", "VTA1A"]
    if test_recordings is None:
        test_recordings = ["Y1", "VTA2"]

    split = RecordingSplit(
        train_recordings=train_recordings,
        val_recordings=val_recordings,
        test_recordings=test_recordings,
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("NAVRIS PHASE 3.0: PRELIMINARY MACHINE LEARNING EXPERIMENT")
    print("=" * 80)
    print(f"Train Recordings:    {split.train_recordings}")
    print(f"Val Recordings:      {split.val_recordings}")
    print(f"Held-out Test:       {split.test_recordings}")
    print("-" * 80)

    extractor = CausalFeatureExtractor(window_size=10)

    def load_split_data(recs: List[str]):
        x_list, y_list, m_list = [], [], []
        per_rec_data = {}
        for rec in recs:
            df_rec = load_recording_ml_data(rec, sync_dir=sync_dir, gate_dir=gate_dir)
            X_rec, y_rec, m_rec = extractor.extract_from_recording(df_rec)
            x_list.append(X_rec)
            y_list.append(y_rec)
            m_list.append(m_rec)
            per_rec_data[rec] = (X_rec, y_rec, m_rec)
            print(f"Loaded {rec:6s}: {len(X_rec):6d} valid causal samples (window=10, 1.0s @ 10Hz)")

        X_cat = pd.concat(x_list, ignore_index=True)
        y_cat = pd.concat(y_list, ignore_index=True)
        m_cat = pd.concat(m_list, ignore_index=True)
        return X_cat, y_cat, m_cat, per_rec_data

    print("\n--- 1. LOADING AND EXTRACTING DATA ---")
    print("Train Split:")
    X_train, y_train, m_train, train_by_rec = load_split_data(split.train_recordings)
    print(f"Total Train Samples: {len(X_train)}")

    print("\nValidation Split:")
    X_val, y_val, m_val, val_by_rec = load_split_data(split.val_recordings)
    print(f"Total Val Samples: {len(X_val)}")

    print("\nHeld-out Test Split:")
    X_test, y_test, m_test, test_by_rec = load_split_data(split.test_recordings)
    print(f"Total Held-out Test Samples: {len(X_test)}")

    print("\n--- 2. AUTOMATED LEAKAGE AUDIT ---")
    audit_pre = run_leakage_audit(
        split=split,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        scaler=None,
    )
    for check_name, res in audit_pre.items():
        if isinstance(res, dict):
            status = "PASS" if res.get("passed", False) else "FAIL"
            print(f"  {check_name}: [{status}]")
        else:
            print(f"  {check_name}: {res}")

    if not audit_pre["overall_leakage_audit_passed"]:
        raise RuntimeError("Pre-training leakage audit failed! Halting execution.")

    print("\n--- 3. TRAINING VELOCITY RESIDUAL XGBOOST REGRESSORS ---")
    model = VelocityResidualXGBoost(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    print("XGBoost regressors fitted successfully for East, North, and Up axes.")

    # Post-training scaler audit
    audit_post = run_leakage_audit(
        split=split,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        scaler=model.scaler,
    )
    status_scaler = "PASS" if audit_post["check_scaler_fit_strictly_on_train_only"]["passed"] else "FAIL"
    print(f"  check_scaler_fit_strictly_on_train_only: [{status_scaler}]")

    print("\n--- 4. FEATURE IMPORTANCES ---")
    df_imp = model.get_feature_importances()
    df_imp.to_csv(out_path / "feature_importances.csv", index=False)
    print(df_imp.head(10).to_string(index=False))

    print("\n--- 5. EVALUATION AND BASELINE COMPARISONS ---")

    # A. Train Partition Evaluation
    pred_train = model.predict(X_train)
    eval_train = evaluate_residual_predictions(m_train, y_train, pred_train)
    print("\n================ TRAIN METRICS (Aggregate) ================")
    print(format_metrics_table(eval_train))

    # B. Validation Partition Evaluation
    pred_val = model.predict(X_val)
    eval_val = evaluate_residual_predictions(m_val, y_val, pred_val)
    print("\n================ VALIDATION METRICS (Aggregate) ================")
    print(format_metrics_table(eval_val))

    # Per-recording validation metrics
    per_rec_val_metrics = {}
    for rec, (X_r, y_r, m_r) in val_by_rec.items():
        pred_r = model.predict(X_r)
        eval_r = evaluate_residual_predictions(m_r, y_r, pred_r)
        per_rec_val_metrics[rec] = eval_r
        print(f"\n--- Validation Recording: {rec} ---")
        print(format_metrics_table(eval_r))

    # C. Held-Out Test Partition Evaluation
    pred_test = model.predict(X_test)
    eval_test = evaluate_residual_predictions(m_test, y_test, pred_test)
    print("\n================ HELD-OUT TEST METRICS (Aggregate) ================")
    print(format_metrics_table(eval_test))

    # Per-recording held-out test metrics
    per_rec_test_metrics = {}
    for rec, (X_r, y_r, m_r) in test_by_rec.items():
        pred_r = model.predict(X_r)
        eval_r = evaluate_residual_predictions(m_r, y_r, pred_r)
        per_rec_test_metrics[rec] = eval_r
        print(f"\n--- Held-Out Test Recording: {rec} ---")
        print(format_metrics_table(eval_r))

    # Compile comprehensive summary dictionary
    summary_bundle = {
        "dataset": {
            "total_recordings": len(split.all_recordings()),
            "train_recordings": split.train_recordings,
            "val_recordings": split.val_recordings,
            "test_recordings": split.test_recordings,
            "train_samples": int(len(X_train)),
            "val_samples": int(len(X_val)),
            "test_samples": int(len(X_test)),
            "causal_window_samples": 10,
            "sampling_rate_hz": 10.0,
        },
        "model_config": {
            "model_type": "VelocityResidualXGBoost",
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "learning_rate": model.learning_rate,
            "subsample": model.subsample,
            "colsample_bytree": model.colsample_bytree,
            "random_state": model.random_state,
        },
        "leakage_audit": audit_post,
        "train_metrics": eval_train,
        "val_metrics": {
            "aggregate": eval_val,
            "by_recording": per_rec_val_metrics,
        },
        "test_metrics": {
            "aggregate": eval_test,
            "by_recording": per_rec_test_metrics,
        },
    }

    # Save summary JSON
    with open(out_path / "preliminary_ml_summary.json", "w") as f:
        json.dump(summary_bundle, f, indent=2)

    # Build flat CSV comparison table across all evaluated recordings
    flat_rows = []
    all_recs_eval = [("TRAIN_ALL", eval_train)]
    for r, ev in per_rec_val_metrics.items():
        all_recs_eval.append((f"VAL_{r}", ev))
    all_recs_eval.append(("VAL_ALL", eval_val))
    for r, ev in per_rec_test_metrics.items():
        all_recs_eval.append((f"TEST_{r}", ev))
    all_recs_eval.append(("TEST_ALL", eval_test))

    for tag, ev in all_recs_eval:
        ml = ev["ml_target_metrics"]
        nav = ev["navigation_metrics"]
        p_d = nav.get("open_loop_position_drift", {})
        row = {
            "partition_or_rec": tag,
            "ml_target_mae_3d_mps": ml["mae_3d_mps"],
            "ml_target_rmse_horiz_mps": ml["horizontal_rmse_mps"],
            "ml_target_rmse_3d_mps": ml["rmse_3d_mps"],
            "ml_bias_east_mps": ml["east"]["bias"],
            "ml_bias_north_mps": ml["north"]["bias"],
            "ml_bias_up_mps": ml["up"]["bias"],
            "nav_base_A_vel_rmse_mps": nav["baseline_A"]["vel_3d_rmse_mps"],
            "nav_base_A_horiz_vel_rmse_mps": nav["baseline_A"]["horiz_vel_rmse_mps"],
            "nav_base_C_ml_vel_rmse_mps": nav["baseline_C_ml"]["vel_3d_rmse_mps"],
            "nav_base_C_ml_horiz_vel_rmse_mps": nav["baseline_C_ml"]["horiz_vel_rmse_mps"],
            "nav_base_B_vel_rmse_mps": nav["baseline_B"]["vel_3d_rmse_mps"] if nav["baseline_B"] else None,
            "pos_base_A_horiz_pos_rmse_m": p_d.get("baseline_A_horiz_pos_rmse_m"),
            "pos_base_A_final_horiz_err_m": p_d.get("baseline_A_final_horiz_err_m"),
            "pos_base_C_ml_horiz_pos_rmse_m": p_d.get("baseline_C_horiz_pos_rmse_m"),
            "pos_base_C_ml_final_horiz_err_m": p_d.get("baseline_C_final_horiz_err_m"),

        }
        flat_rows.append(row)

    df_flat = pd.DataFrame(flat_rows)
    df_flat.to_csv(out_path / "preliminary_ml_summary.csv", index=False)

    print("\n" + "=" * 80)
    print("FLAT COMPARISON SUMMARY TABLE:")
    print(df_flat.to_string(index=False))
    print("=" * 80)
    print(f"Artifacts successfully saved to {out_path.resolve()}/")

    return summary_bundle


if __name__ == "__main__":
    run_experiment()
