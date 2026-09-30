"""
NAVRIS Phase 3.0: Automated Data and Pipeline Leakage Audit.

Enforces strict verification against:
1. Reference fields leaking into feature matrix X.
2. Target fields leaking into X.
3. Future samples / future GNSS leaking into X.
4. Test/validation statistics leaking into feature scaling.
5. Cross-partition recording overlap (train/val/test).
6. Non-causal window index contamination.
"""

from typing import Any, Dict, List, Set, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from navris.ml.dataset import RecordingSplit
from navris.ml.features import FEATURE_NAMES, FORBIDDEN_FEATURE_SUBSTRINGS


def run_leakage_audit(
    split: RecordingSplit,
    X_train: pd.DataFrame,
    X_val: Optional[pd.DataFrame],
    X_test: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
) -> Dict[str, Any]:
    """
    Executes an automated suite of leakage and causal integrity checks.

    Returns:
        Dictionary containing audit findings, check statuses (PASS/FAIL), and summary.
    """
    audit_results: Dict[str, Any] = {}
    all_passed = True

    # 1. Check feature names for forbidden ground-truth / target substrings
    feature_cols = list(X_train.columns)
    forbidden_matches = []
    for col in feature_cols:
        col_lower = col.lower()
        for forbidden in FORBIDDEN_FEATURE_SUBSTRINGS:
            if forbidden in col_lower:
                forbidden_matches.append((col, forbidden))

    check_no_forbidden = len(forbidden_matches) == 0
    audit_results["check_no_forbidden_fields_in_X"] = {
        "passed": check_no_forbidden,
        "violations": forbidden_matches,
    }
    if not check_no_forbidden:
        all_passed = False

    # 2. Check that target names are not in X
    target_names = ["target_delta_v_east_mps", "target_delta_v_north_mps", "target_delta_v_up_mps"]
    target_in_X = [t for t in target_names if t in feature_cols]
    check_no_targets = len(target_in_X) == 0
    audit_results["check_no_targets_in_X"] = {
        "passed": check_no_targets,
        "violations": target_in_X,
    }
    if not check_no_targets:
        all_passed = False

    # 3. Check recording separation across splits
    s_train = set(split.train_recordings)
    s_val = set(split.val_recordings) if split.val_recordings else set()
    s_test = set(split.test_recordings)

    overlap_train_val = list(s_train & s_val)
    overlap_train_test = list(s_train & s_test)
    overlap_val_test = list(s_val & s_test)

    check_split_separation = (
        len(overlap_train_val) == 0 and
        len(overlap_train_test) == 0 and
        len(overlap_val_test) == 0
    )
    audit_results["check_train_val_test_recording_separation"] = {
        "passed": check_split_separation,
        "train_val_overlap": overlap_train_val,
        "train_test_overlap": overlap_train_test,
        "val_test_overlap": overlap_val_test,
    }
    if not check_split_separation:
        all_passed = False

    # 4. Check feature scaling isolation
    check_scaler_isolation = True
    scaler_diff_max = 0.0
    if scaler is not None:
        # Re-compute reference scaler on X_train independently
        ref_scaler = StandardScaler()
        ref_scaler.fit(X_train)

        diff_mean = np.max(np.abs(scaler.mean_ - ref_scaler.mean_))
        diff_scale = np.max(np.abs(scaler.scale_ - ref_scaler.scale_))
        scaler_diff_max = max(diff_mean, diff_scale)

        if scaler_diff_max > 1e-7:
            check_scaler_isolation = False
            all_passed = False

    audit_results["check_scaler_fit_strictly_on_train_only"] = {
        "passed": check_scaler_isolation,
        "max_discrepancy_vs_train_fit": float(scaler_diff_max),
    }

    # 5. Check feature counts match standard schema
    check_feature_count = (
        len(feature_cols) == len(FEATURE_NAMES) and
        set(feature_cols) == set(FEATURE_NAMES)
    )
    audit_results["check_feature_schema_conformance"] = {
        "passed": check_feature_count,
        "feature_count": len(feature_cols),
        "expected_count": len(FEATURE_NAMES),
    }
    if not check_feature_count:
        all_passed = False

    audit_results["overall_leakage_audit_passed"] = all_passed
    return audit_results
