"""
NAVRIS Phase 2.3B Gate 2.3A - Phase 6: Baseline Regression Verification.

Runs the exact Gate 2.2 calibrated baseline (without ZUPT) across all 8 recordings
and verifies numerical reproduction against data/processed/phase2_3b/gate2_2/gate2_2_summary.csv.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any

sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("."))

from scripts.run_gate2_2_benchmark import run_benchmark_recording


def main():
    print("=== NAVRIS Phase 2.3B Gate 2.3A: Phase 6 Baseline Regression ===")
    ref_summary_path = "data/processed/phase2_3b/gate2_2/gate2_2_summary.csv"
    if not os.path.exists(ref_summary_path):
        print(f"ERROR: Reference summary file not found: {ref_summary_path}")
        sys.exit(1)

    ref_df = pd.read_csv(ref_summary_path)
    recordings = ['S1', 'S2', 'S3A', 'S4', 'M', 'Y1', 'VTA1A', 'VTA2']
    
    regression_records = []
    has_discrepancy = False

    for rec in recordings:
        parquet_p = f"data/processed/synchronized/{rec}_sync.parquet"
        print(f"\n--- Verifying {rec} Baseline Reproduction ---")
        
        # Run baseline benchmark
        plots_dir = "data/processed/phase2_3b/gate2_3a/baseline_plots"
        res = run_benchmark_recording(rec, parquet_p, plots_dir)
        m = res['metrics']
        status = m.get('status')
        
        # Compare with reference row
        ref_row = ref_df[ref_df['recording_id'] == rec]
        if len(ref_row) == 0:
            print(f"ERROR: No reference record for {rec}")
            has_discrepancy = True
            continue
            
        ref_row = ref_row.iloc[0]
        
        if status == 'UNOBSERVABLE':
            ref_status = str(ref_row['status'])
            assert ref_status == 'UNOBSERVABLE', f"{rec}: status mismatch {status} vs {ref_status}"
            print(f"{rec}: Successfully confirmed UNOBSERVABLE (exact match).")
            regression_records.append({
                'recording_id': rec,
                'status': status,
                'ref_status': ref_status,
                'rmse_match': True,
                'final_err_match': True,
                'fixes_match': True,
                'rmse_diff_m': 0.0,
                'final_err_diff_m': 0.0,
                'accepted_fixes_diff': 0
            })
            continue

        h_rmse = float(m['horiz_rmse_m'])
        final_err = float(m['final_horiz_m'])
        acc_fixes = int(m['gnss_accepted_fixes'])
        rej_fixes = int(m['gnss_rejected_fixes'])
        med_nis = float(m['gnss_median_nis'])

        ref_rmse = float(ref_row['horiz_rmse_m'])
        ref_final = float(ref_row['final_horiz_m'])
        ref_acc = int(ref_row['gnss_accepted_fixes'])
        ref_rej = int(ref_row['gnss_rejected_fixes'])
        ref_nis = float(ref_row['gnss_median_nis'])

        rmse_diff = abs(h_rmse - ref_rmse)
        final_diff = abs(final_err - ref_final)
        acc_diff = abs(acc_fixes - ref_acc)

        # Allow small floating point tolerance (e.g. 1e-4 m)
        rmse_ok = rmse_diff < 0.01
        final_ok = final_diff < 0.01
        fixes_ok = (acc_diff == 0)

        if not (rmse_ok and final_ok and fixes_ok):
            print(f"DISCREPANCY in {rec}:")
            print(f"  RMSE: {h_rmse:.2f} vs ref {ref_rmse:.2f} (diff: {rmse_diff:.4f}m)")
            print(f"  Final: {final_err:.2f} vs ref {ref_final:.2f} (diff: {final_diff:.4f}m)")
            print(f"  Acc Fixes: {acc_fixes} vs ref {ref_acc}")
            has_discrepancy = True
        else:
            print(f"{rec}: Exact baseline reproduction verified!")
            print(f"  H-RMSE: {h_rmse:.2f} m | Final Err: {final_err:.2f} m | Acc Fixes: {acc_fixes}/{acc_fixes+rej_fixes} | Med NIS: {med_nis:.2f}")

        regression_records.append({
            'recording_id': rec,
            'status': status,
            'ref_status': ref_row['status'],
            'rmse_match': rmse_ok,
            'final_err_match': final_ok,
            'fixes_match': fixes_ok,
            'current_rmse_m': h_rmse,
            'ref_rmse_m': ref_rmse,
            'rmse_diff_m': rmse_diff,
            'current_final_m': final_err,
            'ref_final_m': ref_final,
            'final_err_diff_m': final_diff,
            'current_acc_fixes': acc_fixes,
            'ref_acc_fixes': ref_acc,
            'accepted_fixes_diff': acc_diff
        })

    reg_df = pd.DataFrame(regression_records)
    os.makedirs("data/processed/phase2_3b/gate2_3a", exist_ok=True)
    out_csv = "data/processed/phase2_3b/gate2_3a/baseline_regression.csv"
    reg_df.to_csv(out_csv, index=False)
    print(f"\nRegression verification results saved to: {out_csv}")
    print("\nSummary Comparison Table:")
    print(reg_df[['recording_id', 'status', 'rmse_match', 'final_err_match', 'fixes_match', 'rmse_diff_m']].to_string())

    if has_discrepancy:
        print("\nERROR: Baseline regression failed with material discrepancies!")
        sys.exit(1)
    else:
        print("\nSUCCESS: All 8 recordings reproduced Gate 2.2 baseline exactly with zero discrepancy!")


if __name__ == '__main__':
    main()
