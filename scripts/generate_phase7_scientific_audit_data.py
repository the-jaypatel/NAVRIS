"""
Generates Phase 8 scientific audit data artifacts for Gate 2.3A:
- data/processed/phase2_3b/gate2_3a/phase7_scientific_audit.csv
- data/processed/phase2_3b/gate2_3a/phase7_scientific_audit.json
"""

import os
import json
import numpy as np
import pandas as pd

def main():
    out_dir = "data/processed/phase2_3b/gate2_3a"
    os.makedirs(out_dir, exist_ok=True)

    df_sum = pd.read_csv(os.path.join(out_dir, "gate2_3a_summary.csv"))
    df_up = pd.read_csv(os.path.join(out_dir, "zupt_updates.csv"))
    df_ev = pd.read_csv(os.path.join(out_dir, "zupt_events.csv"))
    df_st = pd.read_csv(os.path.join(out_dir, "stationary_audit.csv"))

    recordings = ['S1', 'S2', 'S3A', 'S4', 'M', 'Y1', 'VTA1A', 'VTA2']

    audit_records = []
    audit_json_dict = {
        'gate': 'Gate 2.3A',
        'phase': 'Phase 8 Scientific Audit',
        'frozen_parameters': {
            'sigma_zupt_mps': 0.05,
            'r_zupt_mps2': 0.0025,
            'window_W': 8,
            'dwell_D': 5,
            'accel_var_thresh_mps2_sq': 0.005,
            'gyro_var_thresh_radps_sq': 1e-4,
            'gravity_dev_thresh_mps2': 0.40,
            'chi2_threshold_zupt': 16.27
        },
        'recordings': {}
    }

    for rec in recordings:
        sub_a = df_sum[(df_sum['recording_id'] == rec) & (df_sum['configuration'] == 'A_CONTROL_BASELINE')]
        sub_b = df_sum[(df_sum['recording_id'] == rec) & (df_sum['configuration'] == 'B_EXPERIMENT_ZUPT')]

        if rec == 'M' or len(sub_a) == 0 or sub_a.iloc[0]['status'] == 'UNOBSERVABLE':
            rec_row = {
                'recording_id': rec,
                'status': 'UNOBSERVABLE',
                'classification': 'Unobservable',
                'primary_pattern': 'N/A',
                'temporal_causality': 'N/A',
                'duration_s': float(sub_a.iloc[0]['duration_s']) if len(sub_a) > 0 else np.nan,
                'h_rmse_A_m': np.nan,
                'h_rmse_B_m': np.nan,
                'h_rmse_pct_change': np.nan,
                'final_err_A_m': np.nan,
                'final_err_B_m': np.nan,
                'gnss_acc_A_pct': np.nan,
                'gnss_acc_B_pct': np.nan,
                'zupt_events_count': 0,
                'zupt_total_duration_s': 0.0,
                'zupt_attempted': 0,
                'zupt_accepted': 0,
                'zupt_rejected': 0,
                'zupt_acceptance_pct': np.nan,
                'zupt_nis_all_median': np.nan,
                'zupt_nis_all_mean': np.nan,
                'zupt_nis_all_p95': np.nan,
                'zupt_nis_all_max': np.nan,
                'zupt_nis_acc_median': np.nan,
                'zupt_nis_acc_mean': np.nan,
                'zupt_nis_acc_p95': np.nan,
                'zupt_nis_acc_max': np.nan,
                'zupt_nis_rej_median': np.nan,
                'zupt_nis_rej_mean': np.nan,
                'zupt_nis_rej_p95': np.nan,
                'zupt_nis_rej_max': np.nan,
                'v_prior_acc_median_mps': np.nan,
                'v_prior_acc_max_mps': np.nan,
                'v_post_acc_median_mps': np.nan,
                'v_post_acc_max_mps': np.nan,
                'v_prior_rej_median_mps': np.nan,
                'v_prior_rej_max_mps': np.nan,
            }
            audit_records.append(rec_row)
            audit_json_dict['recordings'][rec] = rec_row
            continue

        row_a = sub_a.iloc[0]
        row_b = sub_b.iloc[0]

        h_rmse_a = float(row_a['horiz_rmse_m'])
        h_rmse_b = float(row_b['horiz_rmse_m'])
        h_rmse_pct = ((h_rmse_b - h_rmse_a) / max(1e-6, h_rmse_a)) * 100.0

        final_a = float(row_a['final_horiz_m'])
        final_b = float(row_b['final_horiz_m'])

        gnss_acc_a = float(row_a['gnss_acceptance_pct'])
        gnss_acc_b = float(row_b['gnss_acceptance_pct'])

        rec_up = df_up[df_up['recording_id'] == rec]
        acc_up = rec_up[rec_up['accepted'] == True]
        rej_up = rec_up[rec_up['accepted'] == False]

        n_att = len(rec_up)
        n_acc = len(acc_up)
        n_rej = len(rej_up)
        acc_pct = (n_acc / max(1, n_att)) * 100.0

        # NIS stats
        nis_all = rec_up['nis'].values if n_att > 0 else np.array([])
        nis_acc = acc_up['nis'].values if n_acc > 0 else np.array([])
        nis_rej = rej_up['nis'].values if n_rej > 0 else np.array([])

        v_prior_acc = acc_up['v_norm_prior_mps'].values if n_acc > 0 else np.array([])
        v_post_acc = acc_up['v_norm_post_mps'].values if n_acc > 0 else np.array([])
        v_prior_rej = rej_up['v_norm_prior_mps'].values if n_rej > 0 else np.array([])

        rec_ev = df_ev[df_ev['recording_id'] == rec]
        ev_count = len(rec_ev)
        ev_dur = float(rec_ev['duration_s'].sum()) if ev_count > 0 else 0.0

        # Pattern classification
        # A: Detector misses stationary periods
        # B: Detector detects stationary-looking periods but filter velocity is inconsistent (gated out)
        # C: Detector is too permissive
        # D: Measurement covariance appears inconsistent with observed residuals
        # E: Another identifiable cause (e.g. Covariance starvation / discrete impulse coupling)
        if rec == 'VTA2':
            pattern = 'Nominal / Consistent Operation'
            classification = 'Positive Evidence'
            causality = 'Precedes sustained improvement'
        elif rec == 'S2':
            pattern = 'Pattern B (Filter state divergent / complete ZUPT lockout)'
            classification = 'Negative Evidence'
            causality = 'No operational effect (filter locked out)'
        elif rec == 'S3A':
            pattern = 'Pattern E (Covariance starvation during 327s standstill causing premature GNSS rejection)'
            classification = 'Negative Evidence'
            causality = 'Precedes degradation (covariance starvation precipitates GNSS lockout)'
        elif rec == 'S1':
            pattern = 'Pattern B (Late stops divergent) / Pattern E (Early discrete jump vs ultra-long bounding)'
            classification = 'Mixed Evidence'
            causality = 'Precedes long-term bounding / follows early discrete transient'
        elif rec == 'S4':
            pattern = 'Pattern B (Filter divergence post-GNSS loss at t=339s)'
            classification = 'Mixed Evidence'
            causality = 'Precedes intermediate bounding / follows early GNSS loss'
        elif rec == 'Y1':
            pattern = 'Pattern B (Dominated by heading divergence, but 16s local arrest at t=1791s)'
            classification = 'Mixed Evidence'
            causality = 'Precedes temporary local arrest / follows heading divergence'
        elif rec == 'VTA1A':
            pattern = 'Negative Control (Zero stops post-departure; crawl at 839s rejected; zero moving false positives)'
            classification = 'Negative Control'
            causality = 'No operational effect on cruise'
        else:
            pattern = 'Unspecified'
            classification = 'Mixed Evidence'
            causality = 'Unspecified'

        rec_row = {
            'recording_id': rec,
            'status': 'SUCCESS',
            'classification': classification,
            'primary_pattern': pattern,
            'temporal_causality': causality,
            'duration_s': float(row_a['duration_s']),
            'h_rmse_A_m': round(h_rmse_a, 2),
            'h_rmse_B_m': round(h_rmse_b, 2),
            'h_rmse_pct_change': round(h_rmse_pct, 2),
            'final_err_A_m': round(final_a, 2),
            'final_err_B_m': round(final_b, 2),
            'gnss_acc_A_pct': round(gnss_acc_a, 2),
            'gnss_acc_B_pct': round(gnss_acc_b, 2),
            'zupt_events_count': ev_count,
            'zupt_total_duration_s': round(ev_dur, 2),
            'zupt_attempted': n_att,
            'zupt_accepted': n_acc,
            'zupt_rejected': n_rej,
            'zupt_acceptance_pct': round(acc_pct, 2),
            'zupt_nis_all_median': round(float(np.median(nis_all)), 4) if len(nis_all) > 0 else np.nan,
            'zupt_nis_all_mean': round(float(np.mean(nis_all)), 4) if len(nis_all) > 0 else np.nan,
            'zupt_nis_all_p95': round(float(np.percentile(nis_all, 95)), 4) if len(nis_all) > 0 else np.nan,
            'zupt_nis_all_max': round(float(np.max(nis_all)), 4) if len(nis_all) > 0 else np.nan,
            'zupt_nis_acc_median': round(float(np.median(nis_acc)), 4) if len(nis_acc) > 0 else np.nan,
            'zupt_nis_acc_mean': round(float(np.mean(nis_acc)), 4) if len(nis_acc) > 0 else np.nan,
            'zupt_nis_acc_p95': round(float(np.percentile(nis_acc, 95)), 4) if len(nis_acc) > 0 else np.nan,
            'zupt_nis_acc_max': round(float(np.max(nis_acc)), 4) if len(nis_acc) > 0 else np.nan,
            'zupt_nis_rej_median': round(float(np.median(nis_rej)), 4) if len(nis_rej) > 0 else np.nan,
            'zupt_nis_rej_mean': round(float(np.mean(nis_rej)), 4) if len(nis_rej) > 0 else np.nan,
            'zupt_nis_rej_p95': round(float(np.percentile(nis_rej, 95)), 4) if len(nis_rej) > 0 else np.nan,
            'zupt_nis_rej_max': round(float(np.max(nis_rej)), 4) if len(nis_rej) > 0 else np.nan,
            'v_prior_acc_median_mps': round(float(np.median(v_prior_acc)), 4) if len(v_prior_acc) > 0 else np.nan,
            'v_prior_acc_max_mps': round(float(np.max(v_prior_acc)), 4) if len(v_prior_acc) > 0 else np.nan,
            'v_post_acc_median_mps': round(float(np.median(v_post_acc)), 4) if len(v_post_acc) > 0 else np.nan,
            'v_post_acc_max_mps': round(float(np.max(v_post_acc)), 4) if len(v_post_acc) > 0 else np.nan,
            'v_prior_rej_median_mps': round(float(np.median(v_prior_rej)), 4) if len(v_prior_rej) > 0 else np.nan,
            'v_prior_rej_max_mps': round(float(np.max(v_prior_rej)), 4) if len(v_prior_rej) > 0 else np.nan,
        }
        audit_records.append(rec_row)
        audit_json_dict['recordings'][rec] = rec_row

    # Overall Summary Stats
    audit_json_dict['aggregate_findings'] = {
        'total_recordings': 8,
        'observable_recordings': 7,
        'unobservable_recordings': 1,
        'evidence_breakdown': {
            'positive_evidence': ['VTA2'],
            'mixed_evidence': ['S1', 'S4', 'Y1'],
            'negative_evidence': ['S2', 'S3A'],
            'negative_control': ['VTA1A'],
            'unobservable': ['M']
        },
        'critical_question_verdict': 'NEGATIVE',
        'critical_question_statement': (
            'The current real-data evidence is NOT sufficient to claim that the frozen causal ZUPT '
            'implementation provides generally reliable navigation improvement across IO-VNBD. '
            'Rather, the evidence supports that the ZUPT implementation is technically valid, mathematically sound, '
            'strictly causal, and provides substantial correction in suitable stationary segments on routes with observable '
            'heading and nominal conditioning (as demonstrated in VTA2), but exhibits severe failure modes in unobservable heading '
            'regimes (S2 filter lockout), long standstills (S3A covariance starvation), and unconstrained strapdown drift.'
        )
    }

    df_out = pd.DataFrame(audit_records)
    csv_path = os.path.join(out_dir, "phase7_scientific_audit.csv")
    json_path = os.path.join(out_dir, "phase7_scientific_audit.json")

    df_out.to_csv(csv_path, index=False)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(audit_json_dict, f, indent=2)

    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")

if __name__ == '__main__':
    main()
