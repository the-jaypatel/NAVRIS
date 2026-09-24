import json
import numpy as np
import pandas as pd

exp_ids = ["Baseline", "E1_A", "E1_B", "E1_C"]
base_dir = "data/processed/phase2_3b/experiments"

summary_rows = []

for exp in exp_ids:
    diag_path = f"{base_dir}/{exp}/{exp}_diagnostics.json"
    with open(diag_path, "r") as f:
        data = json.load(f)

    metrics = data["metrics"]
    early = data["early_updates"]

    # Fix 0 (t=156s)
    f0 = early[0] if len(early) > 0 else {}
    # Fix 1 (t=165s)
    f1 = early[1] if len(early) > 1 else {}
    # Fix 2 (t=174s)
    f2 = early[2] if len(early) > 2 else {}

    v0 = np.array(data["v0_info"]["v0"])
    v0_norm = np.linalg.norm(v0)
    # Ground truth reference velocity at t=156s: speed=12.031, heading=319.715 deg
    v_ref_0 = np.array([12.031111 * np.sin(np.radians(319.715)), 12.031111 * np.cos(np.radians(319.715)), 0.0])
    v_err_proxy = np.linalg.norm(v0 - v_ref_0)

    summary_rows.append({
        "Experiment": exp,
        "v0_vector": [round(x, 2) for x in v0],
        "v0_norm": round(v0_norm, 2),
        "sigma_v": data["sigma_v"],
        "v_err_proxy_mps": round(v_err_proxy, 2),
        "f1_res_norm_m": round(f1.get("residual_norm", 0.0), 2),
        "f1_res_enu_m": [round(x, 2) for x in f1.get("residual_enu", [])],
        "f1_nis": round(f1.get("nis", 0.0), 2),
        "f1_d_pitch_deg": round(f1.get("delta_pitch_deg", 0.0), 2),
        "f1_d_v_norm_mps": round(np.linalg.norm(f1.get("delta_v", [0, 0, 0])), 2),
        "f1_d_v_enu_mps": [round(x, 2) for x in f1.get("delta_v", [])],
        "f2_nis": round(f2.get("nis", 0.0), 2),
        "f2_accepted": f2.get("accepted", False),
        "acceptance_rate_pct": round(metrics["acceptance_rate_pct"], 2),
        "time_to_50m_s": round(metrics["time_to_50m_s"], 1),
        "time_to_100m_s": round(metrics["time_to_100m_s"], 1),
        "first_60s_rmse_m": round(metrics["first_60s_rmse_m"], 2),
        "full_run_rmse_m": round(metrics["horiz_rmse_m"], 2),
        "final_error_m": round(metrics["final_horiz_err_m"], 2),
    })

print("=== DETAILED COMPARISON TABLE ===")
df_comp = pd.DataFrame(summary_rows)
print(df_comp.to_string(index=False))

print("\n=== FIX 1 (t=165s) DETAILED INVENTORIES ===")
for exp in exp_ids:
    diag_path = f"{base_dir}/{exp}/{exp}_diagnostics.json"
    with open(diag_path, "r") as f:
        data = json.load(f)
    f1 = data["early_updates"][1]
    print(f"\n--- {exp} Fix 1 (t=165.0s) ---")
    print(f"  Residual ENU:       {np.array(f1['residual_enu']).round(2)} m (Norm: {f1['residual_norm']:.2f} m)")
    print(f"  NIS:                {f1['nis']:.2f} (Accepted: {f1['accepted']})")
    print(f"  v_pre -> v_post:    {np.array(f1['v_pre']).round(2)} -> {np.array(f1['v_post']).round(2)} (dv: {np.array(f1['delta_v']).round(2)} m/s)")
    print(f"  pitch_pre -> post:  {f1['pitch_pre_deg']:.2f}° -> {f1['pitch_post_deg']:.2f}° (dpitch: {f1['delta_pitch_deg']:+.2f}°)")
    print(f"  roll_pre -> post:   {f1['roll_pre_deg']:.2f}° -> {f1['roll_post_deg']:.2f}° (droll: {f1['delta_roll_deg']:+.2f}°)")
    print(f"  yaw_pre -> post:    {f1['yaw_pre_deg']:.2f}° -> {f1['yaw_post_deg']:.2f}° (dyaw: {f1['delta_yaw_deg']:+.2f}°)")
    print(f"  Pos 3D Std (prior): {f1['pos_std_3d']:.2f} m")
    print(f"  Vel 3D Std (prior): {f1['vel_std_3d']:.2f} m/s")
    print(f"  Att 3D Std (prior): {f1['att_std_3d_deg']:.2f}°")
    print(f"  K_pos diag:         {np.array(f1['K_pos_diag']).round(3)}")
    print(f"  K_vel diag:         {np.array(f1['K_vel_diag']).round(3)}")
    print(f"  K_att:              {np.array(f1['K_att']).round(4)}")

print("\n=== FIX 2 (t=174s) DETAILED INVENTORIES ===")
for exp in exp_ids:
    diag_path = f"{base_dir}/{exp}/{exp}_diagnostics.json"
    with open(diag_path, "r") as f:
        data = json.load(f)
    f2 = data["early_updates"][2]
    print(f"\n--- {exp} Fix 2 (t=174.0s) ---")
    print(f"  Residual ENU:       {np.array(f2['residual_enu']).round(2)} m (Norm: {f2['residual_norm']:.2f} m)")
    print(f"  NIS:                {f2['nis']:.2f} (Accepted: {f2['accepted']})")
    print(f"  v_pre -> v_post:    {np.array(f2['v_pre']).round(2)} -> {np.array(f2['v_post']).round(2)} (dv: {np.array(f2['delta_v']).round(2)} m/s)")
    print(f"  pitch_pre -> post:  {f2['pitch_pre_deg']:.2f}° -> {f2['pitch_post_deg']:.2f}° (dpitch: {f2['delta_pitch_deg']:+.2f}°)")
    print(f"  Horiz Err vs Ref:   {f2['horiz_err_ref']:.2f} m")
