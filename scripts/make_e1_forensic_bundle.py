import os
import shutil
import pandas as pd
import numpy as np

dest_dir = "claude_e1_forensic_review"
os.makedirs(dest_dir, exist_ok=True)

print("=" * 80)
print("ASSEMBLING CLAUDE E1 FORENSIC REVIEW PACKAGE")
print("=" * 80)

# 1. E1 report
src_rep = "docs/phase2_3b_e1_initialization_experiment_report.md"
shutil.copy2(src_rep, os.path.join(dest_dir, "phase2_3b_e1_initialization_experiment_report.md"))
print("1. Copied phase2_3b_e1_initialization_experiment_report.md")

# 2. E1 comparison summary
src_comp = "data/processed/phase2_3b/experiments/e1_comparison_summary.csv"
shutil.copy2(src_comp, os.path.join(dest_dir, "e1_comparison_summary.csv"))
print("2. Copied e1_comparison_summary.csv")

# 3, 4, 5. Diagnostics JSON files
shutil.copy2("data/processed/phase2_3b/experiments/E1_A/E1_A_diagnostics.json", os.path.join(dest_dir, "E1_A_diagnostics.json"))
shutil.copy2("data/processed/phase2_3b/experiments/E1_B/E1_B_diagnostics.json", os.path.join(dest_dir, "E1_B_diagnostics.json"))
shutil.copy2("data/processed/phase2_3b/experiments/E1_C/E1_C_diagnostics.json", os.path.join(dest_dir, "E1_C_diagnostics.json"))
print("3, 4, 5. Copied E1_A_diagnostics.json, E1_B_diagnostics.json, E1_C_diagnostics.json")

# 6. E1 implementation
shutil.copy2("scripts/run_phase2_3b_e1.py", os.path.join(dest_dir, "run_phase2_3b_e1.py"))
print("6. Copied run_phase2_3b_e1.py")

# 7. E1 tests
shutil.copy2("tests/test_phase2_3b_e1.py", os.path.join(dest_dir, "test_phase2_3b_e1.py"))
print("7. Copied test_phase2_3b_e1.py")

# 8. Initialization implementation
shutil.copy2("src/navris/inertial/init.py", os.path.join(dest_dir, "init.py"))
print("8. Copied init.py")

# 9. ESKF update implementation
shutil.copy2("src/navris/eskf/update.py", os.path.join(dest_dir, "update.py"))
print("9. Copied update.py")

# 10. ESKF core
shutil.copy2("claude_s1_eskf_review/eskf.py", os.path.join(dest_dir, "eskf.py"))
print("10. Copied eskf.py")

# 11. Original S1 result parquet
shutil.copy2("data/processed/phase2_3b/S1_eskf_results.parquet", os.path.join(dest_dir, "S1_eskf_results.parquet"))
print("11. Copied S1_eskf_results.parquet")

# 12. E1-A result parquet
shutil.copy2("data/processed/phase2_3b/experiments/E1_A/E1_A_results.parquet", os.path.join(dest_dir, "E1_A_results.parquet"))
print("12. Copied E1_A_results.parquet")

# 13. E1-C result parquet
shutil.copy2("data/processed/phase2_3b/experiments/E1_C/E1_C_results.parquet", os.path.join(dest_dir, "E1_C_results.parquet"))
print("13. Copied E1_C_results.parquet")

# 14. Relevant synchronized S1 data (t = 140s to 200s)
sync_path = "data/processed/synchronized/S1_sync.parquet"
df_sync = pd.read_parquet(sync_path)
s1_140_200 = df_sync[(df_sync["time_s"] >= 140.0) & (df_sync["time_s"] <= 200.0)].copy().reset_index(drop=True)
s1_csv_path = os.path.join(dest_dir, "S1_140_200_forensic.csv")
s1_140_200.to_csv(s1_csv_path, index=False)
print(f"14. Created S1_140_200_forensic.csv ({len(s1_140_200)} rows, t=[{s1_140_200['time_s'].iloc[0]:.1f}, {s1_140_200['time_s'].iloc[-1]:.1f}])")

# 15. Relevant E1 trajectories (t = 150s to 200s)
# Baseline
df_base = pd.read_parquet("data/processed/phase2_3b/S1_eskf_results.parquet")
base_150_200 = df_base[(df_base["time_s"] >= 150.0) & (df_base["time_s"] <= 200.0)].copy().reset_index(drop=True)
base_csv_path = os.path.join(dest_dir, "Baseline_150_200.csv")
base_150_200.to_csv(base_csv_path, index=False)
print(f"15a. Created Baseline_150_200.csv ({len(base_150_200)} rows, t=[{base_150_200['time_s'].iloc[0]:.1f}, {base_150_200['time_s'].iloc[-1]:.1f}])")

# E1-A
df_e1a = pd.read_parquet("data/processed/phase2_3b/experiments/E1_A/E1_A_results.parquet")
e1a_150_200 = df_e1a[(df_e1a["time_s"] >= 150.0) & (df_e1a["time_s"] <= 200.0)].copy().reset_index(drop=True)
e1a_csv_path = os.path.join(dest_dir, "E1_A_150_200.csv")
e1a_150_200.to_csv(e1a_csv_path, index=False)
print(f"15b. Created E1_A_150_200.csv ({len(e1a_150_200)} rows, t=[{e1a_150_200['time_s'].iloc[0]:.1f}, {e1a_150_200['time_s'].iloc[-1]:.1f}])")

# E1-C
df_e1c = pd.read_parquet("data/processed/phase2_3b/experiments/E1_C/E1_C_results.parquet")
e1c_150_200 = df_e1c[(df_e1c["time_s"] >= 150.0) & (df_e1c["time_s"] <= 200.0)].copy().reset_index(drop=True)
e1c_csv_path = os.path.join(dest_dir, "E1_C_150_200.csv")
e1c_150_200.to_csv(e1c_csv_path, index=False)
print(f"15c. Created E1_C_150_200.csv ({len(e1c_150_200)} rows, t=[{e1c_150_200['time_s'].iloc[0]:.1f}, {e1c_150_200['time_s'].iloc[-1]:.1f}])")

# Inventory
print("\n" + "=" * 80)
print(f"PACKAGE DIRECTORY: {os.path.abspath(dest_dir)}")
print("=" * 80)
files = sorted(os.listdir(dest_dir))
for f in files:
    fp = os.path.join(dest_dir, f)
    size = os.path.getsize(fp)
    if f.endswith(".csv"):
        n_rows = len(pd.read_csv(fp))
        print(f"  {f:<52} {size:>12,} bytes  ({n_rows} rows)")
    else:
        print(f"  {f:<52} {size:>12,} bytes")
