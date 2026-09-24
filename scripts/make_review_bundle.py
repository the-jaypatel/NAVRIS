import os
import shutil

dest_dir = "claude_s1_eskf_review"
os.makedirs(dest_dir, exist_ok=True)

# 1. Copy S1_eskf_results.parquet
shutil.copy2("data/processed/phase2_3b/S1_eskf_results.parquet", os.path.join(dest_dir, "S1_eskf_results.parquet"))
print("1. Copied S1_eskf_results.parquet")

# 2. Copy phase2_3b_gate1_s1_report.md
shutil.copy2("docs/phase2_3b_gate1_s1_report.md", os.path.join(dest_dir, "phase2_3b_gate1_s1_report.md"))
print("2. Copied phase2_3b_gate1_s1_report.md")

# 3. Copy phase2_3_eskf_core.md
shutil.copy2("docs/phase2_3_eskf_core.md", os.path.join(dest_dir, "phase2_3_eskf_core.md"))
print("3. Copied phase2_3_eskf_core.md")

# 4. Copy frames.py
shutil.copy2("src/navris/inertial/frames.py", os.path.join(dest_dir, "frames.py"))
print("4. Copied frames.py")

# 5. Create self-contained init.py (with fallback imports)
with open("src/navris/inertial/init.py", "r", encoding="utf-8") as f:
    init_content = f.read()

# Replace package imports with robust local/package fallback
old_import = """from navris.inertial.frames import (
    quat_normalize,
    quat_multiply,
    rotate_vector,
    wrap_angle_pi
)
from navris.inertial.gnss_fixes import filter_novel_gnss_fixes"""

new_import = """try:
    from frames import (
        quat_normalize,
        quat_multiply,
        rotate_vector,
        wrap_angle_pi
    )
except ImportError:
    from navris.inertial.frames import (
        quat_normalize,
        quat_multiply,
        rotate_vector,
        wrap_angle_pi
    )

try:
    from navris.inertial.gnss_fixes import filter_novel_gnss_fixes
except ImportError:
    def filter_novel_gnss_fixes(df: pd.DataFrame) -> pd.DataFrame:
        if 'phone_gps_is_new_fix' in df.columns:
            return df[df['phone_gps_is_new_fix'].fillna(False).astype(bool)].copy()
        return df.copy()"""

if old_import in init_content:
    init_content = init_content.replace(old_import, new_import)

with open(os.path.join(dest_dir, "init.py"), "w", encoding="utf-8") as f:
    f.write(init_content)
print("5. Created self-contained init.py")

# 6. Assemble unified eskf.py
eskf_modules = ["state.py", "dynamics.py", "propagation.py", "gating.py", "update.py", "reset.py", "filter.py"]
src_dir = "src/navris/eskf"

lines_out = [
    '"""\n',
    'NAVRIS Complete 15-State Error-State Kalman Filter (ESKF) Review Module.\n',
    '\n',
    'Consolidates all modular ESKF subcomponents for external review:\n',
    '- State definitions: NominalState, ErrorState, InnovationRecord, ESKFConfig\n',
    '- Continuous-time system matrices (F, G, Qc) and exact Van Loan discretization\n',
    '- Strapdown nominal integration and error covariance propagation\n',
    '- Chi-square NIS innovation gating for position and velocity\n',
    '- Kalman gain computation and Joseph stabilized covariance update\n',
    '- Multiplicative quaternion error injection and covariance reset\n',
    '- Main ESKF filter orchestrator class\n',
    '"""\n\n',
    'import numpy as np\n',
    'import pandas as pd\n',
    'from dataclasses import dataclass, field\n',
    'from typing import Optional, Tuple, Dict, Any, List\n',
    'from scipy.linalg import expm\n\n',
    'try:\n',
    '    from frames import (\n',
    '        quat_multiply,\n',
    '        quat_normalize,\n',
    '        quat_to_dcm,\n',
    '        quat_to_euler,\n',
    '        rotvec_to_quat,\n',
    '        rotate_vector,\n',
    '        wrap_angle_pi,\n',
    '    )\n',
    'except ImportError:\n',
    '    from navris.inertial.frames import (\n',
    '        quat_multiply,\n',
    '        quat_normalize,\n',
    '        quat_to_dcm,\n',
    '        quat_to_euler,\n',
    '        rotvec_to_quat,\n',
    '        rotate_vector,\n',
    '        wrap_angle_pi,\n',
    '    )\n\n',
    '# WGS-84 Normal Gravity Functions\n',
    'def normal_gravity_wgs84(lat_deg: float, alt_m: float = 0.0) -> float:\n',
    '    lat_rad = np.radians(lat_deg)\n',
    '    sin_lat = np.sin(lat_rad)\n',
    '    sin2_lat = sin_lat * sin_lat\n',
    '    gamma_0 = 9.7803253359 * (1.0 + 0.00193185265241 * sin2_lat) / np.sqrt(1.0 - 0.00669437999014 * sin2_lat)\n',
    '    h_corr = (2.0 * 9.7803253359 / 6378137.0) * (1.0 + 1.0/298.257223563 + 0.00344978650684 - 2.0/298.257223563 * sin2_lat)\n',
    '    return float(gamma_0 - h_corr * alt_m)\n\n',
    'def gravity_vector_enu(lat_deg: float = 52.4, alt_m: float = 0.0) -> np.ndarray:\n',
    '    g = normal_gravity_wgs84(lat_deg, alt_m)\n',
    '    return np.array([0.0, 0.0, -g], dtype=np.float64)\n\n',
]

for mod in eskf_modules:
    path = os.path.join(src_dir, mod)
    lines_out.append(f"\n# " + "=" * 78 + f"\n# MODULE: {mod}\n# " + "=" * 78 + "\n\n")
    with open(path, "r", encoding="utf-8") as f:
        mod_lines = f.readlines()
    skip_multiline = False
    for line in mod_lines:
        stripped = line.strip()
        if skip_multiline:
            if ")" in stripped:
                skip_multiline = False
            continue
        if stripped.startswith("from navris."):
            if "(" in stripped and ")" not in stripped:
                skip_multiline = True
            continue
        if stripped.startswith("import numpy") or stripped.startswith("from dataclasses") or stripped.startswith("from typing") or stripped.startswith("from scipy.linalg"):
            if "(" in stripped and ")" not in stripped:
                skip_multiline = True
            continue
        lines_out.append(line)

with open(os.path.join(dest_dir, "eskf.py"), "w", encoding="utf-8") as f:
    f.writelines(lines_out)
print("6. Created unified eskf.py")

# 7. Create self-contained run_phase2_3b_s1.py
with open("scripts/run_phase2_3b_s1.py", "r", encoding="utf-8") as f:
    run_content = f.read()

# Make imports fallback to local review directory modules if navris is not installed
old_run_import = """from navris.eskf import (
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
from navris.inertial.init import align_attitude_causal
from navris.inertial.frames import (
    quat_to_euler,
    wrap_angle_pi,
)
from navris.inertial.metrics import compute_dr_metrics"""

new_run_import = """try:
    from eskf import (
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
    from init import align_attitude_causal
    from frames import (
        quat_to_euler,
        wrap_angle_pi,
    )
except ImportError:
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
    from navris.inertial.init import align_attitude_causal
    from navris.inertial.frames import (
        quat_to_euler,
        wrap_angle_pi,
    )"""

if old_run_import in run_content:
    run_content = run_content.replace(old_run_import, new_run_import)

with open(os.path.join(dest_dir, "run_phase2_3b_s1.py"), "w", encoding="utf-8") as f:
    f.write(run_content)
print("7. Created self-contained run_phase2_3b_s1.py")

print("\nAll files assembled in claude_s1_eskf_review/:")
for item in sorted(os.listdir(dest_dir)):
    size = os.path.getsize(os.path.join(dest_dir, item))
    print(f"  {item} ({size:,} bytes)")
