"""
NAVRIS Pure Inertial Navigation and Dead-Reckoning Subsystem.
Phase 2.2: Classical 3D Quaternion Strapdown INS Baseline.
"""

from navris.inertial.frames import (
    quat_multiply,
    quat_conjugate,
    quat_normalize,
    quat_to_dcm,
    rotate_vector,
    rotvec_to_quat,
    wrap_angle_pi,
    wrap_angle_2pi,
    euler_to_quat,
    quat_to_euler
)
from navris.inertial.gravity import (
    normal_gravity_wgs84,
    gravity_vector_enu
)
from navris.inertial.segments import (
    detect_discontinuities,
    split_into_contiguous_segments
)
from navris.inertial.gnss_fixes import (
    filter_novel_gnss_fixes
)
from navris.inertial.stationary import (
    detect_stationary_epochs,
    estimate_stationary_imu_stats
)
from navris.inertial.init import (
    align_attitude_causal,
    AlignmentResult
)
from navris.inertial.strapdown import (
    propagate_strapdown_segment,
    run_strapdown_dr,
    StrapdownTrajectory
)
from navris.inertial.planar import (
    run_planar_dr,
    PlanarTrajectory
)
from navris.inertial.oracle import (
    run_oracle_heading_dr,
    run_oracle_heading_bias_dr
)
from navris.inertial.metrics import (
    compute_dr_metrics,
    compute_along_cross_track_errors
)
