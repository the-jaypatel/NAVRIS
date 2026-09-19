"""
NAVRIS 15-State Error-State Kalman Filter (ESKF) Package.
"""

from navris.eskf.state import (
    NominalState,
    ESKFConfig,
    STATE_DIM,
    NOMINAL_DIM,
    IDX_POS,
    IDX_VEL,
    IDX_ATT,
    IDX_ACC_BIAS,
    IDX_GYR_BIAS,
)
from navris.eskf.dynamics import (
    skew,
    continuous_f_matrix,
    continuous_g_matrix,
    continuous_qc_matrix,
    discretize_dynamics_van_loan,
)
from navris.eskf.propagation import (
    propagate_nominal,
    propagate_covariance,
)
from navris.eskf.gating import (
    InnovationRecord,
    compute_mahalanobis_sq,
    gate_measurement,
)
from navris.eskf.update import (
    compute_kalman_gain,
    joseph_covariance_update,
)
from navris.eskf.reset import inject_and_reset
from navris.eskf.filter import ESKF

__all__ = [
    "NominalState",
    "ESKFConfig",
    "STATE_DIM",
    "NOMINAL_DIM",
    "IDX_POS",
    "IDX_VEL",
    "IDX_ATT",
    "IDX_ACC_BIAS",
    "IDX_GYR_BIAS",
    "skew",
    "continuous_f_matrix",
    "continuous_g_matrix",
    "continuous_qc_matrix",
    "discretize_dynamics_van_loan",
    "propagate_nominal",
    "propagate_covariance",
    "InnovationRecord",
    "compute_mahalanobis_sq",
    "gate_measurement",
    "compute_kalman_gain",
    "joseph_covariance_update",
    "inject_and_reset",
    "ESKF",
]
