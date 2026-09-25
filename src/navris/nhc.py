"""
NAVRIS Causal Non-Holonomic Constraints (NHC) Module.

Phase 2.3B Gate 2.3B-2 additive module.
Maintains 100% frozen ESKF core integrity (src/navris/eskf/* untouched).

Implements:
- Strictly causal NHC detector enforcing non-holonomic validity:
    1. Forward speed v_vehicle[x] > min_forward_speed (positive forward motion along vehicle chassis).
    2. Vehicle not stationary (ZUPT inactive).
    3. Angular rate ||omega_meas|| <= max_angular_rate (conservative engineering validity safeguard inhibiting updates during turns where NHC assumptions become unreliable).
    4. Specific force norm deviation | ||f_meas|| - g | <= max_acc_deviation (rejects road shocks, potholes, speed bumps).
- Analytical NHC error-state measurement Jacobian H_nhc in R^{2 x 15}:
    H_v = M_nhc * C_n^v
    H_theta = + M_nhc * C_n^v * [v_hat^n]_x
    All other blocks (position, accel bias, gyro bias) identically zero.
- Sequential Kalman measurement update using existing Joseph-form covariance update
  and multiplicative quaternion error injection machinery.
- 2-DOF Chi-square innovation gating (DOF=2, chi2_threshold=13.82, p=0.001).
- Pre-declared conservative engineering hypothesis measurement covariance:
    R_nhc = diag(sigma_lat^2, sigma_vert^2) = diag(0.0625, 0.0225) m^2/s^2.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from navris.eskf import (
    ESKF,
    NominalState,
    STATE_DIM,
    IDX_VEL,
    IDX_ATT,
)
from navris.eskf.dynamics import skew
from navris.eskf.gating import (
    InnovationRecord,
    gate_measurement,
)
from navris.eskf.update import (
    compute_kalman_gain,
    joseph_covariance_update,
)
from navris.eskf.reset import inject_and_reset
from navris.inertial.frames import quat_to_dcm


# Pre-declared frozen measurement noise parameters (conservative engineering hypothesis)
DEFAULT_SIGMA_LAT: float = 0.25   # m/s (25 cm/s lateral noise)
DEFAULT_SIGMA_VERT: float = 0.15  # m/s (15 cm/s vertical noise)
DEFAULT_R_NHC: np.ndarray = np.diag([DEFAULT_SIGMA_LAT**2, DEFAULT_SIGMA_VERT**2])
NHC_CHI2_THRESHOLD: float = 13.82  # 2 DOF at 99.9% confidence (p = 0.001)

# NHC 2x3 selection matrix: selects lateral (index 1) and vertical (index 2) components
M_NHC: np.ndarray = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)


@dataclass
class NHCConfig:
    """
    Configuration parameters for NHC measurement update.
    Pre-declared and frozen before benchmark evaluation as experimental hypotheses.
    """
    sigma_lat: float = DEFAULT_SIGMA_LAT
    sigma_vert: float = DEFAULT_SIGMA_VERT
    chi2_threshold: float = NHC_CHI2_THRESHOLD


@dataclass
class NHCDetectorConfig:
    """
    Configuration parameters for causal NHC validity detector.
    Pre-declared and frozen before benchmark evaluation as experimental hypotheses.
    """
    min_forward_speed: float = 1.5         # m/s (forward speed along vehicle chassis X_v)
    max_angular_rate: float = 0.087        # rad/s (~5.0 deg/s, conservative safeguard inhibiting updates during turns)
    max_acc_deviation: float = 1.0         # m/s^2 (| ||f_meas|| - g | <= 1.0 m/s^2 rejects shocks/potholes)
    gravity_nominal: float = 9.80665       # m/s^2 nominal gravity


class CausalNHCDetector:
    """
    Strictly causal detector evaluating whether non-holonomic constraints are physically valid
    at the current sample tk using only current/past filter state and IMU measurements.
    """

    def __init__(self, config: Optional[NHCDetectorConfig] = None):
        self.config = config if config is not None else NHCDetectorConfig()
        self.total_nhc_attempts: int = 0
        self.total_nhc_accepted_conditions: int = 0

    def update(
        self,
        nominal_state: NominalState,
        C_b_v: np.ndarray,
        f_meas_b: np.ndarray,
        omega_meas_b: np.ndarray,
        is_stationary: bool = False,
    ) -> bool:
        """
        Evaluates whether NHC should be triggered at the current epoch tk.

        Args:
            nominal_state: Current ESKF nominal state (contains v in navigation frame, q_b_n).
            C_b_v: 3x3 extrinsic rotation matrix mapping body frame to vehicle chassis frame.
            f_meas_b: 3-vector body-frame specific force (m/s^2).
            omega_meas_b: 3-vector body-frame angular rate (rad/s).
            is_stationary: Boolean indicator from stationary detector (ZUPT).

        Returns:
            should_apply: True if all NHC validity conditions are satisfied.
        """
        self.total_nhc_attempts += 1

        # 1. Condition 1: Inhibit when vehicle is stationary (handled by ZUPT)
        if is_stationary:
            return False

        # 2. Condition 2: Forward speed along vehicle chassis must exceed threshold
        # v_vehicle = C_b^v * (C_b^n)^T * v_nav
        C_b_n = quat_to_dcm(nominal_state.q)
        C_n_v = C_b_v @ C_b_n.T
        v_vehicle = C_n_v @ nominal_state.v
        forward_speed = float(v_vehicle[0])

        if forward_speed <= self.config.min_forward_speed:
            return False

        # 3. Condition 3: Total angular rate below threshold (conservative validity safeguard during turns)
        w_norm = float(np.linalg.norm(omega_meas_b))
        if w_norm > self.config.max_angular_rate:
            return False

        # 4. Condition 4: Specific force norm deviation from gravity below threshold (rejects potholes/shocks)
        f_norm = float(np.linalg.norm(f_meas_b))
        acc_dev = abs(f_norm - self.config.gravity_nominal)
        if acc_dev > self.config.max_acc_deviation:
            return False

        self.total_nhc_accepted_conditions += 1
        return True

    def reset(self) -> None:
        """Resets detector internal counters."""
        self.total_nhc_attempts = 0
        self.total_nhc_accepted_conditions = 0


def compute_nhc_jacobian(state: NominalState, C_b_v: np.ndarray) -> np.ndarray:
    """
    Computes analytical NHC measurement Jacobian H_nhc in R^{2 x 15}.

    Convention:
        q_true = rotvec_to_quat(delta_theta^n) (x) q_nominal
        C_b^n(q_true) = (I + [delta_theta^n]_x) * C_b^n_hat
        (C_b^n(q_true))^T = C_b^n_hat^T * (I - [delta_theta^n]_x)
        v^v = C_b^v * (C_b^n)^T * v^n
            = C_b^v * C_b^n_hat^T * (I - [delta_theta^n]_x) * (v_hat^n + delta_v^n)
            = v_hat^v + C_n^v * delta_v^n - C_n^v * [delta_theta^n]_x * v_hat^n
            = v_hat^v + C_n^v * delta_v^n + C_n^v * [v_hat^n]_x * delta_theta^n

    Therefore:
        H_v = M_nhc * C_n^v
        H_theta = + M_nhc * C_n^v * [v_hat^n]_x

    Args:
        state: Current NominalState.
        C_b_v: 3x3 extrinsic rotation matrix from body to vehicle frame.

    Returns:
        H: 2x15 measurement Jacobian matrix.
    """
    C_b_n = quat_to_dcm(state.q)
    C_n_v = C_b_v @ C_b_n.T

    H = np.zeros((2, STATE_DIM), dtype=np.float64)

    # Velocity block: 2x3
    H_v = M_NHC @ C_n_v
    H[0:2, IDX_VEL] = H_v

    # Attitude block: 2x3 with positive sign
    v_skew = skew(state.v)
    H_theta = M_NHC @ C_n_v @ v_skew
    H[0:2, IDX_ATT] = H_theta

    return H


def apply_nhc_update(
    eskf: ESKF,
    C_b_v: np.ndarray,
    R_nhc: Optional[np.ndarray] = None,
    chi2_threshold: float = NHC_CHI2_THRESHOLD,
    timestamp: Optional[float] = None,
) -> Tuple[bool, InnovationRecord]:
    """
    Applies a causal Non-Holonomic Constraint (NHC) measurement update to the ESKF state.

    Measurement model:
        z = [0, 0]^T (m/s) in vehicle frame (transverse and vertical axes)
        h(x) = M_nhc * C_b^v * (C_b^n)^T * v_hat^n = [v_y^v, v_z^v]^T
        residual = z - h(x) = - [v_y^v, v_z^v]^T

    Args:
        eskf: Frozen ESKF instance.
        C_b_v: 3x3 extrinsic rotation matrix from body to vehicle frame.
        R_nhc: 2x2 fixed measurement covariance matrix (default: diag(0.0625, 0.0225)).
        chi2_threshold: Chi-square innovation gating threshold (default: 13.82, 2 DOF).
        timestamp: Measurement timestamp (defaults to current state time).

    Returns:
        (accepted, innovation_record)
    """
    t = timestamp if timestamp is not None else eskf.state.t
    C_b_v = np.asarray(C_b_v, dtype=np.float64)
    assert C_b_v.shape == (3, 3), f"C_b_v must have shape (3, 3), got {C_b_v.shape}"

    # 1. Compute predicted vehicle velocity: v_hat^v = C_b^v * (C_b^n)^T * v_hat^n
    C_b_n = quat_to_dcm(eskf.state.q)
    C_n_v = C_b_v @ C_b_n.T
    v_vehicle = C_n_v @ eskf.state.v

    z_meas = np.zeros(2, dtype=np.float64)
    pred_meas = M_NHC @ v_vehicle  # [v_y^v, v_z^v]^T
    residual = z_meas - pred_meas  # - [v_y^v, v_z^v]^T

    # 2. Measurement Jacobian H in R^{2 x 15}
    H = compute_nhc_jacobian(eskf.state, C_b_v)

    # 3. Measurement covariance R in R^{2 x 2}
    R = np.asarray(R_nhc if R_nhc is not None else DEFAULT_R_NHC, dtype=np.float64)
    assert R.shape == (2, 2), f"R_nhc must be (2, 2), got {R.shape}"

    # 4. Kalman Gain and Innovation Covariance S = H * P * H^T + R
    K, S = compute_kalman_gain(eskf.P, H, R)

    # 5. Chi-square gating (2 DOF)
    accepted, d2 = gate_measurement(residual, S, chi2_threshold)

    record = InnovationRecord(
        timestamp=t,
        sensor_type="nhc",
        raw_meas=z_meas,
        pred_meas=pred_meas,
        residual=residual,
        S_cov=S,
        mahalanobis_sq=d2,
        threshold=chi2_threshold,
        accepted=accepted,
        is_new_fix=True,
    )
    eskf.innovations.append(record)

    if not accepted:
        return False, record

    # 6. Joseph-form covariance update: P^+ = (I - K*H) * P * (I - K*H)^T + K*R*K^T
    P_post = joseph_covariance_update(eskf.P, K, H, R)

    # 7. Error state calculation: delta_x = K * residual
    delta_x = K @ residual

    # 8. Multiplicative quaternion error injection and covariance reset
    eskf.state, eskf.P = inject_and_reset(eskf.state, delta_x, P_post)

    return True, record
