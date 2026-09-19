"""
NAVRIS ESKF Propagation.

Handles:
1. Nominal state strapdown integration (quaternion attitude, specific force leveling,
   Euler-trapezoidal velocity and position integration).
2. Error covariance propagation: P_{k+1} = Phi * P_k * Phi^T + Q_d with
   numerical symmetry enforcement.
"""

import numpy as np

from navris.eskf.state import NominalState
from navris.inertial.frames import (
    quat_multiply,
    quat_normalize,
    quat_to_dcm,
    rotvec_to_quat,
)


def propagate_nominal(
    state: NominalState,
    f_meas_b: np.ndarray,
    omega_meas_b: np.ndarray,
    dt: float,
    g_n: np.ndarray
) -> NominalState:
    """
    Propagates nominal state forward by dt using strapdown mechanization.

    Args:
        state: Current nominal state at time t.
        f_meas_b: Raw specific force measurement in body frame (m/s^2).
        omega_meas_b: Raw angular rate measurement in body frame (rad/s).
        dt: Time step (seconds).
        g_n: Local gravity vector in navigation ENU frame (m/s^2), typically [0, 0, -g].

    Returns:
        new_state: Propagated nominal state at time t + dt.
    """
    assert dt > 0.0, f"dt must be positive, got {dt}"

    # 1. Bias-corrected angular rate in body frame
    omega_hat_b = np.asarray(omega_meas_b, dtype=np.float64) - state.bg

    # Incremental rotation vector and quaternion
    delta_q = rotvec_to_quat(omega_hat_b * dt)
    q_new = quat_normalize(quat_multiply(state.q, delta_q))

    # 2. Bias-corrected specific force in body frame
    f_hat_b = np.asarray(f_meas_b, dtype=np.float64) - state.ba

    # Use midpoint attitude for second-order orientation accuracy during interval
    delta_q_half = rotvec_to_quat(omega_hat_b * (0.5 * dt))
    q_mid = quat_normalize(quat_multiply(state.q, delta_q_half))
    C_b_n_mid = quat_to_dcm(q_mid)

    # Net acceleration in navigation frame
    a_n = C_b_n_mid @ f_hat_b + np.asarray(g_n, dtype=np.float64)

    # 3. Velocity and position integration (Euler-trapezoidal / 2nd-order Taylor)
    v_new = state.v + a_n * dt
    p_new = state.p + state.v * dt + 0.5 * a_n * (dt ** 2)

    return NominalState(
        t=state.t + dt,
        p=p_new,
        v=v_new,
        q=q_new,
        ba=state.ba.copy(),
        bg=state.bg.copy()
    )


def propagate_covariance(
    P: np.ndarray,
    Phi: np.ndarray,
    Qd: np.ndarray
) -> np.ndarray:
    """
    Propagates error state covariance forward by dt:
        P_{k+1} = Phi * P_k * Phi^T + Q_d

    Enforces exact numerical symmetry.

    Args:
        P: 15x15 error covariance matrix.
        Phi: 15x15 discrete state transition matrix.
        Qd: 15x15 discrete process noise covariance matrix.

    Returns:
        P_next: 15x15 propagated error covariance matrix.
    """
    P_next = Phi @ P @ Phi.T + Qd
    P_next = 0.5 * (P_next + P_next.T)
    return P_next
