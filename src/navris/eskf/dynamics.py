"""
NAVRIS ESKF Continuous Dynamics & Discretization.

Derivation of Error State Dynamics (Local Navigation Frame ENU):
- Nominal State:
    dot(p^n) = v^n
    dot(v^n) = C_b^n(q) * (f_meas^b - b_a^b) + g^n
    dot(q_b^n) = 0.5 * q_b^n (x) [0, omega_meas^b - b_g^b]^T
    dot(b_a^b) = w_ba
    dot(b_g^b) = w_bg

- True State Perturbations:
    p_true = p + delta_p
    v_true = v + delta_v
    q_true = Delta_q(delta_theta^n) (x) q
    b_a_true = b_a + delta_b_a
    b_g_true = b_g + delta_b_g

- Linearized Error State Equations:
    d(delta_p)/dt = delta_v
    d(delta_v)/dt = - [C_b^n * f_hat^b]_x * delta_theta^n - C_b^n * delta_b_a^b - C_b^n * w_a
    d(delta_theta^n)/dt = - C_b^n * delta_b_g^b - C_b^n * w_g
    d(delta_b_a^b)/dt = w_ba
    d(delta_b_g^b)/dt = w_bg

Discretization:
- Van Loan's method (1978) via scipy.linalg.expm is used to compute the exact
  discrete state transition matrix Phi and discrete process noise covariance Q_d:
      Lambda = [[-F, G * Q_c * G^T], [0, F^T]] * dt
      Xi = expm(Lambda)
      Phi = Xi[15:30, 15:30]^T
      Q_d = Phi * Xi[0:15, 15:30]
"""

import numpy as np
import scipy.linalg
from typing import Tuple

from navris.eskf.state import NominalState, STATE_DIM, ESKFConfig
from navris.inertial.frames import quat_to_dcm


def skew(v: np.ndarray) -> np.ndarray:
    """
    Returns 3x3 skew-symmetric cross-product matrix [v]_x such that
    skew(v) @ u == np.cross(v, u).
    """
    v = np.asarray(v, dtype=np.float64).ravel()
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=np.float64)


def continuous_f_matrix(nominal: NominalState, f_meas_b: np.ndarray) -> np.ndarray:
    """
    Computes 15x15 continuous system matrix F(t).

    Args:
        nominal: Current nominal state (p, v, q, ba, bg).
        f_meas_b: Raw specific force measurement in body frame (m/s^2).

    Returns:
        F: 15x15 continuous dynamics Jacobian matrix.
    """
    F = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float64)

    # Orientation DCM: C_b^n
    C_b_n = quat_to_dcm(nominal.q)

    # Bias-corrected specific force in body frame
    f_hat_b = np.asarray(f_meas_b, dtype=np.float64) - nominal.ba

    # Specific force rotated to navigation frame: a_n = C_b^n * f_hat_b
    f_n = C_b_n @ f_hat_b

    # 1. d(delta_p) / d(delta_v) = I_3
    F[0:3, 3:6] = np.eye(3, dtype=np.float64)

    # 2. d(delta_v) / d(delta_theta^n) = - [f_n]_x
    F[3:6, 6:9] = -skew(f_n)

    # 3. d(delta_v) / d(delta_ba^b) = - C_b^n
    F[3:6, 9:12] = -C_b_n

    # 4. d(delta_theta^n) / d(delta_bg^b) = - C_b^n
    F[6:9, 12:15] = -C_b_n

    return F


def continuous_g_matrix(nominal: NominalState) -> np.ndarray:
    """
    Computes 15x12 continuous noise mapping matrix G_c(t).

    Noise vector w = [w_a, w_g, w_ba, w_bg]^T in R^12.
    """
    G = np.zeros((STATE_DIM, 12), dtype=np.float64)
    C_b_n = quat_to_dcm(nominal.q)

    # Accelerometer white noise affects velocity error
    G[3:6, 0:3] = -C_b_n

    # Gyroscope white noise affects attitude error
    G[6:9, 3:6] = -C_b_n

    # Bias random walk drives bias states directly in body frame
    G[9:12, 6:9] = np.eye(3, dtype=np.float64)
    G[12:15, 9:12] = np.eye(3, dtype=np.float64)

    return G


def continuous_qc_matrix(config: ESKFConfig) -> np.ndarray:
    """
    Constructs 12x12 continuous process noise spectral density matrix Q_c.
    """
    Q_diag = np.concatenate([
        np.full(3, config.sigma_acc ** 2, dtype=np.float64),
        np.full(3, config.sigma_gyr ** 2, dtype=np.float64),
        np.full(3, config.sigma_acc_bias ** 2, dtype=np.float64),
        np.full(3, config.sigma_gyr_bias ** 2, dtype=np.float64),
    ])
    return np.diag(Q_diag)


def discretize_dynamics_van_loan(
    F: np.ndarray,
    G: np.ndarray,
    Qc: np.ndarray,
    dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discretizes continuous system (F, G, Qc) using Van Loan's method (1978).
    Computes exact state transition matrix Phi and discrete process noise Q_d:

        Q_d = integral_0^dt Phi(tau) * (G * Qc * G^T) * Phi(tau)^T dtau

    Guarantees mathematical consistency, symmetry, and positive semi-definiteness.

    Args:
        F: 15x15 continuous dynamics matrix.
        G: 15x12 continuous noise input matrix.
        Qc: 12x12 continuous noise power spectral density matrix.
        dt: Time step (seconds).

    Returns:
        Phi: 15x15 discrete state transition matrix.
        Qd: 15x15 discrete process noise covariance matrix.
    """
    assert dt > 0.0, f"dt must be positive, got {dt}"
    W = G @ Qc @ G.T  # 15x15 continuous process noise diffusion matrix

    # Construct 30x30 Van Loan block matrix:
    # Lambda = [[-F,  W  ],
    #           [ 0,  F^T]] * dt
    Lambda = np.zeros((2 * STATE_DIM, 2 * STATE_DIM), dtype=np.float64)
    Lambda[0:STATE_DIM, 0:STATE_DIM] = -F * dt
    Lambda[0:STATE_DIM, STATE_DIM:2 * STATE_DIM] = W * dt
    Lambda[STATE_DIM:2 * STATE_DIM, STATE_DIM:2 * STATE_DIM] = F.T * dt

    # Matrix exponential
    Xi = scipy.linalg.expm(Lambda)

    # Lower right block is Phi^T
    Phi_T = Xi[STATE_DIM:2 * STATE_DIM, STATE_DIM:2 * STATE_DIM]
    Phi = Phi_T.T

    # Upper right block is Phi^{-1} * Q_d
    Xi_12 = Xi[0:STATE_DIM, STATE_DIM:2 * STATE_DIM]
    Qd = Phi @ Xi_12

    # Enforce exact symmetry
    Qd = 0.5 * (Qd + Qd.T)

    return Phi, Qd
