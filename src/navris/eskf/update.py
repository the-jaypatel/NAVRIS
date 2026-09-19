"""
NAVRIS ESKF Measurement Update.

Implements numerically robust Kalman gain computation and the Joseph-form
covariance update:
    P^+ = (I - K*H) * P^- * (I - K*H)^T + K * R * K^T

The Joseph form guarantees symmetry and positive semi-definiteness even under
ill-conditioned measurement geometries.
"""

import numpy as np
from typing import Tuple

from navris.eskf.state import STATE_DIM


def compute_kalman_gain(
    P: np.ndarray,
    H: np.ndarray,
    R: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Kalman gain K and innovation covariance S.

    Args:
        P: 15x15 prior error covariance matrix P^-.
        H: Mx15 measurement matrix.
        R: MxM measurement noise covariance matrix.

    Returns:
        K: 15xM Kalman gain matrix K = P * H^T * S^{-1}.
        S: MxM innovation covariance matrix S = H * P * H^T + R.
    """
    H = np.asarray(H, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)

    # Innovation covariance
    S = H @ P @ H.T + R
    S = 0.5 * (S + S.T)

    # Solve S * K^T = H * P  ==>  K = P * H^T * S^{-1}
    try:
        L = np.linalg.cholesky(S)
        # S = L * L^T
        # K^T = solve(L^T, solve(L, H @ P))
        y = np.linalg.solve(L, H @ P)
        K_T = np.linalg.solve(L.T, y)
        K = K_T.T
    except np.linalg.LinAlgError:
        inv_S = np.linalg.pinv(S)
        K = P @ H.T @ inv_S

    return K, S


def joseph_covariance_update(
    P: np.ndarray,
    K: np.ndarray,
    H: np.ndarray,
    R: np.ndarray
) -> np.ndarray:
    """
    Performs Joseph-form error covariance update:
        P^+ = (I - K*H) * P^- * (I - K*H)^T + K * R * K^T

    Enforces numerical symmetry.

    Args:
        P: 15x15 prior error covariance matrix.
        K: 15xM Kalman gain matrix.
        H: Mx15 measurement matrix.
        R: MxM measurement noise covariance.

    Returns:
        P_post: 15x15 posterior error covariance matrix.
    """
    I_KH = np.eye(STATE_DIM, dtype=np.float64) - K @ H
    P_post = I_KH @ P @ I_KH.T + K @ R @ K.T
    P_post = 0.5 * (P_post + P_post.T)
    return P_post
