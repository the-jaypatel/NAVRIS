"""
NAVRIS ESKF Innovation Logging & Measurement Gating.

Implements Chi-Square Mahalanobis distance innovation gating to reject spurious
outliers and logs complete innovation statistics (NIS) for filter consistency auditing.
"""

from dataclasses import dataclass
import numpy as np
from typing import Tuple


@dataclass
class InnovationRecord:
    """Detailed record of a measurement update attempt."""
    timestamp: float
    sensor_type: str  # e.g., 'gnss_pos', 'gnss_vel'
    raw_meas: np.ndarray
    pred_meas: np.ndarray
    residual: np.ndarray
    S_cov: np.ndarray
    mahalanobis_sq: float
    threshold: float
    accepted: bool
    is_new_fix: bool


def compute_mahalanobis_sq(residual: np.ndarray, S: np.ndarray) -> float:
    """
    Computes squared Mahalanobis distance (Normalized Innovation Squared, NIS):
        d^2 = r^T * S^{-1} * r

    Uses Cholesky solve with fallback to pseudoinverse for numerical robustness.
    """
    r = np.asarray(residual, dtype=np.float64).ravel()
    S_mat = np.asarray(S, dtype=np.float64)

    try:
        # Cholesky decomposition L * L^T = S
        L = np.linalg.cholesky(S_mat)
        y = np.linalg.solve(L, r)
        d2 = float(np.dot(y, y))
    except np.linalg.LinAlgError:
        try:
            inv_S = np.linalg.pinv(S_mat)
            d2 = float(r.T @ inv_S @ r)
        except Exception:
            return float("nan")

    return d2


def gate_measurement(
    residual: np.ndarray,
    S: np.ndarray,
    threshold: float
) -> Tuple[bool, float]:
    """
    Evaluates whether measurement passes the Chi-square innovation gate.

    Args:
        residual: Measurement residual vector r = z - h(x).
        S: Innovation covariance matrix S = H * P * H^T + R.
        threshold: Chi-square threshold gamma (e.g., 16.27 for 3 DOF at 99.9%).

    Returns:
        accepted: True if d^2 <= threshold and d^2 is finite, False otherwise.
        mahalanobis_sq: Computed d^2 value.
    """
    d2 = compute_mahalanobis_sq(residual, S)

    if not np.isfinite(d2) or d2 < 0.0:
        return False, d2

    accepted = bool(d2 <= threshold)
    return accepted, d2
