"""
NAVRIS ESKF State Error Injection & Covariance Reset.

In Error-State Kalman Filtering, after a measurement update computes the estimated
error vector delta_x_hat, the error is injected into the nominal state:
    p <- p + delta_p_hat
    v <- v + delta_v_hat
    q <- quat_normalize(Delta_q(delta_theta_hat) (x) q)
    ba <- ba + delta_ba_hat
    bg <- bg + delta_bg_hat

The error state is then reset to zero:
    delta_x <- 0

The error state covariance is updated through the reset Jacobian G_reset:
    P <- G_reset * P^+ * G_reset^T

Conventions:
- Navigation-frame attitude error: q_true = Delta_q(delta_theta^n) (x) q.
- Multiplication: Delta_q(delta_theta^n) on the LEFT.
- Reset Jacobian for attitude block:
    G_theta = I_3 + 0.5 * skew(delta_theta_hat^n)
  which is well-conditioned and non-singular for all reasonable error angles.
"""

import numpy as np
from typing import Tuple

from navris.eskf.state import NominalState, STATE_DIM, IDX_POS, IDX_VEL, IDX_ATT, IDX_ACC_BIAS, IDX_GYR_BIAS
from navris.eskf.dynamics import skew
from navris.inertial.frames import quat_multiply, quat_normalize, rotvec_to_quat


def inject_and_reset(
    nominal: NominalState,
    delta_x_hat: np.ndarray,
    P_post: np.ndarray
) -> Tuple[NominalState, np.ndarray]:
    """
    Injects estimated error state into nominal state and resets error covariance.

    Args:
        nominal: Current nominal state before injection.
        delta_x_hat: 15-dim estimated error state vector from Kalman update.
        P_post: 15x15 posterior error covariance from Joseph update.

    Returns:
        updated_nominal: Nominal state with injected error corrections.
        P_reset: Covariance matrix transformed through reset Jacobian and symmetrized.
    """
    delta_x = np.asarray(delta_x_hat, dtype=np.float64).ravel()
    assert delta_x.shape == (STATE_DIM,), f"delta_x must have shape ({STATE_DIM},), got {delta_x.shape}"

    delta_p = delta_x[IDX_POS]
    delta_v = delta_x[IDX_VEL]
    delta_theta = delta_x[IDX_ATT]
    delta_ba = delta_x[IDX_ACC_BIAS]
    delta_bg = delta_x[IDX_GYR_BIAS]

    # 1. Inject additive position and velocity corrections in navigation frame
    p_new = nominal.p + delta_p
    v_new = nominal.v + delta_v

    # 2. Inject multiplicative attitude correction in navigation frame
    # q_true = Delta_q(delta_theta^n) (x) q
    delta_q = rotvec_to_quat(delta_theta)
    q_new = quat_normalize(quat_multiply(delta_q, nominal.q))

    # 3. Inject additive bias corrections in body frame
    ba_new = nominal.ba + delta_ba
    bg_new = nominal.bg + delta_bg

    updated_nominal = NominalState(
        t=nominal.t,
        p=p_new,
        v=v_new,
        q=q_new,
        ba=ba_new,
        bg=bg_new
    )

    # 4. Construct reset Jacobian G_reset
    G_reset = np.eye(STATE_DIM, dtype=np.float64)
    # Attitude error reset Jacobian: G_theta = I_3 + 0.5 * [delta_theta]_x
    G_reset[IDX_ATT, IDX_ATT] = np.eye(3, dtype=np.float64) + 0.5 * skew(delta_theta)

    # 5. Transform covariance and enforce exact numerical symmetry
    P_reset = G_reset @ P_post @ G_reset.T
    P_reset = 0.5 * (P_reset + P_reset.T)

    return updated_nominal, P_reset
