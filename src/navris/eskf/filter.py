"""
NAVRIS 15-State Error-State Kalman Filter (ESKF).

Integrates:
- High-rate (10 Hz) IMU strapdown nominal propagation with gravity compensation
- Continuous-discrete error state propagation via Van Loan discretization
- Innovation logging and Chi-square gating
- Joseph-form covariance updates
- Novel GNSS fix detection (rejecting sample-and-hold duplicates)
- Multiplicative quaternion error injection and covariance reset
"""

from typing import List, Optional
import numpy as np

from navris.eskf.state import (
    NominalState,
    ESKFConfig,
    STATE_DIM,
    IDX_POS,
    IDX_VEL,
)
from navris.eskf.dynamics import (
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
    gate_measurement,
)
from navris.eskf.update import (
    compute_kalman_gain,
    joseph_covariance_update,
)
from navris.eskf.reset import inject_and_reset
from navris.inertial.gravity import gravity_vector_enu


class ESKF:
    """
    15-State Error-State Extended Kalman Filter for GNSS-Inertial Navigation.
    """

    def __init__(
        self,
        initial_state: NominalState,
        initial_cov: np.ndarray,
        config: Optional[ESKFConfig] = None,
        lat_deg: float = 0.0,
        alt_m: float = 0.0,
        g_n: Optional[np.ndarray] = None,
    ):
        """
        Initializes the ESKF.

        Args:
            initial_state: Initial nominal navigation state.
            initial_cov: 15x15 initial error covariance matrix P_0.
            config: Filter configuration parameters (noise PSDs, gating thresholds).
            lat_deg: Geodetic latitude of operation (degrees) for gravity model.
            alt_m: Ellipsoidal height (meters) for gravity model.
            g_n: Optional explicit gravity vector in ENU. If None, derived from WGS-84.
        """
        self.state: NominalState = initial_state.copy()
        P_init = np.asarray(initial_cov, dtype=np.float64)
        assert P_init.shape == (STATE_DIM, STATE_DIM), f"P must be ({STATE_DIM}, {STATE_DIM})"
        self.P: np.ndarray = 0.5 * (P_init + P_init.T)

        self.config: ESKFConfig = config if config is not None else ESKFConfig()

        if g_n is not None:
            self.g_n = np.asarray(g_n, dtype=np.float64)
        else:
            self.g_n = gravity_vector_enu(lat_deg, alt_m)

        self.innovations: List[InnovationRecord] = []
        self.step_count: int = 0

    def predict(
        self,
        f_meas_b: np.ndarray,
        omega_meas_b: np.ndarray,
        dt: float
    ) -> bool:
        """
        Propagates the nominal state and error covariance forward by dt using
        IMU measurements.

        Args:
            f_meas_b: Raw specific force measurement in body frame (m/s^2).
            omega_meas_b: Raw angular rate measurement in body frame (rad/s).
            dt: Time step duration (seconds).

        Returns:
            success: True if propagation succeeded, False if gap was encountered.
        """
        if dt < self.config.min_dt:
            # Step too small, skip to avoid numerical instability
            return False

        if dt > self.config.max_dt_gap:
            # Timestamp gap detected: per NAVRIS gap philosophy, do not bridge large gaps blindly
            return False

        # 1. Continuous system matrices
        F = continuous_f_matrix(self.state, f_meas_b)
        G = continuous_g_matrix(self.state)
        Qc = continuous_qc_matrix(self.config)

        # 2. Exact discretization via Van Loan
        Phi, Qd = discretize_dynamics_van_loan(F, G, Qc, dt)

        # 3. Propagate nominal state
        self.state = propagate_nominal(self.state, f_meas_b, omega_meas_b, dt, self.g_n)

        # 4. Propagate error covariance
        self.P = propagate_covariance(self.P, Phi, Qd)

        self.step_count += 1
        return True

    def update_gnss_pos(
        self,
        pos_meas_enu: np.ndarray,
        pos_cov: Optional[np.ndarray] = None,
        is_new_fix: bool = True,
        timestamp: Optional[float] = None
    ) -> bool:
        """
        Performs GNSS position measurement update with novel-fix check and Chi-square gating.

        Args:
            pos_meas_enu: 3-vector [East, North, Up] in meters.
            pos_cov: Optional 3x3 position measurement noise covariance.
            is_new_fix: True if this is a genuine new GNSS fix; False if duplicate sample-and-hold.
            timestamp: Optional timestamp of the measurement.

        Returns:
            accepted: True if measurement was accepted and filter updated; False if rejected or duplicate.
        """
        t = timestamp if timestamp is not None else self.state.t
        pos_meas = np.asarray(pos_meas_enu, dtype=np.float64).ravel()
        assert pos_meas.shape == (3,), f"pos_meas_enu must have shape (3,), got {pos_meas.shape}"

        # 1. Novel fix check: sample-and-hold duplicates must never trigger a measurement update
        if not is_new_fix:
            self.innovations.append(InnovationRecord(
                timestamp=t,
                sensor_type="gnss_pos",
                raw_meas=pos_meas,
                pred_meas=self.state.p.copy(),
                residual=pos_meas - self.state.p,
                S_cov=np.zeros((3, 3), dtype=np.float64),
                mahalanobis_sq=0.0,
                threshold=self.config.chi2_threshold_pos,
                accepted=False,
                is_new_fix=False
            ))
            return False

        # 2. Measurement matrix H: z = p^n + v  ==>  H = [I_3, 0_{3x12}]
        H = np.zeros((3, STATE_DIM), dtype=np.float64)
        H[0:3, IDX_POS] = np.eye(3, dtype=np.float64)

        # 3. Measurement noise covariance R
        if pos_cov is not None:
            R = np.asarray(pos_cov, dtype=np.float64)
            assert R.shape == (3, 3), f"pos_cov must have shape (3, 3), got {R.shape}"
        else:
            R = np.diag([
                self.config.gnss_pos_std_horiz ** 2,
                self.config.gnss_pos_std_horiz ** 2,
                self.config.gnss_pos_std_vert ** 2,
            ])

        # 4. Residual and Kalman gain
        pred_pos = self.state.p.copy()
        residual = pos_meas - pred_pos

        K, S = compute_kalman_gain(self.P, H, R)

        # 5. Chi-square innovation gating
        accepted, d2 = gate_measurement(residual, S, self.config.chi2_threshold_pos)

        # Log innovation record
        self.innovations.append(InnovationRecord(
            timestamp=t,
            sensor_type="gnss_pos",
            raw_meas=pos_meas,
            pred_meas=pred_pos,
            residual=residual,
            S_cov=S,
            mahalanobis_sq=d2,
            threshold=self.config.chi2_threshold_pos,
            accepted=accepted,
            is_new_fix=True
        ))

        if not accepted:
            return False

        # 6. Joseph-form covariance update
        P_post = joseph_covariance_update(self.P, K, H, R)

        # 7. Estimated error state: delta_x = K * r
        delta_x = K @ residual

        # 8. Multiplicative error injection & covariance reset
        self.state, self.P = inject_and_reset(self.state, delta_x, P_post)

        return True

    def update_gnss_vel(
        self,
        vel_meas_enu: np.ndarray,
        vel_cov: Optional[np.ndarray] = None,
        is_new_fix: bool = True,
        timestamp: Optional[float] = None
    ) -> bool:
        """
        Performs GNSS velocity measurement update with novel-fix check and Chi-square gating.

        Args:
            vel_meas_enu: 3-vector [vE, vN, vU] in m/s.
            vel_cov: Optional 3x3 velocity measurement noise covariance.
            is_new_fix: True if this is a genuine new fix.
            timestamp: Optional timestamp.

        Returns:
            accepted: True if accepted, False otherwise.
        """
        t = timestamp if timestamp is not None else self.state.t
        vel_meas = np.asarray(vel_meas_enu, dtype=np.float64).ravel()
        assert vel_meas.shape == (3,), f"vel_meas_enu must have shape (3,), got {vel_meas.shape}"

        if not is_new_fix:
            self.innovations.append(InnovationRecord(
                timestamp=t,
                sensor_type="gnss_vel",
                raw_meas=vel_meas,
                pred_meas=self.state.v.copy(),
                residual=vel_meas - self.state.v,
                S_cov=np.zeros((3, 3), dtype=np.float64),
                mahalanobis_sq=0.0,
                threshold=self.config.chi2_threshold_vel,
                accepted=False,
                is_new_fix=False
            ))
            return False

        # Measurement matrix H: z = v^n + v  ==>  H = [0_{3x3}, I_3, 0_{3x9}]
        H = np.zeros((3, STATE_DIM), dtype=np.float64)
        H[0:3, IDX_VEL] = np.eye(3, dtype=np.float64)

        if vel_cov is not None:
            R = np.asarray(vel_cov, dtype=np.float64)
        else:
            R = np.diag([
                self.config.gnss_vel_std_horiz ** 2,
                self.config.gnss_vel_std_horiz ** 2,
                self.config.gnss_vel_std_vert ** 2,
            ])

        pred_vel = self.state.v.copy()
        residual = vel_meas - pred_vel

        K, S = compute_kalman_gain(self.P, H, R)
        accepted, d2 = gate_measurement(residual, S, self.config.chi2_threshold_vel)

        self.innovations.append(InnovationRecord(
            timestamp=t,
            sensor_type="gnss_vel",
            raw_meas=vel_meas,
            pred_meas=pred_vel,
            residual=residual,
            S_cov=S,
            mahalanobis_sq=d2,
            threshold=self.config.chi2_threshold_vel,
            accepted=accepted,
            is_new_fix=True
        ))

        if not accepted:
            return False

        P_post = joseph_covariance_update(self.P, K, H, R)
        delta_x = K @ residual
        self.state, self.P = inject_and_reset(self.state, delta_x, P_post)

        return True
