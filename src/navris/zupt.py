"""
NAVRIS Causal Zero-Velocity Update (ZUPT) Module.

Phase 2.3B Gate 2.3A additive module.
Maintains 100% frozen ESKF core integrity (src/navris/eskf/* untouched).

Implements:
- Causal trailing-window stationary detection with dual thresholds (entry/exit hysteresis)
  and minimum dwell time.
- Sequential zero-velocity Kalman measurement update using existing Joseph-form
  covariance update and multiplicative error state injection machinery.
- Chi-square innovation gating (DOF=3, chi2_threshold=16.27).
- Fixed measurement covariance sigma_zupt = 0.05 m/s treated as a pre-declared conservative
  engineering measurement-noise assumption informed by the stationary IMU audit (not a mathematically
  derived bound).
- Detector thresholds are frozen experimental hypotheses, not optimized or validated thresholds.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Deque
from collections import deque
import numpy as np

from navris.eskf import (
    ESKF,
    NominalState,
    STATE_DIM,
    IDX_VEL,
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


@dataclass
class ZUPTDetectorConfig:
    """
    Configuration parameters for causal stationary detector.
    Pre-declared and frozen before benchmark evaluation as experimental hypotheses
    (not optimized or post-hoc validated thresholds).
    """
    window_size: int = 8                   # 8 samples (0.8 s at 10 Hz)
    min_window_samples: int = 4            # Minimum samples required to compute trailing statistics
    dwell_samples: int = 5                 # 5 consecutive entry-satisfying samples (0.5 s) to enter
    gravity_nominal: float = 9.80665       # Nominal gravity (m/s^2)

    # Stricter Entry Thresholds (all must hold)
    acc_inst_dev_entry: float = 0.35       # | ||f_k|| - g | < 0.35 m/s^2
    acc_mean_dev_entry: float = 0.25       # | mean(||f||) - g | < 0.25 m/s^2
    acc_std_entry: float = 0.18            # std(||f||) < 0.18 m/s^2
    gyr_inst_entry: float = 0.055          # ||omega_k|| < 0.055 rad/s (~3.1 deg/s)
    gyr_mean_entry: float = 0.045          # mean(||omega||) < 0.045 rad/s (~2.6 deg/s)
    gyr_std_entry: float = 0.030           # std(||omega||) < 0.030 rad/s

    # Sensitive Exit Thresholds (any triggers immediate exit)
    acc_inst_dev_exit: float = 0.45        # | ||f_k|| - g | > 0.45 m/s^2
    acc_mean_dev_exit: float = 0.40        # | mean(||f||) - g | > 0.40 m/s^2
    acc_std_exit: float = 0.30             # std(||f||) > 0.30 m/s^2
    gyr_inst_exit: float = 0.080           # ||omega_k|| > 0.080 rad/s (~4.6 deg/s)
    gyr_mean_exit: float = 0.075           # mean(||omega||) > 0.075 rad/s (~4.3 deg/s)
    gyr_std_exit: float = 0.050            # std(||omega||) > 0.050 rad/s


class CausalStationaryDetector:
    """
    Strictly causal trailing-window stationary detector with hysteresis and dwell time.
    Uses only IMU samples received up to the current timestamp tk.
    """

    def __init__(self, config: Optional[ZUPTDetectorConfig] = None):
        self.config = config if config is not None else ZUPTDetectorConfig()
        self.acc_window: Deque[np.ndarray] = deque(maxlen=self.config.window_size)
        self.gyr_window: Deque[np.ndarray] = deque(maxlen=self.config.window_size)
        
        self.is_stationary: bool = False
        self.consecutive_entry_hits: int = 0
        self.total_stationary_updates: int = 0

    def update(self, f_meas_b: np.ndarray, omega_meas_b: np.ndarray) -> bool:
        """
        Updates detector state with a new causal IMU sample.
        
        Args:
            f_meas_b: Body-frame specific force [ax, ay, az] (m/s^2)
            omega_meas_b: Body-frame angular velocity [gx, gy, gz] (rad/s)
            
        Returns:
            is_stationary: True if vehicle is determined to be stationary at this sample.
        """
        f = np.asarray(f_meas_b, dtype=np.float64)
        w = np.asarray(omega_meas_b, dtype=np.float64)
        
        self.acc_window.append(f)
        self.gyr_window.append(w)
        
        # Instantaneous metrics
        f_norm_k = float(np.linalg.norm(f))
        w_norm_k = float(np.linalg.norm(w))
        acc_dev_k = abs(f_norm_k - self.config.gravity_nominal)

        # If window does not have enough samples to compute statistics, cannot be stationary yet
        if len(self.acc_window) < self.config.min_window_samples:
            self.is_stationary = False
            self.consecutive_entry_hits = 0
            return False

        # Compute trailing window statistics
        acc_norms = np.array([np.linalg.norm(a) for a in self.acc_window])
        gyr_norms = np.array([np.linalg.norm(g) for g in self.gyr_window])

        acc_mean = float(np.mean(acc_norms))
        acc_std = float(np.std(acc_norms))
        acc_dev = abs(acc_mean - self.config.gravity_nominal)

        gyr_mean = float(np.mean(gyr_norms))
        gyr_std = float(np.std(gyr_norms))

        if self.is_stationary:
            # Check exit conditions (any condition met triggers immediate exit)
            exit_triggered = (
                acc_dev_k > self.config.acc_inst_dev_exit or
                acc_dev > self.config.acc_mean_dev_exit or
                acc_std > self.config.acc_std_exit or
                w_norm_k > self.config.gyr_inst_exit or
                gyr_mean > self.config.gyr_mean_exit or
                gyr_std > self.config.gyr_std_exit
            )
            if exit_triggered:
                self.is_stationary = False
                self.consecutive_entry_hits = 0
            else:
                self.total_stationary_updates += 1
        else:
            # Check entry conditions (all must hold)
            entry_satisfied = (
                acc_dev_k < self.config.acc_inst_dev_entry and
                acc_dev < self.config.acc_mean_dev_entry and
                acc_std < self.config.acc_std_entry and
                w_norm_k < self.config.gyr_inst_entry and
                gyr_mean < self.config.gyr_mean_entry and
                gyr_std < self.config.gyr_std_entry
            )
            if entry_satisfied:
                self.consecutive_entry_hits += 1
                if self.consecutive_entry_hits >= self.config.dwell_samples:
                    self.is_stationary = True
                    self.total_stationary_updates += 1
            else:
                self.consecutive_entry_hits = 0

        return self.is_stationary

    def reset(self) -> None:
        """Resets detector internal state."""
        self.acc_window.clear()
        self.gyr_window.clear()
        self.is_stationary = False
        self.consecutive_entry_hits = 0
        self.total_stationary_updates = 0


# Fixed physical measurement noise standard deviation (5 cm/s isotropic)
DEFAULT_SIGMA_ZUPT = 0.05  # m/s
DEFAULT_R_ZUPT = np.diag([DEFAULT_SIGMA_ZUPT**2, DEFAULT_SIGMA_ZUPT**2, DEFAULT_SIGMA_ZUPT**2])
ZUPT_CHI2_THRESHOLD = 16.27  # 3 DOF at 99.9% confidence


def apply_zupt_update(
    eskf: ESKF,
    R_zupt: Optional[np.ndarray] = None,
    chi2_threshold: float = ZUPT_CHI2_THRESHOLD,
    timestamp: Optional[float] = None
) -> Tuple[bool, InnovationRecord]:
    """
    Applies a causal Zero-Velocity Update (ZUPT) to the ESKF state.
    
    Measurement model:
        z = [0, 0, 0]^T (m/s) in navigation frame (ENU)
        h(x) = v_hat^n
        residual = z - h(x) = -v_hat^n
        H = [0_{3x3}, I_3, 0_{3x9}]
        
    Args:
        eskf: Frozen ESKF instance
        R_zupt: 3x3 fixed measurement covariance matrix (default: diag(0.0025, 0.0025, 0.0025))
        chi2_threshold: Chi-square innovation gating threshold (default: 16.27)
        timestamp: Measurement timestamp
        
    Returns:
        (accepted, innovation_record)
    """
    t = timestamp if timestamp is not None else eskf.state.t
    z_meas = np.zeros(3, dtype=np.float64)
    pred_vel = eskf.state.v.copy()
    residual = z_meas - pred_vel  # -v_hat

    # 1. Measurement Jacobian H for velocity error state: H = [0, I, 0, 0, 0]
    H = np.zeros((3, STATE_DIM), dtype=np.float64)
    H[0:3, IDX_VEL] = np.eye(3, dtype=np.float64)

    # 2. Measurement noise covariance R
    R = np.asarray(R_zupt if R_zupt is not None else DEFAULT_R_ZUPT, dtype=np.float64)
    assert R.shape == (3, 3), f"R_zupt must be (3, 3), got {R.shape}"

    # 3. Kalman Gain and Innovation Covariance
    K, S = compute_kalman_gain(eskf.P, H, R)

    # 4. Chi-square gating
    accepted, d2 = gate_measurement(residual, S, chi2_threshold)

    record = InnovationRecord(
        timestamp=t,
        sensor_type="zupt",
        raw_meas=z_meas,
        pred_meas=pred_vel,
        residual=residual,
        S_cov=S,
        mahalanobis_sq=d2,
        threshold=chi2_threshold,
        accepted=accepted,
        is_new_fix=True
    )
    eskf.innovations.append(record)

    if not accepted:
        return False, record

    # 5. Joseph-form covariance update
    P_post = joseph_covariance_update(eskf.P, K, H, R)

    # 6. Error state calculation: delta_x = K * r
    delta_x = K @ residual

    # 7. Multiplicative quaternion error injection and covariance reset
    eskf.state, eskf.P = inject_and_reset(eskf.state, delta_x, P_post)

    return True, record
