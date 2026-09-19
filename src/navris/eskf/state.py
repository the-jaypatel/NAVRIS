"""
NAVRIS ESKF State Definitions.

Conventions:
- Navigation Frame: Local Cartesian ENU (X=East, Y=North, Z=Up).
- Body Frame: Raw smartphone IMU sensor frame.
- Nominal State:
    p: position in ENU (m) [3]
    v: velocity in ENU (m/s) [3]
    q: scalar-first Hamilton unit quaternion [qw, qx, qy, qz] representing Body-to-Navigation C_b^n(q) [4]
    ba: accelerometer bias in Body frame (m/s^2) [3]
    bg: gyroscope bias in Body frame (rad/s) [3]
  Total nominal dimensions: 16 (quaternion over-parameterization).

- Error State (15 dimensions, fixed ordering):
    0:3   delta_p^n:  position error in navigation frame (m)
    3:6   delta_v^n:  velocity error in navigation frame (m/s)
    6:9   delta_theta^n: attitude error in navigation frame (rad)
    9:12  delta_ba^b: accelerometer bias error in body frame (m/s^2)
    12:15 delta_bg^b: gyroscope bias error in body frame (rad/s)
"""

from dataclasses import dataclass, field
import numpy as np
from typing import Optional

# Error state dimension and index slices
STATE_DIM = 15
NOMINAL_DIM = 16

IDX_POS = slice(0, 3)
IDX_VEL = slice(3, 6)
IDX_ATT = slice(6, 9)
IDX_ACC_BIAS = slice(9, 12)
IDX_GYR_BIAS = slice(12, 15)


@dataclass
class NominalState:
    """Nominal navigation state containing true physical estimates."""
    t: float = 0.0
    p: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    v: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    q: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    ba: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    bg: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def __post_init__(self):
        self.p = np.asarray(self.p, dtype=np.float64)
        self.v = np.asarray(self.v, dtype=np.float64)
        self.q = np.asarray(self.q, dtype=np.float64)
        self.ba = np.asarray(self.ba, dtype=np.float64)
        self.bg = np.asarray(self.bg, dtype=np.float64)
        assert self.p.shape == (3,), f"p must have shape (3,), got {self.p.shape}"
        assert self.v.shape == (3,), f"v must have shape (3,), got {self.v.shape}"
        assert self.q.shape == (4,), f"q must have shape (4,), got {self.q.shape}"
        assert self.ba.shape == (3,), f"ba must have shape (3,), got {self.ba.shape}"
        assert self.bg.shape == (3,), f"bg must have shape (3,), got {self.bg.shape}"

    def copy(self) -> "NominalState":
        return NominalState(
            t=float(self.t),
            p=self.p.copy(),
            v=self.v.copy(),
            q=self.q.copy(),
            ba=self.ba.copy(),
            bg=self.bg.copy()
        )


@dataclass
class ESKFConfig:
    """
    Configuration parameters for 15-state ESKF.
    All noise terms are continuous power spectral densities (PSD) or standard deviations.
    """
    # Continuous noise parameters (PSD)
    # sigma_acc: accelerometer white noise standard deviation (m/s^2 / sqrt(Hz))
    sigma_acc: float = 0.1
    # sigma_gyr: gyroscope white noise standard deviation (rad/s / sqrt(Hz))
    sigma_gyr: float = 0.01
    # sigma_acc_bias: accelerometer bias random walk PSD (m/s^3 / sqrt(Hz))
    sigma_acc_bias: float = 1e-4
    # sigma_gyr_bias: gyroscope bias random walk PSD (rad/s^2 / sqrt(Hz))
    sigma_gyr_bias: float = 1e-5

    # Measurement noise standard deviations
    gnss_pos_std_horiz: float = 2.0  # meters
    gnss_pos_std_vert: float = 4.0   # meters
    gnss_vel_std_horiz: float = 0.2  # m/s
    gnss_vel_std_vert: float = 0.4   # m/s

    # Chi-square gating thresholds
    # Chi-square distribution with 3 DOF:
    # 95% = 7.815, 99% = 11.345, 99.9% = 16.266
    chi2_threshold_pos: float = 16.27
    chi2_threshold_vel: float = 16.27

    # Timestamp sanity limits
    max_dt_gap: float = 1.0  # seconds; step sizes larger than this are rejected / treated as gaps
    min_dt: float = 1e-6     # seconds
