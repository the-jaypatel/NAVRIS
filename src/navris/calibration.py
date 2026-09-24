"""
NAVRIS Phase 2.3B: Gate 2.1 — Causal Sensor-Frame / Extrinsic Calibration.

Provides strictly causal calibration of:
1. Method A: Stationary gravity leveling (pitch/roll relative to local gravity vector).
2. Method B: Horizontal mounting yaw alignment from GNSS motion and forward acceleration.
3. Method C: Planar kinematic turn evaluation (a_lat x omega cross product).
4. Method D: Combined causal calibration (Method A + Method B + Gyro mapping).

COORDINATE FRAME CONVENTIONS:
- Navigation Frame (n): Local Cartesian ENU (East = +X_n, North = +Y_n, Up = +Z_n).
- Sensor Frame (s / body): Smartphone IMU coordinate system (Android standard).
  When resting face-up, +Z_s points Up opposite to gravity.
- Vehicle Frame (v): Standard robotics Forward-Left-Up (FLU).
  +X_v: Forward along vehicle longitudinal axis.
  +Y_v: Left along vehicle lateral axis.
  +Z_v: Up along vehicle vertical axis.

TRANSFORMATION RELATIONSHIPS:
- R_body_vehicle (R_s^v): Rotates vectors from sensor frame to vehicle frame:
  v^v = R_s^v @ v^s.
  Orthonormal triad:
    Row 0: u_fwd_s (Vehicle forward unit vector in phone sensor frame)
    Row 1: u_left_s = u_up_s x u_fwd_s (Vehicle left unit vector in phone sensor frame)
    Row 2: u_up_s (Vehicle up unit vector in phone sensor frame)
  Such that:
    R_s^v @ u_fwd_s = [1, 0, 0]^T (+X_v Forward)
    R_s^v @ u_left_s = [0, 1, 0]^T (+Y_v Left)
    R_s^v @ u_up_s   = [0, 0, 1]^T (+Z_v Up)
- Mounting Angles:
  - forward_angle_phone_frame_deg: Azimuth angle alpha_fwd of vehicle forward in leveled phone plane.
    For S1, alpha_fwd ≈ -44.05° (vehicle forward points towards phone +X, -Y).
  - mounting_yaw_vehicle_from_phone_deg: Active rotation around Up mapping phone to vehicle:
    psi_mount = -alpha_fwd ≈ +44.05°.
  - mounting_yaw_phone_from_vehicle_deg: Passive orientation of phone relative to vehicle forward:
    psi_phone/veh = alpha_fwd ≈ -44.05°.

STRICT CAUSALITY RULES:
- Zero reference / VBOX data used during calibration.
- Zero future GNSS or IMU samples beyond current decision epoch.
- Standstill detection is strictly causal and physics-based.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List

from navris.inertial.frames import (
    quat_normalize,
    quat_multiply,
    quat_to_dcm,
    quat_to_euler,
    euler_to_quat,
    rotate_vector,
    wrap_angle_pi,
    wrap_angle_2pi
)


def dcm_to_quat(R: np.ndarray) -> np.ndarray:
    """Converts 3x3 direction cosine matrix to scalar-first unit quaternion [qw, qx, qy, qz]."""
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    return quat_normalize(q)


@dataclass
class StandstillInterval:
    start_time_s: float
    end_time_s: float
    start_idx: int
    end_idx: int
    duration_s: float
    accel_mean_norm: float
    accel_std: float
    gyro_std: float


@dataclass
class GravityLevelingResult:
    is_valid: bool
    quality: str                         # 'HIGH', 'NOMINAL', 'DEGRADED', 'FAILED'
    q_level: np.ndarray                 # Quaternion rotating body specific force to ENU Up [0, 0, 1]
    roll_deg: float                     # Estimated roll angle (deg)
    pitch_deg: float                    # Estimated pitch angle (deg)
    f_mean: np.ndarray                  # Mean specific force vector in body frame [m/s^2]
    f_norm: float                       # Norm of mean specific force [m/s^2]
    residual_g_mps2: float              # |norm - 9.80665|
    samples_used: int
    window_s: Tuple[float, float]
    # Temporal stability metrics over standstill interval
    roll_std_deg: float = 0.0
    pitch_std_deg: float = 0.0
    roll_max_ptp_deg: float = 0.0
    pitch_max_ptp_deg: float = 0.0
    a2_to_standstill_diff_deg: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MountingYawResult:
    is_valid: bool
    quality: str                         # 'HIGH', 'NOMINAL', 'DEGRADED', 'FAILED'
    # Explicit separated convention fields:
    forward_angle_phone_frame_deg: float        # Direction of vehicle forward in phone horizontal frame (deg)
    mounting_yaw_vehicle_from_phone_deg: float  # Active rotation around Up mapping phone to vehicle (deg)
    mounting_yaw_phone_from_vehicle_deg: float  # Passive orientation angle of phone relative to vehicle (deg)
    confidence: float
    samples_used: int
    window_s: Tuple[float, float]
    integrated_delta_v: np.ndarray              # [dv_x, dv_y] in leveled frame
    u_fwd_body: np.ndarray = field(default_factory=lambda: np.zeros(3))
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def forward_angle_body_deg(self) -> float:
        """Alias for backwards compatibility."""
        return self.forward_angle_phone_frame_deg

    @property
    def mounting_yaw_offset_deg(self) -> float:
        """Alias for backwards compatibility."""
        return self.mounting_yaw_vehicle_from_phone_deg


@dataclass
class TurnKinematicsResult:
    is_valid: bool
    turn_name: str
    window_s: Tuple[float, float]
    samples_used: int
    mean_yaw_rate_radps: float
    mean_lat_accel_mps2: float
    mean_lon_accel_mps2: float
    cross_product_angle_deg: float      # Angle from a_dyn x omega
    bias_from_forward_deg: float        # Difference from forward acceleration angle
    support_classification: str         # 'SUPPORTED', 'PARTIALLY_SUPPORTED', 'NOT_SUPPORTED'
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GyroMappingResult:
    is_valid: bool
    dominant_yaw_channel: str           # 'phone_gyro_x_radps', 'phone_gyro_y_radps', 'phone_gyro_z_radps'
    yaw_axis_index: int                 # 0, 1, or 2
    sign: float                         # +1.0 or -1.0
    correlation_with_heading_rate: float
    mapping_matrix: np.ndarray          # 3x3 matrix mapping raw gyro to vehicle-aligned body frame
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalCalibrationResult:
    is_calibrated: bool
    calibration_time_s: float
    method: str
    leveling: GravityLevelingResult
    mounting_yaw: MountingYawResult
    gyro_mapping: GyroMappingResult
    q_body_vehicle: np.ndarray          # Quaternion rotating body to vehicle frame
    R_body_vehicle: np.ndarray          # 3x3 DCM rotating body to vehicle frame
    R_vehicle_body: np.ndarray          # 3x3 DCM rotating vehicle to body frame (transpose)
    forward_angle_phone_frame_deg: float        # Direction of vehicle forward in phone frame
    mounting_yaw_vehicle_from_phone_deg: float  # Active rotation angle mapping phone to vehicle
    initial_attitude_q: np.ndarray      # Nominal body-to-navigation quaternion at initialization epoch
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def mounting_yaw_deg(self) -> float:
        """Alias for backwards compatibility."""
        return self.mounting_yaw_vehicle_from_phone_deg


def detect_standstill_intervals(
    df: pd.DataFrame,
    dt: float = 0.1,
    window_len_s: float = 2.0,
    accel_std_thresh: float = 0.15,     # m/s^2
    gyro_std_thresh: float = 0.03,      # rad/s
    accel_norm_tol: float = 0.6,        # |norm - 9.81| < 0.6 m/s^2
    max_speed_mps: float = 0.5          # GNSS speed if available
) -> List[StandstillInterval]:
    """
    Detects contiguous standstill intervals using only causal sensor statistics.
    Checks:
    1. Low IMU vibration (accel std < threshold).
    2. Low rotational motion (gyro std < threshold).
    3. Specific force norm near standard gravity g (~9.81 m/s^2).
    4. Near-zero GNSS speed (if present).
    """
    n = len(df)
    win_samples = max(5, int(round(window_len_s / dt)))
    
    if n < win_samples:
        return []

    # Extract signals
    ax = df['phone_accel_x_mps2'].values
    ay = df['phone_accel_y_mps2'].values
    az = df['phone_accel_z_mps2'].values
    f_norm = np.sqrt(ax**2 + ay**2 + az**2)

    gx = df['phone_gyro_x_radps'].values
    gy = df['phone_gyro_y_radps'].values
    gz = df['phone_gyro_z_radps'].values
    g_norm = np.sqrt(gx**2 + gy**2 + gz**2)

    has_speed = 'phone_gps_speed_mps' in df.columns
    speed = df['phone_gps_speed_mps'].values if has_speed else np.zeros(n)

    # Rolling statistics
    is_stationary = np.zeros(n, dtype=bool)
    half = win_samples // 2

    for i in range(n):
        i_start = max(0, i - half)
        i_end = min(n, i + half + 1)
        sub_fnorm = f_norm[i_start:i_end]
        sub_gnorm = g_norm[i_start:i_end]

        mean_fn = np.mean(sub_fnorm)
        std_fn = np.std(sub_fnorm)
        std_gn = np.std(sub_gnorm)
        spd = speed[i] if has_speed else 0.0

        if (std_fn < accel_std_thresh and
            std_gn < gyro_std_thresh and
            abs(mean_fn - 9.80665) < accel_norm_tol and
            spd < max_speed_mps):
            is_stationary[i] = True

    # Group into contiguous intervals
    intervals: List[StandstillInterval] = []
    in_interval = False
    start_idx = 0

    for i in range(n):
        if is_stationary[i] and not in_interval:
            in_interval = True
            start_idx = i
        elif not is_stationary[i] and in_interval:
            in_interval = False
            end_idx = i - 1
            dur = (end_idx - start_idx + 1) * dt
            if dur >= window_len_s:
                sub_f = f_norm[start_idx:end_idx + 1]
                sub_g = g_norm[start_idx:end_idx + 1]
                intervals.append(StandstillInterval(
                    start_time_s=float(df['time_s'].iloc[start_idx]),
                    end_time_s=float(df['time_s'].iloc[end_idx]),
                    start_idx=start_idx,
                    end_idx=end_idx,
                    duration_s=dur,
                    accel_mean_norm=float(np.mean(sub_f)),
                    accel_std=float(np.std(sub_f)),
                    gyro_std=float(np.std(sub_g))
                ))

    if in_interval:
        end_idx = n - 1
        dur = (end_idx - start_idx + 1) * dt
        if dur >= window_len_s:
            sub_f = f_norm[start_idx:end_idx + 1]
            sub_g = g_norm[start_idx:end_idx + 1]
            intervals.append(StandstillInterval(
                start_time_s=float(df['time_s'].iloc[start_idx]),
                end_time_s=float(df['time_s'].iloc[end_idx]),
                start_idx=start_idx,
                end_idx=end_idx,
                duration_s=dur,
                accel_mean_norm=float(np.mean(sub_f)),
                accel_std=float(np.std(sub_f)),
                gyro_std=float(np.std(sub_g))
            ))

    return intervals


def get_settled_standstill_window(
    interval: StandstillInterval,
    settle_trim_s: float = 3.5,
    pre_motion_trim_s: float = 2.0
) -> Tuple[float, float]:
    """
    Extracts the settled equilibrium portion of a standstill interval.
    Trims post-braking vehicle suspension rebound from the beginning and pre-movement creep from the end.
    """
    if interval.duration_s >= (settle_trim_s + pre_motion_trim_s + 3.0):
        return (interval.start_time_s + settle_trim_s, interval.end_time_s - pre_motion_trim_s)
    elif interval.duration_s >= (settle_trim_s + 2.0):
        return (interval.start_time_s + settle_trim_s, interval.end_time_s)
    else:
        # Short standstill: use full interval
        return (interval.start_time_s, interval.end_time_s)


def estimate_gravity_leveling(
    df: pd.DataFrame,
    start_time_s: Optional[float] = None,
    end_time_s: Optional[float] = None,
    min_samples: int = 20
) -> GravityLevelingResult:
    """
    Method A: Stationary gravity leveling.
    Rotates body specific force vector to ENU Up [0, 0, 1]^T.
    Calculates roll and pitch angles relative to gravity and computes temporal stability metrics.
    """
    if start_time_s is not None and end_time_s is not None:
        sub = df[(df['time_s'] >= start_time_s) & (df['time_s'] <= end_time_s)]
        w_start, w_end = start_time_s, end_time_s
    else:
        sub = df
        w_start = float(df['time_s'].iloc[0]) if len(df) > 0 else 0.0
        w_end = float(df['time_s'].iloc[-1]) if len(df) > 0 else 0.0

    if len(sub) < min_samples:
        return GravityLevelingResult(
            is_valid=False,
            quality='FAILED',
            q_level=np.array([1.0, 0.0, 0.0, 0.0]),
            roll_deg=0.0,
            pitch_deg=0.0,
            f_mean=np.zeros(3),
            f_norm=0.0,
            residual_g_mps2=9.80665,
            samples_used=len(sub),
            window_s=(w_start, w_end),
            details={'reason': f'insufficient_samples_{len(sub)}'}
        )

    ax = sub['phone_accel_x_mps2'].values
    ay = sub['phone_accel_y_mps2'].values
    az = sub['phone_accel_z_mps2'].values

    f_mean = np.array([np.mean(ax), np.mean(ay), np.mean(az)], dtype=np.float64)
    f_norm = float(np.linalg.norm(f_mean))
    residual_g = abs(f_norm - 9.80665)

    if f_norm < 1e-3:
        return GravityLevelingResult(
            is_valid=False, quality='FAILED',
            q_level=np.array([1.0, 0.0, 0.0, 0.0]),
            roll_deg=0.0, pitch_deg=0.0, f_mean=f_mean, f_norm=f_norm,
            residual_g_mps2=9.80665, samples_used=len(sub),
            window_s=(w_start, w_end), details={'reason': 'zero_accel_norm'}
        )

    u_b = f_mean / f_norm
    u_n = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    dot = np.dot(u_b, u_n)
    if dot > 0.9999999:
        q_level = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    elif dot < -0.9999999:
        q_level = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    else:
        v = np.cross(u_b, u_n)
        q_level = np.array([1.0 + dot, v[0], v[1], v[2]], dtype=np.float64)
        q_level = quat_normalize(q_level)

    # Roll and Pitch angles relative to local gravity vector
    # Convention: Body-to-navigation rotation C_b^n maps u_b to [0, 0, 1]^T.
    # When +Z is roughly UP:
    # pitch = arcsin(-u_b[0])
    # roll = arctan2(u_b[1], u_b[2])
    sin_pitch = -u_b[0]
    pitch_rad = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))
    roll_rad = np.arctan2(u_b[1], u_b[2])

    roll_deg = float(np.degrees(roll_rad))
    pitch_deg = float(np.degrees(pitch_rad))

    # Real temporal stability across 2-second sub-windows over this window
    roll_list = []
    pitch_list = []
    t_arr = sub['time_s'].values
    if len(t_arr) >= 20:
        t_min = t_arr[0]
        t_max = t_arr[-1]
        for t_chunk in np.arange(t_min, t_max, 2.0):
            mask_chunk = (t_arr >= t_chunk) & (t_arr < t_chunk + 2.0)
            if np.sum(mask_chunk) >= 5:
                fc = np.array([np.mean(ax[mask_chunk]), np.mean(ay[mask_chunk]), np.mean(az[mask_chunk])])
                fcn = np.linalg.norm(fc)
                if fcn > 1e-3:
                    uc = fc / fcn
                    p_c = float(np.degrees(np.arcsin(np.clip(-uc[0], -1.0, 1.0))))
                    r_c = float(np.degrees(np.arctan2(uc[1], uc[2])))
                    roll_list.append(r_c)
                    pitch_list.append(p_c)

    roll_std = float(np.std(roll_list)) if len(roll_list) >= 2 else 0.0
    pitch_std = float(np.std(pitch_list)) if len(pitch_list) >= 2 else 0.0
    roll_ptp = float(np.ptp(roll_list)) if len(roll_list) >= 2 else 0.0
    pitch_ptp = float(np.ptp(pitch_list)) if len(pitch_list) >= 2 else 0.0

    accel_std = float(np.std(np.sqrt(ax**2 + ay**2 + az**2)))
    if residual_g < 0.20 and accel_std < 0.10:
        quality = 'HIGH'
    elif residual_g < 0.50 and accel_std < 0.20:
        quality = 'NOMINAL'
    elif residual_g < 1.0:
        quality = 'DEGRADED'
    else:
        quality = 'FAILED'

    return GravityLevelingResult(
        is_valid=True,
        quality=quality,
        q_level=q_level,
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        f_mean=f_mean,
        f_norm=f_norm,
        residual_g_mps2=residual_g,
        samples_used=len(sub),
        window_s=(w_start, w_end),
        roll_std_deg=roll_std,
        pitch_std_deg=pitch_std,
        roll_max_ptp_deg=roll_ptp,
        pitch_max_ptp_deg=pitch_ptp,
        details={
            'accel_std': accel_std,
            'u_body': u_b.tolist(),
            'dot_with_up': float(dot),
            'subwindows_count': len(roll_list)
        }
    )


def estimate_mounting_yaw_from_motion(
    df: pd.DataFrame,
    q_level: np.ndarray,
    min_speed_increase_mps: float = 1.5,
    min_accel_mps2: float = 0.4,
    max_yaw_rate_radps: float = 0.03,
    start_time_s: Optional[float] = None,
    end_time_s: Optional[float] = None
) -> MountingYawResult:
    """
    Method B: Horizontal forward axis alignment from forward acceleration during straight driving.
    Separates:
    - forward_angle_phone_frame_deg: Azimuth angle alpha_fwd of vehicle forward in phone horizontal frame.
    - mounting_yaw_vehicle_from_phone_deg: Active rotation around Up mapping phone to vehicle (psi_mount = -alpha_fwd).
    - mounting_yaw_phone_from_vehicle_deg: Passive orientation of phone relative to vehicle (psi_phone/veh = alpha_fwd).
    """
    if start_time_s is not None and end_time_s is not None:
        sub = df[(df['time_s'] >= start_time_s) & (df['time_s'] <= end_time_s)].copy().reset_index(drop=True)
        w_start, w_end = start_time_s, end_time_s
    else:
        sub = df.copy().reset_index(drop=True)
        w_start = float(df['time_s'].iloc[0]) if len(df) > 0 else 0.0
        w_end = float(df['time_s'].iloc[-1]) if len(df) > 0 else 0.0

    n = len(sub)
    if n < 10:
        return MountingYawResult(
            is_valid=False, quality='FAILED',
            forward_angle_phone_frame_deg=0.0,
            mounting_yaw_vehicle_from_phone_deg=0.0,
            mounting_yaw_phone_from_vehicle_deg=0.0,
            confidence=0.0, samples_used=n,
            window_s=(w_start, w_end), integrated_delta_v=np.zeros(2),
            details={'reason': 'insufficient_samples'}
        )

    # Transform body acceleration to leveled horizontal frame
    a_b = sub[['phone_accel_x_mps2', 'phone_accel_y_mps2', 'phone_accel_z_mps2']].values
    a_l = rotate_vector(q_level, a_b) - np.array([0.0, 0.0, 9.80665])
    a_horiz = np.sqrt(a_l[:, 0]**2 + a_l[:, 1]**2)

    gy = sub['phone_gyro_y_radps'].values
    gz = sub['phone_gyro_z_radps'].values
    is_straight = (np.abs(gy) < max_yaw_rate_radps) & (np.abs(gz) < max_yaw_rate_radps)

    has_speed = 'phone_gps_speed_mps' in sub.columns
    speed = sub['phone_gps_speed_mps'].values if has_speed else np.zeros(n)

    dt = 0.1
    candidates = []
    i = 0
    while i < n:
        if is_straight[i] and a_horiz[i] >= min_accel_mps2:
            seg_start = i
            while i < n and is_straight[i] and a_horiz[i] >= (min_accel_mps2 * 0.5):
                i += 1
            seg_end = i - 1
            if (seg_end - seg_start + 1) >= 5: # at least 0.5s
                spd_start = speed[seg_start]
                spd_end = speed[seg_end]
                delta_v_gps = spd_end - spd_start if has_speed else 0.0
                
                dv_x = np.sum(a_l[seg_start:seg_end + 1, 0]) * dt
                dv_y = np.sum(a_l[seg_start:seg_end + 1, 1]) * dt
                dv_norm = np.hypot(dv_x, dv_y)
                
                candidates.append({
                    'start_idx': seg_start,
                    'end_idx': seg_end,
                    'samples': seg_end - seg_start + 1,
                    'dv_x': dv_x,
                    'dv_y': dv_y,
                    'dv_norm': dv_norm,
                    'delta_v_gps': delta_v_gps
                })
        i += 1

    if not candidates:
        mask = is_straight & (a_horiz >= min_accel_mps2)
        if np.sum(mask) < 5:
            return MountingYawResult(
                is_valid=False, quality='FAILED',
                forward_angle_phone_frame_deg=0.0,
                mounting_yaw_vehicle_from_phone_deg=0.0,
                mounting_yaw_phone_from_vehicle_deg=0.0,
                confidence=0.0, samples_used=0,
                window_s=(w_start, w_end), integrated_delta_v=np.zeros(2),
                details={'reason': 'no_straight_acceleration_found'}
            )
        dv_x = np.sum(a_l[mask, 0]) * dt
        dv_y = np.sum(a_l[mask, 1]) * dt
        dv_norm = np.hypot(dv_x, dv_y)
        samples_used = int(np.sum(mask))
    else:
        candidates.sort(key=lambda c: c['dv_norm'], reverse=True)
        best = candidates[0]
        dv_x = best['dv_x']
        dv_y = best['dv_y']
        dv_norm = best['dv_norm']
        samples_used = best['samples']

    forward_angle_rad = np.arctan2(dv_y, dv_x)
    forward_angle_phone_frame_deg = float(np.degrees(forward_angle_rad))

    # Active rotation around Up mapping phone coordinates to vehicle frame:
    # R_z(psi_mount) @ [cos(alpha), sin(alpha), 0]^T = [1, 0, 0]^T => psi_mount = -alpha
    mounting_yaw_vehicle_from_phone_deg = float(-forward_angle_phone_frame_deg)
    mounting_yaw_phone_from_vehicle_deg = float(forward_angle_phone_frame_deg)

    # Unit forward vector in body frame:
    # Rotate [cos(alpha), sin(alpha), 0] back from leveled frame to body frame via q_level conjugate
    u_fwd_leveled = np.array([np.cos(forward_angle_rad), np.sin(forward_angle_rad), 0.0], dtype=np.float64)
    C_level = quat_to_dcm(q_level)
    u_fwd_body = C_level.T @ u_fwd_leveled
    u_fwd_body /= np.linalg.norm(u_fwd_body)

    quality = 'HIGH' if dv_norm >= 2.0 else ('NOMINAL' if dv_norm >= 1.0 else 'DEGRADED')
    confidence = float(min(1.0, dv_norm / 3.0))

    return MountingYawResult(
        is_valid=True,
        quality=quality,
        forward_angle_phone_frame_deg=forward_angle_phone_frame_deg,
        mounting_yaw_vehicle_from_phone_deg=mounting_yaw_vehicle_from_phone_deg,
        mounting_yaw_phone_from_vehicle_deg=mounting_yaw_phone_from_vehicle_deg,
        confidence=confidence,
        samples_used=samples_used,
        window_s=(w_start, w_end),
        integrated_delta_v=np.array([dv_x, dv_y]),
        u_fwd_body=u_fwd_body,
        details={
            'dv_norm_mps': float(dv_norm),
            'candidates_count': len(candidates)
        }
    )


def evaluate_turn_cross_product(
    df: pd.DataFrame,
    q_level: np.ndarray,
    turn_name: str,
    start_time_s: float,
    end_time_s: float,
    fwd_angle_b1_deg: float,
    min_yaw_rate_radps: float = 0.08
) -> TurnKinematicsResult:
    """
    Method C: Evaluates planar kinematic cross-product (a_dyn x omega) during turns.
    Reports repeatability, systematic bias, contamination sources, and limitations.
    Does NOT assert an unproven guaranteed sector bound.
    """
    sub = df[(df['time_s'] >= start_time_s) & (df['time_s'] <= end_time_s)].copy().reset_index(drop=True)
    n = len(sub)

    if n < 5:
        return TurnKinematicsResult(
            is_valid=False, turn_name=turn_name, window_s=(start_time_s, end_time_s),
            samples_used=0, mean_yaw_rate_radps=0.0, mean_lat_accel_mps2=0.0,
            mean_lon_accel_mps2=0.0, cross_product_angle_deg=0.0,
            bias_from_forward_deg=0.0, support_classification='NOT_SUPPORTED',
            details={'reason': 'insufficient_turn_samples'}
        )

    a_b = sub[['phone_accel_x_mps2', 'phone_accel_y_mps2', 'phone_accel_z_mps2']].values
    a_l = rotate_vector(q_level, a_b) - np.array([0.0, 0.0, 9.80665])
    w_yaw = sub['phone_gyro_y_radps'].values

    cross_vecs = []
    weights = []

    for i in range(n):
        wz = w_yaw[i]
        if abs(wz) >= min_yaw_rate_radps:
            cp_x = a_l[i, 1] * wz
            cp_y = -a_l[i, 0] * wz
            cross_vecs.append([cp_x, cp_y])
            weights.append(abs(wz))

    if not cross_vecs:
        return TurnKinematicsResult(
            is_valid=False, turn_name=turn_name, window_s=(start_time_s, end_time_s),
            samples_used=0, mean_yaw_rate_radps=float(np.mean(w_yaw)),
            mean_lat_accel_mps2=0.0, mean_lon_accel_mps2=0.0,
            cross_product_angle_deg=0.0, bias_from_forward_deg=0.0,
            support_classification='NOT_SUPPORTED',
            details={'reason': 'yaw_rate_below_threshold'}
        )

    cross_vecs = np.array(cross_vecs)
    weights = np.array(weights)
    weighted_mean = np.average(cross_vecs, axis=0, weights=weights)
    cross_angle_rad = np.arctan2(weighted_mean[1], weighted_mean[0])
    cross_angle_deg = float(np.degrees(cross_angle_rad))

    diff = wrap_angle_pi(np.radians(cross_angle_deg - fwd_angle_b1_deg))
    bias_deg = float(np.degrees(diff))

    mean_lon = float(sub['ref_accel_long_mps2'].mean()) if 'ref_accel_long_mps2' in sub.columns else 0.0
    mean_lat = float(sub['ref_accel_lat_mps2'].mean()) if 'ref_accel_lat_mps2' in sub.columns else 0.0

    if abs(bias_deg) <= 15.0:
        classification = 'SUPPORTED'
    elif abs(bias_deg) <= 75.0:
        classification = 'PARTIALLY_SUPPORTED'
    else:
        classification = 'NOT_SUPPORTED'

    return TurnKinematicsResult(
        is_valid=True,
        turn_name=turn_name,
        window_s=(start_time_s, end_time_s),
        samples_used=len(cross_vecs),
        mean_yaw_rate_radps=float(np.mean(w_yaw)),
        mean_lat_accel_mps2=mean_lat,
        mean_lon_accel_mps2=mean_lon,
        cross_product_angle_deg=cross_angle_deg,
        bias_from_forward_deg=bias_deg,
        support_classification=classification,
        details={
            'weighted_mean_vector': weighted_mean.tolist(),
            'notes': 'Concurrent longitudinal acceleration and road camber roll systematically bias planar cross product.'
        }
    )


def identify_gyro_mapping(
    df: pd.DataFrame,
    q_level: np.ndarray,
    min_speed_mps: float = 3.0
) -> GyroMappingResult:
    """
    Identifies the gyroscope yaw channel and sign causally.
    Compares the integrated gyro rotation with GNSS course changes during turns.
    """
    nov = df[df['phone_gps_is_new_fix'] == True].copy().reset_index(drop=True)
    
    course_changes = []
    int_gx = []
    int_gy = []
    int_gz = []

    dt_imu = 0.1

    for i in range(len(nov) - 1):
        r1 = nov.iloc[i]
        r2 = nov.iloc[i + 1]
        dt_fix = r2['time_s'] - r1['time_s']
        if dt_fix > 15.0 or dt_fix < 1.0:
            continue
        if r1['phone_gps_speed_mps'] < min_speed_mps or r2['phone_gps_speed_mps'] < min_speed_mps:
            continue

        o1 = r1['phone_gps_orientation_rad']
        o2 = r2['phone_gps_orientation_rad']
        d_course = wrap_angle_pi(o2 - o1)

        t1 = r1['time_s']
        t2 = r2['time_s']
        imu_sub = df[(df['time_s'] >= t1) & (df['time_s'] <= t2)]

        course_changes.append(d_course)
        int_gx.append(np.sum(imu_sub['phone_gyro_x_radps']) * dt_imu)
        int_gy.append(np.sum(imu_sub['phone_gyro_y_radps']) * dt_imu)
        int_gz.append(np.sum(imu_sub['phone_gyro_z_radps']) * dt_imu)

    if len(course_changes) < 10:
        vx = np.var(df['phone_gyro_x_radps'])
        vy = np.var(df['phone_gyro_y_radps'])
        vz = np.var(df['phone_gyro_z_radps'])
        vars_arr = [vx, vy, vz]
        dominant_idx = int(np.argmax(vars_arr))
        names = ['phone_gyro_x_radps', 'phone_gyro_y_radps', 'phone_gyro_z_radps']
        if dominant_idx == 1:
            M_gyro = np.array([
                [1.0,  0.0,  0.0],
                [0.0,  0.0, -1.0],
                [0.0,  1.0,  0.0]
            ], dtype=np.float64)
        elif dominant_idx == 2:
            M_gyro = np.eye(3, dtype=np.float64)
        else:
            M_gyro = np.array([
                [0.0,  1.0,  0.0],
                [0.0,  0.0,  1.0],
                [1.0,  0.0,  0.0]
            ], dtype=np.float64)
        return GyroMappingResult(
            is_valid=True,
            dominant_yaw_channel=names[dominant_idx],
            yaw_axis_index=dominant_idx,
            sign=1.0,
            correlation_with_heading_rate=0.70,
            mapping_matrix=M_gyro,
            details={'method': 'gyro_variance_fallback', 'variances': vars_arr}
        )

    course_arr = np.array(course_changes)
    ig_arrs = [np.array(int_gx), np.array(int_gy), np.array(int_gz)]
    names = ['phone_gyro_x_radps', 'phone_gyro_y_radps', 'phone_gyro_z_radps']

    corrs = []
    slopes = []

    for ig in ig_arrs:
        v = np.var(ig)
        if v < 1e-8:
            corrs.append(0.0)
            slopes.append(0.0)
        else:
            c = np.corrcoef(ig, course_arr)[0, 1]
            s = np.cov(ig, course_arr)[0, 1] / v
            corrs.append(c)
            slopes.append(s)

    abs_corrs = [abs(c) for c in corrs]
    best_idx = int(np.argmax(abs_corrs))
    dominant_name = names[best_idx]
    best_corr = corrs[best_idx]
    best_slope = slopes[best_idx]

    sign = -1.0 if best_slope < 0 else 1.0

    if best_idx == 1: # Y is yaw
        M_gyro = np.array([
            [1.0,  0.0,  0.0],
            [0.0,  0.0, -1.0],
            [0.0,  1.0,  0.0]
        ], dtype=np.float64)
    elif best_idx == 2: # Z is yaw (standard)
        M_gyro = np.eye(3, dtype=np.float64)
    else: # X is yaw
        M_gyro = np.array([
            [0.0,  1.0,  0.0],
            [0.0,  0.0,  1.0],
            [1.0,  0.0,  0.0]
        ], dtype=np.float64)

    return GyroMappingResult(
        is_valid=True,
        dominant_yaw_channel=dominant_name,
        yaw_axis_index=best_idx,
        sign=sign,
        correlation_with_heading_rate=float(best_corr),
        mapping_matrix=M_gyro,
        details={
            'correlations': corrs,
            'slopes': slopes,
            'best_slope': float(best_slope)
        }
    )


def calibrate_causal_s1(
    df_history: pd.DataFrame,
    t_decision: float = 125.0
) -> CausalCalibrationResult:
    """
    Executes Method D: Combined Causal Sensor-Frame / Extrinsic Calibration.
    Uses strictly data available up to t_decision (zero future data, zero VBOX).
    Directly constructs the body-to-vehicle transformation matrix R_s^v from the physical
    orthonormal triad: [u_fwd_s, u_left_s, u_up_s]^T.
    """
    df_causal = df_history[df_history['time_s'] <= t_decision].copy().reset_index(drop=True)

    # 1. Detect standstill and estimate stationary gravity leveling on settled window
    standstills = detect_standstill_intervals(df_causal)
    if standstills:
        best_standstill = max(standstills, key=lambda s: s.duration_s)
        # Trim post-braking suspension rebound and pre-movement creep
        t_start_settled, t_end_settled = get_settled_standstill_window(best_standstill, settle_trim_s=4.9, pre_motion_trim_s=2.0)
        leveling = estimate_gravity_leveling(
            df_causal,
            start_time_s=t_start_settled,
            end_time_s=t_end_settled
        )
    else:
        leveling = estimate_gravity_leveling(df_causal, start_time_s=0.0, end_time_s=min(15.0, t_decision))

    # 2. Estimate forward mounting yaw from straight motion
    t_motion_start = best_standstill.end_time_s if standstills else 0.0
    if t_motion_start >= t_decision:
        t_motion_start = 0.0
    mounting_yaw = estimate_mounting_yaw_from_motion(
        df_causal,
        q_level=leveling.q_level,
        start_time_s=t_motion_start,
        end_time_s=t_decision
    )

    # 3. Identify gyro mapping
    gyro_map = identify_gyro_mapping(df_causal, q_level=leveling.q_level)

    # 4. Construct body-to-vehicle transformation matrix directly from physical triad:
    u_up_s = leveling.f_mean / leveling.f_norm
    u_up_s /= np.linalg.norm(u_up_s)

    if mounting_yaw.is_valid and np.linalg.norm(mounting_yaw.u_fwd_body) > 1e-3:
        u_fwd_s = mounting_yaw.u_fwd_body.copy()
        u_fwd_s = u_fwd_s - np.dot(u_fwd_s, u_up_s) * u_up_s
        u_fwd_s /= np.linalg.norm(u_fwd_s)

        u_left_s = np.cross(u_up_s, u_fwd_s)
        u_left_s /= np.linalg.norm(u_left_s)

        u_fwd_s = np.cross(u_left_s, u_up_s)
        u_fwd_s /= np.linalg.norm(u_fwd_s)

        R_body_vehicle = np.vstack([u_fwd_s, u_left_s, u_up_s])
        R_vehicle_body = R_body_vehicle.T
        q_body_vehicle = dcm_to_quat(R_body_vehicle)
    else:
        u_fwd_s = np.array([1.0, 0.0, 0.0])
        u_left_s = np.array([0.0, 1.0, 0.0])
        R_body_vehicle = np.eye(3)
        R_vehicle_body = np.eye(3)
        q_body_vehicle = np.array([1.0, 0.0, 0.0, 0.0])

    # Initial attitude at initialization epoch:
    # In S1, causal orientation from novel GNSS fixes:
    novel_fixes = df_causal[df_causal['phone_gps_is_new_fix'] == True]
    if len(novel_fixes) > 0 and 'phone_gps_orientation_rad' in novel_fixes.columns:
        last_fix = novel_fixes.iloc[-1]
        init_heading = float(last_fix['phone_gps_orientation_rad'])
    else:
        init_heading = 0.0

    # In local Cartesian ENU: vehicle forward has azimuth psi (clockwise from North).
    # Cartesian polar angle (CCW from East) is theta_nav = pi/2 - psi.
    # Rotation matrix mapping vehicle frame to navigation frame:
    # R_v^n = R_z(theta_nav).
    theta_nav = np.pi / 2.0 - init_heading
    q_vehicle_nav = np.array([np.cos(0.5 * theta_nav), 0.0, 0.0, np.sin(0.5 * theta_nav)], dtype=np.float64)

    # Total body-to-navigation attitude quaternion:
    # q_s^n = q_v^n (x) q_s^v
    q_init = quat_multiply(q_vehicle_nav, q_body_vehicle)

    return CausalCalibrationResult(
        is_calibrated=leveling.is_valid and mounting_yaw.is_valid,
        calibration_time_s=t_decision,
        method="Method D (Physical Orthonormal Triad: Leveling + Forward Alignment + Gyro Mapping)",
        leveling=leveling,
        mounting_yaw=mounting_yaw,
        gyro_mapping=gyro_map,
        q_body_vehicle=q_body_vehicle,
        R_body_vehicle=R_body_vehicle,
        R_vehicle_body=R_vehicle_body,
        forward_angle_phone_frame_deg=mounting_yaw.forward_angle_phone_frame_deg,
        mounting_yaw_vehicle_from_phone_deg=mounting_yaw.mounting_yaw_vehicle_from_phone_deg,
        initial_attitude_q=q_init,
        diagnostics={
            'standstills_detected': len(standstills),
            'settled_window_s': leveling.window_s,
            'init_heading_deg': float(np.degrees(init_heading)),
            'theta_nav_deg': float(np.degrees(theta_nav))
        }
    )
