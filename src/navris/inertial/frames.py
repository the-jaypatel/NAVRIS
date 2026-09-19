"""
NAVRIS Coordinate Frames & Quaternion Algebra.

Conventions:
- Navigation Frame: Local Cartesian ENU (X=East, Y=North, Z=Up).
- Body Frame: Raw smartphone IMU sensor frame (Android standard).
- Quaternion: Scalar-first Hamilton convention q = [qw, qx, qy, qz]^T.
- Quaternion represents Body-to-Navigation rotation C_b^n(q).
- Vector rotation: v^n = C_b^n(q) @ v^b.
- All floating point calculations: float64.
"""

import numpy as np
from typing import Tuple, Union


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalizes quaternion to unit length. Handles near-zero gracefully."""
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Returns conjugate (inverse for unit quat) of scalar-first quaternion."""
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Computes Hamilton quaternion product q_out = q1 (x) q2.
    Scalar-first: q = [qw, qx, qy, qz].
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z], dtype=np.float64)


def rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
    """
    Converts rotation vector theta in R^3 to incremental quaternion Delta_q
    via exact exponential map with series expansion for small angles.
    """
    rotvec = np.asarray(rotvec, dtype=np.float64)
    theta_sq = np.dot(rotvec, rotvec)
    theta = np.sqrt(theta_sq)

    if theta < 1e-8:
        # Taylor series expansion around theta = 0
        w = 1.0 - 0.125 * theta_sq
        s = 0.5 - theta_sq / 48.0
    else:
        half_theta = 0.5 * theta
        w = np.cos(half_theta)
        s = np.sin(half_theta) / theta

    q = np.array([w, s * rotvec[0], s * rotvec[1], s * rotvec[2]], dtype=np.float64)
    return quat_normalize(q)


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """
    Converts unit quaternion q (representing body-to-navigation rotation)
    to 3x3 Direction Cosine Matrix C_b^n.
    v^n = C_b^n @ v^b.
    """
    q = quat_normalize(q)
    w, x, y, z = q[0], q[1], q[2], q[3]

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    C = np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)]
    ], dtype=np.float64)
    return C


def rotate_vector(q: np.ndarray, v_body: np.ndarray) -> np.ndarray:
    """
    Rotates 3D vector from body frame to navigation frame: v_nav = C_b^n(q) @ v_body.
    Supports single 3-vector or (N, 3) batch.
    """
    C = quat_to_dcm(q)
    v = np.asarray(v_body, dtype=np.float64)
    if v.ndim == 1:
        return C @ v
    return (C @ v.T).T


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Converts Euler angles (roll, pitch, yaw in radians) to scalar-first quaternion.
    Rotation order: Z-Y-X (yaw around Z, pitch around Y', roll around X'').
    """
    cr = np.cos(0.5 * roll)
    sr = np.sin(0.5 * roll)
    cp = np.cos(0.5 * pitch)
    sp = np.sin(0.5 * pitch)
    cy = np.cos(0.5 * yaw)
    sy = np.sin(0.5 * yaw)

    qw = cy * cp * cr + sy * sp * sr
    qx = cy * cp * sr - sy * sp * cr
    qy = cy * sp * cr + sy * cp * sr
    qz = sy * cp * cr - cy * sp * sr

    return quat_normalize(np.array([qw, qx, qy, qz], dtype=np.float64))


def quat_to_euler(q: np.ndarray) -> Tuple[float, float, float]:
    """
    Extracts Euler angles (roll, pitch, yaw) in radians from scalar-first quaternion.
    Returns: (roll, pitch, yaw).
    """
    q = quat_normalize(q)
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1.0:
        pitch = np.copysign(np.pi / 2.0, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return float(roll), float(pitch), float(yaw)


def wrap_angle_pi(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Wraps angle to [-pi, +pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def wrap_angle_2pi(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Wraps angle to [0, 2*pi)."""
    return angle % (2.0 * np.pi)
