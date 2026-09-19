"""
NAVRIS Local Normal Gravity Model.

WGS-84 Somigliana theoretical gravity formula with latitude and ellipsoidal height correction:
    g(phi, h) = g_0 * (1 + k * sin^2(phi)) / sqrt(1 - e^2 * sin^2(phi)) - (2 * g_0 / a) * (1 + f + m - 2*f*sin^2(phi)) * h

In local Cartesian ENU coordinates:
    g^n = [0, 0, -g(phi, h)]^T

Explicitly Neglected Terms (documented for short-duration smartphone baseline):
- Earth Rotation Rate (omega_ie ~ 7.292115e-5 rad/s): max Coriolis acceleration 2*omega*v < 0.003 m/s^2 at 20 m/s.
- Transport Rate (v / R_e ~ 3e-6 rad/s).
- Earth Curvature over local trajectories (< 10 km).
"""

import numpy as np
from typing import Tuple, Optional

# WGS-84 Ellipsoidal Constants
WGS84_A = 6378137.0                # Semi-major axis (m)
WGS84_F = 1.0 / 298.257223563      # Flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)# Semi-minor axis (m)
WGS84_E2 = 2.0 * WGS84_F - WGS84_F**2 # Eccentricity squared

# Somigliana Formula Constants
G_EQUATOR = 9.7803253359           # Equatorial gravity (m/s^2)
G_POLE = 9.8321849378              # Polar gravity (m/s^2)
K_SOMIGLIANA = 0.00193185265241    # Somigliana formula constant
M_FACTOR = 0.00344978650684        # Centrifugal acceleration ratio at equator


def normal_gravity_wgs84(lat_deg: float, alt_m: float = 0.0) -> float:
    """
    Computes local WGS-84 normal gravitational acceleration magnitude g(lat, alt) in m/s^2.
    """
    lat_rad = np.radians(lat_deg)
    sin_lat = np.sin(lat_rad)
    sin2_lat = sin_lat * sin_lat

    # Surface gravity on ellipsoid
    gamma_0 = G_EQUATOR * (1.0 + K_SOMIGLIANA * sin2_lat) / np.sqrt(1.0 - WGS84_E2 * sin2_lat)

    # Free-air / height correction
    height_factor = (2.0 * G_EQUATOR / WGS84_A) * (1.0 + WGS84_F + M_FACTOR - 2.0 * WGS84_F * sin2_lat)
    g = gamma_0 - height_factor * alt_m

    return float(g)


def gravity_vector_enu(lat_deg: float = 52.4, alt_m: float = 0.0) -> np.ndarray:
    """
    Returns gravity vector in local Cartesian ENU coordinates:
    g^n = [0.0, 0.0, -g(lat, alt)]^T.
    """
    g = normal_gravity_wgs84(lat_deg, alt_m)
    return np.array([0.0, 0.0, -g], dtype=np.float64)
