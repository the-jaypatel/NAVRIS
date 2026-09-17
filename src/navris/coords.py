"""
NAVRIS Geodetic Coordinate Transformations.
Standard WGS-84 to Local Cartesian ENU (East-North-Up) frame.

Conventions:
- WGS-84 Ellipsoid: semi-major axis a = 6378137.0 m, flattening f = 1/298.257223563.
- Local Cartesian Frame (ENU):
  - X-axis: East [meters]
  - Y-axis: North [meters]
  - Z-axis: Up [meters]
- Reference Origin (phi0, lambda0, h0): First valid reference position in recording.
"""

import numpy as np
from typing import Tuple, Union

# WGS-84 Ellipsoid Constants
WGS84_A = 6378137.0  # semi-major axis (meters)
WGS84_F = 1.0 / 298.257223563  # flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)  # semi-minor axis
WGS84_E2 = 2.0 * WGS84_F - WGS84_F**2  # first eccentricity squared
WGS84_E_PRIME2 = (WGS84_A**2 - WGS84_B**2) / (WGS84_B**2)  # second eccentricity squared


def geodetic_to_ecef(
    lat_deg: Union[float, np.ndarray],
    lon_deg: Union[float, np.ndarray],
    alt_m: Union[float, np.ndarray] = 0.0
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Convert WGS-84 Geodetic coordinates (lat, lon in degrees, alt in meters)
    to Earth-Centered Earth-Fixed (ECEF) Cartesian coordinates (X, Y, Z in meters).
    """
    phi = np.radians(lat_deg)
    lam = np.radians(lon_deg)
    h = alt_m

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_lam = np.sin(lam)
    cos_lam = np.cos(lam)

    # Prime vertical radius of curvature
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_phi**2)

    x = (n + h) * cos_phi * cos_lam
    y = (n + h) * cos_phi * sin_lam
    z = (n * (1.0 - WGS84_E2) + h) * sin_phi

    return x, y, z


def ecef_to_geodetic(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    z: Union[float, np.ndarray]
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Convert ECEF Cartesian coordinates (X, Y, Z in meters) to WGS-84 Geodetic
    coordinates (lat, lon in degrees, alt in meters) using Bowring's method.
    """
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * WGS84_A, p * WGS84_B)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    phi = np.arctan2(
        z + WGS84_E_PRIME2 * WGS84_B * sin_theta**3,
        p - WGS84_E2 * WGS84_A * cos_theta**3
    )
    lam = np.arctan2(y, x)

    sin_phi = np.sin(phi)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_phi**2)
    h = p / np.cos(phi) - n

    return np.degrees(phi), np.degrees(lam), h


def ecef_to_enu(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
    lat0_deg: float,
    lon0_deg: float,
    alt0_m: float = 0.0
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Convert ECEF coordinates (X, Y, Z in meters) to Local Cartesian ENU
    (East, North, Up in meters) relative to origin (lat0_deg, lon0_deg, alt0_m).
    """
    x0, y0, z0 = geodetic_to_ecef(lat0_deg, lon0_deg, alt0_m)

    dx = x - x0
    dy = y - y0
    dz = z - z0

    phi0 = np.radians(lat0_deg)
    lam0 = np.radians(lon0_deg)

    sin_phi0 = np.sin(phi0)
    cos_phi0 = np.cos(phi0)
    sin_lam0 = np.sin(lam0)
    cos_lam0 = np.cos(lam0)

    e = -sin_lam0 * dx + cos_lam0 * dy
    n = -sin_phi0 * cos_lam0 * dx - sin_phi0 * sin_lam0 * dy + cos_phi0 * dz
    u = cos_phi0 * cos_lam0 * dx + cos_phi0 * sin_lam0 * dy + sin_phi0 * dz

    return e, n, u


def enu_to_ecef(
    e: Union[float, np.ndarray],
    n: Union[float, np.ndarray],
    u: Union[float, np.ndarray],
    lat0_deg: float,
    lon0_deg: float,
    alt0_m: float = 0.0
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Convert Local Cartesian ENU (East, North, Up in meters) to ECEF (X, Y, Z in meters)
    relative to origin (lat0_deg, lon0_deg, alt0_m).
    """
    x0, y0, z0 = geodetic_to_ecef(lat0_deg, lon0_deg, alt0_m)

    phi0 = np.radians(lat0_deg)
    lam0 = np.radians(lon0_deg)

    sin_phi0 = np.sin(phi0)
    cos_phi0 = np.cos(phi0)
    sin_lam0 = np.sin(lam0)
    cos_lam0 = np.cos(lam0)

    # Inverse rotation matrix (transpose of orthogonal matrix R)
    dx = -sin_lam0 * e - sin_phi0 * cos_lam0 * n + cos_phi0 * cos_lam0 * u
    dy = cos_lam0 * e - sin_phi0 * sin_lam0 * n + cos_phi0 * sin_lam0 * u
    dz = cos_phi0 * n + sin_phi0 * u

    return x0 + dx, y0 + dy, z0 + dz


def geodetic_to_enu(
    lat_deg: Union[float, np.ndarray],
    lon_deg: Union[float, np.ndarray],
    alt_m: Union[float, np.ndarray],
    lat0_deg: float,
    lon0_deg: float,
    alt0_m: float
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Direct transformation from WGS-84 Geodetic to Local Cartesian ENU.
    """
    x, y, z = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    return ecef_to_enu(x, y, z, lat0_deg, lon0_deg, alt0_m)


def enu_to_geodetic(
    e: Union[float, np.ndarray],
    n: Union[float, np.ndarray],
    u: Union[float, np.ndarray],
    lat0_deg: float,
    lon0_deg: float,
    alt0_m: float
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Direct transformation from Local Cartesian ENU to WGS-84 Geodetic.
    """
    x, y, z = enu_to_ecef(e, n, u, lat0_deg, lon0_deg, alt0_m)
    return ecef_to_geodetic(x, y, z)
