import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from navris.coords import (
    geodetic_to_ecef, ecef_to_geodetic,
    ecef_to_enu, enu_to_ecef,
    geodetic_to_enu, enu_to_geodetic
)

def test_geodetic_ecef_roundtrip():
    # Coventry coordinates
    lat = 52.4017192
    lon = -1.5053331
    alt = 110.19

    x, y, z = geodetic_to_ecef(lat, lon, alt)
    lat_r, lon_r, alt_r = ecef_to_geodetic(x, y, z)

    assert np.isclose(lat, lat_r, atol=1e-8)
    assert np.isclose(lon, lon_r, atol=1e-8)
    assert np.isclose(alt, alt_r, atol=1e-4)  # sub-millimeter precision

def test_enu_origin_maps_to_zero():
    lat0 = 52.4017192
    lon0 = -1.5053331
    alt0 = 110.19

    e, n, u = geodetic_to_enu(lat0, lon0, alt0, lat0, lon0, alt0)
    assert np.isclose(e, 0.0, atol=1e-6)
    assert np.isclose(n, 0.0, atol=1e-6)
    assert np.isclose(u, 0.0, atol=1e-6)

def test_enu_directional_monotonicity():
    lat0 = 52.4017192
    lon0 = -1.5053331
    alt0 = 110.19

    # Moving slightly North (+0.001 deg lat)
    e, n, u = geodetic_to_enu(lat0 + 0.001, lon0, alt0, lat0, lon0, alt0)
    assert n > 100.0   # ~111 meters North
    assert np.isclose(e, 0.0, atol=1.0)

    # Moving slightly East (+0.001 deg lon)
    e, n, u = geodetic_to_enu(lat0, lon0 + 0.001, alt0, lat0, lon0, alt0)
    assert e > 50.0    # ~68 meters East at 52 deg latitude
    assert np.isclose(n, 0.0, atol=1.0)

    # Moving Up (+50 meters)
    e, n, u = geodetic_to_enu(lat0, lon0, alt0 + 50.0, lat0, lon0, alt0)
    assert np.isclose(u, 50.0, atol=1e-3)

def test_enu_geodetic_roundtrip():
    lat0 = 52.4017192
    lon0 = -1.5053331
    alt0 = 110.19

    target_e = 1250.5
    target_n = -3420.2
    target_u = 15.3

    lat, lon, alt = enu_to_geodetic(target_e, target_n, target_u, lat0, lon0, alt0)
    e_r, n_r, u_r = geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0)

    assert np.isclose(target_e, e_r, atol=1e-3)
    assert np.isclose(target_n, n_r, atol=1e-3)
    assert np.isclose(target_u, u_r, atol=1e-3)
