import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from navris.schema import (
    KMH_TO_MPS, MPS_TO_KMH, G_TO_MPS2, DEG_TO_RAD, RAD_TO_DEG,
    PHONE_SCHEMA_COLUMNS, REFERENCE_SCHEMA_COLUMNS, SYNCHRONIZED_SCHEMA_COLUMNS
)

def test_speed_conversion_factors():
    # 36 km/h is exactly 10 m/s
    assert np.isclose(36.0 * KMH_TO_MPS, 10.0, atol=1e-7)
    # 100 km/h is 27.7777778 m/s
    assert np.isclose(100.0 * KMH_TO_MPS, 27.7777778, atol=1e-5)
    # Round-trip
    v_mps = 25.0
    assert np.isclose((v_mps * MPS_TO_KMH) * KMH_TO_MPS, v_mps, atol=1e-7)

def test_acceleration_conversion():
    # 1 g is 9.80665 m/s^2
    assert np.isclose(1.0 * G_TO_MPS2, 9.80665, atol=1e-5)
    assert np.isclose(0.5 * G_TO_MPS2, 4.903325, atol=1e-5)

def test_angle_conversion():
    assert np.isclose(180.0 * DEG_TO_RAD, np.pi, atol=1e-7)
    assert np.isclose(90.0 * DEG_TO_RAD, np.pi / 2.0, atol=1e-7)
    assert np.isclose(360.0 * DEG_TO_RAD, 2.0 * np.pi, atol=1e-7)

def test_schema_definitions_integrity():
    assert len(PHONE_SCHEMA_COLUMNS) >= 24
    assert len(REFERENCE_SCHEMA_COLUMNS) == 29
    assert 'time_s' in SYNCHRONIZED_SCHEMA_COLUMNS
    assert 'phone_accel_x_mps2' in SYNCHRONIZED_SCHEMA_COLUMNS
    assert 'ref_speed_mps' in SYNCHRONIZED_SCHEMA_COLUMNS
