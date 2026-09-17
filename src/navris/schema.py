"""
NAVRIS Standardized Schema Definitions and Physical Constants.

Every sensor measurement is converted strictly to standard SI units:
- Length: meters [m]
- Velocity: meters per second [m/s]
- Acceleration: meters per second squared [m/s^2]
- Angular velocity: radians per second [rad/s]
- Angles / Heading / Orientations: radians [rad]
- Time: seconds [s]
- Magnetic Field: microteslas [uT]
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

# Physical Conversion Constants
KMH_TO_MPS = 1.0 / 3.6               # 1 km/h = 0.2777777777777778 m/s
MPS_TO_KMH = 3.6
G_TO_MPS2 = 9.80665                 # Standard gravitational acceleration (m/s^2)
DEG_TO_RAD = np.pi / 180.0          # Degrees to Radians
RAD_TO_DEG = 180.0 / np.pi          # Radians to Degrees
PSI_TO_BAR = 0.0689475729           # Pound per square inch to bar

# Standardized Column Definitions for Smartphone Data
PHONE_SCHEMA_COLUMNS = [
    'phone_time_s',              # Float: time since start of recording [s]
    'phone_accel_x_mps2',        # Float: IMU acceleration body X [m/s^2]
    'phone_accel_y_mps2',        # Float: IMU acceleration body Y [m/s^2]
    'phone_accel_z_mps2',        # Float: IMU acceleration body Z [m/s^2]
    'phone_gravity_x_mps2',      # Float: Gravity vector body X [m/s^2]
    'phone_gravity_y_mps2',      # Float: Gravity vector body Y [m/s^2]
    'phone_gravity_z_mps2',      # Float: Gravity vector body Z [m/s^2]
    'phone_gyro_x_radps',        # Float: IMU angular velocity body X [rad/s]
    'phone_gyro_y_radps',        # Float: IMU angular velocity body Y [rad/s]
    'phone_gyro_z_radps',        # Float: IMU angular velocity body Z [rad/s]
    'phone_mag_x_uT',            # Float: Magnetometer X [uT] (NaN if missing)
    'phone_mag_y_uT',            # Float: Magnetometer Y [uT] (NaN if missing)
    'phone_mag_z_uT',            # Float: Magnetometer Z [uT] (NaN if missing)
    'phone_has_mag',             # Bool: True if magnetometer present in recording
    'phone_orient_azimuth_rad',  # Float: Orientation Azimuth / Yaw [rad]
    'phone_orient_pitch_rad',    # Float: Orientation Pitch [rad]
    'phone_orient_roll_rad',     # Float: Orientation Roll [rad]
    'phone_has_orientation',     # Bool: True if orientation angles present
    'phone_gps_lat_deg',         # Float: Raw GPS latitude [deg]
    'phone_gps_lon_deg',         # Float: Raw GPS longitude [deg]
    'phone_gps_alt_m',           # Float: GPS altitude [m]
    'phone_gps_east_m',          # Float: Local Cartesian East [m] from origin
    'phone_gps_north_m',         # Float: Local Cartesian North [m] from origin
    'phone_gps_up_m',            # Float: Local Cartesian Up [m] from origin
    'phone_gps_speed_mps',       # Float: GPS reported speed [m/s]
    'phone_gps_accuracy_m',      # Float: Estimated GPS accuracy radius [m]
    'phone_gps_orientation_rad', # Float: GPS course over ground [rad]
    'phone_gps_satellites_str',  # Str: Raw / cleaned satellite string
    'phone_gps_satellites_in_view', # Float: Satellites in view count (NaN if unparsed)
    'phone_gps_satellites_used',    # Float: Satellites used count (NaN if unparsed)
    'phone_gps_is_new_fix',      # Bool: True ONLY when coordinates actually updated
    'phone_gps_is_valid',        # Bool: True if valid non-zero fix
]

# Standardized Column Definitions for Reference (Vehicle CAN / VBOX) Data
REFERENCE_SCHEMA_COLUMNS = [
    'ref_time_s',                # Float: time since start of recording [s]
    'ref_lat_deg',               # Float: VBOX GPS latitude [deg]
    'ref_lon_deg',               # Float: VBOX GPS longitude [deg]
    'ref_alt_m',                 # Float: VBOX GPS altitude [m] (corrected from Height km)
    'ref_east_m',                # Float: Local Cartesian East [m] from origin
    'ref_north_m',               # Float: Local Cartesian North [m] from origin
    'ref_up_m',                  # Float: Local Cartesian Up [m] from origin
    'ref_speed_mps',             # Float: VBOX Doppler velocity [m/s]
    'ref_vertical_speed_mps',    # Float: VBOX vertical velocity [m/s]
    'ref_heading_rad',           # Float: VBOX true heading [rad]
    'ref_heading_deg',           # Float: VBOX true heading [deg]
    'ref_yaw_rate_radps',        # Float: CAN chassis yaw rate [rad/s]
    'ref_steering_angle_rad',    # Float: Steering wheel angle [rad]
    'ref_wheel_speed_fl_radps',  # Float: Wheel angular speed Front-Left [rad/s]
    'ref_wheel_speed_fr_radps',  # Float: Wheel angular speed Front-Right [rad/s]
    'ref_wheel_speed_rl_radps',  # Float: Wheel angular speed Rear-Left [rad/s]
    'ref_wheel_speed_rr_radps',  # Float: Wheel angular speed Rear-Right [rad/s]
    'ref_accel_long_mps2',       # Float: CAN longitudinal acceleration [m/s^2]
    'ref_accel_lat_mps2',        # Float: CAN lateral acceleration [m/s^2]
    'ref_indicated_speed_mps',   # Float: Speedometer CAN speed [m/s]
    'ref_engine_speed_rpm',      # Float: Engine speed [RPM]
    'ref_brake_pressure_psi',    # Float: Brake line pressure [psi]
    'ref_throttle_pct',          # Float: Throttle pedal percentage [0-100%]
    'ref_gear',                  # Float: Engaged gear number [1-5]
    'ref_handbrake',             # Float: Handbrake state [0 or 1]
    'ref_gps_satellites_raw',    # Float: Raw satellite count with 0x80 bit
    'ref_gps_satellites_count',  # Float: True satellite count (minus 128 if DGPS)
    'ref_gps_is_dgps',           # Bool: True if differential / valid fix flag set
    'ref_gps_dropout',           # Bool: True if zero satellites / signal loss
]

# Combined Synchronized Schema Columns
SYNCHRONIZED_SCHEMA_COLUMNS = (
    ['time_s'] +
    [c for c in REFERENCE_SCHEMA_COLUMNS if c != 'ref_time_s'] +
    [c for c in PHONE_SCHEMA_COLUMNS if c != 'phone_time_s']
)
