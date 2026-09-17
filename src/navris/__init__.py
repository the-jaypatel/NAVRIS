"""
NAVRIS: Intelligent Navigation & Inertial System.
Core preprocessing, synchronization, coordinates, and windowing package.
"""

__version__ = '0.1.0'

from navris.coords import geodetic_to_enu, enu_to_geodetic, geodetic_to_ecef, ecef_to_geodetic
from navris.schema import (
    KMH_TO_MPS, G_TO_MPS2, DEG_TO_RAD,
    PHONE_SCHEMA_COLUMNS, REFERENCE_SCHEMA_COLUMNS, SYNCHRONIZED_SCHEMA_COLUMNS
)
from navris.ingest import ingest_smartphone_data, ingest_reference_data
from navris.sync import synchronize_recordings
from navris.windowing import generate_causal_windows, WindowConfig, save_windows_npz
from navris.splitting import (
    assign_standard_recording_split,
    assign_unseen_driver_split,
    assign_unseen_region_split,
    assign_unseen_vehicle_phone_split,
    verify_no_split_leakage
)
from navris.pipeline import process_single_recording, run_pipeline
