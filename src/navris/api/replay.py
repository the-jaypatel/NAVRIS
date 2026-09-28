"""
Replay engine for NAVRIS benchmark datasets.

Loads real IO-VNBD benchmark trajectory and sensor files.
Maps actual measured values to the TBA Frame specification.
Maintains scientific integrity: never invents confidence, covariance, or NIS.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _find_data_root() -> Path:
    """Find data/processed/ directory."""
    here = Path(__file__).resolve().parent
    for ancestor in [here] + list(here.parents):
        candidate = ancestor / "data" / "processed"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Cannot find data/processed/ relative to {here}")


DATA_ROOT = _find_data_root()
GATE_DIR = DATA_ROOT / "phase2_3b" / "gate2_3b"
SYNC_DIR = DATA_ROOT / "synchronized"

# All recognized benchmark recordings in Gate 2.3B
ALL_RECORDINGS = ["S1", "S2", "S3A", "S4", "M", "Y1", "VTA1A", "VTA2"]


class RecordingData:
    """In-memory cache for a single benchmark recording."""

    def __init__(self, recording_id: str):
        self.recording_id = recording_id
        traj_path = GATE_DIR / f"{recording_id}_trajectory_comparison.csv"
        sync_path = SYNC_DIR / f"{recording_id}_sync.parquet"

        if not traj_path.exists():
            if recording_id == "M":
                raise ValueError("Recording M is UNOBSERVABLE: GNSS baseline failed to initialize.")
            raise FileNotFoundError(f"Trajectory comparison not found for {recording_id}: {traj_path}")

        # 1. Read trajectory comparison CSV (10 Hz)
        df_traj = pd.read_csv(traj_path)

        # 2. Read sync parquet if available
        if sync_path.exists():
            sync_cols = [
                "time_s", "ref_lat_deg", "ref_lon_deg", "ref_alt_m",
                "phone_accel_x_mps2", "phone_accel_y_mps2", "phone_accel_z_mps2",
                "phone_gyro_z_radps", "phone_gps_east_m", "phone_gps_north_m",
                "phone_gps_satellites_used", "phone_gps_satellites_in_view",
                "phone_gps_is_valid", "phone_gps_is_new_fix"
            ]
            df_sync = pd.read_parquet(sync_path, columns=sync_cols)
            df_traj["time_key"] = (df_traj["time_s"] * 10).round().astype(int)
            df_sync["time_key"] = (df_sync["time_s"] * 10).round().astype(int)
            self.df = pd.merge(df_traj, df_sync, on="time_key", how="left", suffixes=("", "_sync"))
            
            row0 = df_sync.iloc[0]
            self.origin_lat = float(row0["ref_lat_deg"])
            self.origin_lon = float(row0["ref_lon_deg"])
            self.origin_alt = float(row0["ref_alt_m"])
        else:
            self.df = df_traj
            self.origin_lat = 52.4017
            self.origin_lon = -1.5054
            self.origin_alt = 110.0

        self.n_frames = len(self.df)
        self.times = self.df["time_s"].values
        self.duration_s = float(self.times[-1] - self.times[0]) if self.n_frames > 1 else 0.0

        # Pre-compute last fix timestamps
        self._precompute_fixes()

    def _precompute_fixes(self) -> None:
        """Find the timestamp of the last valid fix for each epoch."""
        has_new_fix = ("phone_gps_is_new_fix" in self.df.columns) and ("phone_gps_is_valid" in self.df.columns)
        last_t = float(self.times[0])
        self.last_fix_times = np.zeros(self.n_frames, dtype=float)

        for i in range(self.n_frames):
            if has_new_fix:
                row = self.df.iloc[i]
                if bool(row.get("phone_gps_is_new_fix", False)) and bool(row.get("phone_gps_is_valid", False)):
                    last_t = float(row["time_s"])
            self.last_fix_times[i] = last_t

    def frame_at(self, idx: int) -> dict:
        """Build Frame dictionary for row `idx`."""
        idx = max(0, min(idx, self.n_frames - 1))
        row = self.df.iloc[idx]

        t = float(row["time_s"])

        # Reference ground truth (VBOX)
        true_e = float(row["ref_east_m"])
        true_n = float(row["ref_north_m"])
        true_speed = float(row["ref_speed_mps"])
        true_heading = math.degrees(float(row["ref_heading_rad"])) % 360.0

        # Estimated solution: Config B (ESKF + ZUPT + NHC)
        sol_e = float(row["pos_B_east_m"])
        sol_n = float(row["pos_B_north_m"])
        sol_up = float(row["pos_B_up_m"])
        vel_e = float(row["vel_B_east_mps"])
        vel_n = float(row["vel_B_north_mps"])
        sol_speed = math.hypot(vel_e, vel_n)
        sol_heading = math.degrees(float(row["head_B_rad"])) % 360.0

        # Exact measured errors
        err_e = sol_e - true_e
        err_n = sol_n - true_n
        horiz_err = float(row["horiz_err_B_m"])
        vel_err = abs(sol_speed - true_speed)
        hdg_err = abs(sol_heading - true_heading)
        if hdg_err > 180.0:
            hdg_err = 360.0 - hdg_err

        # GNSS status
        has_gnss_valid = "phone_gps_is_valid" in row
        gnss_avail = bool(row["phone_gps_is_valid"]) if has_gnss_valid and not pd.isna(row["phone_gps_is_valid"]) else True

        if "phone_gps_east_m" in row and not pd.isna(row["phone_gps_east_m"]):
            gnss_e = float(row["phone_gps_east_m"])
            gnss_n = float(row["phone_gps_north_m"])
        else:
            gnss_e = sol_e
            gnss_n = sol_n

        last_fix_t = float(self.last_fix_times[idx])
        fix_type = "3D FIX" if gnss_avail else "NO FIX"

        sats_used = int(row["phone_gps_satellites_used"]) if "phone_gps_satellites_used" in row and not pd.isna(row["phone_gps_satellites_used"]) else 0
        sats_visible = int(row["phone_gps_satellites_in_view"]) if "phone_gps_satellites_in_view" in row and not pd.isna(row["phone_gps_satellites_in_view"]) else sats_used

        # IMU status
        ax = float(row["phone_accel_x_mps2"]) if "phone_accel_x_mps2" in row and not pd.isna(row["phone_accel_x_mps2"]) else 0.0
        ay = float(row["phone_accel_y_mps2"]) if "phone_accel_y_mps2" in row and not pd.isna(row["phone_accel_y_mps2"]) else 0.0
        az = float(row["phone_accel_z_mps2"]) if "phone_accel_z_mps2" in row and not pd.isna(row["phone_accel_z_mps2"]) else 9.81
        accel_mag = math.sqrt(ax**2 + ay**2 + az**2)

        gz = float(row["phone_gyro_z_radps"]) if "phone_gyro_z_radps" in row and not pd.isna(row["phone_gyro_z_radps"]) else 0.0
        gyro_z = math.degrees(gz)

        # State classification
        if not gnss_avail:
            state = "ins"
        elif horiz_err > 100.0:
            state = "degraded"
        else:
            state = "fused"

        inertial_only = max(0.0, t - last_fix_t) if not gnss_avail else 0.0

        return {
            "t": t,
            "state": state,
            "trueE": true_e,
            "trueN": true_n,
            "trueSpeed": true_speed,
            "trueHeading": true_heading,
            "solE": sol_e,
            "solN": sol_n,
            "solSpeed": sol_speed,
            "solHeading": sol_heading,
            "alt": sol_up,
            "errE": err_e,
            "errN": err_n,
            "horizErr": horiz_err,
            # Scientific integrity: unmeasured / uncomputed values are strictly None
            "sigmaE": None,
            "sigmaN": None,
            "sigmaH": None,
            "confidence": None,
            "velErr": vel_err,
            "hdgErr": hdg_err,
            "gnssAvail": gnss_avail,
            "gnssE": gnss_e,
            "gnssN": gnss_n,
            "lastFixT": last_fix_t,
            "fixType": fix_type,
            "satsUsed": sats_used,
            "satsVisible": sats_visible,
            "hdop": None,  # accuracy_m is not HDOP
            "cn0": None,   # carrier-to-noise is not in dataset
            "accelMag": accel_mag,
            "gyroZ": gyro_z,
            "roll": 0.0,
            "pitch": 0.0,
            "accelBiasMg": None,
            "gyroBiasDph": None,
            "imuTemp": None,
            "nis": None,   # per-frame NIS not in trajectory CSV
            "accepted": idx,
            "rejected": 0,
            "inertialOnly": inertial_only,
            "sigmaV": None,
            "sigmaPsi": None,
            "aiStage": "idle",
            "aiPredictedErr": None,
            "aiCorrections": 0,
            "aiInferenceMs": None,
        }

    def get_history(self, start_idx: int = 0, end_idx: Optional[int] = None, step: int = 10) -> List[dict]:
        """
        Return history samples from `start_idx` to `end_idx` subsampled by `step`.
        Default step=10 converts 10 Hz recording to 1 Hz history for analytics charts.
        """
        if end_idx is None:
            end_idx = self.n_frames
        end_idx = min(end_idx, self.n_frames)

        samples = []
        for i in range(start_idx, end_idx, step):
            f = self.frame_at(i)
            samples.append({
                "t": f["t"],
                "state": f["state"],
                "errE": f["errE"],
                "errN": f["errN"],
                "horizErr": f["horizErr"],
                "sigE": None,
                "sigN": None,
                "sigH": None,
                "conf": None,
                "velErr": f["velErr"],
                "hdgErr": f["hdgErr"],
                "nis": None,
                "rej": f["rejected"],
                "gnss": 1 if f["gnssAvail"] else 0,
                "accel": f["accelMag"],
                "gyro": abs(f["gyroZ"]),
            })
        return samples


class ReplayEngine:
    """Manages recordings and playback session state."""

    def __init__(self):
        self.recordings: Dict[str, RecordingData] = {}
        self.current_recording: str = "S1"
        self.current_idx: int = 0
        self.is_playing: bool = False
        self.speed: float = 1.0

        # Check which recordings are available on disk
        self.available_recordings = []
        for r in ALL_RECORDINGS:
            p = GATE_DIR / f"{r}_trajectory_comparison.csv"
            if p.exists():
                self.available_recordings.append(r)

    def get_recording(self, rec_id: str) -> RecordingData:
        """Get or lazily load a recording."""
        rec_id = rec_id.upper()
        if rec_id not in self.available_recordings:
            if rec_id == "M":
                raise ValueError("Recording M is UNOBSERVABLE: GNSS baseline failed to initialize.")
            raise ValueError(f"Recording '{rec_id}' not available. Available: {self.available_recordings}")

        if rec_id not in self.recordings:
            self.recordings[rec_id] = RecordingData(rec_id)
        return self.recordings[rec_id]

    def select(self, rec_id: str) -> RecordingData:
        data = self.get_recording(rec_id)
        self.current_recording = rec_id
        self.current_idx = 0
        self.is_playing = False
        return data

    def current_data(self) -> RecordingData:
        return self.get_recording(self.current_recording)
