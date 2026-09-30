"""
NAVRIS Phase 3.0: Causal Feature Extraction.

Constructs strictly causal features over a short causal window [k-9 ... k] (1 second at 10 Hz):
- Instantaneous IMU kinematics (accel XYZ, gyro XYZ, norms, gravity deviation).
- Instantaneous NAVRIS filter states (estimated velocity XYZ, speed, heading orientation).
- Temporal causal window statistics (rolling mean, rolling std, delta acceleration, delta gyro, delta velocity).

Strict Leakage Prevention:
- Zero reference fields
- Zero VBOX ground truth fields
- Zero target residuals
- Zero future samples (strictly past and present [k-9 ... k])
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


STANDARD_GRAVITY_MPS2 = 9.80665
WINDOW_SAMPLES = 10  # 1.0 second at 10 Hz sampling rate: [k-9, ..., k]

# Exact feature names extracted for the ML model
FEATURE_NAMES: List[str] = [
    # Instantaneous IMU
    "phone_accel_x_mps2",
    "phone_accel_y_mps2",
    "phone_accel_z_mps2",
    "phone_gyro_x_radps",
    "phone_gyro_y_radps",
    "phone_gyro_z_radps",
    "accel_magnitude_mps2",
    "gyro_magnitude_radps",
    "gravity_deviation_mps2",
    # Instantaneous NAVRIS filter estimates
    "navris_vel_east_mps",
    "navris_vel_north_mps",
    "navris_vel_up_mps",
    "navris_speed_mps",
    "navris_heading_sin",
    "navris_heading_cos",
    # Temporal Causal Window [k-9 ... k] Statistics (1 second history)
    "accel_mag_rolling_mean_1s",
    "accel_mag_rolling_std_1s",
    "gyro_mag_rolling_mean_1s",
    "gyro_mag_rolling_std_1s",
    "delta_accel_x_1s",
    "delta_accel_y_1s",
    "delta_accel_z_1s",
    "delta_gyro_x_1s",
    "delta_gyro_y_1s",
    "delta_gyro_z_1s",
    "navris_delta_vel_east_1s",
    "navris_delta_vel_north_1s",
    "navris_delta_vel_up_1s",
]

TARGET_NAMES: List[str] = [
    "target_delta_v_east_mps",
    "target_delta_v_north_mps",
    "target_delta_v_up_mps",
]

FORBIDDEN_FEATURE_SUBSTRINGS: List[str] = [
    "ref_",
    "vbox",
    "target",
    "ground_truth",
    "target_delta_v",
    "_truth",
    "oracle",
]



class CausalFeatureExtractor:
    """
    Extracts strictly causal feature matrices and targets from aligned recording DataFrames.
    Operates strictly within per-recording boundaries to eliminate cross-recording boundary leakage.
    """

    def __init__(self, window_size: int = WINDOW_SAMPLES):
        self.window_size = window_size

    def extract_from_recording(
        self, df_rec: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extracts causal features, targets, and metadata for a single recording.

        Args:
            df_rec: Aligned recording DataFrame from load_recording_ml_data.

        Returns:
            Tuple of:
            - X: DataFrame of strictly causal features (length N - window_size + 1)
            - y: DataFrame of target residuals [delta_v_E, delta_v_N, delta_v_U]
            - meta: DataFrame with metadata (time_s, recording_id, v_ref, v_navris) for audit & evaluation
        """
        n_samples = len(df_rec)
        if n_samples < self.window_size:
            raise ValueError(
                f"Recording has only {n_samples} samples, less than causal window {self.window_size}"
            )

        # 1. Raw IMU signals
        ax = df_rec["phone_accel_x_mps2"].values.astype(np.float64)
        ay = df_rec["phone_accel_y_mps2"].values.astype(np.float64)
        az = df_rec["phone_accel_z_mps2"].values.astype(np.float64)

        gx = df_rec["phone_gyro_x_radps"].values.astype(np.float64)
        gy = df_rec["phone_gyro_y_radps"].values.astype(np.float64)
        gz = df_rec["phone_gyro_z_radps"].values.astype(np.float64)

        # 2. Instantaneous IMU derived kinematics
        accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
        gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        grav_dev = np.abs(accel_mag - STANDARD_GRAVITY_MPS2)

        # 3. Instantaneous NAVRIS filter estimates
        v_nav_e = df_rec["navris_vel_east_mps"].values.astype(np.float64)
        v_nav_n = df_rec["navris_vel_north_mps"].values.astype(np.float64)
        v_nav_u = df_rec["navris_vel_up_mps"].values.astype(np.float64)
        nav_spd = np.sqrt(v_nav_e**2 + v_nav_n**2)

        nav_hdg = df_rec["navris_heading_rad"].values.astype(np.float64)
        nav_hdg_sin = np.sin(nav_hdg)
        nav_hdg_cos = np.cos(nav_hdg)

        # 4. Temporal Causal Window Statistics [k - (window_size - 1) ... k]
        # We compute rolling statistics strictly backward-looking
        w = self.window_size  # e.g., 10 samples = 1.0 s

        # Preallocate output arrays for valid causal samples: k from (w-1) to (n_samples - 1)
        k_indices = np.arange(w - 1, n_samples)
        m = len(k_indices)

        # Use fast pandas rolling strictly backwards (closed='both' over window w)
        # Verify that rolling in pandas with default is strictly causal:
        # s.rolling(w).mean()[k] computes mean(s[k-w+1 ... k]), which is strictly causal!
        s_accel_mag = pd.Series(accel_mag)
        s_gyro_mag = pd.Series(gyro_mag)

        accel_mag_mean = s_accel_mag.rolling(w).mean().values[k_indices]
        accel_mag_std = s_accel_mag.rolling(w).std(ddof=0).values[k_indices]
        gyro_mag_mean = s_gyro_mag.rolling(w).mean().values[k_indices]
        gyro_mag_std = s_gyro_mag.rolling(w).std(ddof=0).values[k_indices]

        # Recent delta over 1 second: val[k] - val[k - (w - 1)]
        delta_ax = (ax[k_indices] - ax[k_indices - (w - 1)])
        delta_ay = (ay[k_indices] - ay[k_indices - (w - 1)])
        delta_az = (az[k_indices] - az[k_indices - (w - 1)])

        delta_gx = (gx[k_indices] - gx[k_indices - (w - 1)])
        delta_gy = (gy[k_indices] - gy[k_indices - (w - 1)])
        delta_gz = (gz[k_indices] - gz[k_indices - (w - 1)])

        delta_v_nav_e = (v_nav_e[k_indices] - v_nav_e[k_indices - (w - 1)])
        delta_v_nav_n = (v_nav_n[k_indices] - v_nav_n[k_indices - (w - 1)])
        delta_v_nav_u = (v_nav_u[k_indices] - v_nav_u[k_indices - (w - 1)])

        # Construct feature DataFrame
        features_dict = {
            # Instantaneous IMU
            "phone_accel_x_mps2": ax[k_indices],
            "phone_accel_y_mps2": ay[k_indices],
            "phone_accel_z_mps2": az[k_indices],
            "phone_gyro_x_radps": gx[k_indices],
            "phone_gyro_y_radps": gy[k_indices],
            "phone_gyro_z_radps": gz[k_indices],
            "accel_magnitude_mps2": accel_mag[k_indices],
            "gyro_magnitude_radps": gyro_mag[k_indices],
            "gravity_deviation_mps2": grav_dev[k_indices],
            # Instantaneous NAVRIS filter estimates
            "navris_vel_east_mps": v_nav_e[k_indices],
            "navris_vel_north_mps": v_nav_n[k_indices],
            "navris_vel_up_mps": v_nav_u[k_indices],
            "navris_speed_mps": nav_spd[k_indices],
            "navris_heading_sin": nav_hdg_sin[k_indices],
            "navris_heading_cos": nav_hdg_cos[k_indices],
            # Temporal Causal Window [k-9 ... k] Statistics
            "accel_mag_rolling_mean_1s": accel_mag_mean,
            "accel_mag_rolling_std_1s": accel_mag_std,
            "gyro_mag_rolling_mean_1s": gyro_mag_mean,
            "gyro_mag_rolling_std_1s": gyro_mag_std,
            "delta_accel_x_1s": delta_ax,
            "delta_accel_y_1s": delta_ay,
            "delta_accel_z_1s": delta_az,
            "delta_gyro_x_1s": delta_gx,
            "delta_gyro_y_1s": delta_gy,
            "delta_gyro_z_1s": delta_gz,
            "navris_delta_vel_east_1s": delta_v_nav_e,
            "navris_delta_vel_north_1s": delta_v_nav_n,
            "navris_delta_vel_up_1s": delta_v_nav_u,
        }

        df_X = pd.DataFrame(features_dict, index=range(m))

        # Validate feature column list
        assert list(df_X.columns) == FEATURE_NAMES, f"Feature columns mismatch: {df_X.columns} vs {FEATURE_NAMES}"

        # 5. Extract Targets
        target_dict = {
            "target_delta_v_east_mps": df_rec["target_delta_v_east_mps"].values[k_indices],
            "target_delta_v_north_mps": df_rec["target_delta_v_north_mps"].values[k_indices],
            "target_delta_v_up_mps": df_rec["target_delta_v_up_mps"].values[k_indices],
        }
        df_y = pd.DataFrame(target_dict, index=range(m))

        # 6. Metadata for evaluation
        meta_dict = {
            "time_s": df_rec["time_s"].values[k_indices],
            "recording_id": [df_rec["recording_id"].iloc[0]] * m,
            "v_ref_east_mps": df_rec["v_ref_east_mps"].values[k_indices],
            "v_ref_north_mps": df_rec["v_ref_north_mps"].values[k_indices],
            "v_ref_up_mps": df_rec["v_ref_up_mps"].values[k_indices],
            "ref_speed_mps": df_rec["ref_speed_mps"].values[k_indices],
            "ref_heading_rad": df_rec["ref_heading_rad"].values[k_indices],
            "navris_vel_east_mps": v_nav_e[k_indices],
            "navris_vel_north_mps": v_nav_n[k_indices],
            "navris_vel_up_mps": v_nav_u[k_indices],
        }
        if "navris_B_vel_east_mps" in df_rec.columns:
            meta_dict["navris_B_vel_east_mps"] = df_rec["navris_B_vel_east_mps"].values[k_indices]
            meta_dict["navris_B_vel_north_mps"] = df_rec["navris_B_vel_north_mps"].values[k_indices]
            meta_dict["navris_B_vel_up_mps"] = df_rec["navris_B_vel_up_mps"].values[k_indices]

        df_meta = pd.DataFrame(meta_dict, index=range(m))

        # Strict sanity checks
        if df_X.isna().any().any():
            raise ValueError(f"NaN values detected in feature matrix for {df_rec['recording_id'].iloc[0]}")
        if df_y.isna().any().any():
            raise ValueError(f"NaN values detected in target matrix for {df_rec['recording_id'].iloc[0]}")

        return df_X, df_y, df_meta
