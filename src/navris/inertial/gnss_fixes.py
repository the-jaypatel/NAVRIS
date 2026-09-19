"""
NAVRIS Sparse GNSS Observation Gating.

Isolates genuinely new smartphone GNSS fixes from sample-and-hold duplicates.
Ensures filter updates and causal initializers never treat held values as repeated observations.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def filter_novel_gnss_fixes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts only rows with genuinely novel, valid GNSS fixes.
    Uses 'phone_gps_is_new_fix' if present; otherwise dynamically detects
    coordinate changes causally.
    """
    if 'phone_gps_is_new_fix' in df.columns:
        mask = df['phone_gps_is_new_fix'].fillna(False).astype(bool)
        return df[mask].copy()

    # Dynamic fallback: detect coordinate changes causally
    lat_col = 'phone_gps_lat_deg' if 'phone_gps_lat_deg' in df.columns else 'lat'
    lon_col = 'phone_gps_lon_deg' if 'phone_gps_lon_deg' in df.columns else 'lon'

    if lat_col not in df.columns or lon_col not in df.columns:
        return pd.DataFrame(columns=df.columns)

    lats = df[lat_col].values
    lons = df[lon_col].values

    valid = (np.abs(lats) > 0.1) & (~np.isnan(lats)) & (np.abs(lons) > 0.1) & (~np.isnan(lons))
    is_new = np.zeros(len(df), dtype=bool)

    if len(df) > 0 and valid[0]:
        is_new[0] = True

    for i in range(1, len(df)):
        if valid[i]:
            if not valid[i - 1] or (lats[i] != lats[i - 1]) or (lons[i] != lons[i - 1]):
                is_new[i] = True

    return df[is_new].copy()
