"""
NAVRIS Gap & Discontinuity Handling.

Ensures that numerical integration NEVER occurs across recording discontinuities (dt > 0.25 s).
Segments data into strictly contiguous intervals.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


def detect_discontinuities(time_s: np.ndarray, max_dt: float = 0.25) -> List[int]:
    """
    Finds sample indices i where time_s[i] - time_s[i-1] > max_dt.
    Index i represents the start of a new contiguous segment.
    """
    time_s = np.asarray(time_s, dtype=np.float64)
    if len(time_s) < 2:
        return []
    diffs = np.diff(time_s)
    gap_indices = np.where(diffs > max_dt)[0] + 1
    return gap_indices.tolist()


def split_into_contiguous_segments(
    df: pd.DataFrame,
    time_col: str = 'time_s',
    max_dt: float = 0.25,
    min_samples: int = 20
) -> List[Tuple[int, int, pd.DataFrame]]:
    """
    Splits DataFrame into contiguous temporal segments where dt <= max_dt.
    
    Args:
        df: Input DataFrame containing time_col
        time_col: Timestamp column in seconds
        max_dt: Maximum allowable sampling interval before breaking segment (default 0.25 s)
        min_samples: Minimum number of samples for a segment to be retained
        
    Returns:
        List of (start_idx, end_idx, segment_df) tuples where end_idx is exclusive.
    """
    if len(df) < min_samples:
        return []

    t = df[time_col].values
    gap_indices = detect_discontinuities(t, max_dt=max_dt)
    
    # Boundary points: [0, gap_1, gap_2, ..., N]
    boundaries = [0] + gap_indices + [len(df)]
    
    segments = []
    for i in range(len(boundaries) - 1):
        s_start = boundaries[i]
        s_end = boundaries[i + 1]
        if (s_end - s_start) >= min_samples:
            seg_df = df.iloc[s_start:s_end].copy().reset_index(drop=True)
            segments.append((s_start, s_end, seg_df))
            
    return segments
