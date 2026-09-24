"""
NAVRIS End-to-End Data Ingestion, Cleaning, Normalization & Windowing Pipeline.

Orchestrates:
1. Raw CSV ingestion and anomaly sanitization
2. Geodetic to local ENU coordinate projection
3. 10 Hz timeline synchronization and sparse GNSS preservation
4. Driver F (18-column) handling
5. Causal sliding window generation
6. Group-based train/val/test splitting
7. High-efficiency Parquet and NPZ serialization
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

from navris.ingest import ingest_smartphone_data, ingest_reference_data
from navris.sync import synchronize_recordings
from navris.windowing import generate_causal_windows, save_windows_npz, WindowConfig
from navris.splitting import (
    assign_standard_recording_split,
    assign_unseen_driver_split,
    assign_unseen_region_split,
    assign_unseen_vehicle_phone_split,
    verify_no_split_leakage
)


def process_single_recording(
    rec_id: str,
    manifest_entry: Dict[str, Any],
    base_data_dir: str = 'data/raw/IO-VNBD',
    output_dir: str = 'data/processed',
    window_config: Optional[WindowConfig] = None,
    phone_time_offset_s: float = 0.0,
    auto_align_benchmark: bool = False
) -> Dict[str, Any]:
    """
    Processes a single recording through the complete pipeline.
    
    Args:
        rec_id: Unique recording ID
        manifest_entry: Dictionary from recordings_manifest
        base_data_dir: Path to raw dataset root
        output_dir: Target output root directory
        window_config: Windowing parameters
        phone_time_offset_s: Optional constant offset to add to phone timeline
        auto_align_benchmark: If True, uses vehicle reference to estimate and align timeline lag
        
    Returns:
        Summary dict containing status, file paths, row counts, and window counts.
    """
    if window_config is None:
        window_config = WindowConfig()

    has_v = manifest_entry.get('has_v_data', False)
    has_s = manifest_entry.get('has_s_data', False)
    is_sync = manifest_entry.get('is_synchronized', False)

    v_rel = manifest_entry.get('v_file_path')
    s_rel = manifest_entry.get('s_file_path')

    v_path = os.path.join(base_data_dir, str(v_rel)) if (v_rel is not None and pd.notna(v_rel)) else None
    s_path = os.path.join(base_data_dir, str(s_rel)) if (s_rel is not None and pd.notna(s_rel)) else None

    # Route VTA1A to untrimmed raw VBOX source if present to eliminate author 14.4s truncation defect
    if rec_id == 'VTA1A':
        untrimmed_v = os.path.join(base_data_dir, 'Unsynchronised V and S Dataset/Uncategorised IOVNB (V and S) Dataset/V-Dataset/V-Vta1a.csv')
        if os.path.exists(untrimmed_v):
            v_path = untrimmed_v

    sync_dir = os.path.join(output_dir, 'synchronized')
    norm_dir = os.path.join(output_dir, 'normalized')
    wind_dir = os.path.join(output_dir, 'windows')

    os.makedirs(sync_dir, exist_ok=True)
    os.makedirs(norm_dir, exist_ok=True)
    os.makedirs(wind_dir, exist_ok=True)

    result = {
        'recording_id': rec_id,
        'driver': manifest_entry.get('driver'),
        'vehicle': manifest_entry.get('vehicle'),
        'phone': manifest_entry.get('phone_model'),
        'split': manifest_entry.get('split', 'unassigned'),
        'is_synchronized': is_sync,
        'status': 'SUCCESS',
        'sync_parquet_path': None,
        'windows_npz_path': None,
        'num_samples': 0,
        'num_windows': 0,
        'duration_s': 0.0,
        'num_gps_fixes': 0,
        'estimated_lag_s': 0.0,
        'sync_metadata': {}
    }

    try:
        if has_v and has_s and is_sync:
            # Synchronized Pair (Reference + Phone)
            df_ref, origin = ingest_reference_data(v_path)
            df_phone, _ = ingest_smartphone_data(s_path, origin_geodetic=origin)

            df_sync = synchronize_recordings(
                df_ref,
                df_phone,
                dt=window_config.dt,
                phone_time_offset_s=phone_time_offset_s,
                auto_align_benchmark=auto_align_benchmark
            )

            # Save synchronized Parquet
            out_parquet = os.path.join(sync_dir, f"{rec_id}_sync.parquet")
            df_sync.to_parquet(out_parquet, index=False, engine='pyarrow')
            result['sync_parquet_path'] = out_parquet
            result['sync_metadata'] = df_sync.attrs.get('sync_metadata', {})
            result['estimated_lag_s'] = df_sync.attrs.get('sync_metadata', {}).get('estimated_lag_s', 0.0)

            # Generate Causal Windows
            window_dict = generate_causal_windows(df_sync, manifest_entry, window_config)
            out_npz = os.path.join(wind_dir, f"{rec_id}_windows.npz")
            save_windows_npz(window_dict, out_npz)
            result['windows_npz_path'] = out_npz

            result['num_samples'] = len(df_sync)
            result['num_windows'] = len(window_dict['features'])
            result['duration_s'] = float(df_sync['time_s'].iloc[-1]) if len(df_sync) > 0 else 0.0
            result['num_gps_fixes'] = int(df_sync['phone_gps_is_new_fix'].sum())

        elif has_s:
            # Smartphone-only recording (e.g. Driver F in France or Driver H in England)
            df_phone, origin = ingest_smartphone_data(s_path)
            out_parquet = os.path.join(norm_dir, f"{rec_id}_phone.parquet")
            df_phone.to_parquet(out_parquet, index=False, engine='pyarrow')
            result['sync_parquet_path'] = out_parquet

            # Windowing on phone signals
            window_dict = generate_causal_windows(df_phone.rename(columns={'phone_time_s': 'time_s'}), manifest_entry, window_config)
            out_npz = os.path.join(wind_dir, f"{rec_id}_windows.npz")
            save_windows_npz(window_dict, out_npz)
            result['windows_npz_path'] = out_npz

            result['num_samples'] = len(df_phone)
            result['num_windows'] = len(window_dict['features'])
            result['duration_s'] = float(df_phone['phone_time_s'].iloc[-1]) if len(df_phone) > 0 else 0.0
            result['num_gps_fixes'] = int(df_phone['phone_gps_is_new_fix'].sum())

        elif has_v:
            # Reference-only recording (e.g. Driver C in England)
            df_ref, origin = ingest_reference_data(v_path)
            out_parquet = os.path.join(norm_dir, f"{rec_id}_ref.parquet")
            df_ref.to_parquet(out_parquet, index=False, engine='pyarrow')
            result['sync_parquet_path'] = out_parquet
            result['num_samples'] = len(df_ref)
            result['duration_s'] = float(df_ref['ref_time_s'].iloc[-1]) if len(df_ref) > 0 else 0.0

    except Exception as e:
        result['status'] = f"ERROR: {str(e)}"

    return result


def run_pipeline(
    manifest_path: str = 'data/manifest/recordings_manifest.csv',
    split_strategy: str = 'standard',
    limit: Optional[int] = None,
    output_dir: str = 'data/processed',
    auto_align_benchmark: bool = False
) -> pd.DataFrame:
    """
    Executes NAVRIS preprocessing pipeline across dataset manifest.
    """
    df_manifest = pd.read_csv(manifest_path)
    
    # 1. Assign group-based recording splits
    if split_strategy == 'unseen_driver':
        split_map = assign_unseen_driver_split(df_manifest)
    elif split_strategy == 'unseen_region':
        split_map = assign_unseen_region_split(df_manifest)
    elif split_strategy == 'unseen_vehicle':
        split_map = assign_unseen_vehicle_phone_split(df_manifest)
    else:
        split_map = assign_standard_recording_split(df_manifest)

    # Verify zero leakage
    verify_no_split_leakage(split_map)
    df_manifest['split'] = df_manifest['recording_id'].map(split_map)

    # Save updated manifest with split assignments
    split_manifest_path = os.path.join(output_dir, 'manifest_with_splits.csv')
    os.makedirs(output_dir, exist_ok=True)
    df_manifest.to_csv(split_manifest_path, index=False, encoding='utf-8')

    results = []
    recs_to_process = df_manifest.to_dict(orient='records')
    if limit is not None:
        recs_to_process = recs_to_process[:limit]

    print(f"Executing Phase 1 pipeline on {len(recs_to_process)} recordings (strategy: '{split_strategy}')...")

    for entry in recs_to_process:
        rec_id = entry['recording_id']
        print(f"Processing {rec_id} (Driver: {entry['driver']}, Split: {entry['split']})...")
        res = process_single_recording(
            rec_id=rec_id,
            manifest_entry=entry,
            output_dir=output_dir,
            auto_align_benchmark=auto_align_benchmark
        )
        results.append(res)

    df_results = pd.DataFrame(results)
    summary_csv = os.path.join(output_dir, 'pipeline_execution_summary.csv')
    df_results.to_csv(summary_csv, index=False)
    print(f"Pipeline execution completed. Summary saved to {summary_csv}")
    return df_results
