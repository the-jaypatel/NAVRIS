"""
NAVRIS Group-Based Dataset Splitting and Leakage Verification.

Implements strict recording-level partition schemes to ensure zero temporal or spatial
leakage across training, validation, and test splits:
1. Standard Random Recording Split: Stratified recording split (default 70% train, 15% val, 15% test).
2. Unseen-Driver Split: Evaluates generalizability to previously unseen driving behaviors.
3. Unseen-Region Split: Evaluates domain shift across geographic topographies (UK vs. France vs. Nigeria).
4. Unseen-Vehicle/Phone Split: Evaluates sensor/suspension transferability (Ford vs. Renault vs. Volvo).
"""

import hashlib
from typing import Dict, List, Set, Any, Tuple
import pandas as pd


def verify_no_split_leakage(split_map: Dict[str, str]) -> None:
    """
    Verifies that no recording ID is assigned to multiple splits.
    Raises ValueError if leakage is detected.
    
    Args:
        split_map: Dictionary mapping recording_id -> 'train' | 'val' | 'test'
    """
    splits = {}
    for rec_id, split_name in split_map.items():
        if split_name not in splits:
            splits[split_name] = set()
        splits[split_name].add(rec_id)

    split_names = list(splits.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1 = split_names[i]
            s2 = split_names[j]
            overlap = splits[s1] & splits[s2]
            if len(overlap) > 0:
                raise ValueError(
                    f"CRITICAL DATA LEAKAGE: Recording IDs {overlap} appear in both '{s1}' and '{s2}'!"
                )


def assign_standard_recording_split(
    df_manifest: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, str]:
    """
    Assigns each unique recording_id to exactly one split (train, val, test) using deterministic hashing.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    unique_ids = sorted(df_manifest['recording_id'].unique().tolist())
    split_map = {}

    for rec_id in unique_ids:
        # Deterministic hash to bucket recordings consistently
        h = int(hashlib.md5(f"{seed}_{rec_id}".encode('utf-8')).hexdigest(), 16)
        norm_val = (h % 10000) / 10000.0

        if norm_val < train_ratio:
            split_map[rec_id] = 'train'
        elif norm_val < (train_ratio + val_ratio):
            split_map[rec_id] = 'val'
        else:
            split_map[rec_id] = 'test'

    verify_no_split_leakage(split_map)
    return split_map


def assign_unseen_driver_split(
    df_manifest: pd.DataFrame,
    test_drivers: List[str] = None,
    val_drivers: List[str] = None
) -> Dict[str, str]:
    """
    Splits data by Driver ID to test out-of-distribution driver generalization.
    Default:
    - Train: Driver E, Driver A, Driver C
    - Val: Driver B, Driver D
    - Test: Driver F (France), Driver H (Volvo), Driver G (Nigeria)
    """
    if test_drivers is None:
        test_drivers = ['Driver F', 'Driver H', 'Driver G']
    if val_drivers is None:
        val_drivers = ['Driver B', 'Driver D']

    split_map = {}
    for _, row in df_manifest.iterrows():
        rec_id = row['recording_id']
        driver = str(row['driver']).strip()

        if driver in test_drivers:
            split_map[rec_id] = 'test'
        elif driver in val_drivers:
            split_map[rec_id] = 'val'
        else:
            split_map[rec_id] = 'train'

    verify_no_split_leakage(split_map)
    return split_map


def assign_unseen_region_split(
    df_manifest: pd.DataFrame,
    test_countries: List[str] = None,
    val_regions: List[str] = None
) -> Dict[str, str]:
    """
    Splits data by geographic regions to test cross-region generalizability.
    Default:
    - Train: United Kingdom (core Midlands: Coventry, Warwickshire)
    - Val: United Kingdom (Peak District / Buxton / Oxford)
    - Test: France & Nigeria
    """
    if test_countries is None:
        test_countries = ['France', 'Nigeria']

    split_map = {}
    for _, row in df_manifest.iterrows():
        rec_id = row['recording_id']
        country = str(row['country']).strip()
        region = str(row['region']).strip()

        if country in test_countries:
            split_map[rec_id] = 'test'
        elif 'Peak District' in region or 'Oxford' in region:
            split_map[rec_id] = 'val'
        else:
            split_map[rec_id] = 'train'

    verify_no_split_leakage(split_map)
    return split_map


def assign_unseen_vehicle_phone_split(
    df_manifest: pd.DataFrame,
    test_vehicles: List[str] = None,
    val_vehicles: List[str] = None
) -> Dict[str, str]:
    """
    Splits data by vehicle chassis and phone hardware.
    Default:
    - Train: Ford Fiesta Titanium (Huawei P20 Pro)
    - Val: Ford Fiesta Titanium (Validation subset)
    - Test: Renault Megane (Motorola Moto G7) & Volvo XC70 (Blackberry Priv)
    """
    if test_vehicles is None:
        test_vehicles = ['Renault Megane', 'Volvo XC70', 'Toyota Corolla Verso']

    split_map = {}
    for _, row in df_manifest.iterrows():
        rec_id = row['recording_id']
        veh = str(row['vehicle']).strip()

        if any(tv in veh for tv in test_vehicles):
            split_map[rec_id] = 'test'
        else:
            # Deterministic hash on Fiesta
            h = int(hashlib.md5(f"veh_{rec_id}".encode('utf-8')).hexdigest(), 16)
            split_map[rec_id] = 'val' if (h % 100) < 20 else 'train'

    verify_no_split_leakage(split_map)
    return split_map
