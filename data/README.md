# NAVRIS Data Directory

This directory contains metadata manifests and data processing scripts used in the NAVRIS research pipeline.

## External Dataset: IO-VNBD

NAVRIS is evaluated on the public **IO-VNBD (Input-Output Vehicle Navigation Benchmark Dataset)**:
- **Source Repository:** [onyekpeu/IO-VNBD](https://github.com/onyekpeu/IO-VNBD)
- **Publication Reference:** Onyekpeu et al., "IO-VNBD: An Input-Output Vehicle Navigation Benchmark Dataset"

The IO-VNBD dataset consists of real-world driving data collected simultaneously using:
1. **Reference Ground Truth (V-files):** VBOX Video HD2 logger and vehicle CAN bus (10 Hz).
2. **Smartphone Sensors (S-files):** Consumer smartphones logging 3-axis accelerometer, 3-axis gyroscope, 3-axis magnetometer, and GNSS position/speed (10 Hz).

## Intentionally Excluded Files

To keep this repository focused on code, reproducible methodology, and lightweight scientific summaries, raw and bulk generated data files are **not redistributed** in this repository:
- `data/raw/IO-VNBD/`: Raw external CSV datasets (~2.1 GB)
- `data/processed/normalized/*.parquet`: Intermediate normalized parquets
- `data/processed/synchronized/*.parquet`: Large 10 Hz synchronized benchmark parquets
- `data/processed/windows/*.npz`: Sliced feature arrays

## Manifests Included in Repository

The metadata manifests are tracked directly:
- `data/manifest/recordings_manifest.csv`: Detailed catalog of all 120+ benchmark drives, recording drivers, vehicle types, phone models, coordinate bounding boxes, and duration.
- `data/manifest/recordings_manifest.json`: JSON format of the dataset manifest.

Researchers wishing to reproduce NAVRIS results should obtain the raw dataset directly from the upstream [IO-VNBD repository](https://github.com/onyekpeu/IO-VNBD), place the uncompressed files under `data/raw/IO-VNBD/`, and execute the pipeline via `scripts/`.
