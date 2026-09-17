"""
CLI Runner for NAVRIS Phase 1 Pipeline.
Usage:
    python scripts/run_phase1_pipeline.py [--strategy standard] [--limit 10]
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from navris.pipeline import run_pipeline
from navris.windowing import WindowConfig

def main():
    parser = argparse.ArgumentParser(description="NAVRIS Phase 1 Data Ingestion & Preprocessing Pipeline")
    parser.add_argument('--manifest', type=str, default='data/manifest/recordings_manifest.csv', help='Path to recordings manifest')
    parser.add_argument('--strategy', type=str, default='standard', choices=['standard', 'unseen_driver', 'unseen_region', 'unseen_vehicle'], help='Split strategy')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of recordings to process (for testing/benchmarks)')
    parser.add_argument('--window-len', type=int, default=20, help='Window length in samples (20 samples = 2.0s @ 10Hz)')
    parser.add_argument('--stride', type=int, default=10, help='Window stride in samples (10 samples = 1.0s @ 10Hz)')
    parser.add_argument('--output-dir', type=str, default='data/processed', help='Output directory')

    args = parser.parse_args()

    print("==========================================================")
    print("  NAVRIS Phase 1 Data Ingestion & Preprocessing Pipeline  ")
    print("==========================================================")
    print(f"Manifest:       {args.manifest}")
    print(f"Split Strategy: {args.strategy}")
    print(f"Window Config:  Length={args.window_len} samples, Stride={args.stride} samples")
    print(f"Limit:          {args.limit or 'ALL'}")
    print(f"Output Dir:     {args.output_dir}")
    print("----------------------------------------------------------")

    df_results = run_pipeline(
        manifest_path=args.manifest,
        split_strategy=args.strategy,
        limit=args.limit,
        output_dir=args.output_dir
    )

    print("\nPipeline Execution Results Summary:")
    print(f"Total Processed:  {len(df_results)}")
    print(f"Success Count:    {(df_results['status'] == 'SUCCESS').sum()}")
    print(f"Total Windows:    {df_results['num_windows'].sum()}")
    print(f"Total GPS Fixes:  {df_results['num_gps_fixes'].sum()}")

if __name__ == '__main__':
    main()
