#!/usr/bin/env python3
"""
ADS-B Data Consolidation Tool

Functions:
  - Merge multiple flight segment CSVs into one file.
  - Optionally create a random sample dataset.
  - Optionally split into time-based (hourly) files.

Inputs:
  - flight_segment_*.csv files in a directory.

Outputs:
  - consolidated_flight_data.csv
  - sample_flight_data.csv (optional)
  - time_splits/hour_##_data.csv (optional)

Usage:
  python merge.py --input-dir ./data --output consolidated.csv --sample-size 100000 --time-split
"""

import argparse
import os
import glob
import pandas as pd
from pathlib import Path

def find_segment_files(data_dir, pattern="flight_segment_*.csv"):
    """Find all segment CSV files in the given directory."""
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    print(f"Found {len(files)} segment files in {data_dir}")
    return files

def consolidate(files, output_file, chunk_size=10):
    """Merge CSV files into a single output file in chunks."""
    total_records = 0
    first_chunk = True
    for i in range(0, len(files), chunk_size):
        chunk_files = files[i:i+chunk_size]
        dfs = [pd.read_csv(f) for f in chunk_files]
        chunk_df = pd.concat(dfs, ignore_index=True)
        total_records += len(chunk_df)
        chunk_df.to_csv(output_file, mode='w' if first_chunk else 'a', header=first_chunk, index=False)
        first_chunk = False
    print(f"Merged {len(files)} files -> {output_file} ({total_records:,} rows)")

def create_sample(files, sample_file, sample_size=100000):
    """Create a random sample dataset from segment files."""
    dfs = []
    per_file = sample_size // len(files)
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df.sample(min(len(df), per_file), random_state=42))
    sample_df = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=42)
    sample_df.to_csv(sample_file, index=False)
    print(f"Sample dataset saved: {sample_file} ({len(sample_df):,} rows)")

def split_by_hour(files, output_dir="time_splits"):
    """Split merged files into hourly CSV files."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    hour_map = {}
    for f in files:
        df = pd.read_csv(f)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        for hour, group in df.groupby(df["timestamp"].dt.hour):
            hour_map.setdefault(hour, []).append(group)
    for hour, dfs in hour_map.items():
        combined = pd.concat(dfs, ignore_index=True)
        out_file = Path(output_dir) / f"hour_{hour:02d}_data.csv"
        combined.to_csv(out_file, index=False)
    print(f"Hourly splits saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="ADS-B Data Consolidation")
    parser.add_argument("--input-dir", "-d", default=".", help="Directory with flight_segment_*.csv files")
    parser.add_argument("--output", "-o", default="consolidated_flight_data.csv", help="Merged output CSV")
    parser.add_argument("--sample-file", "-s", default=None, help="Optional sample output CSV")
    parser.add_argument("--sample-size", type=int, default=100000, help="Sample size if sample-file is given")
    parser.add_argument("--time-split", action="store_true", help="Create hourly split files")
    args = parser.parse_args()

    files = find_segment_files(args.input_dir)
    if not files:
        print("No segment files found.")
        return

    consolidate(files, args.output)

    if args.sample_file:
        create_sample(files, args.sample_file, args.sample_size)

    if args.time_split:
        split_by_hour(files)

if __name__ == "__main__":
    main()
