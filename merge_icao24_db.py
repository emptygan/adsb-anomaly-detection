#!/usr/bin/env python3
"""
Merge multiple aircraft database JSON files into a single CSV.

Supports:
  - Dict format: {icao24: {t, desc, ...}}
  - List format: [{icao24, t, desc, ...}, ...]

Inputs:
  - Directory containing .json aircraft database files.

Outputs:
  - Merged CSV with columns: icao24, type, description, source_file.

Usage:
  python merge_icao24_db.py --input-dir ./Aircraft_db --output all_aircraft_db.csv
"""

import argparse
import os
import json
import pandas as pd
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser(description="Merge multiple aircraft DB JSON files into a single CSV.")
    ap.add_argument("--input-dir", "-d", default="./Aircraft_db",
                    help="Directory containing .json aircraft database files.")
    ap.add_argument("--output", "-o", default="all_aircraft_db.csv",
                    help="Output CSV file.")
    return ap.parse_args()

def main():
    args = parse_args()
    folder_path = Path(args.input_dir)
    output_csv = Path(args.output)

    if not folder_path.exists():
        print(f"[ERROR] Input directory not found: {folder_path}")
        return

    records = []

    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".json"):
            fpath = folder_path / fname
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Dict format
                if isinstance(data, dict):
                    for icao24, info in data.items():
                        if isinstance(info, dict):
                            records.append({
                                "icao24": str(icao24).lower(),
                                "type": info.get("t"),
                                "description": info.get("desc", ""),
                                "source_file": fname
                            })

                # List format
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            records.append({
                                "icao24": str(item.get("icao24", "")).lower(),
                                "type": item.get("t"),
                                "description": item.get("desc", ""),
                                "source_file": fname
                            })

            except Exception as e:
                print(f"[WARNING] Failed to parse {fname}: {e}")

    # Save DataFrame
    df = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Merged {len(df):,} records -> {output_csv}")

if __name__ == "__main__":
    main()
