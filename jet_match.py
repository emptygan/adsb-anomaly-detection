#!/usr/bin/env python3
"""
Filter ADS-B snapshot data to include only jet aircraft from a known ICAO24 list.

Inputs:
  - hour_data.csv : ADS-B snapshot (must have icao24 column)
  - jet_db.csv    : list of ICAO24 identifiers for jet aircraft

Outputs:
  - filtered.csv  : only rows whose icao24 is in jet_db list

Usage:
  python jet_match.py
"""

import argparse
import pandas as pd
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser(description="Filter ADS-B data for jet aircraft ICAO24 codes.")
    ap.add_argument("--input", "-i", default="hour_09_data.csv", help="Input ADS-B CSV file.")
    ap.add_argument("--jet-db", "-j", default="jet_only_aircraft_opensky.csv", help="CSV file with jet ICAO24 codes.")
    ap.add_argument("--output", "-o", default="hour_09_filtered.csv", help="Output filtered CSV file.")
    return ap.parse_args()

def main():
    args = parse_args()

    df_hour = pd.read_csv(args.input)
    df_jet = pd.read_csv(args.jet_db)

    # Normalize ICAO24 format
    df_hour["icao24"] = df_hour["icao24"].astype(str).str.lower()
    df_jet["icao24"] = df_jet["icao24"].astype(str).str.lower()

    # Filter
    jet_icao_set = set(df_jet["icao24"].unique())
    df_filtered = df_hour[df_hour["icao24"].isin(jet_icao_set)].copy()

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(args.output, index=False)

    # Minimal output
    print(f"Filtered {len(df_filtered):,} / {len(df_hour):,} rows -> {args.output}")

if __name__ == "__main__":
    main()
