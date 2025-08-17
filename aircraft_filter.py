#!/usr/bin/env python3
"""
Filter jet-powered commercial aircraft from a registration database.

Inputs:
  - CSV with at least these columns: icao24, model, typecode
    (default: aircraft-database-complete-2025-02.csv)

Outputs:
  - CSV containing only jet-powered commercial aircraft records
    (default: jet_only_aircraft_opensky.csv)

Usage:
  python aircraft_filter.py
"""

import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Filter jet-powered commercial aircraft.")
    parser.add_argument("--input", "-i", default="aircraft-database-complete-2025-02.csv",
                        help="Path to input aircraft database CSV.")
    parser.add_argument("--output", "-o", default="jet_only_aircraft_opensky.csv",
                        help="Path to output CSV for jet aircraft only.")
    args = parser.parse_args()

    # Load
    df = pd.read_csv(args.input, quotechar="'", on_bad_lines="skip", low_memory=False)
    df.columns = df.columns.str.strip().str.strip("'").str.strip('"')

    # Basic column sanity
    required = {"icao24", "model", "typecode"}
    missing = required - set(df.columns)
    # If some columns are missing, create them to avoid KeyErrors
    for col in missing:
        df[col] = pd.NA

    # Normalize
    df = df[(df["model"].notna()) | (df["typecode"].notna())].copy()
    df["icao24"] = df["icao24"].astype(str).str.lower()

    # Jet keywords (simple heuristic; extend as needed)
    jet_keywords = [
        # Boeing
        "737", "738", "739", "747", "757", "767", "777", "787",
        # Airbus
        "A318", "A319", "A320", "A321", "A330", "A340", "A350", "A380",
        # McDonnell Douglas
        "CRJ", "CL65", "DC9", "DC10", "MD8", "MD9",
        # Embraer E-Jets
        "E170", "E175", "E190", "E195",
    ]

    # Vectorized contains on TYPECODE/MODEL (case-insensitive)
    tc = df["typecode"].astype(str).str.upper()
    md = df["model"].astype(str).str.upper()
    jet_mask = pd.Series(False, index=df.index)
    for kw in jet_keywords:
        jet_mask |= tc.str.contains(kw, na=False) | md.str.contains(kw, na=False)

    jets = df.loc[jet_mask].copy()
    jets.to_csv(args.output, index=False)

    print(f"Filtered jets: {len(jets):,} / {len(df):,} -> {args.output}")

if __name__ == "__main__":
    main()
