#!/usr/bin/env python3
"""
Generate per-aircraft summary statistics from ADS-B flight data.

Inputs:
  - CSV file with at least columns: icao24, time/timestamp, altitude, velocity
    (default: hour_09_filtered.csv)

Outputs:
  - CSV file with aggregated features per aircraft (default: flight_summary.csv)

Features generated:
  records, duration_sec,
  min_altitude, max_altitude, avg_altitude,
  min_velocity, max_velocity, avg_velocity,
  start_altitude, end_altitude,
  start_time, end_time

Usage:
  python flight_summary_generator.py
"""

import argparse
import pandas as pd
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser(description="Generate summary features for each aircraft.")
    ap.add_argument("--input", "-i", default="hour_09_filtered.csv",
                    help="Path to cleaned ADS-B CSV.")
    ap.add_argument("--output", "-o", default="flight_summary.csv",
                    help="Path to output summary CSV.")
    return ap.parse_args()

def main():
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    # Load data
    df = pd.read_csv(in_path)
    df.columns = df.columns.str.lower().str.strip()

    # Identify time column
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], unit="s", errors="coerce")
    elif "timestamp" in df.columns:
        df["time"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        raise ValueError("No 'time' or 'timestamp' column found.")

    # Identify altitude column
    alt_cols = [c for c in df.columns if "altitude" in c and "rate" not in c]
    if "baro_altitude" in alt_cols:
        alt_col = "baro_altitude"
    elif "geo_altitude" in alt_cols:
        alt_col = "geo_altitude"
    elif alt_cols:
        alt_col = alt_cols[0]
    else:
        raise ValueError("No altitude column found.")

    # Identify velocity column
    vel_col = "velocity" if "velocity" in df.columns else None
    if not vel_col:
        raise ValueError("No 'velocity' column found.")

    # Group summarization
    def summarize_group(group):
        group = group.sort_values("time")
        return pd.Series({
            "records": len(group),
            "duration_sec": (group["time"].max() - group["time"].min()).total_seconds(),
            "min_altitude": group[alt_col].min(),
            "max_altitude": group[alt_col].max(),
            "avg_altitude": group[alt_col].mean(),
            "min_velocity": group[vel_col].min(),
            "max_velocity": group[vel_col].max(),
            "avg_velocity": group[vel_col].mean(),
            "start_altitude": group[alt_col].iloc[0],
            "end_altitude": group[alt_col].iloc[-1],
            "start_time": group["time"].min(),
            "end_time": group["time"].max()
        })

    df["icao24"] = df["icao24"].astype(str).str.lower()
    summary_df = df.groupby("icao24").apply(summarize_group).reset_index()

    summary_df.to_csv(out_path, index=False, float_format="%.2f")
    print(f"Saved {len(summary_df):,} aircraft summaries -> {out_path}")

if __name__ == "__main__":
    main()
