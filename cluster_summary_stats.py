#!/usr/bin/env python3
"""
Summarize statistics for each flight behavior cluster and show sample flights.

Inputs:
  - CSV file with at least columns: cluster, icao24, records, duration_sec,
    avg_altitude, avg_velocity, plus other numeric features.

Outputs:
  - Prints mean and standard deviation of selected features for each cluster.
  - Prints a small random sample of flights per cluster.

Usage:
  python cluster_summary_stats.py
"""

import argparse
import pandas as pd

REQUIRED_COLS = ["cluster", "records", "duration_sec",
                 "min_altitude", "max_altitude", "avg_altitude",
                 "min_velocity", "max_velocity", "avg_velocity",
                 "start_altitude", "end_altitude"]

def parse_args():
    ap = argparse.ArgumentParser(description="Summarize per-cluster statistics for flight data.")
    ap.add_argument("--input", "-i", default="flight_summary_kmeans.csv",
                    help="Path to the clustered flight summary CSV.")
    ap.add_argument("--samples", "-s", type=int, default=5,
                    help="Number of sample flights to show per cluster.")
    return ap.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.input)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"[WARNING] Missing expected columns: {missing}")

    # Group summary stats
    group_summary = df.groupby("cluster").agg({
        "records": ["mean", "std"],
        "duration_sec": ["mean", "std"],
        "min_altitude": ["mean"],
        "max_altitude": ["mean"],
        "avg_altitude": ["mean"],
        "min_velocity": ["mean"],
        "max_velocity": ["mean"],
        "avg_velocity": ["mean"],
        "start_altitude": ["mean"],
        "end_altitude": ["mean"]
    }).round(2)

    print("\n=== Cluster Summary Statistics (mean ± std) ===\n")
    print(group_summary)

    # Sample flights
    print(f"\n=== Sample {args.samples} flights per cluster ===")
    for c in sorted(df["cluster"].unique()):
        sample_df = df[df["cluster"] == c][
            ["icao24", "records", "duration_sec", "avg_altitude", "avg_velocity"]
        ].sample(min(args.samples, len(df[df["cluster"] == c])), random_state=42)
        print(f"\n--- Cluster {c} ---")
        print(sample_df.to_string(index=False))

if __name__ == "__main__":
    main()
