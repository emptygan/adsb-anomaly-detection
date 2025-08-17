#!/usr/bin/env python3
"""
Aggregate anomaly detection results from multiple models (Z-score + turning angle, Isolation Forest, KMeans),
compute intersections, and output consensus anomalies.

Inputs:
  - flight_summary_combined.csv  : contains zscore_anomaly / turning_anomaly columns
  - flight_summary_with_iso.csv  : contains iforest_anomaly column
  - flight_summary_kmeans.csv    : contains kmeans_cluster column

Outputs:
  - final_consensus_anomalies.csv  : rows where all three models agree anomaly
  - model_comparison_summary.csv   : summary counts per method

Usage:
  python run_all_detection.py
"""

import argparse
import pandas as pd
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser(description="Combine anomaly detection results from multiple models.")
    ap.add_argument("--zscore", default="flight_summary_combined.csv", help="CSV with zscore_anomaly & turning_anomaly.")
    ap.add_argument("--iso", default="flight_summary_with_iso.csv", help="CSV with iforest_anomaly.")
    ap.add_argument("--kmeans", default="flight_summary_kmeans.csv", help="CSV with kmeans_cluster.")
    ap.add_argument("--output", "-o", default="final_consensus_anomalies.csv", help="Output CSV for consensus anomalies.")
    ap.add_argument("--summary", default="model_comparison_summary.csv", help="CSV for model comparison summary.")
    ap.add_argument("--cluster-id", type=int, default=0, help="KMeans cluster to treat as anomaly.")
    return ap.parse_args()

def main():
    args = parse_args()

    # Load data
    df_z = pd.read_csv(args.zscore)
    df_iso = pd.read_csv(args.iso)
    df_kmeans = pd.read_csv(args.kmeans)

    # Get anomalies from each model
    z_anomalies = set(df_z[(df_z.get("zscore_anomaly", False)) | (df_z.get("turning_anomaly", False))]["icao24"])
    iso_anomalies = set(df_iso[df_iso.get("iforest_anomaly", 0) == 1]["icao24"])
    kmeans_anomalies = set(df_kmeans[df_kmeans.get("kmeans_cluster", -1) == args.cluster_id]["icao24"])

    # Intersection
    common_anomalies = z_anomalies & iso_anomalies & kmeans_anomalies

    # Save consensus anomalies
    result_df = df_z[df_z["icao24"].isin(common_anomalies)]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output, index=False)

    # Save summary stats
    summary_stats = pd.DataFrame({
        "Method": ["Z-score + Turning Angle", "Isolation Forest", f"KMeans Cluster={args.cluster_id}", "All three"],
        "Anomaly count": [len(z_anomalies), len(iso_anomalies), len(kmeans_anomalies), len(common_anomalies)]
    })
    summary_stats.to_csv(args.summary, index=False)

    # Minimal output
    print(f"Z-score + Turning: {len(z_anomalies)} anomalies")
    print(f"Isolation Forest:  {len(iso_anomalies)} anomalies")
    print(f"KMeans (cluster={args.cluster_id}): {len(kmeans_anomalies)} anomalies")
    print(f"Consensus anomalies (all three): {len(common_anomalies)}")
    print(f"Saved: {args.output}, {args.summary}")

if __name__ == "__main__":
    main()
