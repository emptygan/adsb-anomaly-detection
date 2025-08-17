#!/usr/bin/env python3
"""
Z-score anomaly detection + turning angle anomaly detection for flight summary data.

- Reads per-aircraft summary stats CSV and raw ADS-B snapshot.
- Flags anomalies based on:
    1) Z-score on selected numeric features.
    2) Large turning angles (> angle_threshold).
- Outputs combined anomaly CSV with new flags.
"""

import argparse
import pandas as pd
import numpy as np
from scipy.stats import zscore
from geopy.distance import geodesic

def parse_args():
    ap = argparse.ArgumentParser(description="Z-score + turning-angle anomaly detection.")
    ap.add_argument("--summary", "-s", default="flight_summary.csv", help="Per-aircraft summary CSV.")
    ap.add_argument("--raw", "-r", default="hour_09_filtered.csv", help="Raw ADS-B snapshot CSV.")
    ap.add_argument("--out", "-o", default="flight_summary_combined.csv", help="Output CSV.")
    ap.add_argument("--z-thresh", type=float, default=3.0, help="Z-score threshold.")
    ap.add_argument("--angle-thresh", type=float, default=120.0, help="Turning angle threshold in degrees.")
    return ap.parse_args()

def compute_turning_angles(group):
    group = group.sort_values("timestamp").dropna(subset=["latitude", "longitude"])
    coords = group[["latitude", "longitude"]].values
    angles = []

    for i in range(1, len(coords) - 1):
        dist1 = geodesic(coords[i - 1], coords[i]).km
        dist2 = geodesic(coords[i], coords[i + 1]).km
        if dist1 < 0.5 or dist2 < 0.5:
            continue
        a, b, c = np.array(coords[i - 1]), np.array(coords[i]), np.array(coords[i + 1])
        ba, bc = a - b, c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle_deg = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        angles.append(angle_deg)

    return angles

def main():
    args = parse_args()

    # Load summary
    summary_df = pd.read_csv(args.summary)
    features = [
        "min_altitude", "max_altitude", "mean_altitude",
        "min_velocity", "max_velocity", "mean_velocity",
        "latitude_range", "longitude_range", "duration"
    ]
    z_scores = summary_df[features].apply(zscore)
    summary_df["z_score_max"] = z_scores.abs().max(axis=1)
    summary_df["zscore_anomaly"] = summary_df["z_score_max"] > args.z_thresh

    # Turning angle anomalies
    df_raw = pd.read_csv(args.raw)
    turning_anomalies = set()
    for icao, group in df_raw.groupby("icao24"):
        if any(angle > args.angle_thresh for angle in compute_turning_angles(group)):
            turning_anomalies.add(icao)
    summary_df["turning_anomaly"] = summary_df["icao24"].isin(turning_anomalies)

    # Summary
    z_set = set(summary_df.loc[summary_df["zscore_anomaly"], "icao24"])
    turning_set = turning_anomalies
    both_set = z_set & turning_set
    print(f"Z-score anomalies: {len(z_set)}")
    print(f"Turning anomalies: {len(turning_set)}")
    print(f"Combined: {len(both_set)}")

    # Save
    summary_df.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
