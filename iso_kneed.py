#!/usr/bin/env python3
"""
Isolation Forest anomaly detection with automatic threshold via KneeLocator.

Inputs:
  - flight_summary.csv (default) containing features per aircraft.

Outputs:
  - flight_summary_with_iso.csv: original data + iforest_score + iforest_anomaly
  - isoforest_scores.png: sorted anomaly scores with threshold line

Features used:
  min_altitude, max_altitude, mean_altitude,
  min_velocity, max_velocity, mean_velocity,
  latitude_range, longitude_range, duration

Usage:
  python iso_kneed.py
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from kneed import KneeLocator
import matplotlib.pyplot as plt

FEATURES = [
    "min_altitude", "max_altitude", "mean_altitude",
    "min_velocity", "max_velocity", "mean_velocity",
    "latitude_range", "longitude_range", "duration"
]

def parse_args():
    ap = argparse.ArgumentParser(description="Isolation Forest anomaly detection with knee point threshold.")
    ap.add_argument("--input", "-i", default="flight_summary.csv", help="Input flight summary CSV.")
    ap.add_argument("--output", "-o", default="flight_summary_with_iso.csv", help="Output CSV with anomaly labels.")
    ap.add_argument("--plot", "-p", default="isoforest_scores.png", help="Output PNG for score distribution.")
    return ap.parse_args()

def main():
    args = parse_args()

    df = pd.read_csv(args.input)
    df_clean = df.dropna(subset=FEATURES).copy()
    X = df_clean[FEATURES]

    # Train Isolation Forest
    clf = IsolationForest(random_state=42)
    clf.fit(X)
    scores = -clf.decision_function(X)  # higher = more anomalous

    # Knee point detection
    scores_sorted = np.sort(scores)
    knee = KneeLocator(range(len(scores_sorted)), scores_sorted, curve="convex", direction="increasing")
    if knee.knee is not None:
        threshold = scores_sorted[knee.knee]
    else:
        threshold = np.percentile(scores_sorted, 95)
    threshold *= 0.9  # relax threshold by 10%

    # Label anomalies
    df_clean["iforest_score"] = scores
    df_clean["iforest_anomaly"] = (scores > threshold).astype(int)

    # Merge results back
    df_final = df.merge(df_clean[["icao24", "iforest_anomaly", "iforest_score"]], on="icao24", how="left")
    df_final.to_csv(args.output, index=False)

    # Plot score distribution
    plt.figure(figsize=(8, 4))
    plt.plot(scores_sorted, label="Anomaly score")
    if knee.knee is not None:
        plt.axvline(x=knee.knee, color="r", linestyle="--", label=f"Knee point = {knee.knee}")
    plt.axhline(y=threshold, color="g", linestyle="--", label=f"Threshold = {threshold:.4f}")
    plt.title("Isolation Forest anomaly scores")
    plt.xlabel("Sample index (sorted)")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.plot, dpi=150)
    plt.close()

    print(f"Anomalies detected: {df_clean['iforest_anomaly'].sum()} / {len(df_clean)}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Saved: {args.output}, {args.plot}")

if __name__ == "__main__":
    main()
