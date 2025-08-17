#!/usr/bin/env python3
"""
Cluster flights using PCA + KMeans, then save results and a PCA scatter plot.

Inputs:
  - flight_summary.csv (default) with numeric columns:
    records, duration_sec, min_altitude, max_altitude, avg_altitude,
    min_velocity, max_velocity, avg_velocity, start_altitude, end_altitude

Outputs:
  - flight_summary_kmeans.csv (default): original rows + columns [cluster, pca_x, pca_y]
  - pca_kmeans_clusters.png (default): PCA projection colored by cluster

Usage:
  python cluster_and_plot.py
"""

import argparse
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

REQUIRED_NUMERIC_COLS = [
    "records", "duration_sec",
    "min_altitude", "max_altitude", "avg_altitude",
    "min_velocity", "max_velocity", "avg_velocity",
    "start_altitude", "end_altitude",
]

def parse_args():
    ap = argparse.ArgumentParser(description="PCA + KMeans clustering for flight summaries.")
    ap.add_argument("--input", "-i", default="flight_summary.csv", help="Input CSV path.")
    ap.add_argument("--output-csv", "-o", default="flight_summary_kmeans.csv", help="Output CSV with clusters.")
    ap.add_argument("--output-plot", "-p", default="pca_kmeans_clusters.png", help="Output PNG path.")
    ap.add_argument("--k", type=int, default=4, help="Number of KMeans clusters.")
    return ap.parse_args()

def main():
    args = parse_args()

    # Load
    df = pd.read_csv(args.input)

    # Check required columns
    missing = [c for c in REQUIRED_NUMERIC_COLS if c not in df.columns]
    if missing:
        sys.exit(f"[ERROR] Missing required columns in {args.input}: {missing}")

    # Select and drop rows with NaNs on required numeric columns
    df_num = df[REQUIRED_NUMERIC_COLS].copy().dropna()
    # Keep only rows that survived the NaN drop in the main frame
    df = df.loc[df_num.index].copy()

    # Scale
    X = StandardScaler().fit_transform(df_num)

    # KMeans
    kmeans = KMeans(n_clusters=args.k, random_state=42, n_init="auto")
    df["cluster"] = kmeans.fit_predict(X)

    # PCA (2D)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    df["pca_x"] = X_pca[:, 0]
    df["pca_y"] = X_pca[:, 1]

    # Save CSV
    df.to_csv(args.output_csv, index=False, float_format="%.4f")

    # Plot (plain Matplotlib to keep dependencies minimal)
    plt.figure(figsize=(8, 5))
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        plt.scatter(sub["pca_x"], sub["pca_y"], s=18, label=f"Cluster {c}")
    plt.title("Flight Behavior Clusters (PCA)")
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.legend(title="Cluster", frameon=False)
    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=150)
    plt.close()

    print(f"Done: {args.output_csv}, {args.output_plot}")

if __name__ == "__main__":
    main()
