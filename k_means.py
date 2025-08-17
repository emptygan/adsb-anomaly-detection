#!/usr/bin/env python3
"""
Perform PCA + KMeans clustering on flight summary data.

Inputs:
  - flight_summary.csv (default) with numeric flight features.

Outputs:
  - flight_summary_kmeans.csv : original data + PCA coords + cluster labels
  - pca_kmeans_clusters.png   : scatter plot of PCA-reduced clusters

Usage:
  python k_means.py
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

FEATURES = [
    "min_altitude", "max_altitude", "mean_altitude",
    "min_velocity", "max_velocity", "mean_velocity",
    "latitude_range", "longitude_range", "duration"
]

def parse_args():
    ap = argparse.ArgumentParser(description="PCA + KMeans clustering for flight summaries.")
    ap.add_argument("--input", "-i", default="flight_summary.csv", help="Input CSV with flight summary features.")
    ap.add_argument("--output", "-o", default="flight_summary_kmeans.csv", help="Output CSV with PCA & cluster labels.")
    ap.add_argument("--plot", "-p", default="pca_kmeans_clusters.png", help="Output PNG for PCA scatter plot.")
    ap.add_argument("--clusters", "-k", type=int, default=3, help="Number of KMeans clusters.")
    return ap.parse_args()

def main():
    args = parse_args()

    df = pd.read_csv(args.input)
    df_clean = df.dropna(subset=FEATURES).copy()

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[FEATURES])

    # PCA to 2D
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df_clean["PCA1"] = X_pca[:, 0]
    df_clean["PCA2"] = X_pca[:, 1]

    # KMeans clustering
    kmeans = KMeans(n_clusters=args.clusters, random_state=42)
    df_clean["kmeans_cluster"] = kmeans.fit_predict(X_pca)

    # Merge back to original
    df_out = df.merge(df_clean[["icao24", "PCA1", "PCA2", "kmeans_cluster"]], on="icao24", how="left")

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output, index=False)

    # Scatter plot
    plt.figure(figsize=(8, 6))
    for cluster in sorted(df_clean["kmeans_cluster"].unique()):
        subset = df_clean[df_clean["kmeans_cluster"] == cluster]
        plt.scatter(subset["PCA1"], subset["PCA2"], label=f"Cluster {cluster}", alpha=0.6)
    plt.title("PCA + KMeans Clustering of Flight Summaries")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(args.plot, dpi=150)
    plt.close()

    print(f"Saved: {args.output} ({len(df_out)} rows), {args.plot}")

if __name__ == "__main__":
    main()
