#!/usr/bin/env python3
"""
Basic exploratory visualization for ADS-B flight data.

Features:
- Summary stats: shape, info, describe, missing values.
- Histograms: velocity, barometric altitude, vertical rate.
- Top 10 origin countries (bar chart).
- Scatter plot of aircraft positions (colored by origin country).
- Flag simple anomalies (velocity < 1 m/s or missing baro_altitude).

Usage:
  python vis.py --input flight_data.csv
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    ap = argparse.ArgumentParser(description="Basic visualization for ADS-B flight data.")
    ap.add_argument("--input", "-i", required=True, help="Path to flight data CSV file.")
    return ap.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.input)

    # Summary info
    print(f"Records: {df.shape[0]}, Columns: {df.shape[1]}")
    print(df.info())
    print(df.describe())
    print("Missing values per column:\n", df.isnull().sum())

    # Velocity distribution
    plt.figure(figsize=(15, 5))
    sns.histplot(df['velocity'], bins=100, kde=True)
    plt.title("Velocity Distribution")
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Frequency")
    plt.show()

    # Barometric altitude distribution
    plt.figure(figsize=(15, 5))
    sns.histplot(df['baro_altitude'], bins=100, kde=True)
    plt.title("Barometric Altitude Distribution")
    plt.xlabel("Altitude (m)")
    plt.ylabel("Frequency")
    plt.show()

    # Vertical rate distribution
    plt.figure(figsize=(15, 5))
    sns.histplot(df['vertical_rate'], bins=100, kde=True)
    plt.title("Vertical Rate Distribution")
    plt.xlabel("Vertical Rate (m/s)")
    plt.ylabel("Frequency")
    plt.show()

    # Top 10 origin countries
    top_countries = df['origin_country'].value_counts().head(10)
    sns.barplot(x=top_countries.values, y=top_countries.index)
    plt.title("Top 10 Aircraft Origin Countries")
    plt.xlabel("Count")
    plt.ylabel("Country")
    plt.show()

    # Position scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x='longitude', y='latitude',
        hue='origin_country', legend=False, s=10, alpha=0.5
    )
    plt.title("Aircraft Position Distribution")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.show()

    # Simple anomaly check
    abnormal = df[(df['velocity'] < 1) | (df['baro_altitude'].isnull())]
    print(f"Possible anomalies: {abnormal.shape[0]} records")

if __name__ == "__main__":
    main()
