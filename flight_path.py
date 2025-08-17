#!/usr/bin/env python3
"""
Plot a single aircraft's trajectory and time-series from an ADS-B snapshot CSV.

Inputs:
  - A CSV with at least columns:
    icao24, timestamp, longitude, latitude, velocity, vertical_rate, geo_altitude

Outputs (saved PNGs by default):
  - flight_path_<icao24>.png         : Trajectory map (with Cartopy if available)
  - velocity_<icao24>.png            : Velocity vs elapsed seconds
  - vertical_rate_<icao24>.png       : Vertical rate vs elapsed seconds
  - altitude_<icao24>.png            : Geometric altitude vs elapsed seconds

Usage:
  python flight_path.py
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Try optional Cartopy; fall back to plain plot if missing.
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False


def parse_args():
    ap = argparse.ArgumentParser(description="Plot trajectory and time-series for one aircraft.")
    ap.add_argument("--input", "-i", default="flight_data_20250706_180507.csv",
                    help="Path to ADS-B snapshot CSV.")
    ap.add_argument("--icao24", "-a", default=None,
                    help="ICAO24 hex id to plot; if not provided, use the most frequent one.")
    ap.add_argument("--outdir", "-o", default=".",
                    help="Directory to save output figures.")
    ap.add_argument("--show", action="store_true",
                    help="Also display figures interactively.")
    return ap.parse_args()


def select_aircraft(df: pd.DataFrame, icao24: str | None) -> str:
    if icao24:
        return icao24
    return df["icao24"].value_counts().idxmax()


def prepare_plane_df(df: pd.DataFrame, icao24: str) -> pd.DataFrame:
    need_cols = ["longitude", "latitude", "timestamp"]
    sub = df[df["icao24"] == icao24].dropna(subset=need_cols).copy()
    sub["timestamp"] = pd.to_datetime(sub["timestamp"])
    sub.sort_values("timestamp", inplace=True)
    sub["elapsed_seconds"] = (sub["timestamp"] - sub["timestamp"].iloc[0]).dt.total_seconds()
    return sub


def plot_map(df_plane: pd.DataFrame, icao24: str, outpath: Path, show: bool):
    if HAS_CARTOPY:
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        lon_min = df_plane["longitude"].min() - 2
        lon_max = df_plane["longitude"].max() + 2
        lat_min = df_plane["latitude"].min() - 2
        lat_max = df_plane["latitude"].max() + 2
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
        ax.gridlines(draw_labels=True)

        ax.plot(df_plane["longitude"], df_plane["latitude"],
                color="red", marker="o", linewidth=2, markersize=3, label=f"ICAO24: {icao24}")
        ax.plot(df_plane.iloc[0]["longitude"], df_plane.iloc[0]["latitude"], "go", label="Start")
        ax.plot(df_plane.iloc[-1]["longitude"], df_plane.iloc[-1]["latitude"], "bx", label="End")
        ax.set_title(f"Flight Path for ICAO24: {icao24}")
        ax.legend()
    else:
        # Fallback: plain lon/lat plot
        plt.figure(figsize=(10, 6))
        plt.plot(df_plane["longitude"], df_plane["latitude"],
                 color="red", marker="o", linewidth=1.5, markersize=3)
        plt.scatter([df_plane.iloc[0]["longitude"]], [df_plane.iloc[0]["latitude"]], c="g", label="Start")
        plt.scatter([df_plane.iloc[-1]["longitude"]], [df_plane.iloc[-1]["latitude"]], c="b", label="End")
        plt.title(f"Flight Path (no basemap) for ICAO24: {icao24}")
        plt.xlabel("Longitude"); plt.ylabel("Latitude")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_series(df_plane: pd.DataFrame, col: str, ylabel: str, title: str, outpath: Path, show: bool):
    sub = df_plane.dropna(subset=[col]).copy()
    if sub.empty:
        return
    plt.figure(figsize=(10, 5))
    plt.plot(sub["elapsed_seconds"], sub[col], marker="o", linewidth=1.5)
    plt.xlabel("Elapsed Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    # Add zero line for vertical rate
    if col == "vertical_rate":
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    if show:
        plt.show()
    plt.close()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    if "icao24" not in df.columns:
        raise SystemExit(f"[ERROR] 'icao24' column not found in {args.input}")

    icao = select_aircraft(df, args.icao24)
    plane = prepare_plane_df(df, icao)
    if plane.empty:
        raise SystemExit(f"[ERROR] No valid rows for ICAO24={icao} in {args.input}")

    # Paths
    map_png = outdir / f"flight_path_{icao}.png"
    vel_png = outdir / f"velocity_{icao}.png"
    vr_png  = outdir / f"vertical_rate_{icao}.png"
    alt_png = outdir / f"altitude_{icao}.png"

    # Plots
    plot_map(plane, icao, map_png, args.show)
    plot_series(plane, "velocity", "Velocity (m/s)", f"Velocity Trend for ICAO24: {icao}", vel_png, args.show)
    plot_series(plane, "vertical_rate", "Vertical Rate (m/s)", f"Vertical Rate for ICAO24: {icao}", vr_png, args.show)
    plot_series(plane, "geo_altitude", "Geometric Altitude (m)", f"Altitude Change for ICAO24: {icao}", alt_png, args.show)

    print(f"Saved: {map_png}, {vel_png}, {vr_png}, {alt_png}")


if __name__ == "__main__":
    main()
