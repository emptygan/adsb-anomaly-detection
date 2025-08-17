#!/usr/bin/env python3
"""
Advanced anomaly detection and security-oriented visualization for ADS-B data.

Methods:
  - Z-score (on per-aircraft aggregated features)
  - IQR outlier rule (robust)
  - Isolation Forest (unsupervised)
  - Behavior rules (extreme speed/altitude, large jumps, long gaps)

Consensus:
  - An aircraft is "combined_anomaly" if at least 2 methods flag it.

Inputs (CSV):
  - Cleaned ADS-B snapshot with columns at least:
    icao24, timestamp, origin_country, longitude, latitude,
    baro_altitude or geo_altitude, velocity, vertical_rate, on_ground

Outputs:
  - PNG (9-panel figure): adsb_anomaly_detection_<ts>.png
  - TXT (security report): adsb_security_analysis_<ts>.txt
  - CSV (per-aircraft aggregated metrics + flags): flight_params_<ts>.csv

Usage:
  python vis_pro.py --input hour_09_filtered.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.ensemble import IsolationForest



# Arguments

def parse_args():
    ap = argparse.ArgumentParser(description="ADS-B anomaly detection (Z-score / IQR / IF) + security report.")
    ap.add_argument("--input", "-i", default="hour_09_filtered.csv", help="Input cleaned ADS-B CSV.")
    ap.add_argument("--outdir", "-o", default=".", help="Output directory for figures, reports and CSV.")
    ap.add_argument("--contamination", type=float, default=0.02, help="IsolationForest contamination (default 0.02).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--no-figure", action="store_true", help="Skip figure generation.")
    return ap.parse_args()



# Utilities

def _choose_alt_col(df: pd.DataFrame) -> str:
    if "baro_altitude" in df.columns:
        return "baro_altitude"
    if "geo_altitude" in df.columns:
        return "geo_altitude"
    # fallback: try any column containing 'altitude' but not 'rate'
    for c in df.columns:
        lc = c.lower()
        if "altitude" in lc and "rate" not in lc:
            return c
    raise ValueError("No altitude column found (expected baro_altitude or geo_altitude).")


def _as_dt(series: pd.Series) -> pd.Series:
    # support unix seconds or ISO string
    s = pd.to_datetime(series, errors="coerce", utc=True)
    if s.isna().all():
        # try unix seconds
        s = pd.to_datetime(series, unit="s", errors="coerce", utc=True)
    return s


def _haversine(lon1, lat1, lon2, lat2):
    # distances in meters
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(a))


# Aggregation per aircraft

def build_flight_params(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    alt_col = _choose_alt_col(df)

    # normalize types
    df["icao24"] = df["icao24"].astype(str).str.lower()
    df["timestamp"] = _as_dt(df.get("timestamp", df.get("time")))
    df = df.dropna(subset=["icao24", "timestamp", "longitude", "latitude"])
    df = df.sort_values(["icao24", "timestamp"])

    # per-aircraft aggregates
    def _agg(g: pd.DataFrame) -> pd.Series:
        # time
        tmin, tmax = g["timestamp"].min(), g["timestamp"].max()
        duration = (tmax - tmin).total_seconds()

        # movement deltas
        lon = g["longitude"].astype(float).to_numpy()
        lat = g["latitude"].astype(float).to_numpy()
        dist = _haversine(lon[:-1], lat[:-1], lon[1:], lat[1:])
        total_dist = float(np.nansum(dist))
        max_jump = float(np.nanmax(dist)) if len(dist) else 0.0

        # temporal gaps
        t = g["timestamp"].astype("int64").to_numpy() // 10**9
        gaps = np.diff(t) if len(t) > 1 else np.array([])
        avg_gap = float(np.nanmean(gaps)) if gaps.size else 0.0
        max_gap = float(np.nanmax(gaps)) if gaps.size else 0.0

        # altitude & velocity
        alt = g[alt_col].astype(float)
        vel = g.get("velocity", pd.Series(index=g.index, dtype=float)).astype(float)
        vr  = g.get("vertical_rate", pd.Series(index=g.index, dtype=float)).astype(float)

        # climb/descent extremes (approx by vertical_rate)
        max_climb = float(np.nanmax(vr)) if len(vr) else np.nan
        max_descent = float(np.nanmin(vr)) if len(vr) else np.nan

        # ground ratio / status changes
        on_ground = g.get("on_ground", pd.Series(False, index=g.index))
        ground_ratio = float(np.mean(on_ground)) if len(on_ground) else 0.0
        status_changes = int(np.sum(on_ground.astype(int).diff().fillna(0).abs()))

        return pd.Series({
            "records": len(g),
            "duration_sec": duration,
            "min_altitude": float(alt.min()),
            "max_altitude": float(alt.max()),
            "avg_altitude": float(alt.mean()),
            "min_velocity": float(vel.min()),
            "max_velocity": float(vel.max()),
            "avg_velocity": float(vel.mean()),
            "latitude_range": float(g["latitude"].max() - g["latitude"].min()),
            "longitude_range": float(g["longitude"].max() - g["longitude"].min()),
            "start_altitude": float(alt.iloc[0]) if len(alt) else np.nan,
            "end_altitude": float(alt.iloc[-1]) if len(alt) else np.nan,
            "max_climb": max_climb,
            "max_descent": max_descent,
            "ground_ratio": ground_ratio,
            "status_changes": status_changes,
            "total_distance_m": total_dist,
            "max_position_jump_m": max_jump,
            "avg_update_gap_s": avg_gap,
            "max_update_gap_s": max_gap,
            "start_time": tmin,
            "end_time": tmax,
            "origin_country": g.get("origin_country", pd.Series(["unknown"])).iloc[0]
        })

    params = df.groupby("icao24", sort=False).apply(_agg).reset_index()
    return params


# Anomaly methods

Z_FEATURES = [
    "avg_altitude", "max_altitude", "avg_velocity", "max_velocity",
    "latitude_range", "longitude_range", "duration_sec",
    "total_distance_m", "max_position_jump_m", "avg_update_gap_s"
]

IQR_FEATURES = [
    "avg_altitude", "avg_velocity", "max_position_jump_m", "avg_update_gap_s"
]

IF_FEATURES = [
    "min_altitude", "max_altitude", "avg_altitude",
    "min_velocity", "max_velocity", "avg_velocity",
    "latitude_range", "longitude_range", "duration_sec"
]


def flag_zscore(df: pd.DataFrame, thresh: float = 4.0) -> pd.Series:
    X = df[Z_FEATURES].astype(float)
    z = np.abs(stats.zscore(X, nan_policy="omit"))
    # row flagged if any feature z > thresh
    flags = (z > thresh).any(axis=1)
    return pd.Series(flags.astype(int), index=df.index, name="zscore_anomaly")


def flag_iqr(df: pd.DataFrame, mult: float = 3.0) -> pd.Series:
    X = df[IQR_FEATURES].astype(float).copy()
    flags = pd.Series(False, index=df.index)
    for col in IQR_FEATURES:
        s = X[col]
        q1, q3 = np.nanpercentile(s, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - mult * iqr, q3 + mult * iqr
        flags |= (s < lo) | (s > hi)
    return pd.Series(flags.astype(int), index=df.index, name="iqr_anomaly")


def flag_isoforest(df: pd.DataFrame, contamination: float = 0.02, seed: int = 42) -> tuple[pd.Series, pd.Series]:
    X = df[IF_FEATURES].astype(float).fillna(method="ffill").fillna(method="bfill")
    clf = IsolationForest(random_state=seed, contamination=contamination)
    clf.fit(X)
    scores = -clf.decision_function(X)  # higher is more anomalous
    preds = (clf.predict(X) == -1).astype(int)
    return pd.Series(preds, index=df.index, name="iforest_anomaly"), pd.Series(scores, index=df.index, name="iforest_score")


def flag_rules(df: pd.DataFrame) -> pd.Series:
    # simple behavior rules
    extreme_alt = (df["max_altitude"] > 15000) | (df["min_altitude"] < -500)  # meters
    extreme_speed = (df["max_velocity"] > 320)  # m/s (~622 knots)
    huge_jump = (df["max_position_jump_m"] > 50000)  # >50 km single-step jump
    long_gap = (df["max_update_gap_s"] > 600)       # >10 minutes without update
    flags = extreme_alt | extreme_speed | huge_jump | long_gap
    return pd.Series(flags.astype(int), index=df.index, name="rule_anomaly")



# Plotting

def plot_nine_panels(params: pd.DataFrame, out_png: Path):
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    ax = axes.ravel()

    # 1: Anomaly counts bar
    counts = [
        ("Z-score", int(params["zscore_anomaly"].sum())),
        ("IQR", int(params["iqr_anomaly"].sum())),
        ("IForest", int(params["iforest_anomaly"].sum())),
        ("Rules", int(params["rule_anomaly"].sum())),
        ("Combined", int(params["combined_anomaly"].sum())),
    ]
    ax[0].bar([c[0] for c in counts], [c[1] for c in counts])
    ax[0].set_title("Anomaly Counts")
    ax[0].set_ylabel("Count")

    # 2: Altitude histogram
    ax[1].hist(params["avg_altitude"].dropna(), bins=40)
    ax[1].set_title("Average Altitude")

    # 3: Velocity histogram
    ax[2].hist(params["avg_velocity"].dropna(), bins=40)
    ax[2].set_title("Average Velocity")

    # 4: Distance vs Duration
    ax[3].scatter(params["duration_sec"], params["total_distance_m"], s=10, alpha=0.6)
    ax[3].set_xlabel("Duration (s)")
    ax[3].set_ylabel("Total Distance (m)")
    ax[3].set_title("Distance vs Duration")

    # 5: Max position jump
    ax[4].hist(params["max_position_jump_m"].dropna(), bins=40)
    ax[4].set_title("Max Position Jump (m)")

    # 6: Update gap
    ax[5].hist(params["avg_update_gap_s"].dropna(), bins=40)
    ax[5].set_title("Average Update Gap (s)")

    # 7: Country top-10
    topc = params["origin_country"].value_counts().head(10)
    ax[6].barh(topc.index[::-1], topc.values[::-1])
    ax[6].set_title("Top Origin Countries")

    # 8: IF score distribution
    ax[7].hist(params["iforest_score"].dropna(), bins=40)
    ax[7].set_title("Isolation Forest Scores")

    # 9: Combined vs non-combined (duration vs jump)
    c1 = params[params["combined_anomaly"] == 1]
    c0 = params[params["combined_anomaly"] != 1]
    ax[8].scatter(c0["duration_sec"], c0["max_position_jump_m"], s=10, alpha=0.5, label="Normal")
    ax[8].scatter(c1["duration_sec"], c1["max_position_jump_m"], s=12, alpha=0.8, label="Combined")
    ax[8].set_xlabel("Duration (s)")
    ax[8].set_ylabel("Max Jump (m)")
    ax[8].set_title("Combined Anomalies")
    ax[8].legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# Report

def write_report(params: pd.DataFrame, out_txt: Path):
    tot = len(params)
    zc = int(params["zscore_anomaly"].sum())
    iq = int(params["iqr_anomaly"].sum())
    ifc = int(params["iforest_anomaly"].sum())
    rc = int(params["rule_anomaly"].sum())
    cmb = int(params["combined_anomaly"].sum())

    lines = []
    lines.append("ADS-B Security Analysis Report\n")
    lines.append(f"Total aircraft analyzed: {tot}\n")
    lines.append(f"Z-score anomalies: {zc}\nIQR anomalies: {iq}\nIsolation Forest anomalies: {ifc}\nRule-based anomalies: {rc}\n")
    lines.append(f"Combined anomalies (>=2 methods): {cmb}\n")

    # top combined examples
    top = params[params["combined_anomaly"] == 1].sort_values("iforest_score", ascending=False)
    lines.append("\nTop combined anomalies (by IF score):")
    for _, r in top.head(10).iterrows():
        lines.append(
            f"- {r['icao24']}: max_jump={r['max_position_jump_m']:.0f} m, "
            f"max_gap={r['max_update_gap_s']:.0f} s, vmax={r['max_velocity']:.1f} m/s, "
            f"country={r.get('origin_country','unknown')}"
        )

    lines.append("\nNotes:")
    lines.append("- Many anomalies may stem from ADS-B reception gaps or message jumps rather than operational events.")
    lines.append("- Consider integrating weather/route/airport proximity for better interpretability.")
    lines.append("- Thresholds used: Z=4.0, IQR=3.0*IQR, IsolationForest contamination as configured.\n")

    out_txt.write_text("\n".join(lines), encoding="utf-8")



# Main

def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    params = build_flight_params(df)

    # anomaly flags
    params["zscore_anomaly"] = flag_zscore(params, thresh=4.0)
    params["iqr_anomaly"] = flag_iqr(params, mult=3.0)
    if_flags, if_scores = flag_isoforest(params, contamination=args.contamination, seed=args.seed)
    params["iforest_anomaly"] = if_flags
    params["iforest_score"] = if_scores
    params["rule_anomaly"] = flag_rules(params)

    # combined (>= 2 methods true)
    methods = params[["zscore_anomaly", "iqr_anomaly", "iforest_anomaly", "rule_anomaly"]].astype(int)
    params["combined_anomaly"] = (methods.sum(axis=1) >= 2).astype(int)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    csv_path = outdir / f"flight_params_{ts}.csv"
    fig_path = outdir / f"adsb_anomaly_detection_{ts}.png"
    txt_path = outdir / f"adsb_security_analysis_{ts}.txt"

    params.to_csv(csv_path, index=False)

    if not args.no_figure:
        plot_nine_panels(params, fig_path)

    write_report(params, txt_path)

    print(f"Saved: {csv_path}")
    if not args.no_figure:
        print(f"Saved: {fig_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
