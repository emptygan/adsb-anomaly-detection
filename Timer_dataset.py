#!/usr/bin/env python3
"""
Timed ADS-B data collector for OpenSky `/states/all`.

Features
- Anonymous, Basic, or OAuth2 auth modes.
- Fixed-interval polling (default 15s), fixed-duration run (e.g., 210 minutes).
- Row-based segmentation: save CSV files every N rows to control memory.
- Optional geographic bounding box filter.
- Optional scheduled start time (local Australia/Adelaide time or UTC).
- Graceful shutdown on Ctrl+C (flushes buffer to a final CSV).

Inputs
- None required at runtime (API access depends on chosen auth mode).

Outputs
- CSV segment files in the output directory (e.g., flight_segment_YYYYmmdd_HHMMSS_seg001.csv)

Usage
  # Anonymous (limited rate)
  python Timer_dataset.py --duration-min 30 --output-dir ./segments

  # Basic auth
  python Timer_dataset.py --mode basic --username YOUR_USER --password YOUR_PASS --duration-min 60

  # OAuth2 (client credentials)
  python Timer_dataset.py --mode oauth2 --client-id CID --client-secret CSECRET --token-url https://example/token

  # With bbox and scheduled start (local time)
  python Timer_dataset.py --duration-min 45 --bbox 110,-45,155,-10 --start-at "2025-08-17T09:00"

Notes
- Times in CSV are UTC unless otherwise stated.
- Scheduled start defaults to local Australia/Adelaide if no timezone offset is provided.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import math
import json
import signal
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
try:
    # Python 3.9+: use zoneinfo for local scheduling
    from zoneinfo import ZoneInfo
    ADELAIDE_TZ = ZoneInfo("Australia/Adelaide")
except Exception:
    ADELAIDE_TZ = None  # fallback if not available

# OpenSky endpoint
OPEN_SKY_STATES_URL = "https://opensky-network.org/api/states/all"

# Columns as documented by OpenSky (keep consistent ordering)
CSV_COLUMNS = [
    "ts_utc",           # our fetch timestamp (UTC, iso)
    "time_position",    # unix
    "last_contact",     # unix
    "icao24",
    "callsign",
    "origin_country",
    "longitude",
    "latitude",
    "baro_altitude",
    "on_ground",
    "velocity",
    "true_track",
    "vertical_rate",
    "sensors",
    "geo_altitude",
    "squawk",
    "spi",
    "position_source"   # 0=ADS-B, 1=ASTERIX, 2=MLAT (per OpenSky)
]


def parse_args():
    ap = argparse.ArgumentParser(description="Timed ADS-B collector for OpenSky /states/all")
    ap.add_argument("--mode", choices=["anonymous", "basic", "oauth2"], default="anonymous",
                    help="Auth mode. anonymous|basic|oauth2 (default: anonymous)")
    ap.add_argument("--username", help="OpenSky username (basic)")
    ap.add_argument("--password", help="OpenSky password (basic)")
    ap.add_argument("--client-id", help="OAuth2 client_id (oauth2)")
    ap.add_argument("--client-secret", help="OAuth2 client_secret (oauth2)")
    ap.add_argument("--token-url", help="OAuth2 token URL (client_credentials)")
    ap.add_argument("--duration-min", type=int, default=210,
                    help="Total run duration in minutes (default: 210)")
    ap.add_argument("--interval-sec", type=int, default=15,
                    help="Polling interval in seconds (default: 15)")
    ap.add_argument("--segment-rows", type=int, default=12000,
                    help="Rows per CSV segment before rotating (default: 12000)")
    ap.add_argument("--output-dir", default="segments",
                    help="Directory to store CSV segments (default: ./segments)")
    ap.add_argument("--bbox", type=str, default=None,
                    help="Optional bbox as 'minlon,minlat,maxlon,maxlat'")
    ap.add_argument("--start-at", type=str, default=None,
                    help="Optional scheduled start time. ")
    return ap.parse_args()


class GracefulExit:
    """Helper for graceful shutdown on SIGINT/SIGTERM."""
    def __init__(self):
        self.stop = False
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum, frame):
        self.stop = True


def parse_bbox(bbox_str: str | None):
    if not bbox_str:
        return None
    try:
        parts = [float(x) for x in bbox_str.split(",")]
        if len(parts) != 4:
            raise ValueError
        minlon, minlat, maxlon, maxlat = parts
        return (minlon, minlat, maxlon, maxlat)
    except Exception:
        sys.exit("[ERROR] Invalid --bbox. Expected 'minlon,minlat,maxlon,maxlat'.")


def build_session(args) -> requests.Session:
    s = requests.Session()
    if args.mode == "basic":
        if not (args.username and args.password):
            sys.exit("[ERROR] Basic auth requires --username and --password")
        s.auth = (args.username, args.password)
    elif args.mode == "oauth2":
        if not (args.client_id and args.client_secret and args.token_url):
            sys.exit("[ERROR] OAuth2 auth requires --client-id, --client-secret and --token-url")
        token = fetch_oauth2_token(args.token_url, args.client_id, args.client_secret)
        s.headers.update({"Authorization": f"Bearer {token}"})
    # anonymous: nothing to set
    s.headers.update({"User-Agent": "ADS-B-Collector/1.0"})
    s.timeout = (10, 20)  # connect, read
    return s


def fetch_oauth2_token(token_url: str, client_id: str, client_secret: str) -> str:
    data = {"grant_type": "client_credentials"}
    resp = requests.post(token_url, data=data, auth=(client_id, client_secret), timeout=20)
    resp.raise_for_status()
    tok = resp.json().get("access_token")
    if not tok:
        raise RuntimeError("No access_token in OAuth2 response")
    return tok


def wait_until(start_at: str):
    """Block until the scheduled start time. Accepts ISO-like strings."""
    # Try parse with timezone first
    try:
        dt = datetime.fromisoformat(start_at)
        if dt.tzinfo is None:
            # Interpret as Adelaide local if tz missing
            if ADELAIDE_TZ is None:
                sys.exit("[ERROR] No zoneinfo available to interpret local time.")
            dt = dt.replace(tzinfo=ADELAIDE_TZ)
    except Exception:
        sys.exit("[ERROR] --start-at must be ISO-like, e.g. 2025-08-17T09:00 or 2025-08-16T23:30+09:30")

    now = datetime.now(tz=dt.tzinfo)
    if dt <= now:
        return
    seconds = (dt - now).total_seconds()
    print(f"Waiting until {dt.isoformat()} (about {int(seconds)}s)...")
    # Sleep in chunks to be interruptible
    end = time.time() + seconds
    while time.time() < end:
        time.sleep(min(30, max(1, end - time.time())))


def states_request(session: requests.Session, bbox=None):
    params = {}
    if bbox:
        minlon, minlat, maxlon, maxlat = bbox
        params.update({"lamin": minlat, "lomin": minlon, "lamax": maxlat, "lomax": maxlon})
    r = session.get(OPEN_SKY_STATES_URL, params=params)
    # capture remaining credits if provided
    rem = r.headers.get("X-Rate-Limit-Remaining") or r.headers.get("X-Rate-Limit-Remaining-Minute")
    r.raise_for_status()
    data = r.json()
    return data, rem


def flatten_states(payload: dict, fetch_ts_utc: str):
    out = []
    states = payload.get("states") or []
    time_position = payload.get("time")  # unix epoch (server time)
    for s in states:
        # OpenSky state vector spec indices
        rec = {
            "ts_utc": fetch_ts_utc,
            "time_position": s[3],
            "last_contact": s[4],
            "icao24": s[0],
            "callsign": s[1].strip() if isinstance(s[1], str) else s[1],
            "origin_country": s[2],
            "longitude": s[5],
            "latitude": s[6],
            "baro_altitude": s[7],
            "on_ground": s[8],
            "velocity": s[9],
            "true_track": s[10],
            "vertical_rate": s[11],
            "sensors": s[12],
            "geo_altitude": s[13],
            "squawk": s[14],
            "spi": s[15],
            "position_source": s[16],
        }
        # If API-level 'time' exists and specific fields are None, keep above values as-is.
        out.append(rec)
    return out


def segment_filename(base_dir: Path, seg_idx: int) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"flight_segment_{ts}_seg{seg_idx:03d}.csv"


def main():
    args = parse_args()
    bbox = parse_bbox(args.bbox)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.start_at:
        wait_until(args.start_at)

    session = build_session(args)
    stopper = GracefulExit()

    interval = max(1, int(args.interval_sec))
    run_seconds = max(1, args.duration_min * 60)

    total_rows = 0
    seg_rows = 0
    seg_index = 1
    buffer = []

    # Prepare first CSV writer on demand
    current_file = segment_filename(out_dir, seg_index)
    csv_file = open(current_file, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    writer.writeheader()

    t_start = time.time()
    next_tick = t_start
    print(f"Started collection (mode={args.mode}, interval={interval}s, duration={args.duration_min}min)")

    try:
        while (time.time() - t_start) < run_seconds and not stopper.stop:
            # Align to interval
            now = time.time()
            if now < next_tick:
                time.sleep(min(0.5, next_tick - now))
                continue
            next_tick = now + interval

            # Fetch
            fetch_iso = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
            try:
                payload, rem = states_request(session, bbox)
            except requests.HTTPError as e:
                # Minimal backoff on error
                time.sleep(5)
                continue
            except Exception:
                time.sleep(5)
                continue

            rows = flatten_states(payload, fetch_iso)
            if rows:
                writer.writerows(rows)
                seg_rows += len(rows)
                total_rows += len(rows)

            # Rotate segment if needed
            if seg_rows >= args.segment_rows:
                csv_file.close()
                print(f"Saved segment: {current_file.name} ({seg_rows} rows)")
                seg_rows = 0
                seg_index += 1
                current_file = segment_filename(out_dir, seg_index)
                csv_file = open(current_file, "w", newline="", encoding="utf-8")
                writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
                writer.writeheader()

    finally:
        # Final flush/close
        if not csv_file.closed:
            csv_file.close()
        elapsed = int(time.time() - t_start)
        mins = elapsed // 60
        secs = elapsed % 60
        print(f"Finished. Total rows: {total_rows}. Elapsed: {mins}m{secs}s. "
              f"Last segment: {current_file.name}")

if __name__ == "__main__":
    main()
