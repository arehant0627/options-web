#!/usr/bin/env python3
"""
record_snapshot.py — take one premium snapshot and save it.

Designed to run on a schedule (GitHub Actions) so that intraday premium history
accumulates whether or not anybody has the dashboard open. The dashboard's
"Movers -> Since your last run" tab reads whatever this leaves in snapshots/.

    python record_snapshot.py                 # next Friday
    python record_snapshot.py --expiry 2026-09-18
    python record_snapshot.py --rung 2 --keep 400
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

import pandas as pd

from core import SNAP_DIR, list_expiries, load_universe, run_screen, save_snapshot


def next_friday(min_dte=1, max_dte=9):
    for e, d, wd in list_expiries():
        if wd == "Fri" and min_dte <= d <= max_dte:
            return e
    return None


def prune(keep: int):
    """Keep the newest `keep` snapshot files so the repo doesn't grow forever."""
    files = sorted(SNAP_DIR.glob("*.csv"))
    for p in files[:-keep] if len(files) > keep else []:
        p.unlink()
        print(f"pruned {p.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expiry", help="YYYY-MM-DD; default = next Friday")
    ap.add_argument("--rung", type=int, default=1)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--keep", type=int, default=500)
    a = ap.parse_args()

    expiry = a.expiry or next_friday()
    if not expiry:
        print("no Friday expiry in window — nothing to record")
        return 0

    tickers, src = load_universe()
    print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC | "
          f"{len(tickers)} tickers ({src}) | expiry {expiry}")

    cfg = {"rung": a.rung, "earnings_mode": "any", "use_last": True,
           "min_bid": 0.05, "min_oi": 0, "max_spread": 1.0,
           "min_history_rows": 25, "hist_lookback": 252, "rate": 0.04,
           "workers": a.workers}

    df, rejects = run_screen(tickers, expiry, cfg)
    if df.empty:
        print(f"nothing quotable ({len(rejects)} rejected)")
        if rejects:
            top = pd.Series([r for _, r in rejects]).value_counts().head(3)
            print(top.to_string())
        return 1                      # non-zero so a broken feed is visible in CI

    p = save_snapshot(df, expiry)
    print(f"saved {p.name}: {len(df)} names, "
          f"median premium {df['prem_pct'].median():.2f}%, "
          f"{(df['price_src'] == 'bid').sum()} live bids")
    prune(a.keep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
