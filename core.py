"""
NDX Premium Screener — core logic
=================================
Covered-call premium screener for the Nasdaq-100, as a web app.

Shared by app.py (the dashboard) and record_snapshot.py (the cron recorder).
No Streamlit imports here, so it runs headless in CI.
"""

from __future__ import annotations

import io
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# Override with the SNAP_DIR env var to persist snapshots (e.g. Google Drive on
# Colab, where the local filesystem is wiped on every disconnect).
SNAP_DIR = Path(os.environ.get("SNAP_DIR", "snapshots"))
SNAP_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════ Black-Scholes

SQRT2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
def _norm_pdf(x): return math.exp(-0.5 * x * x) / SQRT2PI


def _d1(s, k, t, r, q, v):
    return (math.log(s / k) + (r - q + 0.5 * v * v) * t) / (v * math.sqrt(t))


def bs_call(s, k, t, r, q, v):
    if t <= 0 or v <= 0:
        return max(0.0, s * math.exp(-q * t) - k * math.exp(-r * t))
    d1 = _d1(s, k, t, r, q, v)
    d2 = d1 - v * math.sqrt(t)
    return s * math.exp(-q * t) * _norm_cdf(d1) - k * math.exp(-r * t) * _norm_cdf(d2)


def bs_call_delta(s, k, t, r, q, v):
    if t <= 0 or v <= 0:
        return 1.0 if s > k else 0.0
    return math.exp(-q * t) * _norm_cdf(_d1(s, k, t, r, q, v))


def bs_vega(s, k, t, r, q, v):
    if t <= 0 or v <= 0:
        return 0.0
    return s * math.exp(-q * t) * _norm_pdf(_d1(s, k, t, r, q, v)) * math.sqrt(t)


def implied_vol(price, s, k, t, r=0.04, q=0.0, lo=1e-3, hi=6.0):
    """Newton, with bisection fallback where vega collapses."""
    intrinsic = max(0.0, s * math.exp(-q * t) - k * math.exp(-r * t))
    if price <= intrinsic + 1e-8 or t <= 0:
        return None
    v = 0.35
    for _ in range(60):
        diff = bs_call(s, k, t, r, q, v) - price
        if abs(diff) < 1e-7:
            return v
        vega = bs_vega(s, k, t, r, q, v)
        if vega < 1e-8:
            break
        v -= max(min(diff / vega, 1.0), -1.0)
        if not (lo < v < hi):
            break
    a, b = lo, hi
    if (bs_call(s, k, t, r, q, a) - price) * (bs_call(s, k, t, r, q, b) - price) > 0:
        return None
    for _ in range(200):
        m = 0.5 * (a + b)
        if bs_call(s, k, t, r, q, m) - price > 0:
            b = m
        else:
            a = m
    return 0.5 * (a + b)


# ══════════════════════════════════════════════════════════ universe

UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")}
QQQ_HOLDINGS = ("https://www.invesco.com/us/financial-products/etfs/holdings/main/"
                "holdings/0?audienceType=Investor&action=download&ticker=QQQ")
WIKI_NDX = "https://en.wikipedia.org/wiki/Nasdaq-100"
SLICKCHARTS = "https://www.slickcharts.com/nasdaq100"

NDX_SNAPSHOT_DATE = "2026-08-23"
NDX_SNAPSHOT = """
NVDA AAPL MSFT AMZN GOOGL SPCX AVGO TSLA META MU WMT AMD ASML INTC CSCO PLTR
COST LRCX AMAT NFLX PANW ARM TXN KLAC AMGN SNDK LIN MRVL TMUS PEP CRWD STX SHOP
ADI GILD QCOM WDC BKNG VRTX ISRG PDD SBUX FTNT ABNB ADP ADBE APP INTU MELI DASH
CEG CSX CMCSA MNST MAR CDNS REGN DDOG MDLZ CTAS LITE ROST SNPS ORLY WBD PCAR HON
AEP MPWR BKR NBIS FANG FAST TER NXPI ADSK PYPL HONA AXON ALAB WDAY CRWV CCEP XEL
MSTR RKLB TRI FER EXC TTWO PAYX IDXX KDP ODFL MCHP ROP DXCM GEHC ALNY CPRT KHC
""".split()

TICKER_RE = re.compile(r"^[A-Z]{1,5}(-[A-Z])?$")


def _clean(raw):
    out = {str(t).strip().upper().replace(".", "-") for t in raw}
    return sorted(t for t in out if TICKER_RE.match(t))


def _get(url, timeout=20):
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r.text


def _pick_col(df):
    for c in df.columns:
        if str(c).strip().lower() in ("ticker", "symbol", "holding ticker", "ticker symbol"):
            return c
    for c in df.columns:
        if "ticker" in str(c).lower() or "symbol" in str(c).lower():
            return c
    return None


def load_universe():
    """Returns (tickers, source_label). Cached 12h."""
    try:
        df = pd.read_csv(io.StringIO(_get(QQQ_HOLDINGS)))
        col = _pick_col(df)
        if col is not None:
            t = _clean(df[col].dropna())
            if len(t) >= 90:
                return t, "Invesco QQQ holdings"
    except Exception:
        pass
    for url, label in [(SLICKCHARTS, "Slickcharts"), (WIKI_NDX, "Wikipedia")]:
        try:
            for tbl in pd.read_html(io.StringIO(_get(url))):
                col = _pick_col(tbl)
                if col is not None and len(tbl) > 90:
                    t = _clean(tbl[col].dropna())
                    if len(t) >= 90:
                        return t, label
        except Exception:
            continue
    return _clean(NDX_SNAPSHOT), f"built-in snapshot ({NDX_SNAPSHOT_DATE})"


def list_expiries(refs=("AAPL", "NVDA", "MSFT")):
    """Union of expiries across a few liquid names. Cached 1h."""
    seen = set()
    for r in refs:
        try:
            seen.update(yf.Ticker(r).options or [])
        except Exception:
            continue
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    out = []
    for e in sorted(seen):
        d = (pd.Timestamp(e) - today).days
        if 0 <= d <= 400:
            out.append((e, d, pd.Timestamp(e).day_name()[:3]))
    return out


# ══════════════════════════════════════════════════════════ screening

VALID_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}


def _history(tk, period="2y", tries=2):
    if period not in VALID_PERIODS:
        raise ValueError(f"invalid period {period!r}")
    h = pd.DataFrame()
    for i in range(tries):
        try:
            h = tk.history(period=period, auto_adjust=True)
        except Exception:
            h = pd.DataFrame()
        if not h.empty:
            return h
        time.sleep(0.8 * (i + 1))
    return h


def hist_hit_rate(closes, dte, otm_pct, lookback=252):
    c = closes.tail(lookback + max(dte, 1))
    fwd = (c.shift(-max(dte, 1)) / c - 1).dropna()
    n = len(fwd)
    if n < 30:
        return np.nan, np.nan, n
    return (float((fwd > otm_pct / 100).mean() * 100),
            float(fwd.abs().median() * 100), n)


def expected_move(calls, puts, spot):
    """ATM straddle x 0.85. Falls back to last trade when quotes are zeroed."""
    try:
        k = float(calls.iloc[(calls.strike - spot).abs().argsort().iloc[0]].strike)
        c, p = calls[calls.strike == k].iloc[0], puts[puts.strike == k].iloc[0]

        def px(o):
            b, a = float(o.bid or 0), float(o.ask or 0)
            if b > 0 and a > 0 and a >= b:
                return 0.5 * (b + a)
            return float(getattr(o, "lastPrice", 0) or 0)

        cm, pm = px(c), px(p)
        return 0.85 * (cm + pm) if cm > 0 and pm > 0 else np.nan
    except Exception:
        return np.nan


def _rej(reason):
    return {"_reject": reason}


def screen_one(sym, expiry, cfg):
    """One ticker against one specific expiry date."""
    tk = yf.Ticker(sym)

    hist = _history(tk)
    if hist.empty:
        return _rej("no history (rate-limited?)")
    if len(hist) < cfg["min_history_rows"]:
        return _rej(f"short history ({len(hist)} rows)")
    closes = hist["Close"]
    spot = float(closes.iloc[-1])

    try:
        avail = tk.options or []
    except Exception as e:
        return _rej(f"options list raised {type(e).__name__}")
    if expiry not in avail:
        return _rej("does not list this expiry")

    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    dte = (pd.Timestamp(expiry) - today).days
    t = max(dte, 0.5) / 365.0

    try:
        chain = tk.option_chain(expiry)
    except Exception as e:
        return _rej(f"chain fetch raised {type(e).__name__}")
    calls, puts = chain.calls, chain.puts
    if calls is None or calls.empty:
        return _rej("empty call chain")

    em = expected_move(calls, puts, spot)

    earn = None
    try:
        ed = tk.get_earnings_dates(limit=12)
        if ed is not None and not ed.empty:
            idx = pd.to_datetime(ed.index).tz_localize(None)
            fut = idx[idx >= today]
            earn = fut.min() if len(fut) else None
    except Exception:
        pass
    earn_in = bool(earn is not None and today <= earn <= pd.Timestamp(expiry))
    if cfg["earnings_mode"] == "only" and not earn_in:
        return _rej("no earnings before this expiry")
    if cfg["earnings_mode"] == "exclude" and earn_in:
        return _rej("earnings before this expiry")

    otm = calls[calls["strike"] > spot].sort_values("strike").reset_index(drop=True)
    if len(otm) < cfg["rung"]:
        return _rej(f"only {len(otm)} OTM strikes")
    row = otm.iloc[cfg["rung"] - 1]

    strike = float(row["strike"])
    bid, ask = float(row.bid or 0), float(row.ask or 0)
    last = float(getattr(row, "lastPrice", 0) or 0)
    oi, vol = int(row.openInterest or 0), int(row.volume or 0)

    if bid >= cfg["min_bid"]:
        price, price_src = bid, "bid"
    elif cfg["use_last"] and last >= cfg["min_bid"]:
        if oi < cfg["min_oi"] and vol < cfg["min_oi"]:
            return _rej(f"no live bid and thin (oi {oi}, vol {vol})")
        price, price_src = last, "last"
    elif bid > 0 or last > 0:
        return _rej(f"price {max(bid, last):.2f} < min_bid {cfg['min_bid']}")
    else:
        return _rej("no bid and no last trade")

    if bid > 0 and ask > 0 and ask >= bid:
        mid = 0.5 * (bid + ask)
        spread = (ask - bid) / mid
        if spread > cfg["max_spread"]:
            return _rej(f"spread {spread:.0%} too wide")
    else:
        spread = np.nan

    if oi < cfg["min_oi"] and vol < cfg["min_oi"]:
        return _rej(f"oi {oi} / vol {vol} below min_oi")

    # quote staleness — real, from lastTradeDate
    age_min = np.nan
    try:
        ltd = pd.Timestamp(row.lastTradeDate)
        if pd.notna(ltd):
            now = pd.Timestamp.utcnow()
            if ltd.tzinfo is None:
                ltd = ltd.tz_localize("UTC")
            age_min = (now.tz_localize("UTC") - ltd).total_seconds() / 60
    except Exception:
        pass

    iv = implied_vol(price, spot, strike, t, cfg["rate"])
    delta = bs_call_delta(spot, strike, t, cfg["rate"], 0.0, iv) if iv else np.nan
    lr = np.log(closes / closes.shift(1)).dropna()
    hv20 = float(lr.tail(20).std(ddof=1) * math.sqrt(252)) if len(lr) > 25 else np.nan

    otm_pct = (strike / spot - 1) * 100
    hit, typical, hist_n = hist_hit_rate(closes, dte, otm_pct, cfg["hist_lookback"])

    # day-over-day move in the option itself, straight from the chain
    chg = float(getattr(row, "change", np.nan) or np.nan)
    chg_pct = float(getattr(row, "percentChange", np.nan) or np.nan)

    return {
        "ticker": sym, "spot": round(spot, 2), "strike": strike,
        "otm_pct": round(otm_pct, 2), "expiry": expiry, "dte": dte,
        "credit": round(price * 100, 0),
        "prem_pct": round(price / strike * 100, 3),
        "price_src": price_src,
        "quote_age_min": round(age_min, 0) if age_min == age_min else np.nan,
        "if_called_pct": round((price + strike - spot) / spot * 100, 2),
        "ann_pct": round(price / strike * 100 * (365 / max(dte, 1)), 0),
        "prem_chg": round(chg, 2) if chg == chg else np.nan,
        "prem_chg_pct": round(chg_pct, 1) if chg_pct == chg_pct else np.nan,
        "delta": round(delta, 3) if delta == delta else np.nan,
        "hist_hit": round(hit, 1) if hit == hit else np.nan,
        "edge": round(delta * 100 - hit, 1) if (hit == hit and delta == delta) else np.nan,
        "hist_n": hist_n,
        "typical_move": round(typical, 2) if typical == typical else np.nan,
        "iv": round(iv * 100, 1) if iv else np.nan,
        "iv_hv20": round(iv / hv20, 2) if (iv and hv20 == hv20 and hv20 > 0) else np.nan,
        "em_pct": round(em / spot * 100, 2) if em == em else np.nan,
        "strike_vs_em": round((strike - spot) / em, 2) if (em == em and em > 0) else np.nan,
        "spread_pct": round(spread * 100, 1) if spread == spread else np.nan,
        "oi": oi, "volume": vol,
        "earnings": earn.date().isoformat() if earn is not None else "",
        "earn_wk": earn_in,
    }


def run_screen(tickers, expiry, cfg):
    """Screen a list of tickers against one expiry. Pure — no UI."""
    rows, rejects = [], []
    with ThreadPoolExecutor(max_workers=cfg["workers"]) as ex:
        futs = {ex.submit(screen_one, s, expiry, cfg): s for s in tickers}
        for f in as_completed(futs):
            try:
                r = f.result()
                if r and "_reject" in r:
                    rejects.append((futs[f], r["_reject"]))
                elif r:
                    rows.append(r)
            except Exception as e:
                rejects.append((futs[f], f"{type(e).__name__}"))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("prem_pct", ascending=False).reset_index(drop=True)
    return df, rejects


def add_flags(df):
    def f(r):
        s = ""
        s += "E" if r["earn_wk"] else ""
        s += "!" if (r["strike_vs_em"] == r["strike_vs_em"] and r["strike_vs_em"] < 1) else ""
        s += "~" if (r["spread_pct"] == r["spread_pct"] and r["spread_pct"] > 12) else ""
        s += "L" if r["price_src"] == "last" else ""
        s += "?" if r["hist_n"] < 120 else ""
        return s
    df = df.copy()
    df["flags"] = df.apply(f, axis=1)
    return df


# ══════════════════════════════════════════════════════════ snapshots

def save_snapshot(df, expiry):
    if df.empty:
        return None
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    p = SNAP_DIR / f"{expiry}__{ts}.csv"
    df[["ticker", "strike", "prem_pct", "credit", "iv", "spot"]].to_csv(p, index=False)
    return p


def load_snapshots(expiry):
    out = []
    for p in sorted(SNAP_DIR.glob(f"{expiry}__*.csv")):
        try:
            ts = datetime.strptime(p.stem.split("__")[1], "%Y%m%dT%H%M%S")
            out.append((ts.replace(tzinfo=timezone.utc), pd.read_csv(p)))
        except Exception:
            continue
    return out


def snapshot_movers(df, expiry):
    """Diff current run against the most recent earlier snapshot."""
    snaps = load_snapshots(expiry)
    if len(snaps) < 1:
        return None, None
    ts, prev = snaps[-1]
    m = df.merge(prev, on=["ticker", "strike"], suffixes=("", "_prev"))
    if m.empty:
        return None, ts
    m["prem_pct_chg"] = (m["prem_pct"] - m["prem_pct_prev"]).round(3)
    m["prem_rel_chg"] = ((m["prem_pct"] / m["prem_pct_prev"] - 1) * 100).round(1)
    m["iv_chg"] = (m["iv"] - m["iv_prev"]).round(1)
    return m.sort_values("prem_rel_chg", ascending=False), ts


