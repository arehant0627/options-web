"""
datasource.py — where market data comes from.

Two providers behind one interface:

  YahooSource    no key, delayed, free. Blocked from shared IPs (Streamlit
                 Cloud, CI runners) because Yahoo filters on IP reputation.
  TradierSource  free sandbox token, delayed. Authenticated, so shared IPs
                 don't matter. This is the one that actually works in the cloud.

Set TRADIER_TOKEN and Tradier is used automatically; otherwise Yahoo.

Every provider returns the SAME shapes:
    history(sym)            -> pd.Series of closes, DatetimeIndex
    expirations(sym)        -> list[str] "YYYY-MM-DD"
    chain(sym, expiry)      -> (calls_df, puts_df) with columns
                               strike bid ask lastPrice volume openInterest
                               change percentChange lastTradeDate
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

CHAIN_COLS = ["strike", "bid", "ask", "lastPrice", "volume", "openInterest",
              "change", "percentChange", "lastTradeDate"]


def _empty_chain():
    return pd.DataFrame(columns=CHAIN_COLS)


# ───────────────────────────────────────────────────────────── Yahoo

def _yf_session():
    try:
        from curl_cffi import requests as cffi
        return cffi.Session(impersonate="chrome")
    except Exception:
        return None


class YahooSource:
    name = "Yahoo Finance"
    needs_key = False

    def __init__(self):
        import yfinance as yf
        self._yf = yf
        self._session = _yf_session()

    def _tk(self, sym):
        try:
            return self._yf.Ticker(sym, session=self._session) if self._session \
                else self._yf.Ticker(sym)
        except TypeError:
            return self._yf.Ticker(sym)

    def history(self, sym, period="2y"):
        h = self._tk(sym).history(period=period, auto_adjust=True)
        return h["Close"] if not h.empty else pd.Series(dtype=float)

    def expirations(self, sym):
        return list(self._tk(sym).options or [])

    def chain(self, sym, expiry):
        ch = self._tk(sym).option_chain(expiry)
        out = []
        for df in (ch.calls, ch.puts):
            d = df.copy()
            for c in CHAIN_COLS:
                if c not in d.columns:
                    d[c] = pd.NA
            out.append(d[CHAIN_COLS])
        return out[0], out[1]


# ───────────────────────────────────────────────────────────── Tradier

class TradierSource:
    """Free sandbox token at developer.tradier.com. Delayed, but authenticated."""
    needs_key = True

    def __init__(self, token, sandbox=True, timeout=15):
        self.token = token
        self.base = ("https://sandbox.tradier.com/v1" if sandbox
                     else "https://api.tradier.com/v1")
        self.name = f"Tradier {'sandbox' if sandbox else 'production'}"
        self.timeout = timeout
        self._s = requests.Session()
        self._s.headers.update({"Authorization": f"Bearer {token}",
                                "Accept": "application/json"})

    def _get(self, path, **params):
        r = self._s.get(f"{self.base}{path}", params=params, timeout=self.timeout)
        if r.status_code == 401:
            raise PermissionError("Tradier rejected the token (401)")
        r.raise_for_status()
        return r.json()

    def history(self, sym, period="2y"):
        days = {"1y": 370, "2y": 740, "6mo": 190}.get(period, 740)
        start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        j = self._get("/markets/history", symbol=sym, interval="daily",
                      start=start, end=end)
        days_ = (j.get("history") or {}).get("day")
        if not days_:
            return pd.Series(dtype=float)
        if isinstance(days_, dict):
            days_ = [days_]
        df = pd.DataFrame(days_)
        return pd.Series(df["close"].astype(float).values,
                         index=pd.to_datetime(df["date"]))

    def expirations(self, sym):
        j = self._get("/markets/options/expirations", symbol=sym,
                      includeAllRoots="true", strikes="false")
        e = (j.get("expirations") or {}).get("date")
        if not e:
            return []
        return [e] if isinstance(e, str) else list(e)

    def chain(self, sym, expiry):
        j = self._get("/markets/options/chains", symbol=sym, expiration=expiry,
                      greeks="true")
        opts = (j.get("options") or {}).get("option")
        if not opts:
            return _empty_chain(), _empty_chain()
        if isinstance(opts, dict):
            opts = [opts]
        df = pd.DataFrame(opts)

        def norm(sub):
            if sub.empty:
                return _empty_chain()
            d = pd.DataFrame({
                "strike":       pd.to_numeric(sub["strike"], errors="coerce"),
                "bid":          pd.to_numeric(sub.get("bid"), errors="coerce"),
                "ask":          pd.to_numeric(sub.get("ask"), errors="coerce"),
                "lastPrice":    pd.to_numeric(sub.get("last"), errors="coerce"),
                "volume":       pd.to_numeric(sub.get("volume"), errors="coerce"),
                "openInterest": pd.to_numeric(sub.get("open_interest"), errors="coerce"),
                "change":       pd.to_numeric(sub.get("change"), errors="coerce"),
                "percentChange": pd.to_numeric(sub.get("change_percentage"),
                                               errors="coerce"),
            })
            td = sub.get("trade_date")
            d["lastTradeDate"] = (pd.to_datetime(pd.to_numeric(td, errors="coerce"),
                                                 unit="ms", utc=True, errors="coerce")
                                  if td is not None else pd.NaT)
            return d.fillna({"bid": 0, "ask": 0, "lastPrice": 0,
                             "volume": 0, "openInterest": 0}).sort_values("strike")

        return (norm(df[df["option_type"] == "call"]),
                norm(df[df["option_type"] == "put"]))


# ───────────────────────────────────────────────────────────── selection

def get_source():
    """Tradier when a token is present, else Yahoo."""
    tok = os.environ.get("TRADIER_TOKEN", "").strip()
    if tok:
        sandbox = os.environ.get("TRADIER_SANDBOX", "1") not in ("0", "false", "False")
        return TradierSource(tok, sandbox=sandbox)
    return YahooSource()


def probe(source, sym="AAPL"):
    """(ok, detail) — used by the UI to explain a dead feed precisely."""
    try:
        h = source.history(sym, "6mo")
        if h.empty:
            return False, f"{source.name}: price history came back empty (blocked or throttled)"
        exps = source.expirations(sym)
        if not exps:
            return False, (f"{source.name}: prices OK but NO option expirations. "
                           "This is the signature of an IP block on the options endpoint.")
        calls, _ = source.chain(sym, exps[0])
        live = int((pd.to_numeric(calls["bid"], errors="coerce") > 0).sum())
        return True, (f"{source.name}: OK — {len(h)} bars, {len(exps)} expiries, "
                      f"{len(calls)} calls for {exps[0]} ({live} with live bids)")
    except PermissionError as e:
        return False, str(e)
    except Exception as e:
        return False, f"{source.name}: {type(e).__name__}: {str(e)[:110]}"
