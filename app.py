"""
NDX Weekly Premium Dashboard
============================
Covered-call premium screener for the Nasdaq-100, as a web app.

Run locally:   streamlit run app.py
Deploy:        push to GitHub -> share.streamlit.io -> pick the repo
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from core import (SNAP_DIR, _clean, add_flags, list_expiries as _list_expiries,
                  load_universe as _load_universe, load_snapshots, run_screen as _run_screen,
                  save_snapshot, snapshot_movers, feed_check, source_name, source)

st.set_page_config(page_title="NDX Premium Screener", page_icon="◱", layout="wide")


@st.cache_data(ttl=43_200, show_spinner=False)
def load_universe():
    return _load_universe()


@st.cache_data(ttl=3_600, show_spinner=False)
def list_expiries():
    return _list_expiries()


@st.cache_data(ttl=300, show_spinner=False)
def run_screen(tickers: tuple, expiry: str, cfg_items: tuple, _nonce: int = 0):
    """Cached 5 min. Bump _nonce to force a live refetch."""
    return _run_screen(list(tickers), expiry, dict(cfg_items))


# ══════════════════════════════════════════════════════════ UI

st.title("NDX Weekly Premium Screener")

universe, uni_src = load_universe()

with st.sidebar:
    st.header("Settings")

    exps, live_exp = list_expiries()
    if not live_exp:
        st.warning("Yahoo didn't return an expiry list — showing the next 12 Fridays "
                   "instead. Names that don't list a date get skipped at screen time.")

    labels = [f"{e}  ({wd}, {d}d)" for e, d, wd in exps]
    fridays = [i for i, (e, d, wd) in enumerate(exps) if wd == "Fri" and d >= 1]
    default = fridays[0] if fridays else 0
    pick = st.selectbox("Expiry", range(len(exps)), index=default,
                        format_func=lambda i: labels[i])
    expiry, dte, _ = exps[pick]

    side = st.radio("Side", ["Calls", "Puts"], horizontal=True,
                    help="Calls = covered calls on stock you own. "
                         "Puts = cash-secured puts on stock you'd be happy to buy.")
    side_key = "call" if side == "Calls" else "put"

    if side_key == "call":
        rung = st.slider("Strike rung above spot", 1, 6, 1,
                         help="1 = first listed strike above the current price")
        target_delta = 0.25
    else:
        rung = 1
        target_delta = st.slider("Target delta", 0.05, 0.50, 0.25, 0.05,
                                 help="Picks the put whose |delta| is closest to "
                                      "this. Roughly your assignment probability.")

    st.divider()
    scope = st.radio("Universe", ["Nasdaq-100", "Custom"], horizontal=True)
    if scope == "Custom":
        txt = st.text_area("Tickers", "NVDA MRVL SPCX INTU CRWD", height=68)
        tickers = _clean(txt.replace(",", " ").split())
    else:
        tickers = universe
    st.caption(f"{len(tickers)} tickers · source: {uni_src}")

    st.divider()
    earnings_mode = st.selectbox("Earnings", ["any", "only", "exclude"], 0)
    use_last = st.checkbox("Use last trade when market is closed", True)
    with st.expander("Filters"):
        min_bid = st.number_input("Min price", 0.0, 50.0, 0.10, 0.05)
        min_oi = st.number_input("Min OI or volume", 0, 5000, 10, 5)
        max_spread = st.slider("Max bid-ask spread", 0.05, 1.0, 0.25, 0.05)
        workers = st.slider("Parallel workers", 1, 8, 4,
                            help="Above ~4 Yahoo starts rate-limiting")

    with st.expander(f"Data source · {source_name()}"):
        if st.button("Test connection", use_container_width=True):
            ok, detail = feed_check()
            (st.success if ok else st.error)(detail)
        dbg = st.text_input("Inspect one ticker", "AAPL",
                            help="Dumps raw spot and chain values for this expiry.")
        if st.button("Inspect", use_container_width=True):
            try:
                src = source()
                h = src.history(dbg, "6mo")
                h = pd.to_numeric(h, errors="coerce").dropna()
                st.write({"provider": src.name, "bars": len(h),
                          "spot": round(float(h.iloc[-1]), 2) if len(h) else None,
                          "last bar": str(h.index[-1].date()) if len(h) else None})
                exps = src.expirations(dbg)
                st.write({"expirations": len(exps), "first 5": exps[:5],
                          "selected listed?": expiry in exps})
                if expiry in exps:
                    ca, _ = src.chain(dbg, expiry)
                    k = pd.to_numeric(ca["strike"], errors="coerce")
                    spot_v = float(h.iloc[-1])
                    st.write({"call rows": len(ca),
                              "strike dtype": str(ca["strike"].dtype),
                              "strike range": [float(k.min()), float(k.max())],
                              "unparseable strikes": int(k.isna().sum()),
                              "strikes above spot": int((k > spot_v).sum())})
                    st.dataframe(ca.head(6), use_container_width=True)
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")

        if not os.environ.get("TRADIER_TOKEN"):
            st.caption(
                "**Using Yahoo (no key).** Yahoo blocks shared cloud IPs, so this "
                "often fails on Streamlit Cloud even when it works on your laptop.\n\n"
                "**Fix, ~3 minutes and free:**\n"
                "1. Sign up at developer.tradier.com\n"
                "2. Copy your **sandbox** access token\n"
                "3. Streamlit Cloud → Manage app → Settings → Secrets → paste:\n"
                "   `TRADIER_TOKEN = \"your_token\"`\n\n"
                "The app switches over automatically on reboot.")

    go = st.button("Run screen", type="primary", use_container_width=True)
    fresh = st.button("Force refresh (bypass 5-min cache)",
                      use_container_width=True,
                      help="Screens are cached 5 minutes. This refetches live.")

cfg = {"rung": rung, "side": side_key, "target_delta": target_delta, "earnings_mode": earnings_mode, "use_last": use_last,
       "min_bid": min_bid, "min_oi": min_oi, "max_spread": max_spread,
       "min_history_rows": 25, "hist_lookback": 252, "rate": 0.04,
       "workers": workers}

if fresh:
    st.session_state["nonce"] = st.session_state.get("nonce", 0) + 1

if go or fresh:
    df, rejects = run_screen(tuple(tickers), expiry, tuple(sorted(cfg.items())),
                             st.session_state.get("nonce", 0))
    st.session_state["fetched_at"] = datetime.now(timezone.utc)
    st.session_state["df"] = df
    st.session_state["rejects"] = rejects
    st.session_state["expiry"] = expiry
    if not df.empty:
        prev_snaps = load_snapshots(expiry)
        st.session_state["movers"], st.session_state["mv_ts"] = snapshot_movers(df, expiry)
        save_snapshot(df, expiry)

if "df" not in st.session_state:
    st.info("Pick an expiry in the sidebar and hit **Run screen**.")
    st.stop()

df = st.session_state["df"]
rejects = st.session_state["rejects"]
expiry = st.session_state["expiry"]

if df.empty:
    st.error("Nothing quotable.")
    if rejects:
        st.write(pd.Series([r for _, r in rejects]).value_counts().rename("count"))
    st.stop()

df = add_flags(df)

live = (df["price_src"] == "bid").sum()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Names", len(df))
c2.metric("Best premium", f"{df['prem_pct'].max():.2f}%", df.iloc[0]["ticker"])
c3.metric("Median premium", f"{df['prem_pct'].median():.2f}%")
fetched = st.session_state.get("fetched_at")
if fetched:
    age = (datetime.now(timezone.utc) - fetched).total_seconds() / 60
    st.caption(f"Quotes pulled {fetched:%H:%M} UTC "
               f"({age:.0f} min ago) · Yahoo option data is ~15 min delayed. "
               f"Hit **Force refresh** for a live refetch.")

c4.metric("Live bids", f"{live}/{len(df)}",
          "market open" if live > len(df) * 0.5 else "stale quotes",
          delta_color="normal" if live > len(df) * 0.5 else "inverse")

if live < len(df) * 0.5:
    st.warning("Most rows are priced off the last trade, not a live bid — the market is "
               "closed or thinly quoted. Check `quote_age_min` before acting.")

tab1, tab2, tab3, tab4 = st.tabs(["Screener", "Movers", "Detail", "Columns"])

# ─────────────────────────────────────────────── Screener
with tab1:
    is_put = df["side"].iloc[0] == "put"
    MAIN = (["ticker", "spot", "strike", "below_pct", "credit", "prem_pct", "ann_pct",
             "delta", "hist_hit", "edge", "iv", "hv_rank", "iv_hv20", "days_to_earn",
             "breakeven", "be_below_pct", "cash_required", "price_src", "oi", "flags"]
            if is_put else
            ["ticker", "spot", "strike", "otm_pct", "credit", "prem_pct", "if_called_pct",
             "ann_pct", "delta", "hist_hit", "edge", "iv", "hv_rank", "iv_hv20",
             "days_to_earn", "strike_vs_em", "price_src", "quote_age_min", "oi", "flags"])
    only_e = st.checkbox("Earnings names only", False)
    view = df[df["earn_wk"]] if only_e else df
    st.dataframe(
        view[MAIN], use_container_width=True, height=620, hide_index=True,
        column_config={
            "prem_pct": st.column_config.ProgressColumn(
                "prem %", format="%.2f%%", min_value=0.0,
                max_value=float(df["prem_pct"].max())),
            "credit": st.column_config.NumberColumn("credit $", format="$%d"),
            "otm_pct": st.column_config.NumberColumn("OTM %", format="%.2f%%"),
            "below_pct": st.column_config.NumberColumn("below spot", format="%.2f%%"),
            "be_below_pct": st.column_config.NumberColumn("BE below", format="%.2f%%"),
            "cash_required": st.column_config.NumberColumn("cash req", format="$%d"),
            "hv_rank": st.column_config.NumberColumn("HV rank", format="%d"),
            "days_to_earn": st.column_config.NumberColumn("d to earn", format="%d"),
            "if_called_pct": st.column_config.NumberColumn("if called %", format="%.2f%%"),
            "edge": st.column_config.NumberColumn("edge", format="%+.1f"),
            "quote_age_min": st.column_config.NumberColumn("quote age", format="%d min"),
        })
    st.caption("E earnings · ! strike inside expected move · ~ wide spread · "
               "L last-trade price · ? short history")
    if is_put:
        st.caption(
            "**Puts:** `prem_pct` is premium ÷ strike — yield on the cash you post. "
            "`delta` is roughly the chance you get assigned the shares. `hist_hit` is "
            "how often the stock actually fell that far over the same holding period "
            "in the past year, so `edge` positive means the option prices more "
            "downside than the stock has delivered. Only sell these where you'd be "
            "content to own the stock at `breakeven`.")
    st.download_button("Download CSV", df.to_csv(index=False),
                       f"ndx_premium_{expiry}.csv", "text/csv")

    with st.expander(f"Excluded ({len(rejects)})"):
        if rejects:
            rj = pd.DataFrame(rejects, columns=["ticker", "reason"])
            st.write(rj.groupby("reason")["ticker"]
                     .agg([("count", "size"), ("tickers", lambda s: ", ".join(sorted(s)[:14]))])
                     .sort_values("count", ascending=False))

# ─────────────────────────────────────────────── Movers
with tab2:
    st.subheader("Day over day")
    st.caption("Change in the option's own price versus its previous close — straight "
               "from the chain, no history needed.")
    dd = df.dropna(subset=["prem_chg_pct"])
    if dd.empty:
        st.info("No day-change data in this chain (common right after the open).")
    else:
        up = dd.nlargest(15, "prem_chg_pct")[
            ["ticker", "strike", "credit", "prem_pct", "prem_chg", "prem_chg_pct",
             "iv", "iv_hv20", "flags"]]
        st.dataframe(up, use_container_width=True, hide_index=True,
                     column_config={"prem_chg_pct": st.column_config.NumberColumn(
                         "premium chg", format="%+.1f%%"),
                         "prem_chg": st.column_config.NumberColumn("chg $", format="%+.2f"),
                         "credit": st.column_config.NumberColumn("credit $", format="$%d")})
        st.caption("A premium can jump because IV rose **or** because the stock moved "
                   "toward the strike. Check `iv_hv20` to tell which.")

    st.divider()
    st.subheader("Since your last run")
    mv, mv_ts = st.session_state.get("movers"), st.session_state.get("mv_ts")
    if mv is None or mv.empty:
        st.info("No earlier snapshot for this expiry yet. Every run saves one — "
                "run again in an hour and this fills in with intraday changes.\n\n"
                "Nobody publishes free historical intraday option quotes, so this is "
                "the only way to see hour-scale moves: record them yourself.")
    else:
        mins = (datetime.now(timezone.utc) - mv_ts).total_seconds() / 60
        st.caption(f"vs snapshot from {mv_ts:%Y-%m-%d %H:%M} UTC ({mins:.0f} min ago)")
        st.dataframe(
            mv.nlargest(15, "prem_rel_chg")[
                ["ticker", "strike", "prem_pct_prev", "prem_pct", "prem_rel_chg",
                 "iv_prev", "iv", "iv_chg"]],
            use_container_width=True, hide_index=True,
            column_config={"prem_rel_chg": st.column_config.NumberColumn(
                "premium chg", format="%+.1f%%"),
                "iv_chg": st.column_config.NumberColumn("IV chg", format="%+.1f")})

    st.divider()
    st.subheader("Richest vol vs own realized")
    rich = df.dropna(subset=["iv_hv20"]).nlargest(15, "iv_hv20")[
        ["ticker", "prem_pct", "iv", "iv_hv20", "edge", "earnings", "flags"]]
    st.dataframe(rich, use_container_width=True, hide_index=True)
    st.caption("IV/HV20 above ~1.2 means options price more movement than the stock has "
               "recently delivered. Below 1.0 you're being paid less than it actually moves.")

# ─────────────────────────────────────────────── Detail
with tab3:
    sym = st.selectbox("Ticker", df["ticker"].tolist())
    row = df[df["ticker"] == sym].iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("Spot", f"${row['spot']:,.2f}")
    b.metric("Strike", f"${row['strike']:,.2f}", f"{row['otm_pct']:+.2f}% OTM")
    c.metric("Credit", f"${row['credit']:,.0f}", f"{row['prem_pct']:.2f}% of strike")
    d.metric("Delta", f"{row['delta']:.3f}",
             f"hist {row['hist_hit']:.0f}%" if row["hist_hit"] == row["hist_hit"] else "—")

    try:
        tk = yf.Ticker(sym)
        ch = tk.option_chain(expiry).calls
        spot = float(row["spot"])
        lad = ch[ch.strike > spot * 0.98].copy()
        lad["price"] = np.where(lad.bid > 0, (lad.bid + lad.ask) / 2, lad.lastPrice)
        lad = lad[lad.price > 0]
        lad["prem_pct"] = lad.price / lad.strike * 100
        lad["if_called_pct"] = (lad.price + lad.strike - spot) / spot * 100
        st.line_chart(lad.set_index("strike")[["prem_pct", "if_called_pct"]],
                      height=300)
        st.caption("Where the two lines cross, extra premium stops compensating for "
                   "the upside you're giving up.")
        st.dataframe(lad[["strike", "bid", "ask", "lastPrice", "prem_pct",
                          "if_called_pct", "volume", "openInterest"]].head(14),
                     use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Could not load ladder: {e}")

# ─────────────────────────────────────────────── Columns
with tab4:
    st.markdown("""
| column | meaning |
|---|---|
| `prem_pct` | **the ranking metric** — `price / strike x 100` |
| `credit` | cash per contract, `price x 100` |
| `price_src` | `bid` = live quote · `last` = last trade, possibly stale |
| `quote_age_min` | minutes since that contract last traded — the real staleness check |
| `otm_pct` | `(strike/spot - 1) x 100` |
| `if_called_pct` | total return if assigned: `(price + strike - spot) / spot` |
| `ann_pct` | `prem_pct` scaled to a year. Comparison aid, not a forecast |
| `delta` | market's implied probability of finishing above the strike |
| `hist_hit` | realized: share of past-year `dte`-day windows that rose more than `otm_pct` |
| `edge` | `delta x 100 - hist_hit`. Positive = priced richer than the stock has delivered |
| `hist_n` | windows behind `hist_hit`. Under ~120 is a small sample |
| `iv` | implied vol of that strike, solved from the quote |
| `iv_hv20` | `iv` / 20-day realized vol. >1.2 rich, <1.0 underpaid |
| `em_pct` | expected move to expiry (ATM straddle x 0.85), % of spot |
| `strike_vs_em` | `(strike - spot) / expected move`. Below 1.0 = assignment is the base case |
| `prem_chg_pct` | the option's own % change vs its previous close |

**Put mode**

| column | meaning |
|---|---|
| `prem_pct` | premium ÷ strike — yield on the cash you post to secure the put |
| `below_pct` | how far the strike sits below spot |
| `delta` | shown as absolute value; roughly your assignment probability |
| `hist_hit` | share of past-year windows where the stock *fell* more than `below_pct` |
| `edge` | `delta x 100 - hist_hit`. Positive = more downside priced than delivered |
| `breakeven` | `strike - premium` — your effective cost basis if assigned |
| `be_below_pct` | how far breakeven sits below the current price, your cushion |
| `cash_required` | `strike x 100` — collateral per contract |
| `hv_rank` | **realized**-vol rank, 0-100, within the trailing year |
| `days_to_earn` | calendar days until the next earnings date |

**Flags** — `E` earnings before expiry · `!` strike inside the expected move ·
`~` spread wider than 12% · `L` priced off last trade · `?` short history

### On `hv_rank` vs IV rank

True **IV rank** needs a year of stored implied vol. No free feed serves that
history, so it cannot be computed from a single snapshot of today's chain —
anything claiming otherwise is showing you something else.

`hv_rank` is where the stock's current 20-day **realized** volatility sits within
its own trailing-year range. It answers "is this name unusually volatile for
itself right now," which is most of what you want IV rank for, and it is honest
about what it measures. Pair it with `iv_hv20` — that compares what options are
charging against what the stock is actually doing, which is the richness question.

The snapshot recorder does accumulate real implied vol over time. Once you have
a few months of history in `snapshots/`, true IV rank becomes computable from it.

---
Quotes are delayed. This is a research tool, not advice.
At rung 1 a call strike is at-the-money, so delta sits near 0.50 — expect
assignment roughly half the time, and only write where you'd be content to sell.

Selling a cash-secured put is a bullish-to-neutral position with the same payoff
shape as a covered call: capped gain, full downside. A 0.25 delta put assigns
about a quarter of the time, and the quarter where it does is the quarter the
stock fell. Size it as though you will be assigned, because eventually you will.
""")
