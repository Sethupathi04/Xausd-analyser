"""Streamlit dashboard for the XAUUSD 15-minute analyzer.

Deploy on Streamlit Community Cloud: point it at this repo and set the main
file to `app.py`. Runs entirely on the free (delayed ~15m) yfinance GC=F feed.
"""

from __future__ import annotations

import warnings

import pandas as pd
import streamlit as st

from xau15m.backtest import run_backtest
from xau15m.data import fetch_candles
from xau15m.indicators import compute_all
from xau15m.signals import LONG, NEUTRAL, SHORT, generate_signal

st.set_page_config(page_title="XAUUSD 15m Analyzer", page_icon="📈", layout="wide")

_SIGNAL_COLOR = {LONG: "#16a34a", SHORT: "#dc2626", NEUTRAL: "#ca8a04"}


@st.cache_data(ttl=60, show_spinner=False)
def load_enriched(symbol: str, period: str, interval: str) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        df = fetch_candles(symbol=symbol, period=period, interval=interval)
    return compute_all(df)


st.title("📈 XAUUSD — 15-Minute Technical Analysis")
st.caption(
    "Heuristic technical analysis aid for gold (GC=F futures proxy). "
    "**Not financial advice** — data is delayed ~15 min via yfinance."
)

with st.sidebar:
    st.header("Settings")
    symbol = st.text_input("Symbol", value="GC=F")
    period = st.selectbox("History window", ["5d", "15d", "30d", "60d"], index=3)
    interval = st.selectbox("Interval", ["15m", "5m", "30m", "1h"], index=0)
    threshold = st.slider("Signal threshold", 0.0, 1.0, 0.2, 0.05)
    atr_mult = st.slider("ATR stop multiple", 0.5, 4.0, 1.5, 0.5)
    rr = st.slider("Reward : risk", 1.0, 5.0, 2.0, 0.5)
    do_backtest = st.checkbox("Run backtest", value=True)
    if st.button("🔄 Refresh now"):
        st.cache_data.clear()
        st.rerun()

try:
    enriched = load_enriched(symbol, period, interval)
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to load data: {exc}")
    st.stop()

signal = generate_signal(enriched, threshold=threshold, atr_stop_mult=atr_mult, rr=rr)

color = _SIGNAL_COLOR.get(signal.label, "#334155")
st.markdown(
    f"<h2 style='color:{color};margin-bottom:0'>SIGNAL: {signal.label}</h2>",
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Last price", f"{signal.price}")
c2.metric("Confidence", f"{signal.confidence * 100:.1f}%", f"score {signal.score:+.3f}")
c3.metric("ATR(14)", f"{signal.atr}")
c4.metric("ADX(14)", f"{signal.adx}", "trending" if signal.adx >= 25 else "weak/range")

if signal.label in (LONG, SHORT):
    s1, s2, s3 = st.columns(3)
    s1.metric("Suggested stop", f"{signal.stop}")
    s2.metric("Suggested target", f"{signal.target}")
    s3.metric("Reward : risk", f"1 : {signal.risk_reward:g}")

st.caption(f"Last bar (UTC): {enriched.index[-1]} · {len(enriched)} candles")

left, right = st.columns([3, 1])
with left:
    st.subheader("Price & EMAs")
    st.line_chart(enriched[["Close", "ema_fast", "ema_slow", "ema_trend"]], height=320)
    st.subheader("RSI(14)")
    st.line_chart(enriched[["rsi"]], height=160)
    st.subheader("MACD")
    st.line_chart(enriched[["macd", "macd_signal"]], height=160)
with right:
    st.subheader("Factor votes")
    st.bar_chart(pd.Series(signal.factors, name="score"))

if do_backtest:
    st.subheader("Backtest (next-bar directional, no costs)")
    try:
        bt = run_backtest(enriched, threshold=threshold).as_dict()
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Signals", bt["signals"])
        b2.metric("Hit rate", f"{bt['hit_rate'] * 100:.1f}%", f"{bt['wins']}W/{bt['losses']}L")
        b3.metric("Cumulative", f"{bt['cumulative_return_pct']:+.2f}%")
        b4.metric("Buy & hold", f"{bt['buy_hold_return_pct']:+.2f}%")
    except ValueError as exc:
        st.info(f"Backtest skipped: {exc}")

st.divider()
st.caption(
    "Markets are not reliably predictable. This tool organizes technical context — "
    "it is not a trade recommendation."
)
