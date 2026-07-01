"""Data loading for XAUUSD 15-minute candles."""

from __future__ import annotations

import pandas as pd

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean OHLCV frame with a tz-naive DatetimeIndex."""
    if df.empty:
        raise ValueError("No data returned for the requested symbol/range.")

    # yfinance may return a MultiIndex column frame when given a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={c: c.title() for c in df.columns})
    missing = [c for c in ["Open", "High", "Low", "Close"] if c not in df.columns]
    if missing:
        raise ValueError(f"Data is missing required columns: {missing}")

    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    df = df[OHLCV_COLUMNS].astype("float64")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df.index = idx
    df.index.name = "Datetime"
    return df


def fetch_candles(
    symbol: str = "GC=F",
    period: str = "60d",
    interval: str = "15m",
) -> pd.DataFrame:
    """Fetch OHLCV candles via yfinance.

    GC=F is COMEX gold futures, a freely available proxy for spot XAUUSD on the
    15-minute timeframe.
    """
    import yfinance as yf

    raw = yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return _normalize(raw)


def load_csv(path: str) -> pd.DataFrame:
    """Load candles from a CSV file with a datetime index and OHLCV columns."""
    df = pd.read_csv(path)
    time_col = next(
        (c for c in df.columns if c.lower() in {"datetime", "date", "time", "timestamp"}),
        None,
    )
    if time_col is None:
        raise ValueError("CSV must contain a datetime/date/time/timestamp column.")
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.set_index(time_col)
    return _normalize(df)
