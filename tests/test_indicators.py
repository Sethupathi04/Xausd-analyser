import numpy as np
import pandas as pd

from xau15m.indicators import adx, atr, bollinger, compute_all, ema, macd, rsi


def test_ema_matches_pandas(synthetic_candles):
    s = synthetic_candles["Close"]
    expected = s.ewm(span=20, adjust=False).mean()
    pd.testing.assert_series_equal(ema(s, 20), expected)


def test_rsi_bounds(synthetic_candles):
    r = rsi(synthetic_candles["Close"], 14).dropna()
    assert (r >= 0).all() and (r <= 100).all()


def test_macd_columns(synthetic_candles):
    out = macd(synthetic_candles["Close"])
    assert list(out.columns) == ["macd", "macd_signal", "macd_hist"]
    np.testing.assert_allclose(
        out["macd_hist"], out["macd"] - out["macd_signal"], rtol=1e-9
    )


def test_bollinger_ordering(synthetic_candles):
    bb = bollinger(synthetic_candles["Close"]).dropna()
    assert (bb["bb_upper"] >= bb["bb_mid"]).all()
    assert (bb["bb_mid"] >= bb["bb_lower"]).all()


def test_atr_positive(synthetic_candles):
    a = atr(
        synthetic_candles["High"], synthetic_candles["Low"], synthetic_candles["Close"]
    ).dropna()
    assert (a > 0).all()


def test_adx_bounds(synthetic_candles):
    out = adx(
        synthetic_candles["High"], synthetic_candles["Low"], synthetic_candles["Close"]
    ).dropna()
    assert (out["adx"] >= 0).all() and (out["adx"] <= 100).all()


def test_compute_all_adds_columns(synthetic_candles):
    out = compute_all(synthetic_candles)
    for col in ["ema_fast", "ema_slow", "ema_trend", "rsi", "macd_hist", "bb_pct", "atr", "adx"]:
        assert col in out.columns
    assert len(out) == len(synthetic_candles)
