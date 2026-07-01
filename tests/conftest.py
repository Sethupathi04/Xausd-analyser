import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_candles() -> pd.DataFrame:
    """Deterministic uptrending-then-noisy 15m series for indicator tests."""
    n = 400
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    rng = np.random.default_rng(42)
    drift = np.linspace(0, 40, n)
    noise = rng.normal(0, 2, n).cumsum()
    close = 2000.0 + drift + noise
    high = close + np.abs(rng.normal(1.5, 0.5, n))
    low = close - np.abs(rng.normal(1.5, 0.5, n))
    open_ = close - rng.normal(0, 1, n)
    vol = rng.integers(100, 1000, n).astype("float64")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )
