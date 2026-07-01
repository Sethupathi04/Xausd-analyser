"""Signal generation from indicator values.

The signal is a transparent, weighted vote across several classic technical
factors. Each factor returns a score in [-1, 1]; the weighted average is mapped
to a LONG / SHORT / NEUTRAL label with a confidence value. This is a heuristic
analysis aid, not a guaranteed predictor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

LONG = "LONG"
SHORT = "SHORT"
NEUTRAL = "NEUTRAL"

# Factor weights (sum is normalized internally).
WEIGHTS = {
    "trend": 0.30,
    "ema_cross": 0.25,
    "macd": 0.20,
    "rsi": 0.15,
    "bollinger": 0.10,
}


@dataclass
class Signal:
    label: str
    score: float
    confidence: float
    price: float
    atr: float
    adx: float
    stop: float
    target: float
    risk_reward: float
    factors: dict[str, float] = field(default_factory=dict)


def _trend_factor(row: pd.Series) -> float:
    if row["Close"] > row["ema_trend"]:
        return 1.0
    if row["Close"] < row["ema_trend"]:
        return -1.0
    return 0.0


def _ema_cross_factor(row: pd.Series) -> float:
    if row["ema_fast"] > row["ema_slow"]:
        return 1.0
    if row["ema_fast"] < row["ema_slow"]:
        return -1.0
    return 0.0


def _macd_factor(row: pd.Series) -> float:
    hist = row["macd_hist"]
    if hist > 0:
        return 1.0
    if hist < 0:
        return -1.0
    return 0.0


def _rsi_factor(row: pd.Series) -> float:
    r = row["rsi"]
    if r >= 70:
        return -1.0  # overbought -> fade
    if r <= 30:
        return 1.0  # oversold -> bounce
    # Linearly scale 40..60 band around neutral momentum.
    return max(-1.0, min(1.0, (r - 50.0) / 20.0))


def _bollinger_factor(row: pd.Series) -> float:
    pct = row["bb_pct"]
    if pd.isna(pct):
        return 0.0
    if pct >= 1.0:
        return -1.0
    if pct <= 0.0:
        return 1.0
    return max(-1.0, min(1.0, (pct - 0.5) * 2.0))


def score_row(row: pd.Series) -> dict[str, float]:
    return {
        "trend": _trend_factor(row),
        "ema_cross": _ema_cross_factor(row),
        "macd": _macd_factor(row),
        "rsi": _rsi_factor(row),
        "bollinger": _bollinger_factor(row),
    }


def aggregate_score(factors: dict[str, float]) -> float:
    total_w = sum(WEIGHTS.values())
    return sum(factors[k] * WEIGHTS[k] for k in WEIGHTS) / total_w


def label_from_score(score: float, threshold: float = 0.2) -> str:
    if score >= threshold:
        return LONG
    if score <= -threshold:
        return SHORT
    return NEUTRAL


def generate_signal(
    enriched: pd.DataFrame,
    threshold: float = 0.2,
    atr_stop_mult: float = 1.5,
    rr: float = 2.0,
) -> Signal:
    """Build a Signal from the most recent fully-formed candle."""
    row = enriched.iloc[-1]
    factors = score_row(row)
    score = aggregate_score(factors)
    label = label_from_score(score, threshold)
    confidence = min(1.0, abs(score))

    price = float(row["Close"])
    atr_val = float(row["atr"]) if not pd.isna(row["atr"]) else 0.0
    adx_val = float(row["adx"]) if not pd.isna(row["adx"]) else 0.0
    stop_dist = atr_stop_mult * atr_val

    if label == LONG:
        stop = price - stop_dist
        target = price + rr * stop_dist
    elif label == SHORT:
        stop = price + stop_dist
        target = price - rr * stop_dist
    else:
        stop = price
        target = price

    return Signal(
        label=label,
        score=round(score, 4),
        confidence=round(confidence, 4),
        price=round(price, 2),
        atr=round(atr_val, 2),
        adx=round(adx_val, 2),
        stop=round(stop, 2),
        target=round(target, 2),
        risk_reward=rr if label != NEUTRAL else 0.0,
        factors={k: round(v, 3) for k, v in factors.items()},
    )
