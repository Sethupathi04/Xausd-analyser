"""Walk-forward backtest of the next-bar directional signal."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .signals import LONG, NEUTRAL, aggregate_score, label_from_score, score_row


@dataclass
class BacktestResult:
    bars: int
    signals: int
    long_signals: int
    short_signals: int
    wins: int
    losses: int
    hit_rate: float
    avg_return_pct: float
    cumulative_return_pct: float
    buy_hold_return_pct: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "bars": self.bars,
            "signals": self.signals,
            "long_signals": self.long_signals,
            "short_signals": self.short_signals,
            "wins": self.wins,
            "losses": self.losses,
            "hit_rate": round(self.hit_rate, 4),
            "avg_return_pct": round(self.avg_return_pct, 4),
            "cumulative_return_pct": round(self.cumulative_return_pct, 4),
            "buy_hold_return_pct": round(self.buy_hold_return_pct, 4),
        }


def run_backtest(enriched: pd.DataFrame, threshold: float = 0.2) -> BacktestResult:
    """Evaluate the signal on each bar against the realized next-bar return.

    A LONG signal "wins" if the next close is higher; a SHORT wins if it is
    lower. Returns are computed as the directional next-bar percentage move,
    so they reflect a simple always-flat-between-bars strategy.
    """
    df = enriched.dropna(
        subset=["ema_trend", "ema_fast", "ema_slow", "macd_hist", "rsi", "bb_pct", "atr"]
    )
    if len(df) < 3:
        raise ValueError("Not enough data after warm-up to run a backtest.")

    close = df["Close"].to_numpy()
    fwd_ret = np.empty(len(df))
    fwd_ret[:-1] = (close[1:] - close[:-1]) / close[:-1]
    fwd_ret[-1] = np.nan

    signals = 0
    longs = 0
    shorts = 0
    wins = 0
    losses = 0
    strat_returns: list[float] = []

    # Skip the final bar (no realized forward return).
    for i in range(len(df) - 1):
        factors = score_row(df.iloc[i])
        label = label_from_score(aggregate_score(factors), threshold)
        if label == NEUTRAL:
            continue

        r = fwd_ret[i]
        signals += 1
        if label == LONG:
            longs += 1
            trade_ret = r
        else:
            shorts += 1
            trade_ret = -r

        strat_returns.append(trade_ret)
        if trade_ret > 0:
            wins += 1
        elif trade_ret < 0:
            losses += 1

    hit_rate = wins / signals if signals else 0.0
    avg_ret = float(np.mean(strat_returns)) * 100.0 if strat_returns else 0.0
    cum_ret = (np.prod([1.0 + r for r in strat_returns]) - 1.0) * 100.0 if strat_returns else 0.0
    buy_hold = (close[-1] - close[0]) / close[0] * 100.0

    return BacktestResult(
        bars=len(df),
        signals=signals,
        long_signals=longs,
        short_signals=shorts,
        wins=wins,
        losses=losses,
        hit_rate=hit_rate,
        avg_return_pct=avg_ret,
        cumulative_return_pct=cum_ret,
        buy_hold_return_pct=buy_hold,
    )
