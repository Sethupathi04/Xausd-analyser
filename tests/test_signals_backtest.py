from xau15m.backtest import run_backtest
from xau15m.indicators import compute_all
from xau15m.signals import (
    LONG,
    NEUTRAL,
    SHORT,
    aggregate_score,
    generate_signal,
    label_from_score,
)


def test_label_from_score():
    assert label_from_score(0.5) == LONG
    assert label_from_score(-0.5) == SHORT
    assert label_from_score(0.0) == NEUTRAL


def test_aggregate_score_bounds():
    factors = {"trend": 1, "ema_cross": 1, "macd": 1, "rsi": 1, "bollinger": 1}
    assert abs(aggregate_score(factors) - 1.0) < 1e-9
    factors = {k: -1 for k in factors}
    assert abs(aggregate_score(factors) + 1.0) < 1e-9


def test_generate_signal_fields(synthetic_candles):
    enriched = compute_all(synthetic_candles)
    sig = generate_signal(enriched)
    assert sig.label in (LONG, SHORT, NEUTRAL)
    assert 0.0 <= sig.confidence <= 1.0
    assert sig.price > 0
    assert set(sig.factors) == {"trend", "ema_cross", "macd", "rsi", "bollinger"}


def test_long_stop_below_price(synthetic_candles):
    # The synthetic series trends up, so expect a LONG with stop below price.
    enriched = compute_all(synthetic_candles)
    sig = generate_signal(enriched)
    if sig.label == LONG:
        assert sig.stop < sig.price < sig.target
    elif sig.label == SHORT:
        assert sig.target < sig.price < sig.stop


def test_backtest_consistency(synthetic_candles):
    enriched = compute_all(synthetic_candles)
    bt = run_backtest(enriched)
    assert bt.signals == bt.long_signals + bt.short_signals
    assert bt.wins + bt.losses <= bt.signals
    assert 0.0 <= bt.hit_rate <= 1.0
    d = bt.as_dict()
    assert isinstance(d["cumulative_return_pct"], float)
