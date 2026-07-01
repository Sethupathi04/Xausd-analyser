# XAU15M — 15-Minute Technical Analysis for XAUUSD (Gold)

A small, transparent command-line tool that pulls **15-minute gold candles**
(COMEX gold futures `GC=F` via [yfinance](https://github.com/ranaroussi/yfinance),
a free proxy for spot XAUUSD), computes a suite of classic technical indicators,
and prints a **directional signal** (LONG / SHORT / NEUTRAL) with a confidence
score, an ATR-based stop/target, and a **walk-forward backtest** of the strategy's
historical hit rate.

> ⚠️ **This is a heuristic analysis aid, not financial advice and not a crystal
> ball.** Short-term markets are not reliably predictable. Use it to organize
> technical context, not to make trades blindly.

## What it does

1. **Fetches** 15m OHLCV candles (default last 60 days of `GC=F`).
2. **Computes** indicators with no heavy TA dependency:
   - EMA 20 / 50 / 200 (trend & cross)
   - MACD (12, 26, 9)
   - RSI (14)
   - Bollinger Bands (20, 2σ) with %B
   - ATR (14) for volatility-scaled stops
   - ADX / +DI / -DI (14) for trend strength
3. **Scores** a transparent weighted vote of those factors into a signal.
4. **Backtests** the signal bar-by-bar against realized next-bar moves and
   reports hit rate, average return, cumulative return vs. buy & hold.

## Install

```bash
cd xauusd-15m-analyzer
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Usage

```bash
# Default: GC=F, 60 days of 15m candles, with backtest
xau15m

# Or run as a module without installing
python -m xau15m.cli

# Custom window / no backtest / JSON output
xau15m --period 30d --no-backtest
xau15m --json

# Analyze your own CSV (needs a datetime column + OHLC[V])
xau15m --csv my_gold_15m.csv
```

### Key options

| Flag | Default | Meaning |
|------|---------|---------|
| `--symbol` | `GC=F` | yfinance ticker |
| `--period` | `60d` | history window (yfinance caps 15m at ~60d) |
| `--interval` | `15m` | candle interval |
| `--threshold` | `0.2` | abs. score needed to fire LONG/SHORT |
| `--atr-stop-mult` | `1.5` | ATR multiple for the stop distance |
| `--rr` | `2.0` | reward-to-risk ratio for the target |
| `--csv` | – | load candles from a file instead of yfinance |
| `--json` | – | machine-readable output |
| `--no-backtest` | – | skip the historical evaluation |

## How the signal is built

Each factor returns a score in `[-1, 1]`; the weighted average is mapped to a
label. Weights (normalized internally):

| Factor | Weight | Logic |
|--------|--------|-------|
| Trend (price vs EMA200) | 0.30 | above = bullish |
| EMA cross (20 vs 50) | 0.25 | fast above slow = bullish |
| MACD histogram | 0.20 | positive = bullish |
| RSI(14) | 0.15 | oversold→long, overbought→short |
| Bollinger %B | 0.10 | near lower band→long, upper→short |

`score >= +threshold → LONG`, `score <= -threshold → SHORT`, else `NEUTRAL`.
Stops/targets use `ATR × atr-stop-mult` and the `rr` ratio.

## Tests & lint

```bash
pip install -e ".[dev]"
pytest -q
ruff check .
mypy xau15m
```

## Limitations

- `GC=F` (gold futures) ≈ but is not identical to spot XAUUSD; for exact broker
  pricing, feed your own CSV.
- yfinance limits intraday history (~60 days for 15m).
- The backtest is frictionless (no spread/commission/slippage) and assumes you
  can act on each closed bar — real results will be worse.
