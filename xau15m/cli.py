"""Command-line interface for the XAUUSD 15-minute analyzer."""

from __future__ import annotations

import argparse
import json
import sys
import warnings

import pandas as pd

from . import __version__
from .backtest import run_backtest
from .data import fetch_candles, load_csv
from .indicators import compute_all
from .signals import LONG, SHORT, generate_signal

_COLOR = {"LONG": "\033[92m", "SHORT": "\033[91m", "NEUTRAL": "\033[93m"}
_RESET = "\033[0m"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="xau15m",
        description="15-minute technical analysis & signal for XAUUSD (gold).",
    )
    p.add_argument("--symbol", default="GC=F", help="yfinance ticker (default: GC=F gold futures).")
    p.add_argument("--period", default="60d", help="History window, e.g. 30d, 60d (default: 60d).")
    p.add_argument("--interval", default="15m", help="Candle interval (default: 15m).")
    p.add_argument("--csv", default=None, help="Load candles from a CSV instead of yfinance.")
    p.add_argument("--threshold", type=float, default=0.2, help="Signal score threshold.")
    p.add_argument("--atr-stop-mult", type=float, default=1.5, help="ATR multiple for stops.")
    p.add_argument("--rr", type=float, default=2.0, help="Reward-to-risk ratio for the target.")
    p.add_argument("--no-backtest", action="store_true", help="Skip the historical backtest.")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    p.add_argument("--version", action="version", version=f"xau15m {__version__}")
    return p


def _load(args: argparse.Namespace) -> pd.DataFrame:
    if args.csv:
        return load_csv(args.csv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return fetch_candles(symbol=args.symbol, period=args.period, interval=args.interval)


def _render_human(args, df, enriched, signal, bt) -> str:
    last_time = enriched.index[-1]
    color = "" if args.no_color else _COLOR.get(signal.label, "")
    reset = "" if args.no_color else _RESET
    src = args.csv if args.csv else f"{args.symbol} ({args.interval}, {args.period})"

    lines = [
        "=" * 56,
        " XAUUSD 15-MINUTE TECHNICAL ANALYSIS",
        "=" * 56,
        f" Source        : {src}",
        f" Candles        : {len(df)}",
        f" Last bar (UTC) : {last_time}",
        f" Last price     : {signal.price}",
        "-" * 56,
        f" SIGNAL         : {color}{signal.label}{reset}",
        f" Score          : {signal.score:+.3f}  (threshold ±{args.threshold})",
        f" Confidence     : {signal.confidence * 100:.1f}%",
        f" ATR(14)        : {signal.atr}",
        f" ADX(14)        : {signal.adx}  ({'trending' if signal.adx >= 25 else 'weak/range'})",
    ]
    if signal.label in (LONG, SHORT):
        lines += [
            f" Suggested stop : {signal.stop}",
            f" Suggested tgt  : {signal.target}  (R:R 1:{signal.risk_reward:g})",
        ]
    lines.append("-" * 56)
    lines.append(" Factor breakdown:")
    for name, val in signal.factors.items():
        lines.append(f"   {name:<10}: {val:+.2f}")

    if bt is not None:
        d = bt.as_dict()
        lines += [
            "-" * 56,
            " BACKTEST (next-bar directional, no costs):",
            f"   Signals      : {d['signals']} (L {d['long_signals']} / S {d['short_signals']})",
            f"   Hit rate     : {d['hit_rate'] * 100:.1f}%  ({d['wins']}W / {d['losses']}L)",
            f"   Avg / signal : {d['avg_return_pct']:+.4f}%",
            f"   Cumulative   : {d['cumulative_return_pct']:+.2f}%",
            f"   Buy & hold   : {d['buy_hold_return_pct']:+.2f}%",
        ]
    lines += [
        "=" * 56,
        " NOTE: Heuristic analysis aid, NOT financial advice.",
        "       Markets are not reliably predictable.",
        "=" * 56,
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        df = _load(args)
    except Exception as exc:  # noqa: BLE001 - surface a clean message to the CLI user
        print(f"error: failed to load data: {exc}", file=sys.stderr)
        return 2

    enriched = compute_all(df)
    signal = generate_signal(
        enriched, threshold=args.threshold, atr_stop_mult=args.atr_stop_mult, rr=args.rr
    )

    bt = None
    if not args.no_backtest:
        try:
            bt = run_backtest(enriched, threshold=args.threshold)
        except ValueError as exc:
            print(f"warning: backtest skipped: {exc}", file=sys.stderr)

    if args.json:
        payload = {
            "symbol": args.symbol,
            "interval": args.interval,
            "period": args.period,
            "candles": len(df),
            "last_bar_utc": str(enriched.index[-1]),
            "signal": signal.__dict__,
            "backtest": bt.as_dict() if bt is not None else None,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_render_human(args, df, enriched, signal, bt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
