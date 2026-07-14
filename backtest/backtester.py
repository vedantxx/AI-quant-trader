"""Walk-forward allocation backtester.

Refits the HMM on each in-sample window, then trades the out-of-sample window
using only information available at each bar (no look-ahead). Applies slippage on
weight changes and produces an equity curve plus a per-bar regime log.

Stub scaffold: signatures, type hints, and docstrings only — no logic yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class BacktestResult:
    """Output of a walk-forward backtest run."""

    equity_curve: pd.Series
    returns: pd.Series
    regimes: pd.Series
    trades: int
    folds: int
    meta: dict = field(default_factory=dict)


class Backtester:
    """Walk-forward allocation backtester."""

    def __init__(self, cfg: dict) -> None:
        """Read backtest.* config; build the strategy collaborator."""
        ...

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """Single-asset walk-forward backtest over OHLCV ``df``."""
        ...
