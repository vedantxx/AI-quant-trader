"""Performance metrics: Sharpe, Sortino, drawdown, regime breakdown, benchmark.

Stub scaffold: signatures, type hints, and docstrings only — no logic yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class PerformanceReport:
    """Summary performance statistics for a backtest run."""

    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    volatility: float
    regime_breakdown: dict = field(default_factory=dict)
    vs_benchmark: dict = field(default_factory=dict)


class PerformanceAnalyzer:
    """Compute risk-adjusted metrics and regime/benchmark breakdowns."""

    def __init__(self, cfg: dict) -> None:
        """Read backtest.risk_free_rate config."""
        ...

    def analyze(
        self,
        equity: pd.Series,
        returns: pd.Series,
        regimes: pd.Series,
        benchmark: Optional[pd.Series] = None,
    ) -> PerformanceReport:
        """Full performance report, optionally versus a benchmark series."""
        ...

    def sharpe(self, returns: pd.Series) -> float:
        """Annualized Sharpe ratio."""
        ...

    def sortino(self, returns: pd.Series) -> float:
        """Annualized Sortino ratio (downside deviation)."""
        ...

    def max_drawdown(self, equity: pd.Series) -> float:
        """Maximum peak-to-trough drawdown."""
        ...

    def regime_breakdown(
        self, returns: pd.Series, regimes: pd.Series
    ) -> dict:
        """Per-regime bar count, mean return, and Sharpe."""
        ...
