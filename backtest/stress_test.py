"""Stress tests: crash injection, gap simulation, vol shocks.

Perturbs a price series to probe how the strategy and risk breakers behave under
tail events, then re-runs the backtester on the shocked series.

Stub scaffold: signatures, type hints, and docstrings only — no logic yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .backtester import BacktestResult


@dataclass
class StressScenario:
    """A single stress scenario and its backtest outcome."""

    name: str
    result: BacktestResult
    max_drawdown: float


class StressTester:
    """Inject tail events and re-run the backtester."""

    def __init__(self, cfg: dict) -> None:
        """Build the backtester collaborator from config."""
        ...

    def inject_crash(
        self, df: pd.DataFrame, at: float = 0.5, magnitude: float = -0.20
    ) -> pd.DataFrame:
        """Drop close by ``magnitude`` at fractional position ``at``."""
        ...

    def inject_gap(
        self, df: pd.DataFrame, at: float = 0.5, gap: float = -0.10
    ) -> pd.DataFrame:
        """Shift prices by ``gap`` from ``at`` onward (overnight gap)."""
        ...

    def inject_vol_shock(
        self, df: pd.DataFrame, at: float = 0.5, window: int = 20, mult: float = 3.0
    ) -> pd.DataFrame:
        """Amplify return magnitude over a window to simulate a vol spike."""
        ...

    def run_all(self, df: pd.DataFrame) -> list[StressScenario]:
        """Run baseline + crash + gap + vol-shock scenarios."""
        ...
