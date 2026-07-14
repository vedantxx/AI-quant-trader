"""Stress tests: crash injection, gap simulation, vol shocks.

Perturbs a price series to probe how the strategy and risk breakers behave under
tail events, then re-runs the backtester on the shocked series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtester import Backtester, BacktestResult


@dataclass
class StressScenario:
    name: str
    result: BacktestResult
    max_drawdown: float


def inject_crash(
    df: pd.DataFrame, at: float = 0.5, magnitude: float = -0.20
) -> pd.DataFrame:
    """Drop close by ``magnitude`` at fractional position ``at`` (single-bar crash)."""
    out = df.copy()
    i = int(len(out) * at)
    factor = 1 + magnitude
    out.iloc[i:, out.columns.get_loc("close")] *= factor
    out.iloc[i:, out.columns.get_loc("low")] *= factor
    out.iloc[i, out.columns.get_loc("low")] = out.iloc[i]["close"] * (1 + magnitude)
    return out


def inject_gap(df: pd.DataFrame, at: float = 0.5, gap: float = -0.10) -> pd.DataFrame:
    """Overnight gap: open jumps by ``gap`` and prices shift from ``at`` onward."""
    out = df.copy()
    i = int(len(out) * at)
    for col in ("open", "high", "low", "close"):
        out.iloc[i:, out.columns.get_loc(col)] *= 1 + gap
    return out


def inject_vol_shock(
    df: pd.DataFrame, at: float = 0.5, window: int = 20, mult: float = 3.0
) -> pd.DataFrame:
    """Amplify return magnitude over a window to simulate a volatility spike."""
    out = df.copy()
    i = int(len(out) * at)
    close = out["close"].to_numpy(dtype=float)
    for j in range(i, min(i + window, len(close))):
        base = close[j - 1]
        close[j] = base + (close[j] - base) * mult
    out["close"] = close
    return out


class StressTester:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.backtester = Backtester(cfg)

    def run_all(self, df: pd.DataFrame) -> list[StressScenario]:
        scenarios = {
            "baseline": df,
            "crash_-20%": inject_crash(df),
            "gap_-10%": inject_gap(df),
            "vol_shock_3x": inject_vol_shock(df),
        }
        out = []
        for name, shocked in scenarios.items():
            res = self.backtester.run(shocked)
            peak = res.equity_curve.cummax()
            mdd = float((res.equity_curve / peak - 1).min()) if len(res.equity_curve) else 0.0
            out.append(StressScenario(name, res, mdd))
        return out
