"""Stress tests: crash injection, gap risk, regime misclassification.

Perturbs the OHLCV series to probe how the allocation strategy and the drawdown
circuit breaker behave under tail events, then re-runs the backtester on the
shocked series. The regime-misclassification test deliberately mislabels regimes
to confirm the allocation caps — not the HMM's accuracy — are what contain the
damage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data.feature_engineering import atr

from .backtester import BacktestResult, Backtester


@dataclass
class StressScenario:
    """A single stress scenario and its backtest outcome."""

    name: str
    result: BacktestResult
    max_drawdown: float


@dataclass
class StressSummary:
    """Aggregate outcome of a Monte-Carlo stress family."""

    name: str
    mean_max_loss: float
    worst_case: float
    breaker_fired_pct: float
    detail: dict = field(default_factory=dict)


class StressTester:
    """Inject tail events and re-run the backtester."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.bt = Backtester(cfg)

    # --------------------------------------------------------- injections
    def inject_crash(
        self, df: pd.DataFrame, at: float = 0.5, magnitude: float = -0.10
    ) -> pd.DataFrame:
        """Single-day crash: drop one bar's OHLC by ``magnitude`` (transient)."""
        out = df.copy()
        i = int(len(df) * at)
        i = min(max(i, 0), len(df) - 1)
        for col in ("open", "high", "low", "close"):
            out.iloc[i, out.columns.get_loc(col)] *= (1 + magnitude)
        return out

    def inject_gap(
        self, df: pd.DataFrame, at: float = 0.5, gap: float = -0.10
    ) -> pd.DataFrame:
        """Overnight gap: shift all prices from ``at`` onward by ``gap`` (persists)."""
        out = df.copy()
        i = int(len(df) * at)
        i = min(max(i, 0), len(df) - 1)
        for col in ("open", "high", "low", "close"):
            out.iloc[i:, out.columns.get_loc(col)] *= (1 + gap)
        return out

    def inject_vol_shock(
        self, df: pd.DataFrame, at: float = 0.5, window: int = 20, mult: float = 3.0
    ) -> pd.DataFrame:
        """Amplify per-bar return magnitude over a window to simulate a vol spike."""
        out = df.copy()
        i = int(len(df) * at)
        j = min(i + window, len(df))
        close = out["close"].to_numpy(dtype=float)
        for k in range(max(i, 1), j):
            r = close[k] / close[k - 1] - 1
            close[k] = close[k - 1] * (1 + r * mult)
        scale = close / out["close"].to_numpy(dtype=float)
        for col in ("open", "high", "low", "close"):
            out[col] = out[col].to_numpy(dtype=float) * scale
        return out

    def _crashes(self, df: pd.DataFrame, seed: int, n_gaps: int,
                 mag_range: tuple[float, float]) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        out = df
        points = rng.uniform(0.05, 0.95, n_gaps)
        for at in points:
            mag = rng.uniform(*mag_range)
            out = self.inject_crash(out, at, mag)
        return out

    # ----------------------------------------------------------- families
    def crash_test(
        self, df: pd.DataFrame, n_sims: int = 100, n_gaps: int = 10,
        mag_range: tuple[float, float] = (-0.15, -0.05),
    ) -> StressSummary:
        """Insert ``n_gaps`` single-day crashes at random points; ``n_sims`` MC
        runs. Report mean max loss, worst case, % where the breaker fired."""
        losses, fired = [], 0
        for sim in range(n_sims):
            shocked = self._crashes(df, sim, n_gaps, mag_range)
            res = self.bt.run(shocked)
            mdd = self._mdd(res)
            losses.append(mdd)
            fired += bool(res.meta.get("breaker_fired"))
        return StressSummary(
            name="crash_mc",
            mean_max_loss=float(np.mean(losses)) if losses else 0.0,
            worst_case=float(np.min(losses)) if losses else 0.0,
            breaker_fired_pct=fired / n_sims if n_sims else 0.0,
            detail={"n_sims": n_sims, "n_gaps": n_gaps},
        )

    def gap_test(
        self, df: pd.DataFrame, n_sims: int = 50, n_gaps: int = 5,
        atr_mult_range: tuple[float, float] = (2.0, 5.0),
    ) -> StressSummary:
        """Insert overnight gaps of 2-5x ATR at random points; report expected
        (theoretical) loss versus the actual realized drawdown."""
        natr = (atr(df["high"], df["low"], df["close"], 14) / df["close"]).median()
        losses, expected = [], []
        for sim in range(n_sims):
            rng = np.random.default_rng(1000 + sim)
            out = df
            exp = 0.0
            for _ in range(n_gaps):
                at = rng.uniform(0.05, 0.95)
                mult = rng.uniform(*atr_mult_range)
                gap = -mult * float(natr)
                out = self.inject_gap(out, at, gap)
                exp += gap
            res = self.bt.run(out)
            losses.append(self._mdd(res))
            expected.append(exp)
        return StressSummary(
            name="gap_risk",
            mean_max_loss=float(np.mean(losses)) if losses else 0.0,
            worst_case=float(np.min(losses)) if losses else 0.0,
            breaker_fired_pct=0.0,
            detail={"expected_cum_gap": float(np.mean(expected)) if expected else 0.0,
                    "median_natr": float(natr)},
        )

    def regime_misclass_test(self, df: pd.DataFrame, n_sims: int = 20) -> StressSummary:
        """Shuffle the regime->strategy mapping so strategies run on the WRONG
        regimes. If the system blows up, risk management isn't independent
        enough. A well-capped allocation strategy stays bounded."""
        baseline = self._mdd(self.bt.run(df))
        losses = []
        for sim in range(n_sims):
            res = self.bt.run(df, shuffle_regime_seed=sim)
            losses.append(self._mdd(res))
        return StressSummary(
            name="regime_misclass",
            mean_max_loss=float(np.mean(losses)) if losses else 0.0,
            worst_case=float(np.min(losses)) if losses else 0.0,
            breaker_fired_pct=0.0,
            detail={"baseline_mdd": baseline,
                    "worst_vs_baseline": (float(np.min(losses)) - baseline) if losses else 0.0},
        )

    def run_all(self, df: pd.DataFrame) -> list[StressScenario]:
        """Baseline + single crash + single gap + vol-shock scenarios."""
        scenarios = {
            "baseline": df,
            "crash": self.inject_crash(df, 0.5, -0.15),
            "gap": self.inject_gap(df, 0.5, -0.10),
            "vol_shock": self.inject_vol_shock(df, 0.5, 20, 3.0),
        }
        out = []
        for name, sdf in scenarios.items():
            res = self.bt.run(sdf)
            out.append(StressScenario(name=name, result=res, max_drawdown=self._mdd(res)))
        return out

    @staticmethod
    def _mdd(res: BacktestResult) -> float:
        eq = res.equity_curve
        if len(eq) == 0:
            return 0.0
        return float((eq / eq.cummax() - 1).min())
