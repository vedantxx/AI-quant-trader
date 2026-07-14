"""Walk-forward allocation backtester.

Refits the HMM on each in-sample window, then trades the out-of-sample window
using only information available at each bar (no look-ahead). Applies slippage
on weight changes. Produces an equity curve and per-bar regime log.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.hmm_engine import HMMEngine
from core.regime_strategies import RegimeStrategy
from data.feature_engineering import build_features, trend_flag


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    regimes: pd.Series
    trades: int
    folds: int
    meta: dict = field(default_factory=dict)


class Backtester:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        b = cfg["backtest"]
        self.initial_capital = b["initial_capital"]
        self.slippage = b["slippage_pct"]
        self.train_window = b["train_window"]
        self.test_window = b["test_window"]
        self.step_size = b["step_size"]
        self.strategy = RegimeStrategy(cfg)

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """Single-asset walk-forward backtest on OHLCV ``df``."""
        feats = build_features(df)
        close = df["close"].reindex(feats.index)
        trend = trend_flag(df["close"]).reindex(feats.index).fillna(False)
        rets = close.pct_change().fillna(0.0)

        equity = self.initial_capital
        eq_curve, regime_log, ret_log = [], [], []
        index_log = []
        prev_weight = 0.0
        trades = 0
        folds = 0

        n = len(feats)
        start = self.train_window
        while start + self.test_window <= n:
            folds += 1
            train = feats.iloc[start - self.train_window : start].to_numpy()
            engine = HMMEngine(self.cfg)
            try:
                engine.fit(train)
            except Exception:
                start += self.step_size
                continue

            test_slice = slice(start, start + self.test_window)
            for i in range(test_slice.start, test_slice.stop):
                # information up to and including bar i-1 only
                window = feats.iloc[: i].to_numpy()
                regime = engine.predict(window)
                is_trend = bool(trend.iloc[i - 1]) if i > 0 else False
                alloc = self.strategy.target(
                    regime.label, is_trend, regime.confidence
                )
                weight = alloc.exposure * alloc.size_mult
                if abs(weight - prev_weight) >= self.strategy.rebalance_threshold:
                    equity *= 1 - self.slippage * abs(weight - prev_weight)
                    prev_weight = weight
                    trades += 1

                bar_ret = prev_weight * rets.iloc[i]
                equity *= 1 + bar_ret

                eq_curve.append(equity)
                ret_log.append(bar_ret)
                regime_log.append(regime.label)
                index_log.append(feats.index[i])

            start += self.step_size

        idx = pd.DatetimeIndex(index_log)
        return BacktestResult(
            equity_curve=pd.Series(eq_curve, index=idx, name="equity"),
            returns=pd.Series(ret_log, index=idx, name="returns"),
            regimes=pd.Series(regime_log, index=idx, name="regime"),
            trades=trades,
            folds=folds,
            meta={"initial_capital": self.initial_capital},
        )
