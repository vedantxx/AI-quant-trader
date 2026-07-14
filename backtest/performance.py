"""Performance metrics: Sharpe, Sortino, drawdown, regime breakdown, benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_ANNUALIZATION = 252


@dataclass
class PerformanceReport:
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    volatility: float
    regime_breakdown: dict
    vs_benchmark: dict


def _sharpe(returns: pd.Series, rf: float) -> float:
    excess = returns - rf / _ANNUALIZATION
    sd = excess.std()
    return float(np.sqrt(_ANNUALIZATION) * excess.mean() / sd) if sd else 0.0


def _sortino(returns: pd.Series, rf: float) -> float:
    excess = returns - rf / _ANNUALIZATION
    downside = excess[excess < 0].std()
    return float(np.sqrt(_ANNUALIZATION) * excess.mean() / downside) if downside else 0.0


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1).min())


def compute(
    equity: pd.Series,
    returns: pd.Series,
    regimes: pd.Series,
    cfg: dict,
    benchmark: pd.Series | None = None,
) -> PerformanceReport:
    rf = cfg["backtest"]["risk_free_rate"]
    n = len(returns)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1) if n else 0.0
    years = n / _ANNUALIZATION if n else 1
    cagr = float((1 + total_return) ** (1 / years) - 1) if years > 0 else 0.0
    mdd = max_drawdown(equity)

    breakdown = {}
    for label in ("low", "mid", "high"):
        mask = regimes == label
        r = returns[mask]
        breakdown[label] = {
            "bars": int(mask.sum()),
            "mean_ret": float(r.mean()) if len(r) else 0.0,
            "sharpe": _sharpe(r, rf) if len(r) > 1 else 0.0,
        }

    vs_bench = {}
    if benchmark is not None and len(benchmark) == n:
        bench_total = float((1 + benchmark).prod() - 1)
        vs_bench = {
            "benchmark_return": bench_total,
            "excess_return": total_return - bench_total,
            "benchmark_sharpe": _sharpe(benchmark, rf),
        }

    return PerformanceReport(
        total_return=total_return,
        cagr=cagr,
        sharpe=_sharpe(returns, rf),
        sortino=_sortino(returns, rf),
        max_drawdown=mdd,
        calmar=float(cagr / abs(mdd)) if mdd else 0.0,
        win_rate=float((returns > 0).mean()) if n else 0.0,
        volatility=float(returns.std() * np.sqrt(_ANNUALIZATION)),
        regime_breakdown=breakdown,
        vs_benchmark=vs_bench,
    )
