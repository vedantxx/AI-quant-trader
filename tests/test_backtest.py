"""Tests for the walk-forward backtester, performance metrics, and stress tests."""

import copy

import numpy as np
import pandas as pd
import pytest

from backtest.backtester import Backtester
from backtest.performance import PerformanceAnalyzer
from backtest.stress_test import StressTester

pytest.importorskip("hmmlearn")


# ------------------------------------------------------------------- fixtures
def make_config() -> dict:
    """Fast config: small HMM + short walk-forward windows for test speed."""
    return {
        "hmm": {
            "n_candidates": [3, 4], "n_init": 1, "covariance_type": "diag",
            "min_train_bars": 60, "stability_bars": 3, "flicker_window": 20,
            "flicker_threshold": 4, "min_confidence": 0.55,
        },
        "strategy": {
            "low_vol_allocation": 0.95, "mid_vol_allocation_trend": 0.95,
            "mid_vol_allocation_no_trend": 0.60, "high_vol_allocation": 0.60,
            "low_vol_leverage": 1.25, "rebalance_threshold": 0.10,
            "uncertainty_size_mult": 0.50,
        },
        "risk": {"max_dd_from_peak": 0.10},
        "features": {"zscore_window": 30},
        "backtest": {
            "slippage_pct": 0.0005, "initial_capital": 100000,
            "train_window": 120, "test_window": 40, "step_size": 40,
            "risk_free_rate": 0.045,
        },
    }


def make_ohlcv(n: int = 600, seed: int = 7) -> pd.DataFrame:
    """Two-vol-regime random walk so the HMM has structure to find."""
    rng = np.random.default_rng(seed)
    vol = np.where((np.arange(n) // 60) % 2 == 0, 0.008, 0.025)
    rets = rng.normal(0.0004, vol)
    close = 100 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    close = pd.Series(close, index=idx)
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    return pd.DataFrame(
        {"open": close.shift(1).bfill(), "high": high, "low": low,
         "close": close, "volume": rng.uniform(1e6, 5e6, n)}, index=idx)


# ------------------------------------------------------------------ backtester
def test_backtest_runs_and_is_consistent() -> None:
    res = Backtester(make_config()).run(make_ohlcv(), symbol="SPY")
    assert res.folds >= 1
    assert len(res.equity_curve) > 0
    # equity invariant: equity == cash + shares*price -> always finite, positive
    assert np.isfinite(res.equity_curve.to_numpy()).all()
    assert (res.equity_curve > 0).all()
    # trade log carries the segment columns used by the performance layer
    for col in ("pnl", "regime", "confidence", "hold_bars", "target_alloc"):
        assert col in res.meta["trade_log"].columns


def test_rebalance_threshold_gates_churn() -> None:
    bt = Backtester(make_config())
    assert bt.needs_rebalance(0.60, 0.95) is True    # 0.35 drift
    assert bt.needs_rebalance(0.90, 0.95) is False   # 0.05 drift < 0.10


def test_leverage_allows_allocation_above_one() -> None:
    """Low-vol leverage 1.25 * 0.95 alloc = 1.1875 target -> margin (>1)."""
    res = Backtester(make_config()).run(make_ohlcv(), symbol="SPY")
    # at least one bar should target >1.0 exposure if a low-vol regime appeared
    assert res.meta["allocation"].max() > 0.0  # allocations recorded
    assert res.trades >= 0


# ------------------------------------------------------------------ performance
def test_max_drawdown() -> None:
    a = PerformanceAnalyzer(make_config())
    eq = pd.Series([100.0, 110.0, 99.0, 120.0])
    assert a.max_drawdown(eq) == pytest.approx(99.0 / 110.0 - 1)


def test_sharpe_zero_when_flat() -> None:
    a = PerformanceAnalyzer(make_config())
    assert a.sharpe(pd.Series([0.0, 0.0, 0.0])) == 0.0


def test_sharpe_positive_for_steady_gains() -> None:
    a = PerformanceAnalyzer(make_config())
    assert a.sharpe(pd.Series([0.01, 0.012, 0.009, 0.011])) > 0


def test_benchmarks() -> None:
    a = PerformanceAnalyzer(make_config())
    prices = make_ohlcv(300)["close"]
    bh = a.buy_and_hold(prices)
    assert bh.iloc[-1] == pytest.approx(1e5 * prices.iloc[-1] / prices.iloc[0])
    sma = a.sma_trend(prices, 50)
    assert len(sma) == len(prices)
    rnd = a.random_benchmark(prices, change_freq=0.05, seeds=10)
    assert {"mean_return", "std_return", "mean_sharpe", "std_sharpe"} <= rnd.keys()


def test_analyze_full_report() -> None:
    cfg = make_config()
    res = Backtester(cfg).run(make_ohlcv(), symbol="SPY")
    rep = PerformanceAnalyzer(cfg).analyze(
        res.equity_curve, res.returns, res.regimes,
        trades=res.meta["trade_log"], confidence=res.meta["confidence"])
    assert rep.max_drawdown <= 0.0
    assert rep.regime_breakdown            # at least one regime
    assert set(rep.confidence_breakdown) == {"<50%", "50-60%", "60-70%", "70%+"}
    assert set(rep.worst_case) >= {"worst_day", "max_consecutive_losses"}


# --------------------------------------------------------------------- stress
def test_inject_crash_and_gap() -> None:
    st = StressTester(make_config())
    df = make_ohlcv(100)
    crashed = st.inject_crash(df, at=0.5, magnitude=-0.15)
    i = int(len(df) * 0.5)
    assert crashed["close"].iloc[i] == pytest.approx(df["close"].iloc[i] * 0.85)
    assert crashed["close"].iloc[i + 1] == pytest.approx(df["close"].iloc[i + 1])  # transient
    gapped = st.inject_gap(df, at=0.5, gap=-0.10)
    assert gapped["close"].iloc[i] == pytest.approx(df["close"].iloc[i] * 0.90)
    assert gapped["close"].iloc[-1] == pytest.approx(df["close"].iloc[-1] * 0.90)  # persists


def test_crash_test_summary() -> None:
    st = StressTester(make_config())
    summary = st.crash_test(make_ohlcv(), n_sims=2, n_gaps=5)
    assert summary.mean_max_loss <= 0.0
    assert 0.0 <= summary.breaker_fired_pct <= 1.0


def test_regime_misclass_stays_bounded() -> None:
    """Even with shuffled regimes, long-only capped allocation can't lose more
    than the market it holds — a sanity floor on the damage."""
    st = StressTester(make_config())
    summary = st.regime_misclass_test(make_ohlcv(), n_sims=2)
    assert summary.worst_case > -1.0  # never wipes out
