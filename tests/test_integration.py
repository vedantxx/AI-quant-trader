"""Phase 9 integration tests: end-to-end pipeline, look-ahead invariance,
risk stress, and crash recovery."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("hmmlearn")

import main
from broker.alpaca_client import AlpacaClient
from core.regime_strategies import Direction, Signal
from core.risk_manager import Position, PortfolioState, RiskManager


def _cfg() -> dict:
    return {
        "broker": {"paper_trading": True, "timeframe": "1Day", "symbols": ["SYN"],
                   "regime_symbol": "SYN"},
        "hmm": {"n_candidates": [3, 4], "n_init": 1, "covariance_type": "diag",
                "min_train_bars": 60, "stability_bars": 3, "flicker_window": 20,
                "flicker_threshold": 4, "min_confidence": 0.55},
        "strategy": {"low_vol_allocation": 0.95, "mid_vol_allocation_trend": 0.95,
                     "mid_vol_allocation_no_trend": 0.60, "high_vol_allocation": 0.60,
                     "low_vol_leverage": 1.25, "rebalance_threshold": 0.10,
                     "uncertainty_size_mult": 0.50},
        "risk": {"max_risk_per_trade": 0.01, "max_exposure": 0.80, "max_leverage": 1.25,
                 "max_single_position": 0.15, "max_concurrent": 5, "max_daily_trades": 20,
                 "daily_dd_reduce": 0.02, "daily_dd_halt": 0.03, "weekly_dd_reduce": 0.05,
                 "weekly_dd_halt": 0.07, "max_dd_from_peak": 0.10, "size_reduce_factor": 0.50,
                 "min_position_usd": 100, "max_sector_exposure": 0.30, "correlation_window": 60,
                 "correlation_reduce": 0.70, "correlation_reject": 0.85, "max_spread_pct": 0.005,
                 "duplicate_window_seconds": 60, "gap_multiple": 3.0,
                 "overnight_gap_risk_pct": 0.02, "halt_lock_file": "trading_halted.lock"},
        "features": {"zscore_window": 30},
        "backtest": {"slippage_pct": 0.0005, "initial_capital": 100000,
                     "train_window": 120, "test_window": 40, "step_size": 40,
                     "risk_free_rate": 0.045},
        "monitoring": {},
    }


def _ohlcv(n=560, seed=11, multiindex=False):
    rng = np.random.default_rng(seed)
    vol = np.where((np.arange(n) // 60) % 2 == 0, 0.008, 0.025)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, vol)))
    idx = pd.date_range("2019-01-01", periods=n, freq="B", tz="UTC")
    c = pd.Series(close)
    data = {"open": c.shift(1).bfill().to_numpy(), "high": (c * 1.01).to_numpy(),
            "low": (c * 0.99).to_numpy(), "close": c.to_numpy(),
            "volume": rng.uniform(1e6, 5e6, n)}
    if multiindex:
        mi = pd.MultiIndex.from_arrays([["SYN"] * n, idx], names=["symbol", "timestamp"])
        return pd.DataFrame(data, index=mi)
    return pd.DataFrame(data, index=idx.tz_localize(None))


# ----------------------------------------------- look-ahead: end-date invariance
def test_backtest_invariant_to_end_date():
    """Appending future bars must not change past results (no look-ahead)."""
    from backtest.backtester import Backtester

    full = _ohlcv(560)
    short = full.iloc[:480]                       # cut 80 bars off the END
    bt = Backtester(_cfg())
    eq_full = bt.run(full, "SYN").equity_curve
    eq_short = bt.run(short, "SYN").equity_curve

    shared = eq_full.index.intersection(eq_short.index)
    assert len(shared) > 20                       # meaningful overlap
    # identical on every shared bar — deterministic, causal folds
    assert np.allclose(eq_full.loc[shared].to_numpy(),
                       eq_short.loc[shared].to_numpy(), rtol=1e-9, atol=1e-6)


# ------------------------------------------------------------ risk stress gates
def test_extreme_signal_capped():
    """A signal demanding 500% exposure is capped to the single-position limit."""
    rm = RiskManager(_cfg())
    st = PortfolioState(equity=100_000, cash=100_000, buying_power=400_000,
                        peak_equity=100_000, day_start_equity=100_000,
                        week_start_equity=100_000)
    sig = Signal(symbol="SPY", direction=Direction.LONG, confidence=0.9,
                 entry_price=100.0, stop_loss=95.0, position_size_pct=5.0,
                 leverage=1.25, regime_id=0, regime_name="BULL", regime_probability=0.9,
                 strategy_name="x", reasoning="", metadata={})
    d = rm.validate_signal(sig, st)
    assert d.approved and d.modified_signal.position_size_pct <= 0.15


def test_no_double_entry_rapid_fire():
    """Same symbol+direction inside the duplicate window is rejected."""
    rm = RiskManager(_cfg())
    now = datetime.now(timezone.utc)
    st = PortfolioState(equity=100_000, cash=100_000, buying_power=400_000,
                        peak_equity=100_000, day_start_equity=100_000,
                        week_start_equity=100_000,
                        recent_orders={("SPY", Direction.LONG): now - timedelta(seconds=10)})
    sig = Signal(symbol="SPY", direction=Direction.LONG, confidence=0.9, entry_price=100.0,
                 stop_loss=95.0, position_size_pct=0.95, leverage=1.25, regime_id=0,
                 regime_name="BULL", regime_probability=0.9, strategy_name="x",
                 reasoning="", metadata={})
    assert rm.validate_signal(sig, st, now=now).approved is False


def test_no_stop_rejected():
    rm = RiskManager(_cfg())
    st = PortfolioState(equity=100_000, cash=100_000, buying_power=400_000,
                        peak_equity=100_000, day_start_equity=100_000, week_start_equity=100_000)
    sig = Signal(symbol="SPY", direction=Direction.LONG, confidence=0.9, entry_price=100.0,
                 stop_loss=float("nan"), position_size_pct=0.95, leverage=1.0, regime_id=0,
                 regime_name="BULL", regime_probability=0.9, strategy_name="x",
                 reasoning="", metadata={})
    assert rm.validate_signal(sig, st).approved is False


# --------------------------------------------------------------- crash recovery
class _FakeData:
    def get_stock_bars(self, *a, **k):
        return SimpleNamespace(df=_ohlcv(560, multiindex=True))

    def get_stock_latest_quote(self, *a, **k):
        return {"SYN": SimpleNamespace(bid_price=100.0, ask_price=100.02)}


class _FakeTrading:
    def get_clock(self):
        return SimpleNamespace(is_open=False)

    def get_account(self):
        return SimpleNamespace(equity="100000", cash="100000",
                               buying_power="200000", portfolio_value="100000")

    def get_all_positions(self):
        return [SimpleNamespace(symbol="SYN", qty="10", avg_entry_price="100",
                                current_price="105", market_value="1050",
                                unrealized_pl="50", unrealized_plpc="0.05")]


def test_recovery_restores_state_and_reconciles(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(main, "SNAPSHOT_PATH", tmp_path / "snap.json")
    client = AlpacaClient(_cfg(), trading_client=_FakeTrading(), data_client=_FakeData(),
                          max_retries=1)

    s1 = main.TradingSystem(_cfg(), dry_run=True, client=client)
    s1.startup()
    s1._last_regime = "BULL"
    s1.state.peak_equity = 111_000.0
    s1._save_snapshot()

    # "restart": fresh system, same persisted model + snapshot + broker positions
    s2 = main.TradingSystem(_cfg(), dry_run=True, client=client)
    s2.startup()
    assert s2._last_regime == "BULL"               # recovered from snapshot
    assert s2.state.peak_equity >= 111_000.0
    # reconcile picked up the broker's existing position exactly once (no double)
    assert list(s2.tracker.tracked) == ["SYN"]
    assert s2.tracker.tracked["SYN"].qty == 10
