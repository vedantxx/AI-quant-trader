"""Tests for the live orchestration (TradingSystem) — offline, mocked broker."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("hmmlearn")

import main
from broker.alpaca_client import AlpacaClient

CFG = {
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
             "duplicate_window_seconds": 60, "gap_multiple": 3.0, "overnight_gap_risk_pct": 0.02,
             "halt_lock_file": "trading_halted.lock"},
    "features": {"zscore_window": 30},
    "monitoring": {"dashboard_refresh_seconds": 5, "alert_rate_limit_minutes": 15},
}


def _bars(n=900, seed=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    vol = np.where((np.arange(n) // 60) % 2 == 0, 0.008, 0.025)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, vol)))
    idx = pd.date_range("2019-01-01", periods=n, freq="B", tz="UTC")
    mi = pd.MultiIndex.from_arrays([["SYN"] * n, idx], names=["symbol", "timestamp"])
    close = pd.Series(close)
    return pd.DataFrame(
        {"open": close.shift(1).bfill().to_numpy(), "high": (close * 1.01).to_numpy(),
         "low": (close * 0.99).to_numpy(), "close": close.to_numpy(),
         "volume": rng.uniform(1e6, 5e6, n)}, index=mi)


class FakeData:
    def __init__(self, df):
        self._df = df

    def get_stock_bars(self, *a, **k):
        return SimpleNamespace(df=self._df)

    def get_stock_latest_quote(self, *a, **k):
        return {"SYN": SimpleNamespace(bid_price=100.0, ask_price=100.02)}


class FakeTrading:
    def __init__(self, equity=100_000.0):
        self.equity = equity
        self.calls = []

    def get_clock(self):
        return SimpleNamespace(is_open=False)

    def get_account(self):
        return SimpleNamespace(equity=str(self.equity), cash=str(self.equity),
                               buying_power=str(self.equity * 2),
                               portfolio_value=str(self.equity))

    def get_all_positions(self):
        return []

    def submit_order(self, req):
        self.calls.append(req)
        return SimpleNamespace(id="o1", symbol="SYN", side="buy", qty=1, status="accepted")


@pytest.fixture
def system(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(main, "SNAPSHOT_PATH", tmp_path / "state_snapshot.json")
    client = AlpacaClient(CFG, trading_client=FakeTrading(), data_client=FakeData(_bars()),
                          max_retries=1)
    return main.TradingSystem(CFG, dry_run=True, client=client)


# --------------------------------------------------------------------- startup
def test_startup_trains_and_persists(system, tmp_path):
    system.startup()
    assert system.hmm.model is not None
    assert (tmp_path / "models" / "hmm_SYN.pkl").exists()
    assert system.orchestrator is not None


def test_run_once_full_pipeline_dry(system):
    system.startup()
    summary = system.run_once()
    assert summary["status"] == "ok"
    assert summary["regime"]
    assert summary["equity"] == 100_000.0
    # dry-run: no orders submitted
    assert system.client.trading.calls == []


def test_dry_run_places_no_orders(system):
    system.startup()
    system.run(max_iterations=1)
    assert system.client.trading.calls == []


def test_snapshot_roundtrip(system, tmp_path):
    system.startup()
    system._last_regime = "BULL"
    system.state.peak_equity = 123_456.0
    system._save_snapshot()
    assert (tmp_path / "state_snapshot.json").exists()
    system._last_regime = None
    system._load_snapshot()
    assert system._last_regime == "BULL"
    assert system.state.peak_equity == 123_456.0


def test_model_reused_when_fresh(system, tmp_path):
    system.startup()                      # trains + saves
    first = (tmp_path / "models" / "hmm_SYN.pkl").stat().st_mtime
    system2 = main.TradingSystem(CFG, dry_run=True, client=system.client)
    system2.load_or_train_hmm()           # should LOAD, not retrain
    assert (tmp_path / "models" / "hmm_SYN.pkl").stat().st_mtime == first


def test_hmm_error_holds_regime(system, monkeypatch):
    system.startup()
    system._last_regime = "NEUTRAL"
    monkeypatch.setattr(system.hmm, "predict",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    out = system.run_once()
    assert out["status"] == "hmm_error" and out["regime"] == "NEUTRAL"


# ------------------------------------------------------------------------- cli
def test_cli_parses_trade_flags():
    args = main.build_parser().parse_args(
        ["trade", "--dry-run", "--iterations", "1", "--poll", "0"])
    assert args.mode == "trade" and args.dry_run and args.iterations == 1
