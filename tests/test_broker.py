"""Tests for AlpacaClient, MarketData, PositionTracker (mocked, offline)."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from broker.alpaca_client import AlpacaClient
from broker.position_tracker import PositionTracker
from core.risk_manager import CircuitBreaker, PortfolioState
from data.market_data import MarketData

CFG = {"broker": {"paper_trading": True, "timeframe": "1Day"},
       "risk": {"daily_dd_reduce": 0.02, "daily_dd_halt": 0.03,
                "weekly_dd_reduce": 0.05, "weekly_dd_halt": 0.07,
                "max_dd_from_peak": 0.10}}


class FakeTrading:
    def __init__(self, equity=100_000.0, positions=None) -> None:
        self.equity = equity
        self._positions = positions or [
            SimpleNamespace(symbol="SPY", qty="10", avg_entry_price="100",
                            current_price="110", market_value="1100",
                            unrealized_pl="100", unrealized_plpc="0.1")]

    def get_clock(self):
        return SimpleNamespace(is_open=True)

    def get_account(self):
        return SimpleNamespace(equity=str(self.equity), cash="50000",
                               buying_power="200000", portfolio_value=str(self.equity))

    def get_all_positions(self):
        return self._positions


class FakeData:
    def __init__(self, df) -> None:
        self._df = df

    def get_stock_bars(self, *args, **kwargs):
        return SimpleNamespace(df=self._df)

    def get_stock_latest_quote(self, *args, **kwargs):
        return SimpleNamespace(bid_price=99.9, ask_price=100.1)


def make_client(trading=None, data=None) -> AlpacaClient:
    return AlpacaClient(CFG, trading_client=trading or FakeTrading(),
                        data_client=data or object())


# --------------------------------------------------------------- alpaca client
def test_paper_default_and_market_open() -> None:
    c = make_client()
    assert c.paper is True and c.base_url.endswith("paper-api.alpaca.markets")
    assert c.is_market_open() is True


def test_live_requires_confirmation() -> None:
    cfg = {"broker": {"paper_trading": False}}
    with pytest.raises(RuntimeError):
        AlpacaClient(cfg, trading_client=FakeTrading(), data_client=object(),
                     confirm_fn=lambda p: "no", connect=False)
    # correct phrase proceeds
    c = AlpacaClient(cfg, trading_client=FakeTrading(), data_client=object(),
                     confirm_fn=lambda p: "YES I UNDERSTAND THE RISKS", connect=False)
    assert c.paper is False and c.base_url.endswith("api.alpaca.markets")


def test_get_account_maps_snapshot() -> None:
    snap = make_client().get_account()
    assert snap.equity == 100_000.0 and "SPY" in snap.positions


def test_retry_raises_after_exhaustion() -> None:
    class Broken(FakeTrading):
        def get_clock(self):
            raise TimeoutError("down")
    with pytest.raises(ConnectionError):
        AlpacaClient(CFG, trading_client=Broken(), data_client=object(), max_retries=2)


# ------------------------------------------------------------------ market data
def test_clean_bars_flattens_and_sorts() -> None:
    idx = pd.MultiIndex.from_tuples(
        [("SPY", pd.Timestamp("2024-01-03", tz="UTC")),
         ("SPY", pd.Timestamp("2024-01-02", tz="UTC")),
         ("SPY", pd.Timestamp("2024-01-02", tz="UTC"))],  # duplicate
        names=["symbol", "timestamp"])
    df = pd.DataFrame({"open": [3, 2, 2], "high": [3, 2, 2], "low": [3, 2, 2],
                       "close": [3, 2, 2.5], "volume": [1, 1, 1]}, index=idx)
    out = MarketData._clean_bars(df, "SPY")
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out.index.tz is None
    assert len(out) == 2 and out.index.is_monotonic_increasing
    assert out["close"].iloc[0] == 2.5  # keep=last on the duplicate


def test_history_normalized() -> None:
    idx = pd.MultiIndex.from_tuples(
        [("SPY", pd.Timestamp("2024-01-02", tz="UTC")),
         ("SPY", pd.Timestamp("2024-01-03", tz="UTC"))], names=["symbol", "timestamp"])
    df = pd.DataFrame({"open": [1, 2], "high": [1, 2], "low": [1, 2],
                       "close": [1, 2], "volume": [9, 9]}, index=idx)
    md = MarketData(CFG, make_client(data=FakeData(df)))
    hist = md.history("SPY", lookback_days=10)
    assert len(hist) == 2 and hist.index.tz is None


def test_latest_quote_spread() -> None:
    md = MarketData(CFG, make_client(data=FakeData(pd.DataFrame())))
    q = md.get_latest_quote("SPY")
    assert q["spread_pct"] == pytest.approx((100.1 - 99.9) / 100.0, rel=1e-3)


# --------------------------------------------------------------- position tracker
def test_snapshot_and_weights() -> None:
    pt = PositionTracker(make_client())
    assert "SPY" in pt.snapshot()
    assert pt.current_weights(100_000.0)["SPY"] == pytest.approx(0.011)
    assert pt.total_unrealized() == 100.0


def test_on_fill_tracks_and_closes() -> None:
    pt = PositionTracker(make_client())
    pt.on_fill("AAPL", "buy", 5, 150.0, stop_level=145.0, regime="BULL")
    t = pt.tracked["AAPL"]
    assert t.qty == 5 and t.stop_level == 145.0 and t.regime_at_entry == "BULL"
    pt.on_fill("AAPL", "sell", 5, 155.0)          # flatten
    assert "AAPL" not in pt.tracked


def test_fill_updates_circuit_breaker() -> None:
    # account equity 89k vs 100k peak -> peak kill switch on the fill
    state = PortfolioState(equity=100_000, cash=0, buying_power=0,
                           peak_equity=100_000, day_start_equity=100_000,
                           week_start_equity=100_000)
    breaker = CircuitBreaker(CFG, lock_dir=None)
    client = make_client(trading=FakeTrading(equity=89_000))
    pt = PositionTracker(client, circuit_breaker=breaker, portfolio_state=state)
    try:
        pt.on_fill("SPY", "buy", 1, 100.0)
        assert state.circuit_breaker_status in ("peak_dd", "locked")
    finally:
        breaker.clear_lock()


def test_reconcile_drops_stale() -> None:
    pt = PositionTracker(make_client())
    pt.tracked["OLD"] = pt.tracked.get("OLD") or SimpleNamespace()
    pt.reconcile(now=datetime.now(timezone.utc))
    assert "OLD" not in pt.tracked and "SPY" in pt.tracked
