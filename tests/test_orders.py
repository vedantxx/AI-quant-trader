"""Tests for order executor (client mocked, no network, no SDK required)."""

from types import SimpleNamespace

import pytest

from broker.alpaca_client import AlpacaClient
from broker.order_executor import OrderExecutor, OrderResult
from core.regime_strategies import Direction, Signal

CFG = {"broker": {"paper_trading": True}}


class FakeTrading:
    """Records calls; returns SDK-shaped namespaces. Offline requests are dicts."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.order_status = "filled"

    def get_clock(self):
        return SimpleNamespace(is_open=True)

    @staticmethod
    def _f(req, name):
        return req[name] if isinstance(req, dict) else getattr(req, name)

    def submit_order(self, req):
        self.calls.append(("submit", req))
        return SimpleNamespace(id="ord-1", symbol=self._f(req, "symbol"),
                               side=self._f(req, "side"), qty=self._f(req, "qty"),
                               status="accepted")

    def get_order_by_id(self, oid):
        return SimpleNamespace(id=oid, status=self.order_status, symbol="SPY",
                               side="buy", qty=10)

    def cancel_order_by_id(self, oid):
        self.calls.append(("cancel", oid))

    def cancel_orders(self):
        self.calls.append(("cancel_all", None))

    def replace_order_by_id(self, oid, patch):
        self.calls.append(("replace", oid, patch))

    def close_position(self, sym):
        self.calls.append(("close", sym))

    def close_all_positions(self, cancel_orders=True):
        self.calls.append(("close_all", cancel_orders))


def make_client() -> AlpacaClient:
    return AlpacaClient(CFG, trading_client=FakeTrading(), data_client=object(),
                        max_retries=1)


def make_signal(**kw) -> Signal:
    base = dict(symbol="SPY", direction=Direction.LONG, confidence=0.9,
                entry_price=100.0, stop_loss=95.0, position_size_pct=0.95,
                leverage=1.25, regime_id=0, regime_name="BULL", regime_probability=0.9,
                strategy_name="low_vol_bull", reasoning="", metadata={"qty": 10})
    base.update(kw)
    return Signal(**base)


# ---------------------------------------------------------------- legacy path
def test_submit_market_wraps_result() -> None:
    ex = OrderExecutor(make_client())
    res = ex.submit_market("SPY", 10, "buy")
    assert isinstance(res, OrderResult) and res.symbol == "SPY" and res.trade_id


def test_zero_qty_rejected() -> None:
    with pytest.raises(ValueError):
        OrderExecutor(make_client()).submit_market("SPY", 0, "buy")


def test_limit_requires_positive_qty() -> None:
    with pytest.raises(ValueError):
        OrderExecutor(make_client()).submit_limit("SPY", -1, "buy", 100.0)


def test_cancel_all_delegates() -> None:
    c = make_client()
    OrderExecutor(c).cancel_all()
    assert ("cancel_all", None) in c.trading.calls


# ----------------------------------------------------------------- signal path
def test_marketable_limit_price() -> None:
    assert OrderExecutor._limit_price("buy", 100.0) == 100.10   # pay up 0.1%
    assert OrderExecutor._limit_price("sell", 100.0) == 99.90   # concede 0.1%


def test_submit_order_links_trade_id() -> None:
    ex = OrderExecutor(make_client())
    res = ex.submit_order(make_signal(), risk_modifications=["size capped"])
    assert res.trade_id in ex.trades
    link = ex.trades[res.trade_id]
    assert link.symbol == "SPY" and link.qty == 10 and link.risk_modifications


def test_submit_order_needs_qty() -> None:
    sig = make_signal(metadata={})
    with pytest.raises(ValueError):
        OrderExecutor(make_client()).submit_order(sig)


def test_modify_stop_tighten_only() -> None:
    ex = OrderExecutor(make_client())
    ex._stops["SPY"] = 95.0
    assert ex.modify_stop("SPY", 97.0) is True     # raise = tighten
    assert ex._stops["SPY"] == 97.0
    assert ex.modify_stop("SPY", 96.0) is False    # lower = widen -> rejected
    assert ex._stops["SPY"] == 97.0


def test_bracket_order() -> None:
    ex = OrderExecutor(make_client())
    res = ex.submit_bracket_order(make_signal(take_profit=120.0))
    assert res.order_type == "bracket" and res.trade_id in ex.trades


def test_await_fill_returns_on_fill() -> None:
    ex = OrderExecutor(make_client(), sleep=lambda s: None)
    res = ex.await_fill_or_cancel("ord-1", timeout=3, poll=1)
    assert res.status == "filled"


def test_await_cancels_then_market_retry() -> None:
    c = make_client()
    c.trading.order_status = "new"  # never fills
    ex = OrderExecutor(c, sleep=lambda s: None)
    res = ex.await_fill_or_cancel("ord-1", timeout=2, poll=1,
                                  retry_market=True, symbol="SPY", qty=10, side="buy")
    assert ("cancel", "ord-1") in c.trading.calls
    assert res.order_type == "market"


def test_close_all_clears_stops() -> None:
    c = make_client()
    ex = OrderExecutor(c)
    ex._stops["SPY"] = 95.0
    ex.close_all_positions()
    assert ex._stops == {} and ("close_all", True) in c.trading.calls
