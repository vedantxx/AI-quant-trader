"""Tests for order executor. Alpaca client is mocked — no network."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("alpaca", reason="alpaca-py required for order request objects")

from broker.order_executor import OrderExecutor


def _fake_order(oid="abc", symbol="SPY", side="buy", qty=10, status="accepted"):
    return SimpleNamespace(id=oid, symbol=symbol, side=side, qty=qty, status=status)


def _client():
    client = MagicMock()
    client.trading.submit_order.return_value = _fake_order()
    return client


def test_submit_market_wraps_result():
    ex = OrderExecutor(_client())
    res = ex.submit_market("SPY", 10, "buy")
    assert res.symbol == "SPY"
    assert res.qty == 10.0
    assert res.status == "accepted"


def test_zero_qty_rejected():
    ex = OrderExecutor(_client())
    with pytest.raises(ValueError):
        ex.submit_market("SPY", 0, "buy")


def test_limit_requires_positive_qty():
    ex = OrderExecutor(_client())
    with pytest.raises(ValueError):
        ex.submit_limit("SPY", -5, "sell", limit_price=400)


def test_cancel_all_delegates():
    client = _client()
    ex = OrderExecutor(client)
    ex.cancel_all()
    client.trading.cancel_orders.assert_called_once()
