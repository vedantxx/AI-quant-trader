"""Order placement, modification, and cancellation via Alpaca."""

from __future__ import annotations

from dataclasses import dataclass

try:
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
except ImportError:
    MarketOrderRequest = LimitOrderRequest = None
    OrderSide = TimeInForce = None

from .alpaca_client import AlpacaClient


@dataclass
class OrderResult:
    id: str
    symbol: str
    side: str
    qty: float
    status: str


class OrderExecutor:
    def __init__(self, client: AlpacaClient):
        self.client = client

    def submit_market(self, symbol: str, qty: int, side: str) -> OrderResult:
        if qty <= 0:
            raise ValueError("qty must be positive")
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        o = self.client.trading.submit_order(req)
        return self._wrap(o)

    def submit_limit(
        self, symbol: str, qty: int, side: str, limit_price: float
    ) -> OrderResult:
        if qty <= 0:
            raise ValueError("qty must be positive")
        req = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            limit_price=round(limit_price, 2),
        )
        o = self.client.trading.submit_order(req)
        return self._wrap(o)

    def cancel(self, order_id: str) -> None:
        self.client.trading.cancel_order_by_id(order_id)

    def cancel_all(self) -> None:
        self.client.trading.cancel_orders()

    @staticmethod
    def _wrap(o) -> OrderResult:
        return OrderResult(
            id=str(o.id),
            symbol=o.symbol,
            side=str(o.side),
            qty=float(o.qty),
            status=str(o.status),
        )
