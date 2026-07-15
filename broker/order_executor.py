"""Order placement, modification, and cancellation via Alpaca.

Submits marketable LIMIT orders by default (+/-0.1% of price), with an optional
30-second timeout that cancels and retries at market. Supports bracket (OCO)
orders (entry + stop + take-profit) and tighten-only stop modification. Every
order carries a unique ``trade_id`` linking signal -> risk decision -> order ->
fill for auditing.

The alpaca-py request/enum imports are guarded, so this module imports without
the SDK; when the SDK is absent, request objects fall back to plain dicts so the
executor can be driven by an injected/mock trading client offline.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from core.regime_strategies import Signal

    from .alpaca_client import AlpacaClient

logger = logging.getLogger("regime-trader.orders")

LIMIT_OFFSET = 0.001   # +/-0.1% marketable limit
FILL_TIMEOUT_S = 30.0

try:  # guarded SDK imports
    from alpaca.trading.requests import (
        LimitOrderRequest,
        MarketOrderRequest,
        StopLossRequest,
        TakeProfitRequest,
    )
    from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
    _SDK = True
except ImportError:  # pragma: no cover
    _SDK = False


@dataclass
class OrderResult:
    """Normalized result of a submitted order."""

    id: str
    symbol: str
    side: str
    qty: float
    status: str
    trade_id: str = ""
    order_type: str = "market"


@dataclass
class TradeLink:
    """Audit trail: one trade_id ties a signal to its order and fill."""

    trade_id: str
    symbol: str
    side: str
    qty: float
    signal: Optional["Signal"] = None
    risk_modifications: list = field(default_factory=list)
    order_id: str = ""
    stop_order_id: str = ""
    status: str = "new"


class OrderExecutor:
    """Submit, modify, and cancel orders through the Alpaca client."""

    def __init__(self, client: "AlpacaClient", sleep: Callable[[float], None] = time.sleep) -> None:
        self.client = client
        self._sleep = sleep
        self._stops: dict[str, float] = {}          # symbol -> current stop level
        self._stop_orders: dict[str, str] = {}      # symbol -> stop order id
        self.trades: dict[str, TradeLink] = {}       # trade_id -> link

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _side_for(direction: str) -> str:
        return "buy" if direction == "LONG" else "sell"

    @staticmethod
    def _limit_price(side: str, price: float) -> float:
        """Marketable limit: pay up 0.1% to buy, concede 0.1% to sell."""
        return round(price * (1 + LIMIT_OFFSET) if side == "buy"
                     else price * (1 - LIMIT_OFFSET), 2)

    def _submit(self, request) -> object:
        return self.client._with_retry(self.client.trading.submit_order, request)

    @staticmethod
    def _normalize(order, trade_id: str = "", order_type: str = "market") -> OrderResult:
        return OrderResult(
            id=str(getattr(order, "id", "")),
            symbol=getattr(order, "symbol", ""),
            side=str(getattr(order, "side", "")),
            qty=float(getattr(order, "qty", 0) or 0),
            status=str(getattr(order, "status", "")),
            trade_id=trade_id,
            order_type=order_type,
        )

    # ----------------------------------------------------- request builders
    def _market_req(self, symbol: str, qty: int, side: str, trade_id: str):
        if _SDK:
            return MarketOrderRequest(
                symbol=symbol, qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY, client_order_id=trade_id)
        return {"type": "market", "symbol": symbol, "qty": qty, "side": side,
                "time_in_force": "day", "client_order_id": trade_id}

    def _limit_req(self, symbol: str, qty: int, side: str, limit_price: float, trade_id: str):
        if _SDK:
            return LimitOrderRequest(
                symbol=symbol, qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY, limit_price=limit_price,
                client_order_id=trade_id)
        return {"type": "limit", "symbol": symbol, "qty": qty, "side": side,
                "time_in_force": "day", "limit_price": limit_price,
                "client_order_id": trade_id}

    def _bracket_req(self, symbol: str, qty: int, side: str, entry: float,
                     stop: float, take: Optional[float], trade_id: str):
        if _SDK:
            tp = TakeProfitRequest(limit_price=round(take, 2)) if take else None
            sl = StopLossRequest(stop_price=round(stop, 2))
            return LimitOrderRequest(
                symbol=symbol, qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.GTC, limit_price=round(entry, 2),
                order_class=OrderClass.BRACKET, stop_loss=sl, take_profit=tp,
                client_order_id=trade_id)
        return {"type": "bracket", "symbol": symbol, "qty": qty, "side": side,
                "limit_price": round(entry, 2), "stop_price": round(stop, 2),
                "take_profit": round(take, 2) if take else None,
                "client_order_id": trade_id}

    # --------------------------------------------------------- signal entry
    def submit_order(
        self, signal: "Signal", qty: Optional[int] = None,
        risk_modifications: Optional[list] = None,
    ) -> OrderResult:
        """Marketable LIMIT order from a (risk-approved) signal."""
        qty = self._resolve_qty(signal, qty)
        side = self._side_for(signal.direction)
        trade_id = uuid.uuid4().hex
        limit = self._limit_price(side, signal.entry_price)
        link = TradeLink(trade_id, signal.symbol, side, qty, signal,
                         risk_modifications or [])
        order = self._submit(self._limit_req(signal.symbol, qty, side, limit, trade_id))
        res = self._normalize(order, trade_id, "limit")
        link.order_id, link.status = res.id, res.status
        self.trades[trade_id] = link
        self._stops[signal.symbol] = signal.stop_loss
        logger.info("submitted %s %s x%d @limit %.2f trade_id=%s",
                    side, signal.symbol, qty, limit, trade_id)
        return res

    def submit_bracket_order(
        self, signal: "Signal", qty: Optional[int] = None,
    ) -> OrderResult:
        """Entry + stop-loss + take-profit as a single Alpaca bracket (OCO)."""
        qty = self._resolve_qty(signal, qty)
        side = self._side_for(signal.direction)
        trade_id = uuid.uuid4().hex
        order = self._submit(self._bracket_req(
            signal.symbol, qty, side, signal.entry_price, signal.stop_loss,
            signal.take_profit, trade_id))
        res = self._normalize(order, trade_id, "bracket")
        link = TradeLink(trade_id, signal.symbol, side, qty, signal)
        link.order_id, link.status = res.id, res.status
        self.trades[trade_id] = link
        self._stops[signal.symbol] = signal.stop_loss
        return res

    def _resolve_qty(self, signal: "Signal", qty: Optional[int]) -> int:
        if qty is None:
            qty = (signal.metadata or {}).get("qty")
        if not qty or qty <= 0:
            raise ValueError("order qty must be a positive integer")
        return int(qty)

    def await_fill_or_cancel(
        self, order_id: str, timeout: float = FILL_TIMEOUT_S, poll: float = 1.0,
        retry_market: bool = False, symbol: str = "", qty: int = 0, side: str = "buy",
    ) -> OrderResult:
        """Poll until filled or ``timeout``; cancel unfilled, optionally market-retry."""
        waited = 0.0
        while waited < timeout:
            order = self.client._with_retry(self.client.trading.get_order_by_id, order_id)
            if str(getattr(order, "status", "")).lower() in ("filled", "closed"):
                return self._normalize(order)
            self._sleep(poll)
            waited += poll
        self.cancel_order(order_id)
        logger.warning("order %s unfilled after %.0fs — cancelled", order_id, timeout)
        if retry_market and symbol and qty:
            tid = uuid.uuid4().hex
            o = self._submit(self._market_req(symbol, qty, side, tid))
            return self._normalize(o, tid, "market")
        return OrderResult(order_id, symbol, side, qty, "cancelled")

    # --------------------------------------------------------- stop control
    def modify_stop(self, symbol: str, new_stop: float) -> bool:
        """Tighten a stop only — never widen. Returns True if the stop moved."""
        current = self._stops.get(symbol)
        # LONG stops only rise; reject anything that loosens protection.
        if current is not None and new_stop <= current:
            logger.info("modify_stop %s ignored: %.2f would not tighten %.2f",
                        symbol, new_stop, current)
            return False
        order_id = self._stop_orders.get(symbol)
        if order_id:
            self.client._with_retry(
                self.client.trading.replace_order_by_id, order_id,
                {"stop_price": round(new_stop, 2)})
        self._stops[symbol] = new_stop
        logger.info("tightened stop %s -> %.2f", symbol, new_stop)
        return True

    # --------------------------------------------------------- cancellation
    def cancel_order(self, order_id: str) -> None:
        """Cancel a single order by id."""
        self.client._with_retry(self.client.trading.cancel_order_by_id, order_id)

    def cancel(self, order_id: str) -> None:
        """Alias for :meth:`cancel_order` (legacy)."""
        self.cancel_order(order_id)

    def cancel_all(self) -> None:
        """Cancel every open order."""
        self.client._with_retry(self.client.trading.cancel_orders)

    def close_position(self, symbol: str) -> None:
        """Liquidate a single position."""
        self.client._with_retry(self.client.trading.close_position, symbol)
        self._stops.pop(symbol, None)
        self._stop_orders.pop(symbol, None)

    def close_all_positions(self) -> None:
        """Liquidate everything and cancel open orders."""
        self.client._with_retry(self.client.trading.close_all_positions, cancel_orders=True)
        self._stops.clear()
        self._stop_orders.clear()

    # -------------------------------------------------------- legacy direct
    def submit_market(self, symbol: str, qty: int, side: str) -> OrderResult:
        """Submit a DAY market order."""
        if qty <= 0:
            raise ValueError("qty must be positive")
        tid = uuid.uuid4().hex
        order = self._submit(self._market_req(symbol, qty, side, tid))
        return self._normalize(order, tid, "market")

    def submit_limit(self, symbol: str, qty: int, side: str, limit_price: float) -> OrderResult:
        """Submit a DAY limit order."""
        if qty <= 0:
            raise ValueError("qty must be positive")
        tid = uuid.uuid4().hex
        order = self._submit(self._limit_req(symbol, qty, side, limit_price, tid))
        return self._normalize(order, tid, "limit")
