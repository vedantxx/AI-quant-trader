"""Track open positions, portfolio state, and P&L.

Maintains a per-position view (entry, current price, unrealized P&L, stop level,
holding period, regime at entry vs now) and updates the shared ``PortfolioState``
and ``CircuitBreaker`` on every fill — so drawdown breakers react the instant a
fill lands, not on the next polling cycle. Reconciles tracked positions against
Alpaca on startup. The trade-update WebSocket import is guarded for offline use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # break broker import cycle
    from core.risk_manager import CircuitBreaker, PortfolioState

    from .alpaca_client import AlpacaClient

logger = logging.getLogger("regime-trader.positions")


@dataclass
class PositionSnapshot:
    """Point-in-time view of a single open position."""

    symbol: str
    qty: float
    avg_entry: float
    market_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float


@dataclass
class PortfolioSnapshot:
    """Point-in-time view of the whole account."""

    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    positions: dict[str, PositionSnapshot] = field(default_factory=dict)


@dataclass
class TrackedPosition:
    """Live per-position tracking beyond a point-in-time snapshot."""

    symbol: str
    qty: float
    entry_price: float
    entry_time: datetime
    stop_level: float = 0.0
    current_price: float = 0.0
    regime_at_entry: str = ""
    regime_current: str = ""

    @property
    def unrealized_pl(self) -> float:
        return (self.current_price - self.entry_price) * self.qty

    def holding_period(self, now: Optional[datetime] = None) -> float:
        """Holding period in days."""
        now = now or datetime.now(timezone.utc)
        return (now - self.entry_time).total_seconds() / 86400.0


class PositionTracker:
    """Track open positions and P&L via the Alpaca client."""

    def __init__(
        self,
        client: "AlpacaClient",
        circuit_breaker: Optional["CircuitBreaker"] = None,
        portfolio_state: Optional["PortfolioState"] = None,
    ) -> None:
        self.client = client
        self.breaker = circuit_breaker
        self.state = portfolio_state
        self.tracked: dict[str, TrackedPosition] = {}

    # ------------------------------------------------------------- snapshots
    def snapshot(self) -> dict[str, PositionSnapshot]:
        """Current open positions keyed by symbol."""
        return {p.symbol: p for p in self.client.get_positions()}

    def portfolio(self) -> PortfolioSnapshot:
        """Full account snapshot including positions."""
        return self.client.get_account()

    def current_weights(self, equity: float) -> dict[str, float]:
        """Per-symbol market value as a fraction of equity."""
        if equity <= 0:
            return {}
        return {s: p.market_value / equity for s, p in self.snapshot().items()}

    def total_unrealized(self) -> float:
        """Sum of unrealized P&L across open positions."""
        return sum(p.unrealized_pl for p in self.snapshot().values())

    # ------------------------------------------------------------- reconcile
    def reconcile(self, now: Optional[datetime] = None) -> dict[str, TrackedPosition]:
        """Rebuild tracked positions from Alpaca (source of truth) on startup."""
        now = now or datetime.now(timezone.utc)
        actual = self.snapshot()
        # drop tracked positions the broker no longer reports
        for sym in list(self.tracked):
            if sym not in actual:
                logger.warning("reconcile: dropping stale tracked position %s", sym)
                del self.tracked[sym]
        # add/refresh from broker
        for sym, p in actual.items():
            t = self.tracked.get(sym)
            if t is None:
                self.tracked[sym] = TrackedPosition(
                    symbol=sym, qty=p.qty, entry_price=p.avg_entry, entry_time=now,
                    current_price=p.market_price)
            else:
                t.qty, t.current_price = p.qty, p.market_price
        return self.tracked

    # ---------------------------------------------------------------- fills
    def on_fill(
        self, symbol: str, side: str, filled_qty: float, fill_price: float,
        stop_level: float = 0.0, regime: str = "", now: Optional[datetime] = None,
    ) -> None:
        """Apply a fill: update tracking, then refresh state + circuit breaker."""
        now = now or datetime.now(timezone.utc)
        signed = filled_qty if side == "buy" else -filled_qty
        t = self.tracked.get(symbol)
        if t is None:
            self.tracked[symbol] = TrackedPosition(
                symbol=symbol, qty=signed, entry_price=fill_price, entry_time=now,
                stop_level=stop_level, current_price=fill_price, regime_at_entry=regime,
                regime_current=regime)
        else:
            t.qty += signed
            t.current_price = fill_price
            if stop_level:
                t.stop_level = stop_level
            if t.qty == 0:
                del self.tracked[symbol]
        logger.info("fill %s %s x%s @%.2f", side, symbol, filled_qty, fill_price)
        self._refresh_risk()

    def handle_trade_update(self, data) -> None:
        """WebSocket trade_update callback -> :meth:`on_fill`."""
        event = str(getattr(data, "event", "")).lower()
        if event not in ("fill", "partial_fill"):
            return
        order = getattr(data, "order", None)
        if order is None:
            return
        self.on_fill(
            symbol=getattr(order, "symbol", ""),
            side=str(getattr(order, "side", "")).lower().replace("orderside.", ""),
            filled_qty=float(getattr(order, "filled_qty", 0) or 0),
            fill_price=float(getattr(order, "filled_avg_price", 0) or 0),
        )

    def _refresh_risk(self) -> None:
        """Mark equity to market and let the circuit breaker react to the fill."""
        if self.state is None or self.breaker is None:
            return
        try:
            acct = self.portfolio()
        except Exception as exc:  # noqa: BLE001 - broker call may fail
            logger.warning("risk refresh skipped: %s", exc)
            return
        self.breaker.update(self.state, acct.equity)

    # ------------------------------------------------------------ websocket
    def subscribe_fills(self, trading_stream=None) -> None:
        """Subscribe to Alpaca trade updates (fills) via WebSocket.

        Pass an alpaca-py ``TradingStream`` (or a compatible object exposing
        ``subscribe_trade_updates``/``run``). Without the SDK this is a no-op
        guard; production wiring supplies the stream.
        """
        if trading_stream is None:
            logger.info("no trading stream provided; fill WebSocket not started")
            return
        trading_stream.subscribe_trade_updates(self._async_handler)
        trading_stream.run()

    async def _async_handler(self, data) -> None:  # pragma: no cover - async glue
        self.handle_trade_update(data)
