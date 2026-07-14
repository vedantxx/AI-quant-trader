"""Track open positions, portfolio state, and P&L.

Stub scaffold: signatures, type hints, and docstrings only — no logic yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # break broker import cycle
    from .alpaca_client import AlpacaClient


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


class PositionTracker:
    """Track open positions and P&L via the Alpaca client."""

    def __init__(self, client: "AlpacaClient") -> None:
        """Store the broker client."""
        ...

    def snapshot(self) -> dict[str, PositionSnapshot]:
        """Current open positions keyed by symbol."""
        ...

    def portfolio(self) -> PortfolioSnapshot:
        """Full account snapshot including positions."""
        ...

    def current_weights(self, equity: float) -> dict[str, float]:
        """Per-symbol market value as a fraction of equity."""
        ...

    def total_unrealized(self) -> float:
        """Sum of unrealized P&L across open positions."""
        ...
