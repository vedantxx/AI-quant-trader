"""Order placement, modification, and cancellation via Alpaca.

Stub scaffold: signatures, type hints, and docstrings only — no logic yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .alpaca_client import AlpacaClient


@dataclass
class OrderResult:
    """Normalized result of a submitted order."""

    id: str
    symbol: str
    side: str
    qty: float
    status: str


class OrderExecutor:
    """Submit, modify, and cancel orders through the Alpaca client."""

    def __init__(self, client: "AlpacaClient") -> None:
        """Store the broker client."""
        ...

    def submit_market(self, symbol: str, qty: int, side: str) -> OrderResult:
        """Submit a DAY market order."""
        ...

    def submit_limit(
        self, symbol: str, qty: int, side: str, limit_price: float
    ) -> OrderResult:
        """Submit a DAY limit order."""
        ...

    def cancel(self, order_id: str) -> None:
        """Cancel a single order by id."""
        ...

    def cancel_all(self) -> None:
        """Cancel every open order."""
        ...
