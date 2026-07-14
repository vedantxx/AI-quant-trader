"""Alpaca API wrapper.

Thin adapter over alpaca-py trading + data clients. Reads credentials from env
(.env) and the paper/live flag from config. Downstream code talks to this class,
not to alpaca-py directly.

Stub scaffold: signatures, type hints, and docstrings only — no logic yet.
"""

from __future__ import annotations

from .position_tracker import PortfolioSnapshot, PositionSnapshot


class AlpacaClient:
    """Adapter over the Alpaca trading + data clients."""

    def __init__(self, cfg: dict) -> None:
        """Load credentials from env; resolve paper flag from env + config."""
        ...

    def get_account(self) -> PortfolioSnapshot:
        """Fetch account equity, cash, and buying power."""
        ...

    def get_positions(self) -> list[PositionSnapshot]:
        """Fetch all open positions."""
        ...

    def is_market_open(self) -> bool:
        """True if the market is currently open."""
        ...
