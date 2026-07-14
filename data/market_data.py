"""Real-time and historical market data fetching.

Historical bars come from Alpaca's data client; the same interface serves the
backtester (bulk history) and the live loop (latest bar).

Stub scaffold: signatures, type hints, and docstrings only — no logic yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from broker.alpaca_client import AlpacaClient


class MarketData:
    """Historical and latest-bar market data access."""

    def __init__(self, cfg: dict, client: "AlpacaClient") -> None:
        """Store config and broker client; resolve timeframe."""
        ...

    def history(self, symbol: str, lookback_days: int = 500) -> pd.DataFrame:
        """OHLCV history for a symbol, oldest-first, tz-naive index."""
        ...

    def latest_bar(self, symbol: str) -> pd.Series:
        """Most recent OHLCV bar for a symbol."""
        ...
