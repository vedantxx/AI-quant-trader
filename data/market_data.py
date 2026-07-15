"""Real-time and historical market data fetching.

Historical bars come from Alpaca's data client; the same interface serves the
backtester (bulk history) and the live loop (latest bar/quote/snapshot). Bars are
normalized to a tz-naive, de-duplicated, oldest-first OHLCV frame. Gaps from
weekends, holidays, and halts are left as missing dates rather than fabricated
(no forward-fill) so downstream causal features never see invented prices.

alpaca-py imports are guarded; the client is injected so this module works
offline and under test.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Callable, Optional

import pandas as pd

if TYPE_CHECKING:
    from broker.alpaca_client import AlpacaClient

logger = logging.getLogger("regime-trader.data")

try:  # guarded SDK imports
    from alpaca.data.requests import (
        StockBarsRequest,
        StockLatestQuoteRequest,
        StockSnapshotRequest,
    )
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    _SDK = True
except ImportError:  # pragma: no cover
    _SDK = False

_OHLCV = ["open", "high", "low", "close", "volume"]


class MarketData:
    """Historical and latest-bar market data access."""

    def __init__(self, cfg: dict, client: "AlpacaClient") -> None:
        self.cfg = cfg
        self.client = client
        self.timeframe_str: str = cfg.get("broker", {}).get("timeframe", "1Day")

    # ----------------------------------------------------------- timeframe
    def _timeframe(self, tf: Optional[str] = None):
        """Map a config timeframe string to an alpaca TimeFrame (or the string)."""
        tf = tf or self.timeframe_str
        if not _SDK:
            return tf
        table = {
            "1Min": TimeFrame(1, TimeFrameUnit.Minute),
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
            "1Day": TimeFrame(1, TimeFrameUnit.Day),
        }
        return table.get(tf, TimeFrame(1, TimeFrameUnit.Day))

    # --------------------------------------------------------- normalization
    @staticmethod
    def _clean_bars(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
        """Sort, de-dup, tz-naive, OHLCV-only. Leaves calendar gaps untouched."""
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=_OHLCV)
        out = df.copy()
        # alpaca returns a (symbol, timestamp) MultiIndex; flatten to timestamps
        if isinstance(out.index, pd.MultiIndex):
            if symbol and symbol in out.index.get_level_values(0):
                out = out.xs(symbol, level=0)
            else:
                out = out.droplevel(0)
        out.columns = [c.lower() for c in out.columns]
        out = out[[c for c in _OHLCV if c in out.columns]]
        idx = pd.to_datetime(out.index)
        out.index = idx.tz_localize(None) if idx.tz is not None else idx
        out = out[~out.index.duplicated(keep="last")].sort_index()
        return out

    # ---------------------------------------------------------- historical
    def get_historical_bars(
        self, symbol: str, timeframe: Optional[str] = None,
        start: Optional[datetime] = None, end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """OHLCV bars for ``symbol`` over [start, end], normalized."""
        if _SDK:
            req = StockBarsRequest(
                symbol_or_symbols=symbol, timeframe=self._timeframe(timeframe),
                start=start, end=end)
            bars = self.client._with_retry(self.client.data.get_stock_bars, req)
        else:
            bars = self.client._with_retry(
                self.client.data.get_stock_bars, symbol, timeframe, start, end)
        df = getattr(bars, "df", bars)
        return self._clean_bars(df, symbol)

    # free/IEX data feed rejects requests that include the most recent ~15 min
    RECENT_DATA_CUSHION = timedelta(minutes=16)

    def history(self, symbol: str, lookback_days: int = 500) -> pd.DataFrame:
        """OHLCV history for a symbol, oldest-first, tz-naive index."""
        end = datetime.now(timezone.utc) - self.RECENT_DATA_CUSHION
        start = end - timedelta(days=lookback_days)
        return self.get_historical_bars(symbol, self.timeframe_str, start, end)

    # -------------------------------------------------------------- latest
    def get_latest_bar(self, symbol: str) -> pd.Series:
        """Most recent OHLCV bar for a symbol."""
        df = self.history(symbol, lookback_days=5)
        if len(df) == 0:
            return pd.Series(dtype=float)
        return df.iloc[-1]

    def latest_bar(self, symbol: str) -> pd.Series:
        """Most recent OHLCV bar (legacy alias)."""
        return self.get_latest_bar(symbol)

    def get_latest_quote(self, symbol: str) -> dict:
        """Latest bid/ask and spread fraction for a spread check."""
        if _SDK:
            req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            q = self.client._with_retry(self.client.data.get_stock_latest_quote, req)
            q = q[symbol] if isinstance(q, dict) else q
        else:
            q = self.client._with_retry(self.client.data.get_stock_latest_quote, symbol)
        bid, ask = float(getattr(q, "bid_price", 0) or 0), float(getattr(q, "ask_price", 0) or 0)
        mid = (bid + ask) / 2 if (bid and ask) else 0.0
        return {"bid": bid, "ask": ask, "mid": mid,
                "spread_pct": (ask - bid) / mid if mid else float("inf")}

    def get_snapshot(self, symbol: str):
        """Full snapshot (latest trade/quote/bar) for a symbol."""
        if _SDK:
            req = StockSnapshotRequest(symbol_or_symbols=symbol)
            return self.client._with_retry(self.client.data.get_stock_snapshot, req)
        return self.client._with_retry(self.client.data.get_stock_snapshot, symbol)

    # ------------------------------------------------------------ streaming
    def subscribe_bars(
        self, symbols: list[str], callback: Callable, data_stream=None,
    ) -> None:
        """Subscribe to live bar updates via WebSocket (needs a data stream)."""
        if data_stream is None:
            logger.info("no data stream provided; bar subscription not started")
            return
        data_stream.subscribe_bars(callback, *symbols)
        data_stream.run()

    def subscribe_quotes(
        self, symbols: list[str], callback: Callable, data_stream=None,
    ) -> None:
        """Subscribe to live quote updates (spread checks) via WebSocket."""
        if data_stream is None:
            logger.info("no data stream provided; quote subscription not started")
            return
        data_stream.subscribe_quotes(callback, *symbols)
        data_stream.run()
