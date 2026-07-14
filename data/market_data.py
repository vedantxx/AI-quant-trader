"""Real-time and historical market data fetching.

Historical bars come from Alpaca's data client; the same interface serves the
backtester (bulk history) and the live loop (latest bar).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

try:
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
except ImportError:
    StockBarsRequest = None
    TimeFrame = TimeFrameUnit = None

from broker.alpaca_client import AlpacaClient

_TF_MAP = {
    "1Min": ("Minute", 1),
    "5Min": ("Minute", 5),
    "15Min": ("Minute", 15),
    "1Hour": ("Hour", 1),
    "1Day": ("Day", 1),
}


class MarketData:
    def __init__(self, cfg: dict, client: AlpacaClient):
        self.cfg = cfg
        self.client = client
        self.timeframe = cfg["broker"]["timeframe"]

    def _timeframe(self):
        unit_name, amount = _TF_MAP.get(self.timeframe, ("Day", 1))
        unit = getattr(TimeFrameUnit, unit_name)
        return TimeFrame(amount, unit)

    def history(
        self, symbol: str, lookback_days: int = 500
    ) -> pd.DataFrame:
        """OHLCV history for a symbol, oldest-first, tz-naive index."""
        start = datetime.utcnow() - timedelta(days=lookback_days)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=self._timeframe(),
            start=start,
        )
        bars = self.client.data.get_stock_bars(req).df
        if bars.empty:
            return bars
        df = bars.xs(symbol, level="symbol") if "symbol" in bars.index.names else bars
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df.sort_index()

    def latest_bar(self, symbol: str) -> pd.Series:
        df = self.history(symbol, lookback_days=10)
        if df.empty:
            raise RuntimeError(f"no data for {symbol}")
        return df.iloc[-1]
