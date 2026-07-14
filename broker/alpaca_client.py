"""Alpaca API wrapper.

Thin adapter over alpaca-py trading + data clients. Reads credentials from env
(.env) and paper/live flag from config. Everything downstream (executor,
tracker) talks to this class, not to alpaca-py directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

try:
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient
except ImportError:  # keep import-time safe before deps installed
    TradingClient = None
    StockHistoricalDataClient = None


@dataclass
class Account:
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float


class AlpacaClient:
    def __init__(self, cfg: dict):
        load_dotenv()
        self.api_key = os.environ.get("ALPACA_API_KEY", "")
        self.secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
        env_paper = os.environ.get("ALPACA_PAPER", "true").lower() == "true"
        self.paper = cfg["broker"]["paper_trading"] and env_paper

        if not self.api_key or not self.secret_key:
            raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set")
        if TradingClient is None:
            raise ImportError("alpaca-py not installed")

        self.trading = TradingClient(
            self.api_key, self.secret_key, paper=self.paper
        )
        self.data = StockHistoricalDataClient(self.api_key, self.secret_key)

    def get_account(self) -> Account:
        a = self.trading.get_account()
        return Account(
            equity=float(a.equity),
            cash=float(a.cash),
            buying_power=float(a.buying_power),
            portfolio_value=float(a.portfolio_value),
        )

    def get_positions(self):
        return self.trading.get_all_positions()

    def is_market_open(self) -> bool:
        return bool(self.trading.get_clock().is_open)
