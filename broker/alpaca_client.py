"""Alpaca API wrapper.

Thin adapter over the alpaca-py trading + data clients. Reads credentials from
the environment (.env, never hardcoded) and the paper/live flag from config.
Downstream code talks to this class, not to alpaca-py directly.

SECURITY: API keys are loaded from ``ALPACA_API_KEY`` / ``ALPACA_SECRET_KEY`` in
the environment. ``.env`` is gitignored and must never be committed. Secrets are
never logged. Live trading requires an explicit typed confirmation.

The alpaca-py SDK is imported lazily and guarded, so this module imports cleanly
even when the SDK is not installed; the real clients are only needed to connect.
Both clients can be injected for offline testing.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, Optional

from .position_tracker import PortfolioSnapshot, PositionSnapshot

logger = logging.getLogger("regime-trader.broker")

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"
LIVE_CONFIRM_PHRASE = "YES I UNDERSTAND THE RISKS"

try:  # keep import-time safe before the SDK is installed
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.data.historical import StockHistoricalDataClient
    _SDK = True
except ImportError:  # pragma: no cover - exercised only without the SDK
    TradingClient = None
    StockHistoricalDataClient = None
    GetOrdersRequest = None
    QueryOrderStatus = None
    _SDK = False


class AlpacaClient:
    """Adapter over the Alpaca trading + data clients."""

    def __init__(
        self,
        cfg: dict,
        trading_client=None,
        data_client=None,
        confirm_fn: Callable[[str], str] = input,
        env: Optional[dict] = None,
        max_retries: int = 5,
        connect: bool = True,
    ) -> None:
        """Resolve paper flag, load credentials from env, gate live trading.

        ``trading_client``/``data_client`` may be injected (tests / offline).
        Set ``connect=False`` to skip the startup health check.
        """
        self.cfg = cfg
        env = os.environ if env is None else env
        self.paper: bool = bool(cfg.get("broker", {}).get("paper_trading", True))
        self.base_url = PAPER_URL if self.paper else LIVE_URL
        self.max_retries = max_retries

        if not self.paper:
            self._confirm_live(confirm_fn)

        self._api_key = env.get("ALPACA_API_KEY")
        self._secret_key = env.get("ALPACA_SECRET_KEY")

        self.trading = trading_client
        self.data = data_client
        if self.trading is None or self.data is None:
            self._build_clients()

        if connect:
            self.health_check()

    # ------------------------------------------------------------- setup
    def _confirm_live(self, confirm_fn: Callable[[str], str]) -> None:
        resp = confirm_fn(f"LIVE TRADING MODE. Type '{LIVE_CONFIRM_PHRASE}' to confirm: ")
        if (resp or "").strip() != LIVE_CONFIRM_PHRASE:
            raise RuntimeError("live trading not confirmed — aborting")
        logger.warning("LIVE trading confirmed by operator")

    def _build_clients(self) -> None:
        if not _SDK:
            raise RuntimeError(
                "alpaca-py not installed; inject trading_client/data_client or "
                "`pip install alpaca-py`"
            )
        if not self._api_key or not self._secret_key:
            raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set in environment")
        if self.trading is None:
            self.trading = TradingClient(self._api_key, self._secret_key, paper=self.paper)
        if self.data is None:
            self.data = StockHistoricalDataClient(self._api_key, self._secret_key)

    # ---------------------------------------------------- retry / reconnect
    def _with_retry(self, fn: Callable, *args, **kwargs):
        """Call ``fn`` with exponential backoff on transient failures."""
        delay = 1.0
        last: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - broker/network errors vary
                last = exc
                logger.warning("broker call failed (%d/%d): %s", attempt, self.max_retries, exc)
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay = min(delay * 2, 30.0)
        raise ConnectionError(f"broker unreachable after {self.max_retries} attempts") from last

    def health_check(self) -> bool:
        """Verify connectivity by fetching the market clock."""
        self._with_retry(self.trading.get_clock)
        logger.info("broker health check OK (%s)", "paper" if self.paper else "LIVE")
        return True

    # ------------------------------------------------------------ account
    def get_account(self) -> PortfolioSnapshot:
        """Fetch account equity, cash, and buying power."""
        a = self._with_retry(self.trading.get_account)
        return PortfolioSnapshot(
            equity=float(a.equity),
            cash=float(a.cash),
            buying_power=float(a.buying_power),
            portfolio_value=float(getattr(a, "portfolio_value", a.equity)),
            positions={p.symbol: p for p in self.get_positions()},
        )

    def get_positions(self) -> list[PositionSnapshot]:
        """Fetch all open positions."""
        raw = self._with_retry(self.trading.get_all_positions)
        return [
            PositionSnapshot(
                symbol=p.symbol,
                qty=float(p.qty),
                avg_entry=float(p.avg_entry_price),
                market_price=float(p.current_price),
                market_value=float(p.market_value),
                unrealized_pl=float(p.unrealized_pl),
                unrealized_plpc=float(p.unrealized_plpc),
            )
            for p in raw
        ]

    def get_order_history(self, limit: int = 100, status: str = "all") -> list:
        """Recent orders (raw SDK objects), newest first."""
        if GetOrdersRequest is None:
            return self._with_retry(self.trading.get_orders)
        qstatus = {"open": QueryOrderStatus.OPEN, "closed": QueryOrderStatus.CLOSED}.get(
            status, QueryOrderStatus.ALL)
        req = GetOrdersRequest(status=qstatus, limit=limit)
        return self._with_retry(self.trading.get_orders, filter=req)

    # -------------------------------------------------------------- clock
    def get_clock(self):
        """Raw market clock (is_open, next_open, next_close)."""
        return self._with_retry(self.trading.get_clock)

    def is_market_open(self) -> bool:
        """True if the market is currently open."""
        return bool(self.get_clock().is_open)

    def get_available_margin(self) -> float:
        """Buying power available for new positions."""
        return float(self._with_retry(self.trading.get_account).buying_power)
