"""Risk manager: position sizing, exposure/leverage caps, drawdown breakers.

Stub scaffold: signatures, type hints, and docstrings only — no logic yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class RiskState:
    """Mutable risk/equity state carried across the trading session."""

    equity: float
    peak_equity: float
    day_start_equity: float
    week_start_equity: float
    trades_today: int = 0
    trade_date: Optional[date] = None
    halted: bool = False
    halt_reason: str = ""
    size_scale: float = 1.0


@dataclass
class RiskCheck:
    """Result of a risk gate: whether an action is approved and how sized."""

    approved: bool
    reason: str
    size_scale: float = 1.0
    shares: int = 0


class RiskManager:
    """Position sizing, exposure/leverage caps, drawdown circuit breakers."""

    def __init__(self, cfg: dict) -> None:
        """Read risk.* config."""
        ...

    def position_size(
        self, state: RiskState, price: float, stop_distance: float
    ) -> int:
        """Shares risking ``max_risk_per_trade`` to the stop, capped by weight."""
        ...

    def can_open(self, state: RiskState, open_positions: int) -> RiskCheck:
        """Gate a new position on halt, concurrency, and daily-trade caps."""
        ...

    def check_exposure(self, gross_exposure: float, leverage: float) -> bool:
        """True if gross exposure and leverage are within caps."""
        ...

    def update_drawdowns(self, state: RiskState) -> RiskState:
        """Recompute size scaling / halt flags from current equity."""
        ...

    def roll_day(self, state: RiskState, today: date) -> RiskState:
        """Reset per-day counters when the trading date advances."""
        ...
