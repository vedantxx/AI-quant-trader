"""Risk manager: position sizing, exposure/leverage caps, drawdown breakers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class RiskState:
    equity: float
    peak_equity: float
    day_start_equity: float
    week_start_equity: float
    trades_today: int = 0
    trade_date: date | None = None
    halted: bool = False
    halt_reason: str = ""
    size_scale: float = 1.0          # global size multiplier (dd de-risking)


class RiskManager:
    def __init__(self, cfg: dict):
        r = cfg["risk"]
        self.max_risk_per_trade = r["max_risk_per_trade"]
        self.max_exposure = r["max_exposure"]
        self.max_leverage = r["max_leverage"]
        self.max_single_position = r["max_single_position"]
        self.max_concurrent = r["max_concurrent"]
        self.max_daily_trades = r["max_daily_trades"]
        self.daily_dd_reduce = r["daily_dd_reduce"]
        self.daily_dd_halt = r["daily_dd_halt"]
        self.weekly_dd_reduce = r["weekly_dd_reduce"]
        self.weekly_dd_halt = r["weekly_dd_halt"]
        self.max_dd_from_peak = r["max_dd_from_peak"]

    # ------------------------------------------------------------ sizing
    def position_size(
        self, state: RiskState, price: float, stop_distance: float
    ) -> int:
        """Shares to trade risking ``max_risk_per_trade`` of equity to the stop."""
        if stop_distance <= 0 or price <= 0:
            return 0
        risk_dollars = state.equity * self.max_risk_per_trade * state.size_scale
        shares = int(risk_dollars / stop_distance)
        # cap by single-position weight
        max_shares_by_weight = int(
            state.equity * self.max_single_position / price
        )
        return max(0, min(shares, max_shares_by_weight))

    def can_open(self, state: RiskState, open_positions: int) -> tuple[bool, str]:
        if state.halted:
            return False, state.halt_reason
        if open_positions >= self.max_concurrent:
            return False, "max_concurrent reached"
        if state.trades_today >= self.max_daily_trades:
            return False, "max_daily_trades reached"
        return True, ""

    def check_exposure(self, gross_exposure: float, leverage: float) -> bool:
        return (
            gross_exposure <= self.max_exposure + 1e-9
            and leverage <= self.max_leverage + 1e-9
        )

    # --------------------------------------------------- drawdown breakers
    def update_drawdowns(self, state: RiskState) -> RiskState:
        """Recompute size scaling / halt flags from current equity."""
        state.peak_equity = max(state.peak_equity, state.equity)

        daily_dd = 1 - state.equity / state.day_start_equity
        weekly_dd = 1 - state.equity / state.week_start_equity
        peak_dd = 1 - state.equity / state.peak_equity

        state.halted, state.halt_reason, state.size_scale = False, "", 1.0

        if peak_dd >= self.max_dd_from_peak:
            state.halted = True
            state.halt_reason = f"peak drawdown {peak_dd:.1%} >= kill switch"
        elif daily_dd >= self.daily_dd_halt:
            state.halted = True
            state.halt_reason = f"daily drawdown {daily_dd:.1%} halt"
        elif weekly_dd >= self.weekly_dd_halt:
            state.halted = True
            state.halt_reason = f"weekly drawdown {weekly_dd:.1%} halt"
        elif daily_dd >= self.daily_dd_reduce or weekly_dd >= self.weekly_dd_reduce:
            state.size_scale = 0.5

        return state

    def roll_day(self, state: RiskState, today: date) -> RiskState:
        """Reset per-day counters when the trading date advances."""
        if state.trade_date != today:
            state.trade_date = today
            state.trades_today = 0
            state.day_start_equity = state.equity
        return state
