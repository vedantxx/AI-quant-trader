"""Risk manager: position sizing, exposure/leverage caps, drawdown breakers.

The risk manager operates INDEPENDENTLY of the HMM. Even if the HMM is completely
wrong, the circuit breakers fire on ACTUAL P&L — defense in depth. The risk
manager holds ABSOLUTE VETO POWER over every signal: it can reject an order or
shrink it, never enlarge it.

Layers (all thresholds from ``risk.*`` in settings.yaml):
  1. Portfolio limits  — exposure, single-position, sector, concurrency, trades.
  2. Circuit breakers  — daily/weekly/peak drawdown, on realized equity.
  3. Position risk     — mandatory stop, 1% risk-to-stop sizing, overnight gap.
  4. Leverage rules    — 1.0x default; 1.25x only in calm, certain conditions.
  5. Order validation  — buying power, spread, duplicate suppression.
  6. Correlation       — reduce/reject on high correlation to open positions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .regime_strategies import Signal

logger = logging.getLogger("regime-trader.risk")


# ============================================================ legacy state API
@dataclass
class RiskState:
    """Mutable risk/equity state carried across the trading session (legacy)."""

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
    """Result of a legacy risk gate: whether an action is approved and how sized."""

    approved: bool
    reason: str
    size_scale: float = 1.0
    shares: int = 0


# ============================================================ Phase 5 state API
@dataclass
class Position:
    """An open position tracked by the portfolio."""

    symbol: str
    weight: float                 # gross weight as fraction of equity
    entry_price: float
    stop_loss: float
    direction: str = "LONG"
    sector: str = "UNKNOWN"


@dataclass
class PortfolioState:
    """Everything the risk manager needs to decide on a signal."""

    equity: float
    cash: float
    buying_power: float
    positions: dict[str, Position] = field(default_factory=dict)
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    peak_equity: float = 0.0
    day_start_equity: float = 0.0
    week_start_equity: float = 0.0
    trades_today: int = 0
    flicker_rate: int = 0
    regime_uncertain: bool = False
    hmm_regime: str = ""          # regime label at decision time (for wrong-HMM logs)
    price_history: dict = field(default_factory=dict)   # symbol -> price/return Series
    sector_map: dict = field(default_factory=dict)      # symbol -> sector
    recent_orders: dict = field(default_factory=dict)   # (symbol, dir) -> datetime
    circuit_breaker_status: str = "none"

    @property
    def drawdown(self) -> float:
        """Peak-to-current drawdown (negative fraction)."""
        return (self.equity / self.peak_equity - 1) if self.peak_equity else 0.0

    def gross_exposure(self) -> float:
        return sum(p.weight for p in self.positions.values())

    def sector_exposure(self, sector: str) -> float:
        return sum(p.weight for p in self.positions.values() if p.sector == sector)


@dataclass
class RiskDecision:
    """Verdict on a signal. ``modified_signal`` is the (possibly shrunk) order."""

    approved: bool
    modified_signal: Optional["Signal"] = None
    rejection_reason: str = ""
    modifications: list[str] = field(default_factory=list)


@dataclass
class BreakerEvent:
    """A single circuit-breaker trigger, logged for auditing."""

    breaker_type: str
    drawdown: float
    equity: float
    positions_closed: int
    hmm_regime: str
    action: str
    timestamp: str


@dataclass
class BreakerResult:
    """Outcome of evaluating the breakers against a portfolio state."""

    size_scale: float = 1.0
    halt: bool = False
    close_all: bool = False
    active: str = "none"
    reason: str = ""


# ================================================================ circuit breaker
class CircuitBreaker:
    """Drawdown circuit breakers, firing on realized equity — regime-independent."""

    def __init__(self, cfg: dict, lock_dir: Optional[Path] = None) -> None:
        r = cfg["risk"]
        self.daily_reduce = float(r["daily_dd_reduce"])
        self.daily_halt = float(r["daily_dd_halt"])
        self.weekly_reduce = float(r["weekly_dd_reduce"])
        self.weekly_halt = float(r["weekly_dd_halt"])
        self.peak_halt = float(r["max_dd_from_peak"])
        self.reduce_factor = float(r.get("size_reduce_factor", 0.50))
        self.lock_path = (lock_dir or Path.cwd()) / r.get("halt_lock_file", "trading_halted.lock")
        self._history: list[BreakerEvent] = []

    # ---- drawdown helpers (positive fraction, e.g. 0.03 == 3% down)
    @staticmethod
    def _dd(now: float, start: float) -> float:
        return max(0.0, 1 - now / start) if start > 0 else 0.0

    def evaluate(self, state: PortfolioState) -> BreakerResult:
        """Assess all breakers; most-severe action wins."""
        daily = self._dd(state.equity, state.day_start_equity)
        weekly = self._dd(state.equity, state.week_start_equity)
        peak = self._dd(state.equity, state.peak_equity)

        res = BreakerResult()
        # peak kill switch — hard halt + lock file (manual reset required)
        if peak >= self.peak_halt:
            self._trip("peak_dd", peak, state, "halt+close+lock")
            self._write_lock(peak, state)
            return BreakerResult(0.0, True, True, "peak_dd",
                                 f"peak drawdown {peak:.2%} >= {self.peak_halt:.2%}")
        if self.lock_path.exists():
            return BreakerResult(0.0, True, True, "locked", "trading_halted.lock present")

        # daily / weekly halts
        if daily >= self.daily_halt:
            self._trip("daily_dd_halt", daily, state, "halt+close")
            return BreakerResult(0.0, True, True, "daily_dd_halt",
                                 f"daily drawdown {daily:.2%} >= {self.daily_halt:.2%}")
        if weekly >= self.weekly_halt:
            self._trip("weekly_dd_halt", weekly, state, "halt+close")
            return BreakerResult(0.0, True, True, "weekly_dd_halt",
                                 f"weekly drawdown {weekly:.2%} >= {self.weekly_halt:.2%}")

        # reduce (halve size) — take the more severe if both fire
        if daily >= self.daily_reduce or weekly >= self.weekly_reduce:
            which = "daily_dd_reduce" if daily >= self.daily_reduce else "weekly_dd_reduce"
            self._trip(which, max(daily, weekly), state, f"reduce x{self.reduce_factor}")
            res = BreakerResult(self.reduce_factor, False, False, which,
                                f"drawdown reduce active ({which})")
        return res

    def check(self, state: PortfolioState) -> BreakerResult:
        """Alias for :meth:`evaluate` (spec API)."""
        return self.evaluate(state)

    def update(self, state: PortfolioState, current_equity: float) -> PortfolioState:
        """Mark equity to market and lift the running peak."""
        state.equity = current_equity
        state.peak_equity = max(state.peak_equity, current_equity)
        state.circuit_breaker_status = self.evaluate(state).active
        return state

    def reset_daily(self, state: PortfolioState) -> None:
        """Roll the daily baseline at a new session open."""
        state.day_start_equity = state.equity
        state.daily_pnl = 0.0
        state.trades_today = 0

    def reset_weekly(self, state: PortfolioState) -> None:
        """Roll the weekly baseline at a new week."""
        state.week_start_equity = state.equity
        state.weekly_pnl = 0.0

    def is_halted(self) -> bool:
        return self.lock_path.exists()

    def clear_lock(self) -> None:
        """Manual resume: remove the halt lock file."""
        self.lock_path.unlink(missing_ok=True)

    def get_history(self) -> list[BreakerEvent]:
        return list(self._history)

    # ---- internals
    def _trip(self, kind: str, dd: float, state: PortfolioState, action: str) -> None:
        ev = BreakerEvent(
            breaker_type=kind, drawdown=dd, equity=state.equity,
            positions_closed=len(state.positions) if "close" in action else 0,
            hmm_regime=state.hmm_regime, action=action,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._history.append(ev)
        logger.warning("CIRCUIT BREAKER %s dd=%.2f%% equity=%.0f regime=%s action=%s",
                       kind, dd * 100, state.equity, state.hmm_regime or "?", action)

    def _write_lock(self, dd: float, state: PortfolioState) -> None:
        try:
            self.lock_path.write_text(
                f"peak drawdown {dd:.4f} at {datetime.now(timezone.utc).isoformat()}\n"
                f"equity={state.equity} peak={state.peak_equity} regime={state.hmm_regime}\n"
                "Delete this file to resume trading.\n"
            )
            logger.error("TRADING HALTED — wrote %s", self.lock_path)
        except OSError as exc:
            logger.error("could not write halt lock: %s", exc)


# ==================================================================== risk manager
class RiskManager:
    """Position sizing, exposure/leverage caps, drawdown circuit breakers."""

    def __init__(self, cfg: dict, lock_dir: Optional[Path] = None) -> None:
        r = cfg["risk"]
        self.max_risk_per_trade = float(r["max_risk_per_trade"])
        self.max_exposure = float(r["max_exposure"])
        self.max_leverage = float(r["max_leverage"])
        self.max_single_position = float(r["max_single_position"])
        self.max_concurrent = int(r["max_concurrent"])
        self.max_daily_trades = int(r["max_daily_trades"])
        self.min_position_usd = float(r.get("min_position_usd", 100))
        self.max_sector_exposure = float(r.get("max_sector_exposure", 0.30))
        self.corr_window = int(r.get("correlation_window", 60))
        self.corr_reduce = float(r.get("correlation_reduce", 0.70))
        self.corr_reject = float(r.get("correlation_reject", 0.85))
        self.max_spread_pct = float(r.get("max_spread_pct", 0.005))
        self.dup_window_s = float(r.get("duplicate_window_seconds", 60))
        self.gap_multiple = float(r.get("gap_multiple", 3.0))
        self.overnight_gap_risk = float(r.get("overnight_gap_risk_pct", 0.02))
        self.reduce_factor = float(r.get("size_reduce_factor", 0.50))
        self.breaker = CircuitBreaker(cfg, lock_dir)

    # ---------------------------------------------------- primary entry point
    def validate_signal(
        self, signal: "Signal", state: PortfolioState, now: Optional[datetime] = None
    ) -> RiskDecision:
        """Apply every risk gate. Returns approval + a possibly-shrunk signal."""
        now = now or datetime.now(timezone.utc)
        mods: list[str] = []
        equity = state.equity

        # ---- 0. hard halts (lock file / breaker)
        breaker = self.breaker.evaluate(state)
        if breaker.halt:
            return RiskDecision(False, None, f"halted: {breaker.reason}", mods)

        # ---- 1. mandatory stop loss (system refuses orders without one)
        if signal.stop_loss is None or not np.isfinite(signal.stop_loss):
            return RiskDecision(False, None, "no stop loss", mods)
        risk_per_share = abs(signal.entry_price - signal.stop_loss)
        if risk_per_share <= 0:
            return RiskDecision(False, None, "stop loss not below entry", mods)

        # ---- 2. order validation: tradeable, spread, duplicate
        meta = signal.metadata or {}
        if meta.get("tradeable", True) is False:
            return RiskDecision(False, None, "symbol not tradeable", mods)
        spread = meta.get("spread_pct")
        if spread is not None and spread > self.max_spread_pct:
            return RiskDecision(False, None,
                                f"spread {spread:.3%} > {self.max_spread_pct:.3%}", mods)
        key = (signal.symbol, signal.direction)
        last = state.recent_orders.get(key)
        if last is not None and (now - last).total_seconds() < self.dup_window_s:
            return RiskDecision(False, None, "duplicate order within window", mods)

        # ---- 3. concurrency + daily-trade caps (new symbols only)
        is_new = signal.symbol not in state.positions
        if is_new and len(state.positions) >= self.max_concurrent:
            return RiskDecision(False, None,
                                f"max concurrent positions ({self.max_concurrent})", mods)
        if state.trades_today >= self.max_daily_trades:
            return RiskDecision(False, None,
                                f"max daily trades ({self.max_daily_trades})", mods)

        # ---- 4. 1%-risk-to-stop sizing, capped by regime then portfolio
        dollar_risk = equity * self.max_risk_per_trade
        risk_weight = (dollar_risk / risk_per_share) * signal.entry_price / equity
        weight = min(risk_weight, signal.position_size_pct, self.max_single_position)
        if weight < risk_weight:
            mods.append(f"size capped to {weight:.2%}")

        # ---- 5. overnight gap risk: 3x stop gap must cost <= overnight_gap_risk
        if meta.get("overnight", False):
            gap_loss_per_share = self.gap_multiple * risk_per_share
            max_shares_gap = (equity * self.overnight_gap_risk) / gap_loss_per_share
            gap_weight = max_shares_gap * signal.entry_price / equity
            if gap_weight < weight:
                weight = gap_weight
                mods.append(f"overnight gap cap {weight:.2%}")

        # ---- 6. circuit-breaker size reduction
        if breaker.size_scale < 1.0:
            weight *= breaker.size_scale
            mods.append(f"breaker reduce x{breaker.size_scale} ({breaker.active})")

        # ---- 7. correlation with open positions
        corr = self._max_correlation(signal.symbol, state)
        if corr is not None:
            if corr > self.corr_reject:
                return RiskDecision(False, None,
                                    f"correlation {corr:.2f} > {self.corr_reject}", mods)
            if corr > self.corr_reduce:
                weight *= self.reduce_factor
                mods.append(f"correlation {corr:.2f} reduce x{self.reduce_factor}")

        # ---- 8. leverage rules (1.0x unless calm + certain + uncrowded)
        leverage = self._resolve_leverage(signal, state, breaker)
        if leverage != signal.leverage:
            mods.append(f"leverage {signal.leverage}->{leverage}")

        # ---- 9. sector exposure cap (30% per sector)
        sector = meta.get("sector") or state.sector_map.get(signal.symbol, "UNKNOWN")
        existing_sector = state.sector_exposure(sector) - (
            state.positions[signal.symbol].weight if not is_new else 0.0)
        head = self.max_sector_exposure - existing_sector
        if head <= 0:
            return RiskDecision(False, None,
                                f"sector {sector} at cap {self.max_sector_exposure:.0%}", mods)
        if weight > head:
            weight = head
            mods.append(f"sector cap {sector} -> {weight:.2%}")

        # ---- 10. portfolio gross-exposure cap (80%, 20% cash minimum)
        current_gross = state.gross_exposure() - (
            state.positions[signal.symbol].weight if not is_new else 0.0)
        head = self.max_exposure - current_gross
        if head <= 0:
            return RiskDecision(False, None,
                                f"gross exposure at cap {self.max_exposure:.0%}", mods)
        if weight > head:
            weight = head
            mods.append(f"exposure cap -> {weight:.2%}")

        # ---- 11. buying power (leveraged notional must fit)
        notional = weight * leverage * equity
        if notional > state.buying_power:
            affordable = state.buying_power / (leverage * equity) if leverage * equity else 0.0
            weight = max(0.0, affordable)
            mods.append(f"buying-power cap -> {weight:.2%}")

        # ---- 12. minimum position size
        if weight * equity < self.min_position_usd:
            return RiskDecision(False, None,
                                f"position < ${self.min_position_usd:.0f} minimum", mods)

        modified = replace(signal, position_size_pct=round(weight, 6), leverage=leverage)
        return RiskDecision(True, modified, "", mods)

    # ------------------------------------------------------------- leverage
    def _resolve_leverage(
        self, signal: "Signal", state: PortfolioState, breaker: BreakerResult
    ) -> float:
        """1.0x unless the signal asked for more AND conditions are calm."""
        requested = min(signal.leverage, self.max_leverage)
        if requested <= 1.0:
            return 1.0
        force_flat = (
            state.regime_uncertain
            or breaker.active != "none"
            or len(state.positions) >= 3
            or state.flicker_rate > 0
        )
        return 1.0 if force_flat else requested

    # ----------------------------------------------------------- correlation
    def _max_correlation(self, symbol: str, state: PortfolioState) -> Optional[float]:
        """Highest 60-day return correlation between ``symbol`` and open names."""
        hist = state.price_history
        if symbol not in hist or not state.positions:
            return None
        cand = self._returns(hist[symbol]).tail(self.corr_window)
        best: Optional[float] = None
        for other in state.positions:
            if other == symbol or other not in hist:
                continue
            oth = self._returns(hist[other]).tail(self.corr_window)
            joined = pd.concat([cand, oth], axis=1, join="inner").dropna()
            if len(joined) < 5:
                continue
            c = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
            if np.isfinite(c) and (best is None or c > best):
                best = c
        return best

    @staticmethod
    def _returns(series: pd.Series) -> pd.Series:
        """Treat a mostly-positive series as prices; else as returns already."""
        s = pd.Series(series)
        return s.pct_change().dropna() if (s > 0).all() else s.dropna()

    # ============================================= legacy gate API (signal_gen)
    def position_size(self, state: RiskState, price: float, stop_distance: float) -> int:
        """Shares risking ``max_risk_per_trade`` to the stop, capped by weight."""
        if stop_distance <= 0 or price <= 0:
            return 0
        dollar_risk = state.equity * self.max_risk_per_trade * state.size_scale
        shares = int(dollar_risk / stop_distance)
        max_shares = int(state.equity * self.max_single_position / price)
        return max(0, min(shares, max_shares))

    def can_open(self, state: RiskState, open_positions: int) -> RiskCheck:
        """Gate a new position on halt, concurrency, and daily-trade caps."""
        if state.halted:
            return RiskCheck(False, f"halted: {state.halt_reason}")
        if open_positions >= self.max_concurrent:
            return RiskCheck(False, f"max concurrent ({self.max_concurrent})")
        if state.trades_today >= self.max_daily_trades:
            return RiskCheck(False, f"max daily trades ({self.max_daily_trades})")
        return RiskCheck(True, "ok", size_scale=state.size_scale)

    def check_exposure(self, gross_exposure: float, leverage: float) -> bool:
        """True if gross exposure and leverage are within caps."""
        return gross_exposure <= self.max_exposure + 1e-9 and leverage <= self.max_leverage + 1e-9

    def update_drawdowns(self, state: RiskState) -> RiskState:
        """Recompute size scaling / halt flags from current equity."""
        peak_dd = self.breaker._dd(state.equity, state.peak_equity)
        daily_dd = self.breaker._dd(state.equity, state.day_start_equity)
        weekly_dd = self.breaker._dd(state.equity, state.week_start_equity)
        state.size_scale = 1.0
        if peak_dd >= self.breaker.peak_halt:
            state.halted, state.halt_reason = True, "peak drawdown kill switch"
        elif daily_dd >= self.breaker.daily_halt:
            state.halted, state.halt_reason = True, "daily drawdown halt"
        elif weekly_dd >= self.breaker.weekly_halt:
            state.halted, state.halt_reason = True, "weekly drawdown halt"
        elif daily_dd >= self.breaker.daily_reduce or weekly_dd >= self.breaker.weekly_reduce:
            state.size_scale = self.reduce_factor
        if state.equity > state.peak_equity:
            state.peak_equity = state.equity
        return state

    def roll_day(self, state: RiskState, today: date) -> RiskState:
        """Reset per-day counters when the trading date advances."""
        if state.trade_date != today:
            state.trade_date = today
            state.trades_today = 0
            state.day_start_equity = state.equity
            if not (self.breaker._dd(state.equity, state.peak_equity) >= self.breaker.peak_halt):
                state.halted, state.halt_reason = False, ""
        return state
