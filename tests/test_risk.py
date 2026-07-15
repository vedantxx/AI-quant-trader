"""Tests for the risk manager: circuit breakers, veto, sizing, correlation."""

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from core.regime_strategies import Direction, Signal
from core.risk_manager import (
    CircuitBreaker,
    Position,
    PortfolioState,
    RiskCheck,
    RiskManager,
    RiskState,
)

CFG = {
    "risk": {
        "max_risk_per_trade": 0.01, "max_exposure": 0.80, "max_leverage": 1.25,
        "max_single_position": 0.15, "max_concurrent": 5, "max_daily_trades": 20,
        "daily_dd_reduce": 0.02, "daily_dd_halt": 0.03, "weekly_dd_reduce": 0.05,
        "weekly_dd_halt": 0.07, "max_dd_from_peak": 0.10, "size_reduce_factor": 0.50,
        "min_position_usd": 100, "max_sector_exposure": 0.30, "correlation_window": 60,
        "correlation_reduce": 0.70, "correlation_reject": 0.85, "max_spread_pct": 0.005,
        "duplicate_window_seconds": 60, "gap_multiple": 3.0, "overnight_gap_risk_pct": 0.02,
        "halt_lock_file": "trading_halted.lock",
    }
}


def make_signal(**kw) -> Signal:
    base = dict(
        symbol="SPY", direction=Direction.LONG, confidence=0.9, entry_price=100.0,
        stop_loss=95.0, position_size_pct=0.95, leverage=1.25, regime_id=0,
        regime_name="BULL", regime_probability=0.9, strategy_name="low_vol_bull",
        reasoning="", metadata={},
    )
    base.update(kw)
    return Signal(**base)


def make_state(**kw) -> PortfolioState:
    base = dict(
        equity=100_000.0, cash=100_000.0, buying_power=200_000.0,
        peak_equity=100_000.0, day_start_equity=100_000.0, week_start_equity=100_000.0,
    )
    base.update(kw)
    return PortfolioState(**base)


def rm(tmp_path) -> RiskManager:
    return RiskManager(CFG, lock_dir=tmp_path)


# ------------------------------------------------------------------ breakers
def test_daily_dd_reduce(tmp_path) -> None:
    cb = CircuitBreaker(CFG, tmp_path)
    st = make_state(equity=97_500)  # -2.5% on the day -> reduce
    res = cb.evaluate(st)
    assert res.size_scale == 0.50 and not res.halt


def test_daily_dd_halt(tmp_path) -> None:
    cb = CircuitBreaker(CFG, tmp_path)
    res = cb.evaluate(make_state(equity=96_500))  # -3.5% -> halt + close
    assert res.halt and res.close_all and res.active == "daily_dd_halt"


def test_peak_kill_switch_writes_lock(tmp_path) -> None:
    cb = CircuitBreaker(CFG, tmp_path)
    res = cb.evaluate(make_state(equity=89_000))  # -11% from peak
    assert res.halt and res.active == "peak_dd"
    assert (tmp_path / "trading_halted.lock").exists()
    assert cb.is_halted()
    cb.clear_lock()
    assert not cb.is_halted()


def test_breaker_history_logs_regime(tmp_path) -> None:
    cb = CircuitBreaker(CFG, tmp_path)
    cb.evaluate(make_state(equity=96_500, hmm_regime="BULL"))
    hist = cb.get_history()
    assert hist and hist[-1].hmm_regime == "BULL" and hist[-1].positions_closed == 0


# ---------------------------------------------------------------- validation
def test_rejects_missing_stop(tmp_path) -> None:
    d = rm(tmp_path).validate_signal(make_signal(stop_loss=float("nan")), make_state())
    assert not d.approved and "stop" in d.rejection_reason


def test_rejects_stop_not_below_entry(tmp_path) -> None:
    d = rm(tmp_path).validate_signal(make_signal(stop_loss=100.0), make_state())
    assert not d.approved


def test_sizes_one_percent_risk_and_caps(tmp_path) -> None:
    # risk/share = 5, dollar risk = 1000 -> 200 shares -> weight 0.20, capped to 0.15
    d = rm(tmp_path).validate_signal(make_signal(), make_state())
    assert d.approved
    assert d.modified_signal.position_size_pct == pytest.approx(0.15)


def test_leverage_forced_flat_when_uncertain(tmp_path) -> None:
    d = rm(tmp_path).validate_signal(make_signal(), make_state(regime_uncertain=True))
    assert d.modified_signal.leverage == 1.0


def test_leverage_allowed_when_calm(tmp_path) -> None:
    d = rm(tmp_path).validate_signal(make_signal(), make_state())
    assert d.modified_signal.leverage == 1.25


def test_max_concurrent_blocks_new(tmp_path) -> None:
    positions = {f"S{i}": Position(f"S{i}", 0.1, 100, 95) for i in range(5)}
    d = rm(tmp_path).validate_signal(make_signal(), make_state(positions=positions))
    assert not d.approved and "concurrent" in d.rejection_reason


def test_duplicate_blocked(tmp_path) -> None:
    now = datetime.now(timezone.utc)
    st = make_state(recent_orders={("SPY", Direction.LONG): now - timedelta(seconds=30)})
    d = rm(tmp_path).validate_signal(make_signal(), st, now=now)
    assert not d.approved and "duplicate" in d.rejection_reason


def test_sector_cap(tmp_path) -> None:
    positions = {"XLK": Position("XLK", 0.28, 100, 95, sector="TECH")}
    st = make_state(positions=positions, sector_map={"XLK": "TECH", "AAPL": "TECH"})
    d = rm(tmp_path).validate_signal(make_signal(symbol="AAPL"), st)
    assert d.approved and d.modified_signal.position_size_pct == pytest.approx(0.02)


def test_spread_rejected(tmp_path) -> None:
    d = rm(tmp_path).validate_signal(
        make_signal(metadata={"spread_pct": 0.01}), make_state())
    assert not d.approved and "spread" in d.rejection_reason


def test_correlation_reject(tmp_path) -> None:
    idx = pd.date_range("2024-01-01", periods=90, freq="B")
    base = pd.Series(100 + np.cumsum(np.random.default_rng(1).normal(0, 1, 90)), idx)
    st = make_state(
        positions={"QQQ": Position("QQQ", 0.1, 100, 95)},
        price_history={"SPY": base, "QQQ": base * 1.0001},  # ~identical -> corr ~1
    )
    d = rm(tmp_path).validate_signal(make_signal(), st)
    assert not d.approved and "correlation" in d.rejection_reason


def test_overnight_gap_cap(tmp_path) -> None:
    # 3x gap of $5 = $15/share; 2% of 100k = $2000 -> 133 shares -> weight ~0.133
    d = rm(tmp_path).validate_signal(
        make_signal(metadata={"overnight": True}), make_state())
    assert d.approved and d.modified_signal.position_size_pct < 0.15


def test_halt_lock_vetoes_all(tmp_path) -> None:
    (tmp_path / "trading_halted.lock").write_text("halted")
    d = rm(tmp_path).validate_signal(make_signal(), make_state())
    assert not d.approved and "halt" in d.rejection_reason.lower()


# -------------------------------------------------------------- legacy gate
def test_legacy_position_size_and_cap() -> None:
    r = RiskManager(CFG)
    st = RiskState(equity=100_000, peak_equity=100_000,
                   day_start_equity=100_000, week_start_equity=100_000)
    assert r.position_size(st, price=100, stop_distance=5) == 150  # capped by 15% weight


def test_legacy_daily_halt_and_roll() -> None:
    r = RiskManager(CFG)
    st = RiskState(equity=96_000, peak_equity=100_000,
                   day_start_equity=100_000, week_start_equity=100_000)
    r.update_drawdowns(st)
    assert st.halted
    r.roll_day(st, date(2024, 1, 2))
    assert st.trades_today == 0 and not st.halted


def test_legacy_max_concurrent() -> None:
    r = RiskManager(CFG)
    st = RiskState(equity=100_000, peak_equity=100_000,
                   day_start_equity=100_000, week_start_equity=100_000)
    assert r.can_open(st, open_positions=5).approved is False
