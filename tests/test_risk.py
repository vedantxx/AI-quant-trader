"""Tests for the risk manager: sizing, caps, drawdown breakers."""

from datetime import date

from core.risk_manager import RiskManager, RiskState

CFG = {
    "risk": {
        "max_risk_per_trade": 0.01,
        "max_exposure": 0.80,
        "max_leverage": 1.25,
        "max_single_position": 0.15,
        "max_concurrent": 5,
        "max_daily_trades": 20,
        "daily_dd_reduce": 0.02,
        "daily_dd_halt": 0.03,
        "weekly_dd_reduce": 0.05,
        "weekly_dd_halt": 0.07,
        "max_dd_from_peak": 0.10,
    }
}


def _state(equity=100_000.0):
    return RiskState(
        equity=equity,
        peak_equity=equity,
        day_start_equity=equity,
        week_start_equity=equity,
    )


def test_position_size_respects_risk():
    rm = RiskManager(CFG)
    shares = rm.position_size(_state(), price=100, stop_distance=2.0)
    # 1% of 100k = $1000 risk / $2 stop = 500 shares, capped by 15% weight (150)
    assert shares == 150


def test_single_position_cap():
    rm = RiskManager(CFG)
    shares = rm.position_size(_state(), price=100, stop_distance=0.1)
    assert shares <= int(100_000 * 0.15 / 100)


def test_daily_dd_halt():
    rm = RiskManager(CFG)
    st = _state()
    st.equity = 96_500  # -3.5% on the day
    st = rm.update_drawdowns(st)
    assert st.halted is True
    assert "daily" in st.halt_reason


def test_daily_dd_reduce_scales_size():
    rm = RiskManager(CFG)
    st = _state()
    st.equity = 97_500  # -2.5% -> reduce, not halt
    st = rm.update_drawdowns(st)
    assert st.halted is False
    assert st.size_scale == 0.5


def test_peak_kill_switch():
    rm = RiskManager(CFG)
    st = _state()
    st.peak_equity = 120_000
    st.equity = 107_000  # -10.8% from peak
    st = rm.update_drawdowns(st)
    assert st.halted is True
    assert "peak" in st.halt_reason


def test_max_concurrent_blocks_open():
    rm = RiskManager(CFG)
    ok, reason = rm.can_open(_state(), open_positions=5)
    assert ok is False
    assert "concurrent" in reason


def test_roll_day_resets_counters():
    rm = RiskManager(CFG)
    st = _state()
    st.trades_today = 10
    st = rm.roll_day(st, date(2026, 7, 14))
    assert st.trades_today == 0
