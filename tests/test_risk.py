"""Tests for the risk manager: sizing, caps, drawdown breakers (skeletons)."""

from core.risk_manager import RiskCheck, RiskManager, RiskState  # noqa: F401


def test_position_size_respects_risk() -> None:
    """Size risks max_risk_per_trade to the stop, capped by weight."""
    ...


def test_single_position_cap() -> None:
    """Size never exceeds max_single_position weight."""
    ...


def test_daily_dd_halt() -> None:
    """Daily drawdown past the halt threshold halts trading."""
    ...


def test_daily_dd_reduce_scales_size() -> None:
    """Daily drawdown past the reduce threshold scales size down."""
    ...


def test_peak_kill_switch() -> None:
    """Drawdown from peak past max_dd_from_peak halts trading."""
    ...


def test_max_concurrent_blocks_open() -> None:
    """can_open() rejects once max_concurrent positions are open."""
    ...


def test_roll_day_resets_counters() -> None:
    """roll_day() resets the daily trade counter on a new date."""
    ...
