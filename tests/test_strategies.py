"""Tests for regime-based allocation strategy (skeletons)."""

from core.regime_strategies import AllocationSignal, RegimeStrategies  # noqa: F401


def test_low_vol_uses_leverage() -> None:
    """Low-vol regime allocates full exposure with leverage."""
    ...


def test_mid_vol_trend_vs_no_trend() -> None:
    """Mid-vol exposure depends on the trend flag."""
    ...


def test_high_vol_derisks() -> None:
    """High-vol regime cuts exposure and drops leverage to 1.0."""
    ...


def test_uncertainty_haircut() -> None:
    """Low regime confidence applies the uncertainty size multiplier."""
    ...


def test_rebalance_threshold() -> None:
    """needs_rebalance() triggers only past the drift threshold."""
    ...
