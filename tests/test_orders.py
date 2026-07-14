"""Tests for order executor (skeletons — client mocked, no network)."""

from broker.order_executor import OrderExecutor, OrderResult  # noqa: F401


def test_submit_market_wraps_result() -> None:
    """submit_market() returns a normalized OrderResult."""
    ...


def test_zero_qty_rejected() -> None:
    """submit_market() rejects a non-positive quantity."""
    ...


def test_limit_requires_positive_qty() -> None:
    """submit_limit() rejects a non-positive quantity."""
    ...


def test_cancel_all_delegates() -> None:
    """cancel_all() delegates to the client's cancel-orders call."""
    ...
