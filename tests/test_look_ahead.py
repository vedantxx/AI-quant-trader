"""Verify no look-ahead bias in features and backtest (skeletons).

Core invariant: a feature value at time t must not change when future bars are
appended. If it does, the feature peeks into the future.
"""

from data.feature_engineering import FeatureEngineer  # noqa: F401


def test_features_are_causal() -> None:
    """build_features(df[:k]) matches build_features(df) on shared rows."""
    ...


def test_realized_vol_no_future_leak() -> None:
    """realized_vol on a prefix matches the full series on shared rows."""
    ...


def test_rsi_no_future_leak() -> None:
    """rsi on a prefix matches the full series on shared rows."""
    ...


def test_backtest_uses_past_only() -> None:
    """Each backtest bar uses features up to i-1 only."""
    ...
