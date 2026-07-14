"""Verify no look-ahead bias in features and backtest.

Core invariant: a feature value at time t must not change when future bars are
appended. If it does, the feature peeks into the future.
"""

import numpy as np
import pandas as pd

from data.feature_engineering import build_features, realized_vol, rsi


def _ohlcv(n=300, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1e6, 5e6, n),
        },
        index=idx,
    )


def test_features_are_causal():
    """Features on df[:k] must match features on full df at the same rows."""
    df = _ohlcv()
    k = 200
    full = build_features(df)
    partial = build_features(df.iloc[:k])
    common = partial.index.intersection(full.index)
    aligned_full = full.loc[common]
    aligned_partial = partial.loc[common]
    pd.testing.assert_frame_equal(
        aligned_partial, aligned_full, check_exact=False, rtol=1e-9
    )


def test_realized_vol_no_future_leak():
    df = _ohlcv()
    v_full = realized_vol(df["close"], 20)
    v_partial = realized_vol(df["close"].iloc[:150], 20)
    common = v_partial.dropna().index
    assert np.allclose(
        v_partial.loc[common].values, v_full.loc[common].values, equal_nan=True
    )


def test_rsi_no_future_leak():
    df = _ohlcv()
    r_full = rsi(df["close"], 14)
    r_partial = rsi(df["close"].iloc[:150], 14)
    common = r_partial.dropna().index
    assert np.allclose(r_partial.loc[common].values, r_full.loc[common].values)
