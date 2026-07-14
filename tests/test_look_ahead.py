"""Verify NO look-ahead bias in features and filtered regime inference.

The core invariant: a feature value — and a filtered regime estimate — at time
t must not change when future bars are appended. If it does, information from
the future has leaked into the past and every backtest built on it is fiction.
"""

import numpy as np
import pandas as pd
import pytest

from data.feature_engineering import FeatureEngineer, realized_vol, rsi

hmm = pytest.importorskip("hmmlearn")  # engine training needs hmmlearn

from core.hmm_engine import HMMEngine  # noqa: E402

CFG = {
    "hmm": {
        "n_candidates": [3, 4],
        "n_init": 3,
        "covariance_type": "full",
        "min_train_bars": 300,
        "stability_bars": 3,
        "flicker_window": 20,
        "flicker_threshold": 4,
        "min_confidence": 0.55,
    }
}


def _ohlcv(n=1200, seed=1):
    """Synthetic OHLCV with a calm half and a turbulent half."""
    rng = np.random.default_rng(seed)
    calm = rng.normal(0.0004, 0.006, n // 2)
    wild = rng.normal(-0.0002, 0.025, n - n // 2)
    ret = np.concatenate([calm, wild])
    close = 100 * np.exp(np.cumsum(ret))
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close,
         "volume": rng.integers(1_000_000, 5_000_000, n)},
        index=idx,
    )


# ---------------------------------------------------------------- feature causality
def test_features_are_causal():
    """build_features(df[:k]) matches build_features(df) on shared rows."""
    df = _ohlcv()
    fe = FeatureEngineer()
    full = fe.build_features(df)
    partial = fe.build_features(df.iloc[:600])
    common = partial.index.intersection(full.index)
    assert len(common) > 100
    pd.testing.assert_frame_equal(
        partial.loc[common], full.loc[common], check_exact=False, rtol=1e-9
    )


def test_realized_vol_no_future_leak():
    df = _ohlcv()
    v_full = realized_vol(df["close"], 20)
    v_partial = realized_vol(df["close"].iloc[:400], 20)
    common = v_partial.dropna().index
    assert np.allclose(v_partial.loc[common].values, v_full.loc[common].values)


def test_rsi_no_future_leak():
    df = _ohlcv()
    r_full = rsi(df["close"], 14)
    r_partial = rsi(df["close"].iloc[:400], 14)
    common = r_partial.dropna().index
    assert np.allclose(r_partial.loc[common].values, r_full.loc[common].values)


# ------------------------------------------------------- MANDATORY: filtered regime
def test_no_look_ahead_bias():
    """Filtered regime at index T is identical with data[:T] vs data[:T+extra].

    Uses the forward algorithm (predict_regime_filtered), never Viterbi. The
    filtered estimate at a bar depends only on that bar and earlier ones, so
    appending future bars must not change it.
    """
    df = _ohlcv()
    fe = FeatureEngineer()
    features = fe.feature_matrix(df)          # standardized, causal
    assert len(features) >= 500

    engine = HMMEngine(CFG).fit(features)

    idx = 399
    regime_short = engine.predict_regime_filtered(features[:400])[idx]
    regime_long = engine.predict_regime_filtered(features[:500])[idx]
    assert regime_short == regime_long, "LOOK-AHEAD BIAS DETECTED"


def test_filtered_matches_across_many_indices():
    """Robustness: the whole prefix is invariant to appended future data."""
    df = _ohlcv()
    fe = FeatureEngineer()
    features = fe.feature_matrix(df)
    engine = HMMEngine(CFG).fit(features)

    short = engine.predict_regime_filtered(features[:450])
    long = engine.predict_regime_filtered(features[:600])
    assert np.array_equal(short, long[:450]), "filtered prefix changed with future data"


def test_viterbi_would_leak():
    """Sanity check: Viterbi (predict) DOES revise the past — why we avoid it."""
    df = _ohlcv()
    fe = FeatureEngineer()
    features = fe.feature_matrix(df)
    engine = HMMEngine(CFG).fit(features)

    short = engine.model.predict(features[:450])
    long = engine.model.predict(features[:600])[:450]
    # Not asserting inequality (data-dependent), just documenting the contrast:
    # filtered is guaranteed equal above; Viterbi carries no such guarantee.
    assert short.shape == long.shape
