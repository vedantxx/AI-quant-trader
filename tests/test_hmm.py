"""Tests for the HMM regime engine: model selection, labeling, filters."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("hmmlearn")

from core.hmm_engine import HMMEngine, RegimeInfo, RegimeState  # noqa: E402
from data.feature_engineering import FeatureEngineer  # noqa: E402

CFG = {
    "hmm": {
        "n_candidates": [3, 4, 5],
        "n_init": 3,
        "covariance_type": "full",
        "min_train_bars": 300,
        "stability_bars": 3,
        "flicker_window": 20,
        "flicker_threshold": 4,
        "min_confidence": 0.55,
    }
}


def _ohlcv(n=1200, seed=7):
    rng = np.random.default_rng(seed)
    calm = rng.normal(0.0005, 0.006, n // 2)
    wild = rng.normal(-0.0003, 0.024, n - n // 2)
    close = 100 * np.exp(np.cumsum(np.concatenate([calm, wild])))
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": close, "high": close * 1.005, "low": close * 0.995,
         "close": close, "volume": rng.integers(1_000_000, 5_000_000, n)},
        index=idx,
    )


@pytest.fixture(scope="module")
def fitted():
    features = FeatureEngineer().feature_matrix(_ohlcv())
    return HMMEngine(CFG).fit(features), features


def test_selects_model_by_bic(fitted):
    engine, _ = fitted
    assert engine.model is not None
    assert engine.metadata.n_regimes in CFG["hmm"]["n_candidates"]
    # every candidate that fit has a recorded BIC; selected is the minimum
    assert engine.metadata.bic == min(engine.metadata.all_bic.values())


def test_labels_sorted_by_return(fitted):
    engine, _ = fitted
    n = engine.metadata.n_regimes
    returns = [engine.regime_info[s].expected_return for s in range(n)]
    labels = [engine.state_to_label[s] for s in range(n)]
    # state ids are assigned so ascending return order maps to the label list
    ordered_by_ret = [labels[s] for s in np.argsort(returns)]
    from core.hmm_engine import REGIME_LABELS
    assert ordered_by_ret == REGIME_LABELS[n]


def test_min_train_bars_enforced():
    features = FeatureEngineer().feature_matrix(_ohlcv(n=400))
    small = features[:100]
    with pytest.raises(ValueError):
        HMMEngine(CFG).fit(small)


def test_predict_returns_regime_state(fitted):
    engine, features = fitted
    state = engine.predict(features[:410])
    assert isinstance(state, RegimeState)
    assert 0.0 <= state.probability <= 1.0
    assert state.label in set(engine.state_to_label.values())


def test_regime_info_populated(fitted):
    engine, _ = fitted
    for s, info in engine.regime_info.items():
        assert isinstance(info, RegimeInfo)
        assert info.min_confidence_to_act == CFG["hmm"]["min_confidence"]


def test_transition_matrix_rows_sum_to_one(fitted):
    engine, _ = fitted
    tm = engine.get_transition_matrix()
    assert np.allclose(tm.sum(axis=1), 1.0)


def test_stability_filter_requires_persistence():
    """A one-off differing state does not immediately confirm a change."""
    engine = HMMEngine(CFG)
    engine.state_to_label = {0: "BEAR", 1: "BULL"}
    engine._update_stability(0)          # seed confirmed = 0
    assert engine._confirmed_state == 0
    engine._update_stability(1)          # pending 1/3
    assert engine._confirmed_state == 0
    assert engine.detect_regime_change() is False
    engine._update_stability(1)          # 2/3
    engine._update_stability(1)          # 3/3 -> confirm
    assert engine._confirmed_state == 1
    assert engine.detect_regime_change() is True


def test_flicker_detection():
    engine = HMMEngine(CFG)
    engine._bar_index = 100
    # more changes than threshold inside the window -> flickering
    for b in range(96, 101):
        engine._change_bars.append(b)
    assert engine.get_regime_flicker_rate() > CFG["hmm"]["flicker_threshold"]
    assert engine.is_flickering() is True
