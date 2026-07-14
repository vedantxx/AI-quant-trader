"""Tests for the HMM regime engine."""

import numpy as np
import pytest

from core.hmm_engine import HMMEngine

CFG = {
    "hmm": {
        "n_candidates": [2, 3],
        "n_init": 2,
        "covariance_type": "full",
        "min_train_bars": 100,
        "stability_bars": 3,
        "flicker_window": 20,
        "flicker_threshold": 4,
        "min_confidence": 0.55,
    }
}


def _synthetic(n=400, seed=0):
    """Two-regime series: calm then volatile, with return-like col 0."""
    rng = np.random.default_rng(seed)
    calm = rng.normal(0, 0.005, (n // 2, 3))
    wild = rng.normal(0, 0.03, (n // 2, 3))
    return np.vstack([calm, wild])


def test_fit_selects_model():
    engine = HMMEngine(CFG).fit(_synthetic())
    assert engine.model is not None
    assert set(engine.state_to_label.values()) <= {"low", "mid", "high"}


def test_min_train_bars_enforced():
    with pytest.raises(ValueError):
        HMMEngine(CFG).fit(_synthetic(n=50))


def test_predict_returns_confidence():
    engine = HMMEngine(CFG).fit(_synthetic())
    res = engine.predict(_synthetic())
    assert 0.0 <= res.confidence <= 1.0
    assert res.label in {"low", "mid", "high"}


def test_low_confidence_not_accepted():
    engine = HMMEngine(CFG).fit(_synthetic())
    res = engine.predict(_synthetic())
    if res.confidence < CFG["hmm"]["min_confidence"]:
        assert res.accepted is False
