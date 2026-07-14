"""Tests for regime-based allocation strategy."""

from core.regime_strategies import RegimeStrategy

CFG = {
    "strategy": {
        "low_vol_allocation": 0.95,
        "mid_vol_allocation_trend": 0.95,
        "mid_vol_allocation_no_trend": 0.60,
        "high_vol_allocation": 0.60,
        "low_vol_leverage": 1.25,
        "rebalance_threshold": 0.10,
        "uncertainty_size_mult": 0.50,
    },
    "hmm": {"min_confidence": 0.55},
}


def test_low_vol_uses_leverage():
    s = RegimeStrategy(CFG)
    t = s.target("low", trend_up=True, confidence=0.9)
    assert t.exposure == 0.95
    assert t.leverage == 1.25


def test_mid_vol_trend_vs_no_trend():
    s = RegimeStrategy(CFG)
    assert s.target("mid", True, 0.9).exposure == 0.95
    assert s.target("mid", False, 0.9).exposure == 0.60


def test_high_vol_derisks():
    s = RegimeStrategy(CFG)
    t = s.target("high", trend_up=True, confidence=0.9)
    assert t.exposure == 0.60
    assert t.leverage == 1.0


def test_uncertainty_haircut():
    s = RegimeStrategy(CFG)
    t = s.target("low", trend_up=True, confidence=0.4)  # below min_confidence
    assert t.size_mult == 0.50


def test_rebalance_threshold():
    s = RegimeStrategy(CFG)
    assert s.needs_rebalance(0.10, 0.25) is True
    assert s.needs_rebalance(0.20, 0.25) is False
