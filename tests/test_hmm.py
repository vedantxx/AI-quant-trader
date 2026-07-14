"""Tests for the HMM regime engine (skeletons — fill in with implementation)."""

from core.hmm_engine import HMMEngine, RegimeState  # noqa: F401


def test_fit_selects_model_by_bic() -> None:
    """fit() picks the lowest-BIC state count and labels states."""
    ...


def test_min_train_bars_enforced() -> None:
    """fit() rejects histories shorter than hmm.min_train_bars."""
    ...


def test_predict_returns_confidence() -> None:
    """predict() returns a RegimeState with confidence in [0, 1]."""
    ...


def test_low_confidence_not_accepted() -> None:
    """A regime below hmm.min_confidence is not accepted."""
    ...


def test_flicker_freeze() -> None:
    """Excess state switches within the window freeze acceptance."""
    ...
