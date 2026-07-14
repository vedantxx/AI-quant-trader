"""HMM regime detection engine.

Fits a Gaussian HMM over engineered market features, selects the best state
count by BIC, labels states as volatility regimes (low/mid/high), and gates raw
state flips through stability + flicker + confidence filters.

Stub scaffold: signatures, type hints, and docstrings only — no logic yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:  # avoid importing hmmlearn until deps installed
    from hmmlearn.hmm import GaussianHMM


@dataclass
class RegimeState:
    """Detected regime for a single bar."""

    state: int                          # raw HMM state index
    label: str                          # "low" | "mid" | "high"
    confidence: float                   # posterior prob of chosen state
    accepted: bool                      # passed stability + confidence filters
    posteriors: Optional[np.ndarray] = None


class HMMEngine:
    """HMM regime detection engine."""

    def __init__(self, cfg: dict) -> None:
        """Read hmm.* config; initialize filter state."""
        ...

    def fit(self, features: np.ndarray) -> "HMMEngine":
        """Fit HMM, selecting state count by BIC across ``hmm.n_candidates``."""
        ...

    def predict(self, features: np.ndarray) -> RegimeState:
        """Infer regime for the latest bar. Uses data up to last row only."""
        ...

    def _select_by_bic(self, features: np.ndarray) -> "GaussianHMM":
        """Fit each candidate state count and return the lowest-BIC model."""
        ...

    def _bic(self, model: "GaussianHMM", features: np.ndarray, ll: float) -> float:
        """Bayesian information criterion for a fitted model."""
        ...

    def _label_states(self, features: np.ndarray) -> dict[int, str]:
        """Map raw states to low/mid/high ordered by state realized vol."""
        ...

    def _passes_filters(self, state: int, confidence: float) -> bool:
        """Stability + flicker + min-confidence gate for a raw state flip."""
        ...
