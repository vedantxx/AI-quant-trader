"""HMM regime detection engine.

Fits a Gaussian HMM over engineered market features, selects the best state
count by BIC across ``hmm.n_candidates``, and labels each state as a volatility
regime (low / mid / high) ordered by state realized volatility. Includes
stability and flicker filters so a raw state flip does not immediately become a
tradeable regime change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:  # keep import-time safe before deps installed
    GaussianHMM = None


@dataclass
class RegimeResult:
    """Output of a single regime inference step."""

    state: int                       # raw HMM state index
    label: str                       # "low" | "mid" | "high"
    confidence: float                # posterior prob of the chosen state
    accepted: bool                   # passed stability + confidence filters
    posteriors: np.ndarray = field(default_factory=lambda: np.array([]))


class HMMEngine:
    def __init__(self, cfg: dict):
        h = cfg["hmm"]
        self.n_candidates = h["n_candidates"]
        self.n_init = h["n_init"]
        self.covariance_type = h["covariance_type"]
        self.min_train_bars = h["min_train_bars"]
        self.stability_bars = h["stability_bars"]
        self.flicker_window = h["flicker_window"]
        self.flicker_threshold = h["flicker_threshold"]
        self.min_confidence = h["min_confidence"]

        self.model: Optional["GaussianHMM"] = None
        self.state_to_label: dict[int, str] = {}
        self._recent_states: list[int] = []      # for flicker detection
        self._pending_state: Optional[int] = None
        self._pending_count: int = 0
        self._current_label: str = "mid"

    # ------------------------------------------------------------------ fit
    def fit(self, features: np.ndarray) -> "HMMEngine":
        """Fit HMM, choosing state count by BIC. ``features`` is (T, n_feat)."""
        if GaussianHMM is None:
            raise ImportError("hmmlearn not installed")
        if len(features) < self.min_train_bars:
            raise ValueError(
                f"need >= {self.min_train_bars} bars, got {len(features)}"
            )

        best_model, best_bic = None, np.inf
        for n in self.n_candidates:
            model = GaussianHMM(
                n_components=n,
                covariance_type=self.covariance_type,
                n_iter=200,
                random_state=42,
            )
            best_for_n, best_ll = None, -np.inf
            for seed in range(self.n_init):
                m = GaussianHMM(
                    n_components=n,
                    covariance_type=self.covariance_type,
                    n_iter=200,
                    random_state=seed,
                )
                try:
                    m.fit(features)
                    ll = m.score(features)
                except Exception:
                    continue
                if ll > best_ll:
                    best_ll, best_for_n = ll, m
            if best_for_n is None:
                continue
            bic = self._bic(best_for_n, features, best_ll)
            if bic < best_bic:
                best_bic, best_model = bic, best_for_n

        if best_model is None:
            raise RuntimeError("HMM fit failed for all candidates")

        self.model = best_model
        self._assign_labels(features)
        return self

    def _bic(self, model: "GaussianHMM", X: np.ndarray, ll: float) -> float:
        n, d = model.n_components, X.shape[1]
        # transitions + start probs + means + covariances (full)
        n_params = (n * n - 1) + (n - 1) + (n * d) + (n * d * (d + 1) // 2)
        return -2 * ll + n_params * np.log(len(X))

    def _assign_labels(self, features: np.ndarray) -> None:
        """Map raw states to vol labels ordered by state realized vol."""
        states = self.model.predict(features)
        # assume feature column 0 is a return-like series; use its std per state
        vol_by_state = {}
        for s in range(self.model.n_components):
            mask = states == s
            vol_by_state[s] = features[mask, 0].std() if mask.any() else 0.0
        ordered = sorted(vol_by_state, key=vol_by_state.get)
        n = len(ordered)
        self.state_to_label = {}
        for rank, s in enumerate(ordered):
            if rank < n / 3:
                self.state_to_label[s] = "low"
            elif rank < 2 * n / 3:
                self.state_to_label[s] = "mid"
            else:
                self.state_to_label[s] = "high"

    # -------------------------------------------------------------- predict
    def predict(self, features: np.ndarray) -> RegimeResult:
        """Infer regime for the latest bar. ``features`` is (T, n_feat).

        Only information up to the last row is used — no future leakage.
        """
        if self.model is None:
            raise RuntimeError("model not fit")
        posteriors = self.model.predict_proba(features)[-1]
        state = int(np.argmax(posteriors))
        confidence = float(posteriors[state])

        self._recent_states.append(state)
        if len(self._recent_states) > self.flicker_window:
            self._recent_states.pop(0)

        accepted = self._apply_filters(state, confidence)
        label = self.state_to_label.get(state, "mid")
        return RegimeResult(
            state=state,
            label=self._current_label if not accepted else label,
            confidence=confidence,
            accepted=accepted,
            posteriors=posteriors,
        )

    def _apply_filters(self, state: int, confidence: float) -> bool:
        """Stability + flicker + confidence gate."""
        if confidence < self.min_confidence:
            self._pending_state, self._pending_count = None, 0
            return False

        # flicker: too many distinct switches in the window -> freeze
        switches = sum(
            1
            for a, b in zip(self._recent_states, self._recent_states[1:])
            if a != b
        )
        if switches >= self.flicker_threshold:
            return False

        label = self.state_to_label.get(state, "mid")
        if label == self._current_label:
            self._pending_state, self._pending_count = None, 0
            return True

        # require new regime to persist stability_bars before accepting
        if state == self._pending_state:
            self._pending_count += 1
        else:
            self._pending_state, self._pending_count = state, 1

        if self._pending_count >= self.stability_bars:
            self._current_label = label
            self._pending_state, self._pending_count = None, 0
            return True
        return False
