"""HMM regime detection engine — a VOLATILITY / market-state CLASSIFIER.

The HMM classifies the market into a small number of hidden states from
engineered observable features. It does NOT predict price direction; the
strategy layer consumes the classification to set allocation.

Key design points:
  * Automatic model selection by BIC over n_components in ``hmm.n_candidates``.
  * Regimes labeled by ascending mean return (human-readable names only — the
    strategy layer orders regimes by volatility independently).
  * Live inference uses the FORWARD algorithm (filtered posterior
    P(state_t | obs_1:t)) — never Viterbi (``model.predict``) — so no future
    data leaks into past regime estimates. This is the single most important
    correctness property; see ``tests/test_look_ahead.py``.
  * Regime changes are confirmed only after persisting ``stability_bars``; a
    high flicker rate forces an uncertainty mode.
"""

from __future__ import annotations

import logging
import pickle
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:  # keep import-time safe before deps installed
    GaussianHMM = None

logger = logging.getLogger("regime-trader.hmm")

# Return-sorted labels per selected regime count (ascending mean return).
REGIME_LABELS: dict[int, list[str]] = {
    3: ["BEAR", "NEUTRAL", "BULL"],
    4: ["CRASH", "BEAR", "BULL", "EUPHORIA"],
    5: ["CRASH", "BEAR", "NEUTRAL", "BULL", "EUPHORIA"],
    6: ["CRASH", "STRONG_BEAR", "WEAK_BEAR", "WEAK_BULL", "STRONG_BULL", "EUPHORIA"],
    7: [
        "CRASH", "STRONG_BEAR", "WEAK_BEAR", "NEUTRAL",
        "WEAK_BULL", "STRONG_BULL", "EUPHORIA",
    ],
}

# Column indices into the feature matrix (must match feature_engineering).
RET_COL = 0
VOL_COL = 3

RECOMMENDED_MIN_TRAIN_BARS = 504  # ~2 years of daily data


@dataclass
class RegimeInfo:
    """Static description of a labeled regime, derived at training time."""

    regime_id: int
    regime_name: str
    expected_return: float
    expected_volatility: float
    recommended_strategy_type: str
    max_leverage_allowed: float
    max_position_size_pct: float
    min_confidence_to_act: float


@dataclass
class RegimeState:
    """Filtered regime estimate for a single bar (live/backtest inference)."""

    label: str
    state_id: int
    probability: float
    state_probabilities: np.ndarray
    timestamp: Optional[datetime] = None
    is_confirmed: bool = False
    consecutive_bars: int = 0
    is_flickering: bool = False


@dataclass
class ModelMetadata:
    """Persisted alongside the model for provenance and auditing."""

    n_regimes: int
    bic: float
    all_bic: dict
    training_date: str
    labels: list
    feature_dim: int
    log_likelihood: float
    converged: bool
    iterations: int = field(default=0)


class HMMEngine:
    """Gaussian-HMM volatility/market-state classifier with filtered inference."""

    def __init__(self, cfg: dict) -> None:
        h = cfg["hmm"]
        self.n_candidates: list[int] = list(h["n_candidates"])
        self.n_init: int = h["n_init"]
        self.covariance_type: str = h["covariance_type"]
        self.min_train_bars: int = h["min_train_bars"]
        self.stability_bars: int = h["stability_bars"]
        self.flicker_window: int = h["flicker_window"]
        self.flicker_threshold: int = h["flicker_threshold"]
        self.min_confidence: float = h["min_confidence"]

        self.model: Optional["GaussianHMM"] = None
        self.metadata: Optional[ModelMetadata] = None
        self.state_to_label: dict[int, str] = {}
        self.regime_info: dict[int, RegimeInfo] = {}

        self._reset_filter_state()

    # ------------------------------------------------------------- filter state
    def _reset_filter_state(self) -> None:
        self._confirmed_state: Optional[int] = None
        self._consecutive: int = 0
        self._pending_state: Optional[int] = None
        self._pending_count: int = 0
        self._last_change_confirmed: bool = False
        self._change_bars: deque[int] = deque(maxlen=self.flicker_window)
        self._bar_index: int = 0

    # -------------------------------------------------------------------- train
    def fit(
        self, features: np.ndarray, returns: Optional[np.ndarray] = None
    ) -> "HMMEngine":
        """Select a model by BIC, fit it, and label regimes by mean return.

        ``features`` is (T, n_feat), standardized and causal. ``returns`` is an
        optional (T,) raw-return series used to order/describe regimes; if
        omitted, the standardized return column (``RET_COL``) is used.
        """
        if GaussianHMM is None:
            raise ImportError("hmmlearn not installed")
        if len(features) < self.min_train_bars:
            raise ValueError(
                f"need >= {self.min_train_bars} bars, got {len(features)}"
            )
        if len(features) < RECOMMENDED_MIN_TRAIN_BARS:
            logger.warning(
                "training on %d bars (< recommended %d)",
                len(features), RECOMMENDED_MIN_TRAIN_BARS,
            )

        model, meta = self._select_by_bic(features)
        self.model = model
        self.metadata = meta
        self._assign_labels(features, returns)
        self._reset_filter_state()
        logger.info(
            "selected %d-regime model  BIC=%.1f  logL=%.1f  labels=%s",
            meta.n_regimes, meta.bic, meta.log_likelihood,
            [self.state_to_label[i] for i in range(meta.n_regimes)],
        )
        return self

    def _select_by_bic(self, X: np.ndarray) -> tuple["GaussianHMM", ModelMetadata]:
        """Fit each candidate count with n_init restarts; pick lowest BIC."""
        best_model, best_meta, best_bic = None, None, np.inf
        all_bic: dict[int, float] = {}

        for n in self.n_candidates:
            best_for_n, best_ll = None, -np.inf
            for seed in range(self.n_init):
                m = GaussianHMM(
                    n_components=n,
                    covariance_type=self.covariance_type,
                    n_iter=200,
                    random_state=seed,
                )
                try:
                    m.fit(X)
                    ll = m.score(X)
                except Exception as exc:  # singular cov, non-convergence, etc.
                    logger.debug("fit failed n=%d seed=%d: %s", n, seed, exc)
                    continue
                if np.isfinite(ll) and ll > best_ll:
                    best_ll, best_for_n = ll, m
            if best_for_n is None:
                logger.warning("no successful fit for n=%d", n)
                continue

            bic = self._bic(best_for_n, X, best_ll)
            all_bic[n] = bic
            logger.info("candidate n=%d  BIC=%.1f  logL=%.1f", n, bic, best_ll)
            if bic < best_bic:
                best_bic, best_model, best_ll_final = bic, best_for_n, best_ll

        if best_model is None:
            raise RuntimeError("HMM fit failed for all candidates")

        meta = ModelMetadata(
            n_regimes=best_model.n_components,
            bic=best_bic,
            all_bic=all_bic,
            training_date=datetime.now(timezone.utc).isoformat(),
            labels=[],  # filled in _assign_labels
            feature_dim=X.shape[1],
            log_likelihood=best_ll_final,
            converged=bool(best_model.monitor_.converged),
            iterations=len(best_model.monitor_.history),
        )
        return best_model, meta

    def _bic(self, model: "GaussianHMM", X: np.ndarray, ll: float) -> float:
        """BIC = -2*logL + n_params*log(n_samples), full-covariance param count."""
        n, d = model.n_components, X.shape[1]
        n_params = (
            (n * n - 1)                    # transition matrix (rows sum to 1)
            + (n - 1)                      # start probabilities
            + (n * d)                      # means
            + (n * d * (d + 1) // 2)       # full covariances
        )
        return -2 * ll + n_params * np.log(len(X))

    def _assign_labels(
        self, features: np.ndarray, returns: Optional[np.ndarray]
    ) -> None:
        """Sort states by mean return (ascending) and attach labels + metadata."""
        n = self.model.n_components
        states = self.model.predict(features)  # Viterbi OK here: labeling only

        ret_series = (
            returns if returns is not None else features[:, RET_COL]
        )
        mean_ret, mean_vol = {}, {}
        for s in range(n):
            mask = states == s
            if mask.any():
                mean_ret[s] = float(np.nanmean(ret_series[mask]))
                mean_vol[s] = float(np.nanstd(features[mask, RET_COL]))
            else:  # fall back to model params for empty states
                mean_ret[s] = float(self.model.means_[s, RET_COL])
                mean_vol[s] = float(np.sqrt(self._state_var(s, RET_COL)))

        order = sorted(range(n), key=lambda s: mean_ret[s])  # ascending return
        labels = REGIME_LABELS[n]

        self.state_to_label, self.regime_info = {}, {}
        for rank, s in enumerate(order):
            name = labels[rank]
            self.state_to_label[s] = name
            self.regime_info[s] = RegimeInfo(
                regime_id=s,
                regime_name=name,
                expected_return=mean_ret[s],
                expected_volatility=mean_vol[s],
                recommended_strategy_type=self._strategy_for(name),
                max_leverage_allowed=self._leverage_for(name),
                max_position_size_pct=self._max_size_for(name),
                min_confidence_to_act=self.min_confidence,
            )
        if self.metadata is not None:
            self.metadata.labels = [self.state_to_label[i] for i in range(n)]

    def _state_var(self, s: int, col: int) -> float:
        """Variance of feature ``col`` in state ``s`` across covariance types."""
        cov = self.model.covars_[s]
        if cov.ndim == 2:            # full / tied
            return float(cov[col, col])
        return float(np.atleast_1d(cov)[col])  # diag / spherical

    @staticmethod
    def _strategy_for(name: str) -> str:
        if "BULL" in name or name == "EUPHORIA":
            return "trend_following"
        if "BEAR" in name or name == "CRASH":
            return "defensive"
        return "mean_reversion"

    @staticmethod
    def _leverage_for(name: str) -> float:
        return {"CRASH": 0.0, "EUPHORIA": 1.0}.get(
            name, 1.25 if "BULL" in name or name == "NEUTRAL" else 0.5
        )

    @staticmethod
    def _max_size_for(name: str) -> float:
        if name in ("CRASH", "EUPHORIA"):
            return 0.05
        if "STRONG" in name or name in ("BULL", "NEUTRAL"):
            return 0.15
        return 0.10

    # --------------------------------------------------- filtered (forward) core
    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        """Log emission probabilities, shape (T, n_components)."""
        n = self.model.n_components
        out = np.empty((len(X), n))
        for s in range(n):
            out[:, s] = multivariate_normal.logpdf(
                X,
                mean=self.model.means_[s],
                cov=self._full_cov(s),
                allow_singular=True,
            )
        return out

    def _full_cov(self, s: int) -> np.ndarray:
        """Full covariance matrix for state ``s`` regardless of covariance_type."""
        cov = self.model.covars_[s]
        if cov.ndim == 2:
            return cov
        d = self.model.means_.shape[1]
        return np.eye(d) * np.atleast_1d(cov)

    def _forward_log(self, X: np.ndarray) -> np.ndarray:
        """Filtered log-posteriors log P(state_t | obs_1:t), shape (T, n).

        Pure forward algorithm in log space. Row t uses observations 0..t only,
        so the value at any index is invariant to data appended after it — the
        property that eliminates look-ahead bias.
        """
        log_start = np.log(self.model.startprob_ + 1e-300)
        log_trans = np.log(self.model.transmat_ + 1e-300)
        log_emit = self._log_emission(X)

        T, n = log_emit.shape
        log_alpha = np.empty((T, n))

        a = log_start + log_emit[0]
        log_alpha[0] = a - logsumexp(a)                 # normalize (filtered)
        for t in range(1, T):
            # predict: log sum_i alpha_{t-1,i} * trans_{i,j}
            pred = logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0)
            a = pred + log_emit[t]
            log_alpha[t] = a - logsumexp(a)
        return log_alpha

    def predict_regime_filtered(self, features: np.ndarray) -> np.ndarray:
        """Filtered most-likely state per bar (argmax of forward posterior)."""
        if self.model is None:
            raise RuntimeError("model not fit")
        return np.argmax(self._forward_log(features), axis=1)

    def predict_regime_proba(self, features: np.ndarray) -> np.ndarray:
        """Filtered posterior distribution over states for the latest bar."""
        if self.model is None:
            raise RuntimeError("model not fit")
        return np.exp(self._forward_log(features)[-1])

    # ---------------------------------------------------------- stateful predict
    def predict(
        self, features: np.ndarray, timestamp: Optional[datetime] = None
    ) -> RegimeState:
        """Advance the live filter by one bar and return the confirmed regime.

        Applies the stability filter (a new raw state must persist
        ``stability_bars`` before it is confirmed) and the flicker guard.
        """
        proba = self.predict_regime_proba(features)
        raw_state = int(np.argmax(proba))
        confidence = float(proba[raw_state])
        self._bar_index += 1

        self._update_stability(raw_state)
        flickering = self.is_flickering()

        # confidence gate and flicker guard both force "unconfirmed"
        confirmed = (
            self._confirmed_state == raw_state
            and confidence >= self.min_confidence
            and not flickering
        )
        display_state = (
            self._confirmed_state if self._confirmed_state is not None else raw_state
        )
        label = self.state_to_label.get(display_state, "UNKNOWN")

        return RegimeState(
            label=label,
            state_id=display_state,
            probability=confidence,
            state_probabilities=proba,
            timestamp=timestamp,
            is_confirmed=confirmed,
            consecutive_bars=self._consecutive,
            is_flickering=flickering,
        )

    def _update_stability(self, raw_state: int) -> None:
        """Persist-N-bars confirmation logic; records confirmed changes."""
        self._last_change_confirmed = False

        if self._confirmed_state is None:            # first observation
            self._confirmed_state = raw_state
            self._consecutive = 1
            self._pending_state, self._pending_count = None, 0
            return

        if raw_state == self._confirmed_state:
            self._consecutive += 1
            self._pending_state, self._pending_count = None, 0
            return

        # raw state differs -> accumulate persistence before confirming
        if raw_state == self._pending_state:
            self._pending_count += 1
        else:
            self._pending_state, self._pending_count = raw_state, 1

        if self._pending_count >= self.stability_bars:
            prev = self._confirmed_state
            self._confirmed_state = raw_state
            self._consecutive = self._pending_count
            self._pending_state, self._pending_count = None, 0
            self._last_change_confirmed = True
            self._change_bars.append(self._bar_index)
            logger.warning(
                "regime change confirmed: %s -> %s (bar %d)",
                self.state_to_label.get(prev), self.state_to_label.get(raw_state),
                self._bar_index,
            )
        else:
            logger.info(
                "regime transition pending: %s (%d/%d)",
                self.state_to_label.get(raw_state),
                self._pending_count, self.stability_bars,
            )

    # ----------------------------------------------------------- query methods
    def detect_regime_change(self) -> bool:
        """True only if the most recent ``predict`` confirmed a regime change."""
        return self._last_change_confirmed

    def get_regime_stability(self) -> int:
        """Consecutive bars spent in the current confirmed regime."""
        return self._consecutive

    def get_transition_matrix(self) -> np.ndarray:
        """Learned state transition probability matrix."""
        if self.model is None:
            raise RuntimeError("model not fit")
        return self.model.transmat_

    def get_regime_flicker_rate(self) -> int:
        """Confirmed regime changes within the trailing flicker window."""
        cutoff = self._bar_index - self.flicker_window
        return sum(1 for b in self._change_bars if b > cutoff)

    def is_flickering(self) -> bool:
        """True if the flicker rate exceeds the configured threshold."""
        return self.get_regime_flicker_rate() > self.flicker_threshold

    # ------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        """Pickle the model, labels, regime info, and metadata."""
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "state_to_label": self.state_to_label,
                    "regime_info": self.regime_info,
                    "metadata": self.metadata,
                },
                f,
            )

    def load(self, path: str) -> "HMMEngine":
        """Restore a pickled model bundle and reset live filter state.

        SECURITY: pickle is used because hmmlearn/sklearn models are not
        JSON-serializable. Only load bundles produced by ``save`` on trusted,
        self-generated artifacts — never a model file from an untrusted source.
        """
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        self.model = bundle["model"]
        self.state_to_label = bundle["state_to_label"]
        self.regime_info = bundle["regime_info"]
        self.metadata = bundle["metadata"]
        self._reset_filter_state()
        return self
