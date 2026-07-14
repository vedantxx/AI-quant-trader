"""Technical indicators and HMM feature computation.

All features are causal: each row depends only on data at or before that row
(``tests/test_look_ahead.py`` enforces this). The HMM feature matrix leads with
a return-like column so the engine can order states by volatility.

Stub scaffold: signatures, type hints, and docstrings only — no logic yet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Compute causal technical features for the HMM."""

    def log_returns(self, close: pd.Series) -> pd.Series:
        """Log returns of the close series."""
        ...

    def realized_vol(self, close: pd.Series, window: int = 20) -> pd.Series:
        """Rolling standard deviation of log returns."""
        ...

    def rsi(self, close: pd.Series, window: int = 14) -> pd.Series:
        """Relative strength index."""
        ...

    def atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Average true range."""
        ...

    def momentum(self, close: pd.Series, window: int = 10) -> pd.Series:
        """Percent change over ``window`` bars."""
        ...

    def trend_flag(self, close: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
        """True where the fast SMA is above the slow SMA (uptrend)."""
        ...

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """HMM feature matrix; column 0 is return-like. NaN warm-up dropped."""
        ...

    def feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """``build_features`` as a numpy array."""
        ...
