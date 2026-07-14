"""Technical indicators and HMM feature computation.

All features are causal: each row depends only on data at or before that row.
``test_look_ahead.py`` enforces this. The HMM feature matrix leads with a
return-like column (used by the engine to order states by volatility).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    return log_returns(close).rolling(window).std()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def trend_flag(close: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    """True where fast SMA is above slow SMA (uptrend)."""
    return close.rolling(fast).mean() > close.rolling(slow).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """HMM feature matrix. Column order matters: [0] must be return-like.

    Returns a frame with NaN rows (warm-up) dropped.
    """
    close = df["close"]
    feats = pd.DataFrame(index=df.index)
    feats["ret"] = log_returns(close)                 # col 0: return-like
    feats["vol20"] = realized_vol(close, 20)
    feats["rsi14"] = rsi(close, 14) / 100.0
    feats["atr14"] = atr(df, 14) / close
    feats["mom10"] = close.pct_change(10)
    return feats.dropna()


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return build_features(df).to_numpy()
