"""Technical indicators and HMM observable features.

Pure functions computing OHLCV-derived features for the volatility-regime HMM,
plus a thin ``FeatureEngineer`` wrapper that assembles the standardized feature
matrix.

CAUSALITY: every function is strictly causal — a value at row t depends only on
data at or before t. No centered windows, no ``shift(-k)``. This is enforced by
``tests/test_look_ahead.py``. The final feature matrix leads with a return-like
column (``RET_COL``) and carries a realized-vol column (``VOL_COL``) so the
engine can order regimes by return and describe them by volatility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Column indices into the assembled feature matrix (order matters downstream).
FEATURE_COLUMNS = [
    "ret1",        # 0  return-like  -> RET_COL
    "ret5",        # 1
    "ret20",       # 2
    "rvol20",      # 3  realized vol -> VOL_COL
    "vol_ratio",   # 4
    "vol_z",       # 5  volume z-score
    "vol_trend",   # 6
    "adx14",       # 7
    "sma50_slope", # 8
    "rsi_z",       # 9
    "dist_200",    # 10
    "roc10",       # 11
    "roc20",       # 12
    "natr14",      # 13
]
RET_COL = 0
VOL_COL = 3

ZSCORE_WINDOW = 252


# --------------------------------------------------------------------- returns
def log_return(close: pd.Series, period: int = 1) -> pd.Series:
    """Log return over ``period`` bars."""
    return np.log(close / close.shift(period))


def roc(close: pd.Series, period: int = 10) -> pd.Series:
    """Rate of change (percent) over ``period`` bars."""
    return close.pct_change(period)


# ------------------------------------------------------------------ volatility
def realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling standard deviation of 1-bar log returns."""
    return log_return(close, 1).rolling(window).std()


def vol_ratio(close: pd.Series, fast: int = 5, slow: int = 20) -> pd.Series:
    """Ratio of short-window to long-window realized volatility."""
    return realized_vol(close, fast) / realized_vol(close, slow)


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """Wilder average true range."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def normalized_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """ATR as a fraction of price."""
    return atr(high, low, close, window) / close


# ---------------------------------------------------------------------- volume
def volume_zscore(volume: pd.Series, window: int = 50) -> pd.Series:
    """Volume z-score versus a rolling mean/std."""
    mean = volume.rolling(window).mean()
    std = volume.rolling(window).std()
    return (volume - mean) / std


def volume_trend(volume: pd.Series, window: int = 10) -> pd.Series:
    """Slope of the short volume SMA, normalized by a longer volume mean."""
    sma = volume.rolling(window).mean()
    return sma.diff() / volume.rolling(window * 5).mean()


# ----------------------------------------------------------------------- trend
def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """Average directional index (Wilder), trend-strength in [0, 100]."""
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    tr = atr(high, low, close, window)  # already Wilder-smoothed TR
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False).mean() / tr
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / window, adjust=False).mean()


def sma_slope(close: pd.Series, window: int = 50) -> pd.Series:
    """1-bar slope of the ``window``-period SMA, normalized by price."""
    sma = close.rolling(window).mean()
    return sma.diff() / close


# --------------------------------------------------------------- mean reversion
def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder relative strength index in [0, 100]."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def dist_from_sma(close: pd.Series, window: int = 200) -> pd.Series:
    """Distance from the ``window``-period SMA as a fraction of price."""
    return (close - close.rolling(window).mean()) / close


# ------------------------------------------------------------ standardization
def rolling_zscore(series: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    """Causal rolling z-score over ``window`` bars."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


class FeatureEngineer:
    """Assemble the standardized HMM observable feature matrix from OHLCV."""

    def __init__(self, zscore_window: int = ZSCORE_WINDOW) -> None:
        self.zscore_window = zscore_window
        self.columns = list(FEATURE_COLUMNS)

    def raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Unstandardized feature frame (before rolling z-scoring)."""
        close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
        feats = pd.DataFrame(index=df.index)
        feats["ret1"] = log_return(close, 1)
        feats["ret5"] = log_return(close, 5)
        feats["ret20"] = log_return(close, 20)
        feats["rvol20"] = realized_vol(close, 20)
        feats["vol_ratio"] = vol_ratio(close, 5, 20)
        feats["vol_z"] = volume_zscore(vol, 50)
        feats["vol_trend"] = volume_trend(vol, 10)
        feats["adx14"] = adx(high, low, close, 14)
        feats["sma50_slope"] = sma_slope(close, 50)
        feats["rsi_z"] = rsi(close, 14)
        feats["dist_200"] = dist_from_sma(close, 200)
        feats["roc10"] = roc(close, 10)
        feats["roc20"] = roc(close, 20)
        feats["natr14"] = normalized_atr(high, low, close, 14)
        return feats[self.columns]

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardized feature matrix (rolling z-scored), warm-up rows dropped.

        Result columns are ``FEATURE_COLUMNS``; the return-like column is at
        ``RET_COL`` and realized vol at ``VOL_COL``.
        """
        raw = self.raw_features(df)
        std = raw.apply(lambda s: rolling_zscore(s, self.zscore_window))
        return std.replace([np.inf, -np.inf], np.nan).dropna()

    def feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """``build_features`` as a contiguous float64 numpy array."""
        return np.ascontiguousarray(self.build_features(df).to_numpy(dtype=float))
