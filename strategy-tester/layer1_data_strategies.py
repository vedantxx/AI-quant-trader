"""Strategy Tester — Layer 1: Data & Strategy Library.

The foundation for a four-layer strategy-testing system. This layer:

  1. Downloads daily OHLCV for ~30 liquid assets (2010-2025) via yfinance.
  2. Implements the full popular-retail strategy spectrum (~47 strategies
     across six families), each returning a daily position in {-1, 0, 1}
     with NO look-ahead (positions are shifted one bar before use).
  3. Exposes ``build_configs()`` -> list of (name, func, params, category)
     whose parameter grids expand into several hundred configs, so running
     every config across every asset yields thousands of backtests.

Importable by later layers:

    from layer1_data_strategies import download_universe, build_configs, STRATEGIES

Run directly to download the universe (cached to ./data) and print the
config/backtest counts:

    python layer1_data_strategies.py
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Universe
# --------------------------------------------------------------------------- #
START = "2010-01-01"
END = "2025-01-01"
MIN_BARS = 500
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

UNIVERSE: List[str] = [
    # index ETFs
    "SPY", "QQQ", "IWM", "DIA",
    # sector ETFs
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLU", "XLY", "XLP",
    # commodities / rates / international
    "GLD", "USO", "TLT", "HYG", "EFA", "EEM", "EWZ",
    # crypto
    "BTC-USD", "ETH-USD",
    # large caps
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "JPM",
]


def _load_env() -> None:
    """Load repo-root .env into os.environ (stdlib; no python-dotenv needed).
    Never overwrites already-set vars; values are never logged."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, ".env")
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            if k and k not in os.environ:
                os.environ[k] = v.strip().strip('"').strip("'")


# Polygon uses prefixed symbols for crypto/forex; equities/ETFs pass through.
_POLYGON_SYMBOL = {"BTC-USD": "X:BTCUSD", "ETH-USD": "X:ETHUSD"}


def _polygon_ohlcv(ticker: str, key: str, retries: int = 3) -> pd.DataFrame | None:
    """Daily adjusted OHLCV from Polygon aggregates via stdlib urllib.
    Returns None on any failure so the caller can fall back to yfinance."""
    import json
    import ssl
    import time
    import urllib.error
    import urllib.request

    try:  # python.org builds lack system CA certs; use certifi (ships w/ yfinance)
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()

    sym = _POLYGON_SYMBOL.get(ticker, ticker)
    url = (f"https://api.polygon.io/v2/aggs/ticker/{sym}/range/1/day/"
           f"{START}/{END}?adjusted=true&sort=asc&limit=50000&apiKey={key}")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30, context=ctx) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code == 429:  # rate limited (free tier: 5 req/min)
                time.sleep(15 * (attempt + 1))
                continue
            return None
        except Exception:
            return None
        res = data.get("results")
        if not res:
            return None
        df = pd.DataFrame(res).rename(
            columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
        df.index = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(None).dt.normalize()
        return df[["Open", "High", "Low", "Close", "Volume"]]
    return None


def _flatten_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """Reduce a yfinance frame (possibly MultiIndex columns) to OHLCV."""
    if isinstance(raw.columns, pd.MultiIndex):
        # pick whichever level holds the field names (Close/Open/…)
        lvl = 0 if "Close" in raw.columns.get_level_values(0) else 1
        raw = raw.copy()
        raw.columns = raw.columns.get_level_values(lvl)
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
    return raw[cols].dropna()


def download_universe(force: bool = False, verbose: bool = True,
                      retries: int = 3) -> Dict[str, pd.DataFrame]:
    """Download daily OHLCV for the universe, cached as CSV under ./data.

    Returns {ticker: DataFrame[Open, High, Low, Close, Volume]}. Assets with
    fewer than MIN_BARS rows are skipped. Cached so reruns skip the network
    (and dodge yfinance rate limits).
    """
    import time

    _load_env()
    pkey = os.environ.get("POLYGON_API_KEY")
    os.makedirs(DATA_DIR, exist_ok=True)
    out: Dict[str, pd.DataFrame] = {}
    for tkr in UNIVERSE:
        cache = os.path.join(DATA_DIR, f"{tkr.replace('-', '_')}.csv")
        df = None
        if not force and os.path.exists(cache):
            try:
                df = pd.read_csv(cache, index_col=0, parse_dates=True)
            except Exception:
                df = None
        if df is None and pkey:  # primary source: Polygon
            p = _polygon_ohlcv(tkr, pkey)
            if p is not None and len(p) >= MIN_BARS:  # else free-tier stub -> yfinance
                df = p
                df.to_csv(cache)
        if df is None:  # backup source: yfinance
            import yfinance as yf
            for attempt in range(retries):
                try:
                    raw = yf.download(tkr, start=START, end=END, auto_adjust=True,
                                      progress=False, threads=False)
                except Exception:
                    raw = pd.DataFrame()
                df = _flatten_ohlcv(raw) if len(raw) else None
                if df is not None and len(df):
                    df.to_csv(cache)
                    break
                time.sleep(2 * (attempt + 1))  # back off on empty / rate limit
        if df is None or len(df) < MIN_BARS:
            if verbose:
                print(f"  skip {tkr}: {0 if df is None else len(df)} bars (< {MIN_BARS})")
            continue
        df.index = pd.to_datetime(df.index)
        out[tkr] = df
        if verbose:
            print(f"  {tkr:9s} {len(df):5d} bars  {df.index[0].date()} .. {df.index[-1].date()}")
    return out


# --------------------------------------------------------------------------- #
# Indicator helpers (all strictly causal — use data up to and including t)
# --------------------------------------------------------------------------- #
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing (RMA)."""
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    h, l, pc = df["High"], df["Low"], df["Close"].shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return rma(true_range(df), n)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = rma(d.clip(lower=0), n)
    dn = rma((-d).clip(lower=0), n)
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def bbands(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    sd = close.rolling(n).std(ddof=0)
    return mid - k * sd, mid, mid + k * sd


def macd(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    line = ema(close, fast) - ema(close, slow)
    signal = ema(line, sig)
    return line, signal, line - signal


def adx(df: pd.DataFrame, n: int = 14):
    up = df["High"].diff()
    dn = -df["Low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(df)
    atr_n = rma(tr, n)
    plus_di = 100 * rma(pd.Series(plus_dm, index=df.index), n) / atr_n
    minus_di = 100 * rma(pd.Series(minus_dm, index=df.index), n) / atr_n
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return plus_di, minus_di, rma(dx.fillna(0), n)


def stoch(df: pd.DataFrame, k: int = 14, d: int = 3):
    ll = df["Low"].rolling(k).min()
    hh = df["High"].rolling(k).max()
    pk = 100 * (df["Close"] - ll) / (hh - ll).replace(0, np.nan)
    return pk, pk.rolling(d).mean()


def cci(df: pd.DataFrame, n: int = 20):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    m = tp.rolling(n).mean()
    md = (tp - m).abs().rolling(n).mean()
    return (tp - m) / (0.015 * md.replace(0, np.nan))


def williams_r(df: pd.DataFrame, n: int = 14):
    hh = df["High"].rolling(n).max()
    ll = df["Low"].rolling(n).min()
    return -100 * (hh - df["Close"]) / (hh - ll).replace(0, np.nan)


def keltner(df: pd.DataFrame, n: int = 20, mult: float = 2.0):
    mid = ema(df["Close"], n)
    rng = mult * atr(df, n)
    return mid - rng, mid, mid + rng


def donchian(df: pd.DataFrame, n: int = 20):
    return df["Low"].rolling(n).min(), df["High"].rolling(n).max()


def obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["Close"].diff().fillna(0))
    return (sign * df["Volume"]).cumsum()


def adl(df: pd.DataFrame) -> pd.Series:
    hl = (df["High"] - df["Low"]).replace(0, np.nan)
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl
    return (clv.fillna(0) * df["Volume"]).cumsum()


def cmf(df: pd.DataFrame, n: int = 20) -> pd.Series:
    hl = (df["High"] - df["Low"]).replace(0, np.nan)
    mfv = (((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl).fillna(0) * df["Volume"]
    return mfv.rolling(n).sum() / df["Volume"].rolling(n).sum().replace(0, np.nan)


def mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    rmf = tp * df["Volume"]
    pos = rmf.where(tp > tp.shift(1), 0.0)
    neg = rmf.where(tp < tp.shift(1), 0.0)
    mr = pos.rolling(n).sum() / neg.rolling(n).sum().replace(0, np.nan)
    return (100 - 100 / (1 + mr)).fillna(50)


def linreg_slope(s: pd.Series, n: int) -> pd.Series:
    x = np.arange(n)
    xm = x.mean()
    denom = ((x - xm) ** 2).sum()

    def _slope(w):
        return ((x - xm) * (w - w.mean())).sum() / denom

    return s.rolling(n).apply(_slope, raw=True)


def hull(close: pd.Series, n: int) -> pd.Series:
    half = max(1, int(n / 2))
    sq = max(1, int(round(np.sqrt(n))))
    wma = lambda s, p: s.rolling(p).apply(
        lambda w: np.dot(w, np.arange(1, p + 1)) / (p * (p + 1) / 2), raw=True)
    return wma(2 * wma(close, half) - wma(close, n), sq)


def kama(close: pd.Series, n: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    change = (close - close.shift(n)).abs()
    vol = close.diff().abs().rolling(n).sum()
    er = (change / vol.replace(0, np.nan)).fillna(0)
    sc = (er * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    out = close.copy().astype(float)
    vals = close.values
    scv = sc.values
    k = vals[0]
    for i in range(1, len(vals)):
        k = k + (scv[i] if not np.isnan(scv[i]) else 0.0) * (vals[i] - k)
        out.iloc[i] = k
    return out


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> pd.Series:
    hl2 = (df["High"] + df["Low"]) / 2
    a = atr(df, n)
    upper = hl2 + mult * a
    lower = hl2 - mult * a
    close = df["Close"].values
    up, lo = upper.values, lower.values
    dir_ = np.ones(len(df))
    fu, fl = up.copy(), lo.copy()
    for i in range(1, len(df)):
        fu[i] = min(up[i], fu[i - 1]) if close[i - 1] <= fu[i - 1] else up[i]
        fl[i] = max(lo[i], fl[i - 1]) if close[i - 1] >= fl[i - 1] else lo[i]
        if close[i] > fu[i - 1]:
            dir_[i] = 1
        elif close[i] < fl[i - 1]:
            dir_[i] = -1
        else:
            dir_[i] = dir_[i - 1]
    return pd.Series(dir_, index=df.index)


def psar(df: pd.DataFrame, step: float = 0.02, mx: float = 0.2) -> pd.Series:
    high, low = df["High"].values, df["Low"].values
    n = len(df)
    out = np.zeros(n)
    bull = True
    af = step
    ep = high[0]
    sar = low[0]
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if bull:
            if low[i] < sar:
                bull, sar, af, ep = False, ep, step, low[i]
            else:
                if high[i] > ep:
                    ep, af = high[i], min(af + step, mx)
        else:
            if high[i] > sar:
                bull, sar, af, ep = True, ep, step, high[i]
            else:
                if low[i] < ep:
                    ep, af = low[i], min(af + step, mx)
        out[i] = 1 if bull else -1
    return pd.Series(out, index=df.index)


def _breakout_hold(long_trig: pd.Series, short_trig: pd.Series, index) -> pd.Series:
    """Enter/hold position from the last opposing breakout trigger."""
    pos = pd.Series(np.nan, index=index)
    pos[long_trig.fillna(False).astype(bool)] = 1.0
    pos[short_trig.fillna(False).astype(bool)] = -1.0
    return pos.ffill().fillna(0.0)


# --------------------------------------------------------------------------- #
# Strategy registry: each strategy returns a shifted {-1,0,1} position
# --------------------------------------------------------------------------- #
STRATEGIES: Dict[str, Tuple[Callable, str]] = {}


def strategy(name: str, category: str):
    """Register a raw-signal function; the wrapper shifts one bar (no look-ahead)."""
    def deco(fn: Callable) -> Callable:
        def wrapped(df: pd.DataFrame, **p) -> pd.Series:
            raw = fn(df, **p).reindex(df.index)
            return raw.shift(1).fillna(0.0).clip(-1, 1)
        wrapped.raw = fn
        STRATEGIES[name] = (wrapped, category)
        return wrapped
    return deco


C = lambda cond: cond.astype(float)  # bool -> 0/1


# ---- TREND ---------------------------------------------------------------- #
@strategy("ma_crossover", "trend")
def ma_crossover(df, fast=20, slow=50):
    return np.sign(ema(df["Close"], fast) - ema(df["Close"], slow))


@strategy("ts_momentum", "trend")
def ts_momentum(df, lookback=90):
    return np.sign(df["Close"] - df["Close"].shift(lookback))


@strategy("roc_momentum", "trend")
def roc_momentum(df, lookback=60, thr=0.0):
    roc = df["Close"].pct_change(lookback)
    return np.sign(roc - thr) * C(roc.abs() > thr)


@strategy("macd", "trend")
def macd_strat(df, fast=12, slow=26, sig=9):
    line, signal, _ = macd(df["Close"], fast, slow, sig)
    return np.sign(line - signal)


@strategy("donchian_breakout", "trend")
def donchian_breakout(df, n=20):
    lo, hi = donchian(df, n)
    return _breakout_hold(df["Close"] > hi.shift(1), df["Close"] < lo.shift(1), df.index)


@strategy("bollinger_breakout", "trend")
def bollinger_breakout(df, n=20, k=2.0):
    lb, _, ub = bbands(df["Close"], n, k)
    return _breakout_hold(df["Close"] > ub, df["Close"] < lb, df.index)


@strategy("supertrend", "trend")
def supertrend_strat(df, n=10, mult=3.0):
    return supertrend(df, n, mult)


@strategy("parabolic_sar", "trend")
def parabolic_sar(df, step=0.02, mx=0.2):
    return psar(df, step, mx)


@strategy("adx_trend", "trend")
def adx_trend(df, n=14, thr=25):
    pdi, mdi, adx_v = adx(df, n)
    return np.sign(pdi - mdi) * C(adx_v > thr)


@strategy("ichimoku", "trend")
def ichimoku(df, conv=9, base=26, span_b=52):
    hh = lambda p: df["High"].rolling(p).max()
    ll = lambda p: df["Low"].rolling(p).min()
    tenkan = (hh(conv) + ll(conv)) / 2
    kijun = (hh(base) + ll(base)) / 2
    a = ((tenkan + kijun) / 2).shift(base)
    b = ((hh(span_b) + ll(span_b)) / 2).shift(base)
    above = (df["Close"] > a) & (df["Close"] > b)
    below = (df["Close"] < a) & (df["Close"] < b)
    return C(above) - C(below)


@strategy("linreg_slope", "trend")
def linreg_slope_strat(df, n=30):
    return np.sign(linreg_slope(df["Close"], n))


@strategy("aroon", "trend")
def aroon(df, n=25):
    up = 100 * df["High"].rolling(n + 1).apply(lambda w: (len(w) - 1 - np.argmax(w)) / n, raw=True)
    dn = 100 * df["Low"].rolling(n + 1).apply(lambda w: (len(w) - 1 - np.argmin(w)) / n, raw=True)
    return np.sign(up - dn)


@strategy("vortex", "trend")
def vortex(df, n=14):
    tr = true_range(df)
    vip = (df["High"] - df["Low"].shift(1)).abs().rolling(n).sum() / tr.rolling(n).sum()
    vim = (df["Low"] - df["High"].shift(1)).abs().rolling(n).sum() / tr.rolling(n).sum()
    return np.sign(vip - vim)


@strategy("trix", "trend")
def trix(df, n=15, sig=9):
    t = ema(ema(ema(df["Close"], n), n), n)
    tr = t.pct_change() * 100
    return np.sign(tr - tr.rolling(sig).mean())


@strategy("hull_ma", "trend")
def hull_ma(df, n=20):
    h = hull(df["Close"], n)
    return np.sign(h - h.shift(1))


@strategy("kama", "trend")
def kama_strat(df, n=10):
    k = kama(df["Close"], n)
    return np.sign(df["Close"] - k)


@strategy("turtle", "trend")
def turtle(df, entry=20, exit=10):
    lo_en, hi_en = donchian(df, entry)
    lo_ex, hi_ex = donchian(df, exit)
    long_in = df["Close"] > hi_en.shift(1)
    short_in = df["Close"] < lo_en.shift(1)
    long_out = df["Close"] < lo_ex.shift(1)   # exit long on N-low breach
    short_out = df["Close"] > hi_ex.shift(1)  # exit short on N-high breach
    p = _breakout_hold(long_in, short_in, df.index)
    p = p.where(~((p > 0) & long_out), 0.0)
    p = p.where(~((p < 0) & short_out), 0.0)
    return p


@strategy("dual_momentum", "trend")
def dual_momentum(df, lookback=252):
    # absolute momentum: long only when 12m return positive, else flat
    mom = df["Close"].pct_change(lookback)
    return C(mom > 0)


@strategy("elder_ray", "trend")
def elder_ray(df, n=13):
    e = ema(df["Close"], n)
    bull = df["High"] - e
    bear = df["Low"] - e
    up = (e > e.shift(1)) & (bear < 0) & (bull > 0)
    dn = (e < e.shift(1)) & (bull > 0.0) & (bear < 0.0) & (df["Low"] < df["Low"].shift(1))
    return C(up) - C((e < e.shift(1)) & (bull < 0))


# ---- MEAN REVERSION ------------------------------------------------------- #
@strategy("rsi_revert", "meanrev")
def rsi_revert(df, n=14, lo=30, hi=70):
    r = rsi(df["Close"], n)
    return C(r < lo) - C(r > hi)


@strategy("bollinger_revert", "meanrev")
def bollinger_revert(df, n=20, k=2.0):
    lb, _, ub = bbands(df["Close"], n, k)
    return C(df["Close"] < lb) - C(df["Close"] > ub)


@strategy("zscore_revert", "meanrev")
def zscore_revert(df, n=20, z=1.5):
    m = sma(df["Close"], n)
    sd = df["Close"].rolling(n).std(ddof=0)
    zs = (df["Close"] - m) / sd.replace(0, np.nan)
    return C(zs < -z) - C(zs > z)


@strategy("stochastic", "meanrev")
def stochastic_strat(df, k=14, d=3, lo=20, hi=80):
    pk, _ = stoch(df, k, d)
    return C(pk < lo) - C(pk > hi)


@strategy("cci_revert", "meanrev")
def cci_revert(df, n=20, thr=100):
    c = cci(df, n)
    return C(c < -thr) - C(c > thr)


@strategy("williams_r", "meanrev")
def williams_r_strat(df, n=14, lo=-80, hi=-20):
    w = williams_r(df, n)
    return C(w < lo) - C(w > hi)


@strategy("keltner_revert", "meanrev")
def keltner_revert(df, n=20, mult=2.0):
    lb, _, ub = keltner(df, n, mult)
    return C(df["Close"] < lb) - C(df["Close"] > ub)


@strategy("vwap_revert", "meanrev")
def vwap_revert(df, n=20, z=1.5):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    vwap = (tp * df["Volume"]).rolling(n).sum() / df["Volume"].rolling(n).sum()
    dev = (df["Close"] - vwap)
    sd = dev.rolling(n).std(ddof=0)
    zs = dev / sd.replace(0, np.nan)
    return C(zs < -z) - C(zs > z)


@strategy("percent_b", "meanrev")
def percent_b(df, n=20, k=2.0):
    lb, _, ub = bbands(df["Close"], n, k)
    pb = (df["Close"] - lb) / (ub - lb).replace(0, np.nan)
    return C(pb < 0.05) - C(pb > 0.95)


@strategy("connors_rsi", "meanrev")
def connors_rsi(df, lo=20, hi=80):
    r3 = rsi(df["Close"], 3)
    updown = np.sign(df["Close"].diff()).fillna(0)
    streak = updown.copy().astype(float)
    s = 0.0
    vals = updown.values
    out = np.zeros(len(vals))
    for i in range(len(vals)):
        s = s + vals[i] if vals[i] != 0 and vals[i] == np.sign(s if s != 0 else vals[i]) else vals[i]
        out[i] = s
    rstreak = rsi(pd.Series(out, index=df.index), 2)
    roc = df["Close"].pct_change()
    prank = roc.rolling(100).apply(lambda w: (w[-1] > w).mean() * 100, raw=True)
    crsi = (r3 + rstreak + prank.fillna(50)) / 3
    return C(crsi < lo) - C(crsi > hi)


@strategy("ultimate_oscillator", "meanrev")
def ultimate_oscillator(df, lo=30, hi=70):
    pc = df["Close"].shift(1)
    bp = df["Close"] - pd.concat([df["Low"], pc], axis=1).min(axis=1)
    tr = pd.concat([df["High"], pc], axis=1).max(axis=1) - pd.concat([df["Low"], pc], axis=1).min(axis=1)
    avg = lambda p: bp.rolling(p).sum() / tr.rolling(p).sum().replace(0, np.nan)
    uo = 100 * (4 * avg(7) + 2 * avg(14) + avg(28)) / 7
    return C(uo < lo) - C(uo > hi)


@strategy("gap_fade", "meanrev")
def gap_fade(df, thr=0.01):
    gap = df["Open"] / df["Close"].shift(1) - 1
    return C(gap < -thr) - C(gap > thr)


# ---- VOLUME --------------------------------------------------------------- #
@strategy("obv_trend", "volume")
def obv_trend(df, n=20):
    o = obv(df)
    return np.sign(o - o.ewm(span=n, adjust=False).mean())


@strategy("cmf", "volume")
def cmf_strat(df, n=20, thr=0.05):
    c = cmf(df, n)
    return C(c > thr) - C(c < -thr)


@strategy("mfi", "volume")
def mfi_strat(df, n=14, lo=20, hi=80):
    m = mfi(df, n)
    return C(m < lo) - C(m > hi)


@strategy("volume_surge", "volume")
def volume_surge(df, n=20, mult=2.0):
    surge = df["Volume"] > mult * sma(df["Volume"], n)
    up = df["Close"] > df["Close"].shift(1)
    return C(surge & up) - C(surge & ~up)


@strategy("force_index", "volume")
def force_index(df, n=13):
    fi = ((df["Close"] - df["Close"].shift(1)) * df["Volume"]).ewm(span=n, adjust=False).mean()
    return np.sign(fi)


@strategy("chaikin_osc", "volume")
def chaikin_osc(df, fast=3, slow=10):
    a = adl(df)
    return np.sign(ema(a, fast) - ema(a, slow))


# ---- VOLATILITY ----------------------------------------------------------- #
@strategy("atr_breakout", "volatility")
def atr_breakout(df, n=14, mult=2.0):
    a = atr(df, n)
    upper = df["Close"].shift(1) + mult * a
    lower = df["Close"].shift(1) - mult * a
    return _breakout_hold(df["Close"] > upper, df["Close"] < lower, df.index)


@strategy("volatility_breakout", "volatility")
def volatility_breakout(df, n=20, k=1.0):
    rng = (df["High"] - df["Low"]).rolling(n).mean()
    long_t = df["Close"] > df["Open"] + k * rng
    short_t = df["Close"] < df["Open"] - k * rng
    return _breakout_hold(long_t, short_t, df.index)


@strategy("squeeze_breakout", "volatility")
def squeeze_breakout(df, n=20, bb_k=2.0, kc_mult=1.5):
    lb, mid, ub = bbands(df["Close"], n, bb_k)
    kl, _, ku = keltner(df, n, kc_mult)
    squeeze_on = (lb > kl) & (ub < ku)
    fired = squeeze_on.shift(1, fill_value=False) & ~squeeze_on
    mom = df["Close"] - sma(df["Close"], n)
    long_t = fired & (mom > 0)
    short_t = fired & (mom < 0)
    return _breakout_hold(long_t, short_t, df.index)


# ---- PATTERN -------------------------------------------------------------- #
@strategy("engulfing", "pattern")
def engulfing(df, hold=5):
    o, c = df["Open"], df["Close"]
    po, pc = o.shift(1), c.shift(1)
    bull = (c > o) & (pc < po) & (c >= po) & (o <= pc)
    bear = (c < o) & (pc > po) & (c <= po) & (o >= pc)
    pos = pd.Series(np.nan, index=df.index)
    pos[bull] = 1.0
    pos[bear] = -1.0
    return pos.ffill(limit=hold).fillna(0.0)


@strategy("three_bar_reversal", "pattern")
def three_bar_reversal(df, hold=3):
    c = df["Close"]
    down2 = (c.shift(2) > c.shift(1)) & (c.shift(1) > c)
    up2 = (c.shift(2) < c.shift(1)) & (c.shift(1) < c)
    long_t = down2 & (c > c.shift(1))  # reversal up after two-down (evaluated next bar via shift)
    long_sig = (c.shift(2) > c.shift(1)) & (c > c.shift(1))
    short_sig = (c.shift(2) < c.shift(1)) & (c < c.shift(1))
    pos = pd.Series(np.nan, index=df.index)
    pos[long_sig] = 1.0
    pos[short_sig] = -1.0
    return pos.ffill(limit=hold).fillna(0.0)


@strategy("higher_highs_lows", "pattern")
def higher_highs_lows(df, n=20):
    hh = df["High"].rolling(n).max()
    ll = df["Low"].rolling(n).min()
    up = (hh > hh.shift(n)) & (ll > ll.shift(n))
    dn = (hh < hh.shift(n)) & (ll < ll.shift(n))
    return C(up) - C(dn)


@strategy("pivot_bounce", "pattern")
def pivot_bounce(df, n=20, tol=0.01):
    lo = df["Low"].rolling(n).min()
    hi = df["High"].rolling(n).max()
    near_sup = (df["Close"] - lo) / lo <= tol
    near_res = (hi - df["Close"]) / hi <= tol
    return C(near_sup) - C(near_res)


# ---- COMPOSITE ------------------------------------------------------------ #
@strategy("macd_rsi", "composite")
def macd_rsi(df, fast=12, slow=26, sig=9, rsi_n=14):
    line, signal, _ = macd(df["Close"], fast, slow, sig)
    r = rsi(df["Close"], rsi_n)
    long_ = (line > signal) & (r > 50)
    short_ = (line < signal) & (r < 50)
    return C(long_) - C(short_)


@strategy("triple_screen", "composite")
def triple_screen(df, trend_n=50, k=14, lo=30, hi=70):
    e = ema(df["Close"], trend_n)
    trend = np.sign(e - e.shift(1))
    pk, _ = stoch(df, k, 3)
    long_ = (trend > 0) & (pk < lo)
    short_ = (trend < 0) & (pk > hi)
    pos = pd.Series(np.nan, index=df.index)
    pos[long_] = 1.0
    pos[short_] = -1.0
    pos[(trend > 0) & (pk > hi)] = 0.0
    pos[(trend < 0) & (pk < lo)] = 0.0
    return pos.ffill().fillna(0.0)


@strategy("chandelier", "composite")
def chandelier(df, n=22, mult=3.0):
    hh = df["High"].rolling(n).max()
    ll = df["Low"].rolling(n).min()
    a = atr(df, n)
    long_stop = hh - mult * a
    short_stop = ll + mult * a
    long_t = df["Close"] > short_stop.shift(1)
    short_t = df["Close"] < long_stop.shift(1)
    return _breakout_hold(long_t, short_t, df.index)


# --------------------------------------------------------------------------- #
# Parameter grids -> configs
# --------------------------------------------------------------------------- #
def _grid(**kw):
    """Cartesian product of keyword lists -> list of param dicts."""
    keys = list(kw)
    out = [{}]
    for k in keys:
        out = [dict(d, **{k: v}) for d in out for v in kw[k]]
    return out


GRIDS: Dict[str, List[dict]] = {
    # trend
    "ma_crossover": [dict(fast=f, slow=s) for f in [5, 10, 20, 50]
                     for s in [20, 50, 100, 200] if f < s],
    "ts_momentum": _grid(lookback=[20, 60, 90, 120, 180, 252]),
    "roc_momentum": _grid(lookback=[20, 60, 120, 200], thr=[0.0, 0.02]),
    "macd": _grid(fast=[8, 12], slow=[21, 26], sig=[9]),
    "donchian_breakout": _grid(n=[20, 40, 55, 100]),
    "bollinger_breakout": _grid(n=[20, 50], k=[2.0, 2.5, 3.0]),
    "supertrend": _grid(n=[7, 10, 14], mult=[2.0, 3.0]),
    "parabolic_sar": _grid(step=[0.02, 0.03], mx=[0.2]),
    "adx_trend": _grid(n=[14, 20], thr=[20, 25, 30]),
    "ichimoku": _grid(conv=[9], base=[26], span_b=[52]),
    "linreg_slope": _grid(n=[20, 50, 100]),
    "aroon": _grid(n=[14, 25]),
    "vortex": _grid(n=[14, 21]),
    "trix": _grid(n=[9, 15], sig=[9]),
    "hull_ma": _grid(n=[16, 25, 49]),
    "kama": _grid(n=[10, 20]),
    "turtle": _grid(entry=[20, 55], exit=[10, 20]),
    "dual_momentum": _grid(lookback=[126, 200, 252]),
    "elder_ray": _grid(n=[13, 21]),
    # mean reversion
    "rsi_revert": _grid(n=[2, 3, 14], lo=[20, 30], hi=[70, 80]),
    "bollinger_revert": _grid(n=[20, 50], k=[2.0, 2.5, 3.0]),
    "zscore_revert": _grid(n=[20, 50], z=[1.0, 1.5, 2.0]),
    "stochastic": _grid(k=[14, 21], d=[3], lo=[20], hi=[80]),
    "cci_revert": _grid(n=[14, 20], thr=[100, 150]),
    "williams_r": _grid(n=[14, 28]),
    "keltner_revert": _grid(n=[20], mult=[1.5, 2.0, 2.5]),
    "vwap_revert": _grid(n=[20, 50], z=[1.5, 2.0]),
    "percent_b": _grid(n=[20, 50], k=[2.0]),
    "connors_rsi": _grid(lo=[10, 20], hi=[80, 90]),
    "ultimate_oscillator": _grid(lo=[30], hi=[70]),
    "gap_fade": _grid(thr=[0.005, 0.01, 0.02]),
    # volume
    "obv_trend": _grid(n=[20, 50]),
    "cmf": _grid(n=[20, 50], thr=[0.05, 0.1]),
    "mfi": _grid(n=[14], lo=[20], hi=[80]),
    "volume_surge": _grid(n=[20], mult=[1.5, 2.0, 3.0]),
    "force_index": _grid(n=[13, 21]),
    "chaikin_osc": _grid(fast=[3], slow=[10]),
    # volatility
    "atr_breakout": _grid(n=[14, 20], mult=[1.5, 2.0, 3.0]),
    "volatility_breakout": _grid(n=[20, 50], k=[0.5, 1.0, 1.5]),
    "squeeze_breakout": _grid(n=[20], bb_k=[2.0], kc_mult=[1.0, 1.5]),
    # pattern
    "engulfing": _grid(hold=[3, 5, 10]),
    "three_bar_reversal": _grid(hold=[3, 5]),
    "higher_highs_lows": _grid(n=[10, 20, 50]),
    "pivot_bounce": _grid(n=[20, 50], tol=[0.01, 0.02]),
    # composite
    "macd_rsi": _grid(fast=[8, 12], slow=[21, 26], sig=[9], rsi_n=[14]),
    "triple_screen": _grid(trend_n=[50, 100], k=[14], lo=[20, 30], hi=[70, 80]),
    "chandelier": _grid(n=[22], mult=[2.0, 3.0]),
}


def build_configs() -> List[Tuple[str, Callable, dict, str]]:
    """Expand grids into (name, func, params, category) tuples."""
    configs: List[Tuple[str, Callable, dict, str]] = []
    for name, (fn, category) in STRATEGIES.items():
        for params in GRIDS.get(name, [{}]):
            tag = ",".join(f"{k}={v}" for k, v in params.items()) or "default"
            configs.append((f"{name}[{tag}]", fn, params, category))
    return configs


def config_summary(configs) -> pd.DataFrame:
    df = pd.DataFrame([(c[3], c[0]) for c in configs], columns=["category", "name"])
    by_cat = df.groupby("category").size().rename("configs")
    by_strat = df["name"].str.extract(r"^([^\[]+)")[0].nunique()
    return by_cat, by_strat


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    configs = build_configs()
    by_cat, n_strats = config_summary(configs)
    print("=" * 60)
    print("STRATEGY LIBRARY")
    print("=" * 60)
    print(f"  strategies : {n_strats}")
    print(f"  configs    : {len(configs)}")
    print("  by category:")
    for cat, n in by_cat.items():
        print(f"    {cat:12s} {n:4d}")
    print(f"  assets     : {len(UNIVERSE)}")
    print(f"  backtests  : {len(configs) * len(UNIVERSE):,} (configs x assets)")
    print("=" * 60)

    if os.environ.get("SKIP_DOWNLOAD"):
        print("SKIP_DOWNLOAD set — not downloading.")
    else:
        print("Downloading universe (cached to ./data)…")
        data = download_universe()
        print(f"Loaded {len(data)} assets with >= {MIN_BARS} bars.")
        print(f"Total backtests over loaded assets: {len(configs) * len(data):,}")
