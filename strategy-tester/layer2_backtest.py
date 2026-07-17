"""Strategy Tester — Layer 2: Backtest & the Six-Filter Funnel.

Builds on Layer 1. For every (config x asset):

  * backtest naked: strat return = position * asset return - per-side cost;
  * walk-forward: 5 sequential windows, 70% in-sample / 30% out-of-sample,
    stitch the 5 OOS tails into one series and score it — that stitched OOS
    number is the one that matters, because the strategy never saw it;
  * record IS Sharpe, OOS Sharpe, OOS max drawdown, trade count.

Then a six-filter funnel strips out everything weak, risky, or overfit and
reports the attrition, survival by category/family, and the top survivors.

Run:
    python layer2_backtest.py                 # full sweep -> sweep_results.csv
    python layer2_backtest.py --cost 0.0002 --crypto-cost 0.001
    python layer2_backtest.py --min-sharpe 0.5 --max-dd -0.35
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

import layer1_data_strategies as L

CRYPTO = {"BTC-USD", "ETH-USD"}
ANN = np.sqrt(252)
SWEEP_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_results.csv")


# --------------------------------------------------------------------------- #
# Config (all thresholds/costs configurable)
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    cost: float = 0.0001          # per-side transaction cost (1 bp)
    crypto_cost: float = 0.0010   # per-side cost for crypto (10 bp)
    n_windows: int = 5
    is_frac: float = 0.70
    # six filters
    max_dd: float = -0.35         # 1: OOS max drawdown better than this
    min_sharpe: float = 0.5       # 2: OOS Sharpe above this
    max_sharpe: float = 2.5       # 3: OOS Sharpe below this (too good = the asset)
    overfit_gap: float = 0.30     # 4: OOS Sharpe not > IS Sharpe * (1+gap)
    min_trades: int = 30          # 5: at least this many OOS trades
    # 6: IS Sharpe > 0


# --------------------------------------------------------------------------- #
# Core metrics (on any daily return series)
# --------------------------------------------------------------------------- #
def sharpe(r: pd.Series) -> float:
    if len(r) < 2:
        return 0.0
    s = r.std(ddof=0)
    return 0.0 if s == 0 or np.isnan(s) else float(r.mean() / s * ANN)


def max_drawdown(r: pd.Series) -> float:
    if len(r) == 0:
        return 0.0
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def count_trades(pos: pd.Series) -> int:
    if len(pos) == 0:
        return 0
    d = pos.diff()
    d.iloc[0] = pos.iloc[0]  # establishing the initial position is a trade
    return int((d != 0).sum())


def strat_returns(pos: pd.Series, ret: pd.Series, cost: float) -> pd.Series:
    """Position * return, minus per-side cost on every unit of turnover.
    pos is already lagged one bar in Layer 1, so pos[t]*ret[t] is causal."""
    turn = pos.diff().abs()
    turn.iloc[0] = abs(pos.iloc[0])
    return pos * ret - turn * cost


# --------------------------------------------------------------------------- #
# Walk-forward: 5 windows, 70/30, stitch the OOS tails
# --------------------------------------------------------------------------- #
def walk_forward(pos: pd.Series, ret: pd.Series, cost: float,
                 n_windows: int, is_frac: float, return_series: bool = False):
    n = len(pos)
    bounds = np.linspace(0, n, n_windows + 1).astype(int)
    is_parts: List[pd.Series] = []
    oos_parts: List[pd.Series] = []
    oos_trades = 0
    for i in range(n_windows):
        a, b = bounds[i], bounds[i + 1]
        if b - a < 20:
            continue
        split = a + int((b - a) * is_frac)
        wpos, wret = pos.iloc[a:b], ret.iloc[a:b]
        sr = strat_returns(wpos, wret, cost)
        k = split - a
        is_parts.append(sr.iloc[:k])
        oos_parts.append(sr.iloc[k:])
        oos_trades += count_trades(wpos.iloc[k:])
    if not oos_parts:
        empty = pd.Series(dtype=float)
        stats = dict(is_sharpe=0.0, oos_sharpe=0.0, oos_maxdd=0.0, trades=0)
        return (stats, empty) if return_series else stats
    is_r = pd.concat(is_parts)
    oos_r = pd.concat(oos_parts)
    stats = dict(is_sharpe=sharpe(is_r), oos_sharpe=sharpe(oos_r),
                 oos_maxdd=max_drawdown(oos_r), trades=oos_trades)
    return (stats, oos_r) if return_series else stats


def oos_returns(df: pd.DataFrame, fn, params: dict, cfg: "Config",
                asset: str = "") -> pd.Series:
    """Stitched out-of-sample daily return series for one config on one asset."""
    ret = df["Close"].pct_change().fillna(0.0)
    cost = cfg.crypto_cost if asset in CRYPTO else cfg.cost
    pos = fn(df, **params)
    _, oos_r = walk_forward(pos, ret, cost, cfg.n_windows, cfg.is_frac,
                            return_series=True)
    return oos_r


# --------------------------------------------------------------------------- #
# Sweep: every config x asset
# --------------------------------------------------------------------------- #
def run_sweep(data: Dict[str, pd.DataFrame], cfg: Config,
              verbose: bool = True) -> pd.DataFrame:
    configs = L.build_configs()
    total = len(configs) * len(data)
    if verbose:
        print(f"Sweeping {len(configs)} configs x {len(data)} assets = {total:,} backtests…")
    rows = []
    done = 0
    for name, fn, params, category in configs:
        strat = name.split("[")[0]
        for asset, df in data.items():
            ret = df["Close"].pct_change().fillna(0.0)
            cost = cfg.crypto_cost if asset in CRYPTO else cfg.cost
            try:
                pos = fn(df, **params)
                wf = walk_forward(pos, ret, cost, cfg.n_windows, cfg.is_frac)
            except Exception:
                wf = dict(is_sharpe=0.0, oos_sharpe=0.0, oos_maxdd=0.0, trades=0)
            rows.append(dict(config=name, strategy=strat, category=category,
                             asset=asset, **wf))
            done += 1
        if verbose and done % 1000 < len(data):
            print(f"  {done:,}/{total:,}", end="\r")
    if verbose:
        print(f"  {total:,}/{total:,} done.        ")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Six-filter funnel
# --------------------------------------------------------------------------- #
def apply_filters(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    f1 = df["oos_maxdd"] > cfg.max_dd
    f2 = df["oos_sharpe"] > cfg.min_sharpe
    f3 = df["oos_sharpe"] < cfg.max_sharpe
    f4 = df["oos_sharpe"] <= df["is_sharpe"] * (1 + cfg.overfit_gap)
    f5 = df["trades"] >= cfg.min_trades
    f6 = df["is_sharpe"] > 0
    out = df.copy()
    out["f1_dd"], out["f2_floor"], out["f3_ceiling"] = f1, f2, f3
    out["f4_overfit"], out["f5_trades"], out["f6_is_pos"] = f4, f5, f6
    out["survived"] = f1 & f2 & f3 & f4 & f5 & f6
    return out


def funnel_report(df: pd.DataFrame, cfg: Config) -> None:
    n = len(df)
    pos_oos = (df["oos_sharpe"] > 0).sum()
    cleared = (df["oos_sharpe"] > cfg.min_sharpe).sum()
    r = apply_filters(df, cfg)
    surv = r["survived"].sum()

    line = "=" * 64
    print(f"\n{line}\nTHE FUNNEL\n{line}")
    print(f"  total backtests            {n:>8,}")
    print(f"  positive OOS Sharpe        {pos_oos:>8,}  ({pos_oos/n:6.1%})")
    print(f"  cleared OOS Sharpe > {cfg.min_sharpe:<4} {cleared:>8,}  ({cleared/n:6.1%})")
    print(f"  survived all six filters   {surv:>8,}  ({surv/n:6.1%})")

    print(f"\n{line}\nATTRITION (each filter alone, on all backtests)\n{line}")
    for col, label in [("f1_dd", f"1 OOS maxDD > {cfg.max_dd:.0%}"),
                       ("f2_floor", f"2 OOS Sharpe > {cfg.min_sharpe}"),
                       ("f3_ceiling", f"3 OOS Sharpe < {cfg.max_sharpe}"),
                       ("f4_overfit", f"4 OOS <= IS*(1+{cfg.overfit_gap:.0%})"),
                       ("f5_trades", f"5 trades >= {cfg.min_trades}"),
                       ("f6_is_pos", "6 IS Sharpe > 0")]:
        p = r[col].sum()
        print(f"  {label:28} pass {p:>7,}  ({p/n:6.1%})")

    print(f"\n{line}\nSURVIVAL BY CATEGORY\n{line}")
    _rate_table(r, "category")
    print(f"\n{line}\nSURVIVAL BY FAMILY (strategy)\n{line}")
    _rate_table(r, "strategy", top=15)

    surv_df = r[r["survived"]].sort_values("oos_sharpe", ascending=False)
    print(f"\n{line}\nTOP SURVIVORS\n{line}")
    if surv_df.empty:
        print("  none survived — loosen thresholds or the edge isn't there.")
    else:
        show = surv_df.head(25)[["config", "asset", "category",
                                 "is_sharpe", "oos_sharpe", "oos_maxdd", "trades"]]
        with pd.option_context("display.max_columns", None, "display.width", 200,
                               "display.float_format", lambda x: f"{x:.2f}"):
            print(show.to_string(index=False))
    print(line)


def _rate_table(r: pd.DataFrame, key: str, top: int | None = None) -> None:
    g = r.groupby(key).agg(n=("survived", "size"),
                           survived=("survived", "sum"),
                           mean_oos=("oos_sharpe", "mean"))
    g["rate"] = g["survived"] / g["n"]
    g = g.sort_values("rate", ascending=False)
    if top:
        g = g.head(top)
    print(f"  {key:16} {'n':>7} {'surv':>6} {'rate':>7} {'mean_oos':>9}")
    for k, row in g.iterrows():
        print(f"  {str(k):16} {int(row['n']):>7} {int(row['survived']):>6} "
              f"{row['rate']:>6.1%} {row['mean_oos']:>9.2f}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=Config.cost)
    ap.add_argument("--crypto-cost", type=float, default=Config.crypto_cost)
    ap.add_argument("--windows", type=int, default=Config.n_windows)
    ap.add_argument("--is-frac", type=float, default=Config.is_frac)
    ap.add_argument("--max-dd", type=float, default=Config.max_dd)
    ap.add_argument("--min-sharpe", type=float, default=Config.min_sharpe)
    ap.add_argument("--max-sharpe", type=float, default=Config.max_sharpe)
    ap.add_argument("--overfit-gap", type=float, default=Config.overfit_gap)
    ap.add_argument("--min-trades", type=int, default=Config.min_trades)
    ap.add_argument("--reuse", action="store_true", help="reuse existing sweep_results.csv")
    a = ap.parse_args()
    cfg = Config(cost=a.cost, crypto_cost=a.crypto_cost, n_windows=a.windows,
                 is_frac=a.is_frac, max_dd=a.max_dd, min_sharpe=a.min_sharpe,
                 max_sharpe=a.max_sharpe, overfit_gap=a.overfit_gap,
                 min_trades=a.min_trades)

    import db

    sweep = None
    if a.reuse:
        sweep = db.load_sweep()
        if sweep is None and os.path.exists(SWEEP_CSV):
            sweep = pd.read_csv(SWEEP_CSV)
        if sweep is not None:
            print(f"Reusing saved sweep ({len(sweep):,} rows).")
    if sweep is None:
        print("Loading universe (cached)…")
        data = L.download_universe(verbose=False)
        print(f"  {len(data)} assets loaded.")
        sweep = run_sweep(data, cfg)
        if db.available():
            db.save_sweep(sweep)
            print(f"Wrote {len(sweep):,} rows to MySQL (sweep_results).")
        else:
            sweep.to_csv(SWEEP_CSV, index=False)
            print(f"MySQL unavailable — wrote {SWEEP_CSV} ({len(sweep):,} rows).")

    funnel_report(sweep, cfg)


if __name__ == "__main__":
    main()
