"""Strategy Tester — Layer 4: Cross-Sectional Momentum Check.

Standalone. In the main sweep, momentum was tested on each asset by itself and
scored near zero. But the strongest documented form is *cross-sectional* —
ranking assets against each other. This builds that and compares, honestly.

Every 21 trading days rank all assets by trailing return, go long the top third
and short the bottom third (equal weight), hold to the next rebalance, and pay
the same per-asset costs on turnover. Lookbacks: 3m (63d), 6m (126d), and the
standard 12-1 (t-252 → t-21, skipping the last month to dodge short-term
reversal). Scored the same way as the sweep — in-sample vs out-of-sample split,
walk-forward OOS Sharpe + drawdown — so it's directly comparable.

Reports what it actually shows (no tuning) and writes xsec_momentum.csv.

Run:
    python layer4_xsec_momentum.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import layer1_data_strategies as L
import layer2_backtest as B

OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xsec_momentum.csv")
REBALANCE = 21           # ~monthly
MIN_BREADTH = 6          # need at least this many ranked assets to trade

# lookback -> function(prices) giving trailing-return signal at each date
LOOKBACKS = {
    "3m": lambda p: p / p.shift(63) - 1,
    "6m": lambda p: p / p.shift(126) - 1,
    "12-1m": lambda p: p.shift(21) / p.shift(252) - 1,  # 12m minus most recent month
}


# --------------------------------------------------------------------------- #
def build_panel(data: dict) -> pd.DataFrame:
    """Aligned Close-price panel (dates x assets) on the union calendar,
    forward-filled. Leading pre-listing gaps stay NaN (excluded from ranking)."""
    prices = pd.DataFrame({t: df["Close"] for t, df in data.items()}).sort_index()
    return prices.ffill()


def xsec_returns(prices: pd.DataFrame, signal: pd.DataFrame,
                 cost_vec: pd.Series) -> pd.Series:
    """Net daily return of the long-top-third / short-bottom-third portfolio."""
    rets = prices.pct_change().fillna(0.0)
    assets = list(prices.columns)
    idx = prices.index
    n = len(idx)

    target = pd.DataFrame(0.0, index=idx, columns=assets)
    rebalanced = pd.Series(False, index=idx)
    for i in range(252, n, REBALANCE):
        d = idx[i]
        valid = signal.loc[d].dropna()
        valid = valid[np.isfinite(valid)]
        if len(valid) < MIN_BREADTH:
            continue
        m = max(1, len(valid) // 3)
        ranked = valid.sort_values()
        w = pd.Series(0.0, index=assets)
        w[ranked.index[-m:]] = 1.0 / m     # long strongest third
        w[ranked.index[:m]] = -1.0 / m     # short weakest third
        target.loc[d] = w.values
        rebalanced[d] = True

    # hold between rebalances, then lag one bar for execution (no look-ahead)
    target = target.where(rebalanced, np.nan).ffill().fillna(0.0)
    exec_w = target.shift(1).fillna(0.0)

    gross = (exec_w * rets).sum(axis=1)
    turnover = exec_w.diff().abs()
    turnover.iloc[0] = exec_w.iloc[0].abs()
    cost = (turnover * cost_vec).sum(axis=1)
    net = gross - cost
    # start once the book is actually on
    first = exec_w.abs().sum(axis=1).gt(0).idxmax()
    return net.loc[first:]


def wf_returns(r: pd.Series, n_windows: int = 5, is_frac: float = 0.70) -> dict:
    """Same 5-window 70/30 walk-forward as the sweep, on a ready return series."""
    r = r.dropna()
    n = len(r)
    bounds = np.linspace(0, n, n_windows + 1).astype(int)
    is_parts, oos_parts, per_window = [], [], []
    for i in range(n_windows):
        a, b = bounds[i], bounds[i + 1]
        if b - a < 20:
            continue
        split = a + int((b - a) * is_frac)
        is_parts.append(r.iloc[a:split])
        oos_parts.append(r.iloc[split:b])
        per_window.append(round(B.sharpe(r.iloc[split:b]), 2))
    is_r, oos_r = pd.concat(is_parts), pd.concat(oos_parts)
    return dict(is_sharpe=B.sharpe(is_r), oos_sharpe=B.sharpe(oos_r),
                oos_maxdd=B.max_drawdown(oos_r), full_sharpe=B.sharpe(r),
                full_maxdd=B.max_drawdown(r), per_window_oos=per_window)


# --------------------------------------------------------------------------- #
def single_asset_momentum_baseline(sweep: pd.DataFrame) -> dict:
    """Single-asset time-series momentum result from the main sweep."""
    ts = sweep[sweep["strategy"] == "ts_momentum"]
    if ts.empty:
        return dict(mean=float("nan"), median=float("nan"), best=float("nan"), n=0)
    return dict(mean=float(ts["oos_sharpe"].mean()),
                median=float(ts["oos_sharpe"].median()),
                best=float(ts["oos_sharpe"].max()), n=int(len(ts)))


def main() -> None:
    print("Loading universe…")
    data = L.download_universe(verbose=False)
    print(f"  {len(data)} assets.")
    prices = build_panel(data)
    cost_vec = pd.Series({t: (B.Config.crypto_cost if t in B.CRYPTO else B.Config.cost)
                          for t in prices.columns})

    import db
    sweep = db.load_sweep()
    if sweep is None and os.path.exists(B.SWEEP_CSV):
        sweep = pd.read_csv(B.SWEEP_CSV)
    base = single_asset_momentum_baseline(sweep) if sweep is not None else \
        dict(mean=float("nan"), median=float("nan"), best=float("nan"), n=0)

    rows = []
    for name, fn in LOOKBACKS.items():
        sig = fn(prices)
        net = xsec_returns(prices, sig, cost_vec)
        wf = wf_returns(net)
        rows.append(dict(lookback=name, **wf))
    res = pd.DataFrame(rows)

    # persist
    save = res.drop(columns=["per_window_oos"]).copy()
    save["per_window_oos"] = res["per_window_oos"].apply(lambda x: " ".join(map(str, x)))
    save.to_csv(OUT_CSV, index=False)
    if db.available():
        try:
            _save_db(db, save)
            print("Wrote xsec_momentum to MySQL + CSV.")
        except Exception:
            print("Wrote xsec_momentum.csv (MySQL save skipped).")
    else:
        print("Wrote xsec_momentum.csv.")

    # ---- report ----
    line = "=" * 70
    print(f"\n{line}\nCROSS-SECTIONAL MOMENTUM  (long top third / short bottom third)\n{line}")
    print(f"  {'lookback':10}{'OOS Sharpe':>12}{'OOS maxDD':>12}{'full Sharpe':>13}{'full maxDD':>12}")
    for _, r in res.iterrows():
        print(f"  {r['lookback']:10}{r['oos_sharpe']:>12.2f}{r['oos_maxdd']:>12.0%}"
              f"{r['full_sharpe']:>13.2f}{r['full_maxdd']:>12.0%}")

    print(f"\n{line}\nSIDE BY SIDE vs SINGLE-ASSET MOMENTUM (ts_momentum, main sweep)\n{line}")
    print(f"  single-asset ts_momentum OOS Sharpe: mean {base['mean']:.2f}  "
          f"median {base['median']:.2f}  best {base['best']:.2f}  (n={base['n']})")
    best_x = res.loc[res["oos_sharpe"].idxmax()]
    print(f"  cross-sectional best OOS Sharpe:     {best_x['oos_sharpe']:.2f}  "
          f"({best_x['lookback']})")
    verdict = ("Ranking assets against each other BEAT single-asset momentum."
               if best_x["oos_sharpe"] > base["mean"] + 0.1 else
               "Cross-sectional did NOT clearly beat single-asset momentum here.")
    print(f"  → {verdict}")

    print(f"\n{line}\nREGIME DEPENDENCE  (OOS Sharpe per walk-forward window)\n{line}")
    for _, r in res.iterrows():
        pw = r["per_window_oos"]
        lean = "leans on one window" if (max(pw) - min(pw)) > 1.0 or \
            sum(1 for x in pw if x > 0) <= len(pw) // 2 else "spread across windows"
        print(f"  {r['lookback']:8} windows {pw}   → {lean}")
    print(f"\n  Drawdowns reported as they come out (cross-sectional momentum runs deep).")
    print(line)


def _save_db(db, save: pd.DataFrame) -> None:
    conn = db.connect(create_db=True)
    try:
        with conn.cursor() as cur:
            cur.execute("""CREATE TABLE IF NOT EXISTS xsec_momentum (
                lookback VARCHAR(16), is_sharpe DOUBLE, oos_sharpe DOUBLE,
                oos_maxdd DOUBLE, full_sharpe DOUBLE, full_maxdd DOUBLE,
                per_window_oos VARCHAR(128))""")
            cur.execute("TRUNCATE TABLE xsec_momentum")
            for _, r in save.iterrows():
                cur.execute(
                    "INSERT INTO xsec_momentum (lookback,is_sharpe,oos_sharpe,"
                    "oos_maxdd,full_sharpe,full_maxdd,per_window_oos) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    (r["lookback"], float(r["is_sharpe"]), float(r["oos_sharpe"]),
                     float(r["oos_maxdd"]), float(r["full_sharpe"]),
                     float(r["full_maxdd"]), r["per_window_oos"]))
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
