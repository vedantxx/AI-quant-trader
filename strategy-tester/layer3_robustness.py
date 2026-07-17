"""Strategy Tester — Layer 3: Robustness Checks.

Builds on the funnel. A strategy can survive the six filters and still be
fragile — working only by luck or on one magic parameter. Two checks catch it:

  * PARAMETER SENSITIVITY — per family, the spread of OOS Sharpe across all its
    settings (mean, std, fraction positive). A tight spread with a high positive
    fraction means the edge is real, not curve-fit to one lucky number.
  * BOOTSTRAP STRESS TEST — for each survivor, resample its OOS daily returns a
    few hundred times to get a distribution of outcomes instead of the single
    path history happened to take. Report p5/p50/p95 Sharpe and the worst-case
    drawdown, and flag each survivor solid or fragile.

Reads the sweep from MySQL (CSV fallback); writes both result tables back to
MySQL and to CSV so they can be charted later.

Run:
    python layer3_robustness.py
    python layer3_robustness.py --n-boot 500 --fragile-dd -0.40
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import layer1_data_strategies as L
import layer2_backtest as B

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PARAM_CSV = os.path.join(OUT_DIR, "param_sensitivity.csv")
BOOT_CSV = os.path.join(OUT_DIR, "bootstrap_results.csv")


# --------------------------------------------------------------------------- #
# 1. Parameter sensitivity
# --------------------------------------------------------------------------- #
def parameter_sensitivity(sweep: pd.DataFrame) -> pd.DataFrame:
    """Per family: spread of OOS Sharpe across all its settings."""
    g = sweep.groupby("strategy")
    rows = []
    for strat, sub in g:
        rows.append(dict(
            strategy=strat,
            category=sub["category"].iloc[0],
            n_configs=int(sub["config"].nunique()),
            mean_oos=float(sub["oos_sharpe"].mean()),
            std_oos=float(sub["oos_sharpe"].std(ddof=0)),
            frac_pos=float((sub["oos_sharpe"] > 0).mean()),
        ))
    df = pd.DataFrame(rows)
    # robust = decent mean, tight spread, mostly positive
    df["robust"] = (df["mean_oos"] > 0) & (df["frac_pos"] >= 0.6)
    return df.sort_values(["robust", "mean_oos"], ascending=[False, False])


# --------------------------------------------------------------------------- #
# 2. Bootstrap stress test
# --------------------------------------------------------------------------- #
def bootstrap(oos_r: pd.Series, n_boot: int, seed: int = 0) -> dict:
    """Resample OOS returns with replacement n_boot times; distribution of
    Sharpe and the worst-case drawdown across resamples."""
    r = oos_r.to_numpy()
    r = r[~np.isnan(r)]
    if len(r) < 20:
        return dict(p5_sharpe=0.0, p50_sharpe=0.0, p95_sharpe=0.0, worst_dd=0.0)
    rng = np.random.default_rng(seed)
    n = len(r)
    sharpes = np.empty(n_boot)
    dds = np.empty(n_boot)
    for i in range(n_boot):
        s = r[rng.integers(0, n, n)]
        sd = s.std()
        sharpes[i] = 0.0 if sd == 0 else s.mean() / sd * B.ANN
        eq = np.cumprod(1 + s)
        dds[i] = (eq / np.maximum.accumulate(eq) - 1).min()
    return dict(
        p5_sharpe=float(np.percentile(sharpes, 5)),
        p50_sharpe=float(np.percentile(sharpes, 50)),
        p95_sharpe=float(np.percentile(sharpes, 95)),
        worst_dd=float(dds.min()),  # deepest drawdown seen across all resamples
    )


def stress_survivors(sweep: pd.DataFrame, cfg: B.Config, n_boot: int,
                     fragile_dd: float, verbose: bool = True) -> pd.DataFrame:
    survivors = B.apply_filters(sweep, cfg)
    survivors = survivors[survivors["survived"]].sort_values(
        "oos_sharpe", ascending=False)
    if survivors.empty:
        return pd.DataFrame()

    data = L.download_universe(verbose=False)
    cfg_map = {name: (fn, params) for name, fn, params, _ in L.build_configs()}

    rows = []
    for _, s in survivors.iterrows():
        name, asset = s["config"], s["asset"]
        if name not in cfg_map or asset not in data:
            continue
        fn, params = cfg_map[name]
        oos_r = B.oos_returns(data[asset], fn, params, cfg, asset)
        bt = bootstrap(oos_r, n_boot)
        verdict = "solid" if bt["worst_dd"] > fragile_dd else "fragile"
        rows.append(dict(config=name, asset=asset, category=s["category"],
                         oos_sharpe=float(s["oos_sharpe"]), **bt, verdict=verdict))
        if verbose:
            print(f"  {verdict:8} {name[:42]:42} {asset:8} "
                  f"OOS {s['oos_sharpe']:.2f}  worstDD {bt['worst_dd']:.0%}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def report(param: pd.DataFrame, boot: pd.DataFrame, fragile_dd: float) -> None:
    line = "=" * 74
    print(f"\n{line}\nPARAMETER SENSITIVITY  (spread of OOS Sharpe across a family's settings)\n{line}")
    print(f"  {'strategy':22}{'cat':11}{'cfgs':>5}{'mean':>8}{'std':>7}{'frac+':>7}  robust")
    for _, r in param.iterrows():
        print(f"  {r['strategy']:22}{r['category']:11}{int(r['n_configs']):>5}"
              f"{r['mean_oos']:>8.2f}{r['std_oos']:>7.2f}{r['frac_pos']:>7.0%}"
              f"  {'yes' if r['robust'] else ''}")
    print("  (robust = mean OOS Sharpe > 0 and >=60% of settings positive → not curve-fit)")

    print(f"\n{line}\nBOOTSTRAP STRESS TEST  ({'no survivors' if boot.empty else str(len(boot)) + ' survivors, resampled'})\n{line}")
    if boot.empty:
        print("  no survivors to stress — run Layer 2 first.")
        print(line)
        return
    solid = (boot["verdict"] == "solid").sum()
    print(f"  solid {solid} / {len(boot)}   fragile {len(boot) - solid} / {len(boot)}   "
          f"(fragile = worst-case bootstrap DD below {fragile_dd:.0%})")
    show = boot.sort_values("p50_sharpe", ascending=False).head(25)
    print(f"\n  {'config':40}{'asset':7}{'OOS':>6}{'p5':>7}{'p50':>7}{'p95':>7}{'worstDD':>9}  verdict")
    for _, r in show.iterrows():
        print(f"  {r['config'][:39]:40}{r['asset']:7}{r['oos_sharpe']:>6.2f}"
              f"{r['p5_sharpe']:>7.2f}{r['p50_sharpe']:>7.2f}{r['p95_sharpe']:>7.2f}"
              f"{r['worst_dd']:>9.0%}  {r['verdict']}")
    print(line)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--fragile-dd", type=float, default=-0.40)
    # funnel thresholds (must match the sweep's intended filters)
    ap.add_argument("--min-sharpe", type=float, default=B.Config.min_sharpe)
    ap.add_argument("--max-dd", type=float, default=B.Config.max_dd)
    a = ap.parse_args()
    cfg = B.Config(min_sharpe=a.min_sharpe, max_dd=a.max_dd)

    import db
    sweep = db.load_sweep()
    src = "MySQL"
    if sweep is None and os.path.exists(B.SWEEP_CSV):
        sweep, src = pd.read_csv(B.SWEEP_CSV), "CSV"
    if sweep is None:
        raise SystemExit("No sweep found — run layer2_backtest.py first.")
    print(f"Loaded {len(sweep):,} sweep rows from {src}.")

    param = parameter_sensitivity(sweep)
    print("\nStress-testing survivors "
          f"(n_boot={a.n_boot}, fragile if worst DD < {a.fragile_dd:.0%})…")
    boot = stress_survivors(sweep, cfg, a.n_boot, a.fragile_dd)

    # persist: MySQL primary, CSV for charting
    if db.available():
        db.save_param_sensitivity(param)
        if not boot.empty:
            db.save_bootstrap(boot)
        print("Wrote param_sensitivity + bootstrap_results to MySQL.")
    param.to_csv(PARAM_CSV, index=False)
    boot.to_csv(BOOT_CSV, index=False)
    print(f"Wrote {os.path.basename(PARAM_CSV)} and {os.path.basename(BOOT_CSV)}.")

    report(param, boot, a.fragile_dd)


if __name__ == "__main__":
    main()
