# Strategy Tester

A four-layer system that runs thousands of backtests across every popular
retail strategy, then puts each through walk-forward validation, realistic
costs, and robustness checks to surface the few that actually hold up.

Each layer is one runnable script, importable by the next.

## Layer 1 — Data & Strategy Library (`layer1_data_strategies.py`)

The foundation.

- **Data.** Daily OHLCV, 2010-01-01 → 2025-01-01, ~30 liquid assets: index
  ETFs, sector ETFs, commodities/rates/international, crypto, and large caps.
  **Primary source Polygon** (`POLYGON_API_KEY` in repo-root `.env`, loaded with
  a stdlib parser — no python-dotenv), **yfinance as backup**. A Polygon result
  under 500 bars (free-tier history cap) is discarded and yfinance fills in.
  Assets with < 500 bars are skipped. Cached to `./data/*.csv` so reruns skip
  the network. Keys are read from env only, never logged or committed.
- **Strategies.** 47 strategies across six families — **trend, meanrev,
  volume, volatility, pattern, composite** — each a function
  `f(df, **params) -> position∈{-1,0,1}` (long / flat / short). Every signal is
  shifted one bar, so today's position uses only data up to yesterday: **no
  look-ahead** (verified — a signal recomputed on a truncated series matches
  exactly on the overlap).
- **Configs.** `build_configs()` expands per-family parameter grids into
  **181 configs**. Across the loaded universe that is **~5,000 backtests** —
  each strategy tested *naked* (raw signal, no stops/sizing) so you see what the
  base edge is actually worth.

### Run

```bash
pip install -r requirements.txt
python layer1_data_strategies.py          # downloads universe, prints counts
SKIP_DOWNLOAD=1 python layer1_data_strategies.py   # counts only, no network
```

### Import from later layers

```python
from layer1_data_strategies import download_universe, build_configs, STRATEGIES

data = download_universe()                # {ticker: OHLCV DataFrame}
for name, fn, params, category in build_configs():
    position = fn(data["SPY"], **params)  # shifted {-1,0,1} series
```

## Layer 2 — Backtest & the Funnel (`layer2_backtest.py`)

The core. Scores every strategy on data it never saw, then six filters strip
out everything weak, risky, or overfit.

- **Backtest.** Strat return = position × next-day asset return − per-side cost
  (1 bp default, configurable; 10 bp for crypto). Metrics on any return series:
  annualized Sharpe (`mean/std·√252`), max drawdown, trade count.
- **Walk-forward (the part that matters).** Each asset split into 5 sequential
  windows; within each, first 70% in-sample / last 30% out-of-sample. Keep only
  the OOS tails, stitch the 5 into one series, score Sharpe + maxDD on it — the
  strategy was never tuned on it.
- **Sweep.** Every config × asset → `sweep_results.csv` with IS Sharpe, OOS
  Sharpe, OOS maxDD, trade count.
- **Six-filter funnel** (all thresholds configurable; survive = pass ALL):
  1. OOS maxDD better than −35%
  2. OOS Sharpe > 0.5
  3. OOS Sharpe < 2.5 (higher = the asset did the work, not the strategy)
  4. OOS Sharpe not > IS Sharpe × 1.30 (a big gap is the overfit signature)
  5. ≥ 30 OOS trades (statistical meaning)
  6. IS Sharpe > 0
- **Report.** Attrition per filter, positive-OOS / cleared-0.5 / survived counts,
  survival rate by category and by family with mean OOS Sharpe, top survivors.

```bash
python layer2_backtest.py                          # full sweep + funnel
python layer2_backtest.py --cost 0.0002 --crypto-cost 0.001
python layer2_backtest.py --min-sharpe 0.4 --reuse # re-funnel a saved sweep
```

## Layer 3 — Robustness Checks (`layer3_robustness.py`)

A strategy can survive the funnel and still be fragile — working by luck or on
one magic setting. Two checks catch it:

- **Parameter sensitivity.** Per family, the spread of OOS Sharpe across all its
  settings: mean, std, fraction positive. `robust = mean > 0 and ≥60% of
  settings positive` — a tight, mostly-positive spread means the edge is real,
  not curve-fit to one lucky number.
- **Bootstrap stress test.** For each funnel survivor, resample its OOS daily
  returns with replacement (default 200×) → a distribution of outcomes instead
  of the one path history took. Reports p5/p50/p95 Sharpe and the worst-case
  drawdown, and flags each survivor **solid** or **fragile** (fragile = worst
  bootstrap DD below −40%, configurable).

Writes `param_sensitivity` and `bootstrap_results` to MySQL and to CSV for
charting.

```bash
python layer3_robustness.py
python layer3_robustness.py --n-boot 500 --fragile-dd -0.35
```

## Storage — local MySQL

All artifacts live in a local MySQL database (`strategy_tester`), CSV is only a
fallback when the server is down:

- `bars` — OHLCV per (ticker, dt)
- `sweep_results` — every config × asset backtest (Layer 2)
- `param_sensitivity`, `bootstrap_results` — robustness (Layer 3)

Connection is read from repo-root `.env` (`DB_HOST`/`DB_PORT`/`DB_USER`/
`DB_PASSWORD`/`DB_NAME`) with local-dev defaults (`127.0.0.1:3306`, `root`, no
password, `strategy_tester`). Credentials are never logged. Every layer degrades
to CSV if MySQL is unavailable, so the pipeline never hard-fails.

## Layer 4 (next)

Regime routing (momentum in trends / mean-reversion in chop) + combine
uncorrelated solid survivors + sizing → the signal engine.
