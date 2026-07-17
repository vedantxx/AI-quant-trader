# Strategy Tester

A four-layer system that runs thousands of backtests across every popular
retail strategy, then puts each through walk-forward validation, realistic
costs, and robustness checks to surface the few that actually hold up.

Each layer is one runnable script, importable by the next.

## Layer 1 — Data & Strategy Library (`layer1_data_strategies.py`)

The foundation.

- **Data.** Daily OHLCV via `yfinance` (`auto_adjust=True`), 2010-01-01 →
  2025-01-01, ~30 liquid assets: index ETFs, sector ETFs, commodities/rates/
  international, crypto, and large caps. Assets with < 500 bars are skipped.
  Cached to `./data/*.csv` so reruns skip the network (and dodge rate limits).
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

## Layers 2–4 (next)

2. Backtest engine + metrics over every config × asset.
3. Walk-forward OOS validation, six filters, realistic costs.
4. Robustness checks + survivor ranking.
