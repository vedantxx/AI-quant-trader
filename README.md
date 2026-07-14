# AI-Quant-trader

Regime-aware systematic equity trader. An HMM detects volatility regimes from
market features; a volatility-based allocation strategy maps each regime to a
target exposure; a risk manager enforces sizing, leverage, and drawdown circuit
breakers; orders route to Alpaca (paper by default).

## Status

Stub scaffold: every class/dataclass and method signature is in place with full
type hints and docstrings, but method bodies are `...` — no logic implemented
yet. Tests are skeletons naming the cases to fill in.

## Structure

```
config/       settings.yaml (all params) + credentials example
core/         hmm_engine, regime_strategies, risk_manager, signal_generator
broker/       alpaca_client, order_executor, position_tracker
data/         market_data, feature_engineering
monitoring/   logger, dashboard, alerts
backtest/     backtester (walk-forward), performance, stress_test
tests/        hmm, look-ahead, strategies, risk, orders
main.py       entry point
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # fill in Alpaca keys
cp config/credentials.yaml.example config/credentials.yaml   # optional
```

Set `ALPACA_PAPER=true` and `broker.paper_trading: true` (default) until fully
validated.

## Usage

```bash
python main.py backtest        # walk-forward backtest over configured symbols
python main.py trade           # live/paper trading loop
python main.py dashboard       # terminal dashboard only
```

## Config

Everything tunable lives in [config/settings.yaml](config/settings.yaml),
grouped by section: `broker`, `hmm`, `strategy`, `risk`, `backtest`,
`monitoring`.

## Safety

- Paper trading is the default. Live trading requires flipping
  `broker.paper_trading` **and** `ALPACA_PAPER=false`.
- Drawdown kill switches (`risk.*_dd_halt`, `risk.max_dd_from_peak`) halt trading
  automatically.
- `tests/test_look_ahead.py` guards against look-ahead bias in features/signals.

## Disclaimer

Educational / research use. Not financial advice. Trading involves risk of loss.
