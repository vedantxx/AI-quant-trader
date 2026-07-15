# AI-Quant-trader

A regime-aware, long-only systematic equity trader. A Gaussian HMM classifies the
market's **volatility environment**; a volatility-based allocation strategy maps
each regime to a target exposure; an independent risk manager enforces sizing,
leverage, and drawdown circuit breakers; orders route to Alpaca (paper by default).

## Philosophy: risk management > signal generation

The HMM does not predict price direction — it detects **volatility regimes**.
Stocks drift upward most of the time in calm markets; the worst drawdowns cluster
in high-volatility spikes. So the edge is not a clever entry signal — it is
**avoiding big drawdowns** by sizing down when volatility rises. Cut your worst
drawdown in half and compounding does the rest.

Consequences that shape the whole codebase:

- **Always long, never short.** Shorting was tested exhaustively in walk-forward
  backtests and consistently destroyed returns: markets drift up, and V-shaped
  recoveries arrive faster than the HMM detects them. The response to high
  volatility is **reducing** allocation, not reversing it.
- **The risk manager has absolute veto power.** It runs independently of the HMM.
  Even if the model is completely wrong, circuit breakers fire on realized P&L.
- **No look-ahead, ever.** Live/backtest inference uses the forward algorithm
  (filtered posterior), never Viterbi. Guarded by `tests/test_look_ahead.py` and
  an end-date-invariance test.

## Architecture

```
 OHLCV bars
     │
     ▼
 feature engineering      14 causal indicators, 252-bar rolling z-score
     │
     ▼
 HMM engine               Gaussian HMM, BIC model selection (3–7 regimes),
     │                    forward-filtered inference, stability + flicker guard
     ▼
 volatility rank          regimes sorted by expected volatility (NOT by label)
     │
     ▼
 allocation strategy      low-vol → 95% @1.25x · mid → 95%/60% by trend ·
     │                    high-vol → 60% @1.0x   (long-only)
     ▼
 risk manager  ◄── absolute veto: 1%-risk-to-stop sizing, 15% single / 80% gross /
     │           30% sector caps, correlation & spread gates, leverage rules,
     │           daily/weekly/peak drawdown circuit breakers
     ▼
 order executor           marketable-limit / bracket (OCO), tighten-only stops
     │
     ▼
 Alpaca (paper default)   position tracker updates state + breakers on every fill
```

## Quick start (6 steps)

```bash
# 1. create a virtualenv
python -m venv .venv && source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. add Alpaca PAPER keys to .env  (gitignored, never committed)
#    ALPACA_API_KEY=...
#    ALPACA_SECRET_KEY=...
printf 'ALPACA_API_KEY=\nALPACA_SECRET_KEY=\n' > .env   # then edit

# 4. sanity-check the pipeline against your paper account (no orders placed)
python main.py trade --dry-run --iterations 1

# 5. backtest a symbol (drop OHLCV at data/SPY.csv or let it pull from Alpaca)
python main.py backtest --symbols SPY --compare

# 6. run the live paper loop
python main.py trade
```

## CLI reference

```
python main.py trade                      live/paper loop (data→regime→risk→orders)
python main.py trade --dry-run            full pipeline, logs orders, places none
python main.py trade --train-only         (re)train the HMM and exit
python main.py trade --poll 300           seconds between bar checks (default 60)
python main.py trade --iterations N        stop after N loop iterations

python main.py backtest --symbols SPY QQQ  walk-forward backtest
        --start 2019-01-01 --end 2024-12-31
        --compare                          add buy&hold / 200-SMA / random benchmarks
        --stress-test [--sims 100]         crash / gap / regime-misclassification MC

python main.py dashboard                   one-shot dashboard snapshot
```

Outputs land in `backtest_output/` (`*_equity_curve.csv`, `*_trade_log.csv`,
`*_regime_history.csv`, `*_benchmark_comparison.csv`). Logs are rotating JSON
lines in `logs/` (`main`, `trades`, `regime`, `alerts`).

## Configuration

Everything tunable lives in [config/settings.yaml](config/settings.yaml):

| Section     | Key examples                                            | What it controls |
|-------------|---------------------------------------------------------|------------------|
| `broker`    | `paper_trading`, `symbols`, `timeframe`, `regime_symbol`| account, universe, bar size, which symbol drives regime |
| `hmm`       | `n_candidates`, `min_confidence`, `stability_bars`, `flicker_threshold` | model selection + filtering |
| `strategy`  | `low_vol_allocation`, `low_vol_leverage`, `rebalance_threshold` | per-regime exposure & leverage |
| `risk`      | `max_risk_per_trade`, `max_single_position`, `*_dd_halt`, `max_dd_from_peak` | sizing caps + circuit breakers |
| `backtest`  | `train_window`, `test_window`, `step_size`, `slippage_pct` | walk-forward windows |
| `monitoring`| `dashboard_refresh_seconds`, `alert_rate_limit_minutes` | dashboard + alerts |

Secrets come from `.env` (keys) and optional `config/credentials.yaml`
(email/webhook for alerts). Both are gitignored.

## FAQ

**Why the forward algorithm and not Viterbi?** Viterbi (`model.predict`) computes
the most-likely *path* using the entire sequence — a bar's label can change when
future data arrives. That is look-ahead bias. The forward algorithm's filtered
posterior `P(state_t | obs_1:t)` uses only data up to `t`, so past decisions never
change. Appending future bars leaves earlier results bit-identical (tested).

**How is the number of regimes chosen?** By BIC across `hmm.n_candidates` (3–7).
BIC balances fit against parameter count, so it avoids over-fitting to noise. The
best model is refit each walk-forward fold and on the weekly live retrain.

**Why did my trade get rejected / resized?** The risk manager vetoes or shrinks
orders for: missing stop, position < \$100, over the 15% single-position / 80%
gross / 30% sector caps, correlation > 0.85 with an open name, spread > 0.5%,
duplicate within 60s, an active circuit breaker, or zero account equity. Every
rejection is logged with its reason.

**"BULL" regime but low allocation?** Labels are sorted by **return**, allocation
by **volatility rank** — they are independent. A high-return regime can be
high-volatility and correctly get the defensive 60% allocation.

**How do I switch to live trading?** Set `broker.paper_trading: false` and use
live API keys. On startup the system requires typing
`YES I UNDERSTAND THE RISKS`. Paper-trade first, for a long time.

## Testing

```bash
pytest                                   # full offline suite (mocks the broker)
ALPACA_LIVE_TEST=1 pytest tests/test_paper_live.py -s   # live paper order round-trip
```

Coverage includes look-ahead bias, end-date invariance, risk stress (extreme
signals capped, rapid-fire blocked, no-stop rejected), crash recovery, and a
full offline dry-run of the orchestration loop.

## Disclaimer

Educational and research use only. This is **not** financial advice and makes
**no** guarantee of profit — trading involves substantial risk of loss. Paper
trade first and understand every component before risking real capital. The
authors accept no liability for any losses incurred.
