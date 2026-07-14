"""AI-Quant-trader entry point.

Usage:
    python main.py backtest     walk-forward backtest over configured symbols
    python main.py trade        live/paper trading loop
    python main.py dashboard    terminal dashboard only

Config is read from config/settings.yaml; secrets from .env.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent / "config" / "settings.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_credentials() -> dict:
    path = Path(__file__).parent / "config" / "credentials.yaml"
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


# --------------------------------------------------------------------- modes
def run_backtest(cfg: dict) -> None:
    from broker.alpaca_client import AlpacaClient
    from data.market_data import MarketData
    from backtest.backtester import Backtester
    from backtest.performance import compute
    from monitoring.logger import get_logger

    log = get_logger()
    client = AlpacaClient(cfg)
    md = MarketData(cfg, client)
    bt = Backtester(cfg)

    for symbol in cfg["broker"]["symbols"]:
        df = md.history(symbol, lookback_days=1500)
        if df.empty:
            log.warning(f"no data for {symbol}, skipping")
            continue
        res = bt.run(df)
        bench = df["close"].pct_change().reindex(res.returns.index).fillna(0.0)
        perf = compute(res.equity_curve, res.returns, res.regimes, cfg, bench)
        log.info(
            f"{symbol}: return={perf.total_return:.1%} "
            f"sharpe={perf.sharpe:.2f} maxDD={perf.max_drawdown:.1%} "
            f"trades={res.trades} folds={res.folds}"
        )


def run_trade(cfg: dict) -> None:
    from broker.alpaca_client import AlpacaClient
    from broker.order_executor import OrderExecutor
    from broker.position_tracker import PositionTracker
    from core.regime_strategies import RegimeStrategy
    from core.risk_manager import RiskManager
    from core.signal_generator import SignalGenerator
    from monitoring.logger import get_logger

    log = get_logger()
    client = AlpacaClient(cfg)
    strategy = RegimeStrategy(cfg)
    risk = RiskManager(cfg)
    _ = SignalGenerator(cfg, strategy, risk)
    _ = OrderExecutor(client)
    _ = PositionTracker(client)

    mode = "PAPER" if client.paper else "LIVE"
    log.info(f"trading loop starting [{mode}] — wire the schedule loop here")
    # TODO: schedule.every().day.at(...).do(step); per-bar step:
    #   fetch data -> features -> regime -> signals -> risk gate -> orders


def run_dashboard(cfg: dict) -> None:
    from monitoring.dashboard import Dashboard, DashboardState

    dash = Dashboard(cfg)
    demo = DashboardState(
        equity=100_000.0,
        regime="mid",
        confidence=0.72,
        positions={},
        halted=False,
        halt_reason="",
    )
    dash.render_once(demo)


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] not in {"backtest", "trade", "dashboard"}:
        print(__doc__)
        return 1
    cfg = load_config()
    {"backtest": run_backtest, "trade": run_trade, "dashboard": run_dashboard}[
        argv[1]
    ](cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
