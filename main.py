"""AI-Quant-trader entry point.

Usage:
    python main.py backtest --symbols SPY --start 2019-01-01 --end 2024-12-31
    python main.py backtest --symbols SPY --start 2019-01-01 --end 2024-12-31 --compare
    python main.py backtest --stress-test --symbols SPY
    python main.py trade        live/paper trading loop
    python main.py dashboard    terminal dashboard only

Config is read from config/settings.yaml; secrets from .env.

Backtest data is loaded from ``data/<SYMBOL>.csv`` (columns: date/open/high/low/
close/volume). If a symbol CSV is absent, the Alpaca data layer is used when
credentials are configured.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config" / "settings.yaml"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "backtest_output"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load settings.yaml into a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_credentials() -> dict:
    """Load optional config/credentials.yaml, or {} if absent."""
    path = ROOT / "config" / "credentials.yaml"
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


# --------------------------------------------------------------------- data
def load_ohlcv(
    symbol: str, start: str | None, end: str | None, cfg: dict
) -> pd.DataFrame:
    """OHLCV for ``symbol`` from ``data/<SYMBOL>.csv`` (falls back to Alpaca).

    CSV must have a date column (date/timestamp/index) plus open/high/low/
    close/volume. Rows are filtered to [start, end] and sorted oldest-first.
    """
    csv = DATA_DIR / f"{symbol}.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        date_col = next((c for c in ("date", "timestamp", "Date", "index")
                         if c in df.columns), df.columns[0])
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        df.columns = [c.lower() for c in df.columns]
        if start:
            df = df[df.index >= pd.to_datetime(start)]
        if end:
            df = df[df.index <= pd.to_datetime(end)]
        return df[["open", "high", "low", "close", "volume"]]

    # fall back to the broker data layer (needs credentials)
    from broker.alpaca_client import AlpacaClient
    from data.market_data import MarketData

    md = MarketData(cfg, AlpacaClient(cfg))
    return md.history(symbol, lookback_days=2000)


# --------------------------------------------------------------------- modes
def run_backtest(cfg: dict, args: argparse.Namespace) -> None:
    """Walk-forward backtest each requested symbol; log performance + CSVs."""
    from rich.console import Console

    from backtest.backtester import Backtester
    from backtest.performance import PerformanceAnalyzer, render_report

    console = Console()
    OUTPUT_DIR.mkdir(exist_ok=True)
    bt = Backtester(cfg)
    analyzer = PerformanceAnalyzer(cfg)

    for symbol in args.symbols:
        try:
            df = load_ohlcv(symbol, args.start, args.end, cfg)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]No data for {symbol}: {exc}[/red]")
            console.print(f"[yellow]Place OHLCV at {DATA_DIR / (symbol + '.csv')}[/yellow]")
            continue

        console.rule(f"[bold]{symbol}[/bold]  {len(df)} bars")
        res = bt.run(df, symbol=symbol)
        if len(res.equity_curve) == 0:
            console.print(f"[red]{symbol}: not enough history for any fold[/red]")
            continue

        report = analyzer.analyze(
            res.equity_curve, res.returns, res.regimes,
            trades=res.meta["trade_log"], confidence=res.meta["confidence"],
        )
        render_report(report, console)

        # ------------------------------------------------------------ CSVs
        pre = OUTPUT_DIR / symbol
        res.equity_curve.to_csv(f"{pre}_equity_curve.csv")
        res.meta["trade_log"].to_csv(f"{pre}_trade_log.csv", index=False)
        res.regimes.to_csv(f"{pre}_regime_history.csv")

        if args.compare:
            prices = res.meta["price"].dropna()
            change_freq = res.trades / max(len(res.equity_curve), 1)
            bench = analyzer.compare_benchmarks(
                prices, report.total_return, report.sharpe, change_freq,
                capital=res.meta["initial_capital"],
            )
            bench.to_csv(f"{pre}_benchmark_comparison.csv", index=False)
            from rich.table import Table
            t = Table(title="Benchmarks")
            for c in ["strategy", "total_return", "sharpe"]:
                t.add_column(c, justify="right")
            for _, r in bench.iterrows():
                t.add_row(r["strategy"], f"{r['total_return']:.2%}", f"{r['sharpe']:.2f}")
            console.print(t)

        console.print(f"[green]CSVs -> {OUTPUT_DIR}/[/green]")


def run_stress(cfg: dict, args: argparse.Namespace) -> None:
    """Run the Monte-Carlo stress families on the first requested symbol."""
    from rich.console import Console
    from rich.table import Table

    from backtest.stress_test import StressTester

    console = Console()
    symbol = args.symbols[0]
    df = load_ohlcv(symbol, args.start, args.end, cfg)
    st = StressTester(cfg)
    console.rule(f"[bold]Stress: {symbol}[/bold]  {len(df)} bars")

    summaries = [
        st.crash_test(df, n_sims=args.sims),
        st.gap_test(df, n_sims=max(args.sims // 2, 5)),
        st.regime_misclass_test(df, n_sims=max(args.sims // 5, 5)),
    ]
    t = Table(title="Stress Summary")
    for c in ["scenario", "mean max loss", "worst case", "breaker fired %"]:
        t.add_column(c, justify="right")
    for s in summaries:
        t.add_row(s.name, f"{s.mean_max_loss:.2%}", f"{s.worst_case:.2%}",
                  f"{s.breaker_fired_pct:.0%}")
    console.print(t)


def run_trade(cfg: dict) -> None:
    """Live/paper trading loop: data -> regime -> signals -> risk -> orders."""
    ...


def run_dashboard(cfg: dict) -> None:
    """Render the terminal dashboard."""
    ...


# --------------------------------------------------------------------- cli
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="main.py")
    sub = p.add_subparsers(dest="mode", required=True)

    bt = sub.add_parser("backtest", help="walk-forward backtest")
    bt.add_argument("--symbols", nargs="+", default=["SPY"])
    bt.add_argument("--start", default=None)
    bt.add_argument("--end", default=None)
    bt.add_argument("--compare", action="store_true", help="run benchmark comparisons")
    bt.add_argument("--stress-test", action="store_true", help="run stress tests instead")
    bt.add_argument("--sims", type=int, default=100, help="Monte-Carlo sims for stress")

    sub.add_parser("trade", help="live/paper trading loop")
    sub.add_parser("dashboard", help="terminal dashboard")
    return p


def main(argv: list[str]) -> int:
    """Dispatch to backtest / trade / dashboard."""
    args = build_parser().parse_args(argv[1:])
    cfg = load_config()
    if args.mode == "backtest":
        run_stress(cfg, args) if args.stress_test else run_backtest(cfg, args)
    elif args.mode == "trade":
        run_trade(cfg)
    elif args.mode == "dashboard":
        run_dashboard(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
