"""AI-Quant-trader entry point.

Usage:
    python main.py trade                 live/paper loop (data->regime->risk->orders)
    python main.py trade --dry-run       full pipeline, no orders
    python main.py trade --train-only    train/refresh the HMM and exit
    python main.py backtest --symbols SPY --start 2019-01-01 --end 2024-12-31
    python main.py backtest --symbols SPY --stress-test
    python main.py dashboard             one-shot dashboard snapshot

Config is read from config/settings.yaml; secrets from .env (never committed).
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config" / "settings.yaml"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "backtest_output"
MODELS_DIR = ROOT / "models"
SNAPSHOT_PATH = ROOT / "state_snapshot.json"
MODEL_MAX_AGE_DAYS = 7


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


# ============================================================ live trading system
class TradingSystem:
    """Wires data -> HMM regime -> allocation -> risk -> orders, with recovery."""

    def __init__(self, cfg: dict, dry_run: bool = False, client=None) -> None:
        from broker.alpaca_client import AlpacaClient
        from broker.order_executor import OrderExecutor
        from broker.position_tracker import PositionTracker
        from core.hmm_engine import HMMEngine
        from core.risk_manager import PortfolioState, RiskManager
        from data.feature_engineering import FeatureEngineer
        from data.market_data import MarketData
        from monitoring.alerts import AlertManager
        from monitoring.dashboard import Dashboard
        from monitoring.logger import TradingLogger

        self.cfg = cfg
        self.dry_run = dry_run
        self.tlog = TradingLogger()
        self.log = self.tlog.get()
        self.alert = AlertManager(cfg, load_credentials(), log_sink=self.tlog.alert_log)
        self.recent_signals: list[dict] = []

        self.symbols: list[str] = list(cfg["broker"]["symbols"])
        self.regime_symbol: str = cfg["broker"].get("regime_symbol", self.symbols[0])
        self.sector_map: dict = cfg["broker"].get("sectors", {})

        self.client = client or AlpacaClient(cfg)
        self.market = MarketData(cfg, self.client)
        feat_cfg = cfg.get("features", {})
        self.fe = FeatureEngineer(**({"zscore_window": feat_cfg["zscore_window"]}
                                     if "zscore_window" in feat_cfg else {}))
        self.hmm = HMMEngine(cfg)
        self.risk = RiskManager(cfg, lock_dir=ROOT)
        self.executor = OrderExecutor(self.client)
        self.state = PortfolioState(equity=0, cash=0, buying_power=0)
        self.tracker = PositionTracker(self.client, self.risk.breaker, self.state)
        self.dashboard = Dashboard(cfg)

        self.orchestrator = None
        self._stop = False
        self._last_regime = None
        self._last_train = datetime.now(timezone.utc)
        self._session_start = datetime.now(timezone.utc)
        self._n_orders = 0

    # ----------------------------------------------------------------- startup
    def startup(self) -> None:
        """Connect, verify account, load/train the HMM, sync positions, recover."""
        acct = self.client.get_account()
        self.log.info("account verified: equity=%.2f buying_power=%.2f paper=%s",
                      acct.equity, acct.buying_power, self.client.paper)

        self.load_or_train_hmm()
        self._refresh_portfolio_state()
        self.tracker.reconcile()
        self._load_snapshot()

        self.log.info("System online — %d symbols, regime symbol %s, %s mode",
                      len(self.symbols), self.regime_symbol,
                      "DRY-RUN" if self.dry_run else "LIVE-ORDER")

    def load_or_train_hmm(self) -> None:
        """Load a fresh persisted model, else train from history and save."""
        MODELS_DIR.mkdir(exist_ok=True)
        path = MODELS_DIR / f"hmm_{self.regime_symbol}.pkl"
        if path.exists() and self._model_age_days(path) <= MODEL_MAX_AGE_DAYS:
            self.hmm.load(str(path))
            self.log.info("loaded HMM %s (age %.1fd)", path.name, self._model_age_days(path))
        else:
            self.train_hmm(save_path=path)

    def train_hmm(self, save_path: Path | None = None) -> None:
        """Refit the HMM on recent history and rebuild the orchestrator."""
        from core.regime_strategies import StrategyOrchestrator

        bars = self.market.history(self.regime_symbol, lookback_days=1000)
        feats = self.fe.build_features(bars)
        returns = bars["close"].pct_change().reindex(feats.index).to_numpy()
        self.hmm.fit(feats.to_numpy(dtype=float), returns=returns)
        if save_path is not None:
            self.hmm.save(str(save_path))
            self.log.info("trained + saved HMM -> %s", save_path.name)
        self.orchestrator = StrategyOrchestrator(self.cfg, self.hmm.regime_info)

    def _model_age_days(self, path: Path) -> float:
        try:
            self.hmm.load(str(path))
            td = self.hmm.metadata.training_date
            age = datetime.now(timezone.utc) - datetime.fromisoformat(td)
            return age.total_seconds() / 86400.0
        except Exception:  # noqa: BLE001 - unreadable/old model -> force retrain
            return float("inf")

    # ------------------------------------------------------------- one iteration
    def run_once(self, now: datetime | None = None) -> dict:
        """A single bar-close pass. Never raises: degrades and holds on failure."""
        now = now or datetime.now(timezone.utc)
        if self.orchestrator is None:
            from core.regime_strategies import StrategyOrchestrator
            self.orchestrator = StrategyOrchestrator(self.cfg, self.hmm.regime_info)

        # --- 1-2. pull bars + build causal features
        try:
            bars = {s: self.market.history(s, lookback_days=1000) for s in self.symbols}
        except Exception as exc:  # noqa: BLE001 - data feed drop: pause, keep stops
            self.log.error("data fetch failed, holding: %s", exc)
            self.alert.alert("data_drop", "Data feed drop", str(exc))
            return {"status": "data_drop"}

        regime_bars = bars[self.regime_symbol]
        feats = self.fe.build_features(regime_bars)
        if len(feats) < 5:
            self.log.warning("insufficient features, holding")
            return {"status": "warming_up"}

        # --- 3-5. filtered regime + stability + flicker
        try:
            state = self.hmm.predict(feats.to_numpy(dtype=float), now)
        except Exception as exc:  # noqa: BLE001 - HMM error: hold current regime
            self.log.error("HMM error, holding last regime: %s", exc)
            self.alert.alert("hmm_error", "HMM prediction error", str(exc))
            return {"status": "hmm_error", "regime": str(self._last_regime)}
        flickering = self.hmm.is_flickering()
        prev_regime = self._last_regime
        self._last_regime = state.label
        self.tlog.set_context(regime=state.label, probability=round(state.probability, 4))
        self.tlog.regime("regime", regime=state.label, probability=state.probability,
                         stability=self.hmm.get_regime_stability(), flickering=flickering)
        if prev_regime and prev_regime != state.label and state.is_confirmed:
            self.alert.regime_change(prev_regime, state.label, state.probability)
            self.log.warning("REGIME CHANGE %s -> %s", prev_regime, state.label)
        if flickering:
            self.alert.flicker_exceeded(self.hmm.get_regime_flicker_rate(),
                                        self.hmm.flicker_threshold)

        # --- 6. target allocation per symbol
        self._refresh_portfolio_state(bars)
        if self.state.equity <= 0:
            self.log.warning("account equity is 0 — regime %s detected, holding (no sizing)",
                             state.label)
            return {"status": "no_equity", "regime": state.label,
                    "regime_prob": state.probability, "equity": 0.0,
                    "last_bar": str(regime_bars.index[-1].date())}
        self.state.regime_uncertain = (state.probability < self.hmm.min_confidence) or flickering
        self.state.flicker_rate = self.hmm.get_regime_flicker_rate()
        self.state.hmm_regime = state.label
        signals = self.orchestrator.generate_signals(
            self.symbols, bars, state, is_flickering=flickering)

        # --- 7. risk gate + order routing
        approved = rejected = modified = 0
        for sig in signals:
            price = float(bars[sig.symbol]["close"].iloc[-1])
            self._annotate(sig, price)
            decision = self.risk.validate_signal(sig, self.state, now=now)
            if not decision.approved:
                rejected += 1
                self.log.info("REJECT %s: %s", sig.symbol, decision.rejection_reason)
                continue
            ms = decision.modified_signal
            qty = int(ms.position_size_pct * self.state.equity / price)
            if decision.modifications:
                modified += 1
                self.log.info("MODIFY %s: %s", sig.symbol, "; ".join(decision.modifications))
            if qty <= 0:
                continue
            approved += 1
            self._record_signal(now, sig.symbol, ms, state.label)
            if self.dry_run:
                self.log.info("DRY-RUN would submit %s x%d @~%.2f (alloc %.1f%% lev %.2f)",
                              sig.symbol, qty, price, ms.position_size_pct * 100, ms.leverage)
            else:
                res = self.executor.submit_order(ms, qty=qty, risk_modifications=decision.modifications)
                self.state.recent_orders[(sig.symbol, sig.direction)] = now
                self._n_orders += 1
                self.log.info("ORDER %s %s x%d id=%s", res.side, res.symbol, qty, res.id)
                self.tlog.trade("order", symbol=res.symbol, side=res.side, qty=qty,
                                order_id=res.id, trade_id=res.trade_id, regime=state.label)

        # --- 8. trailing stops (tighten-only) per current regime
        self._update_stops(signals)

        # --- 9. circuit breaker (independent of regime)
        breaker = self.risk.breaker.evaluate(self.state)
        if breaker.close_all and not self.dry_run:
            self.executor.close_all_positions()
            self.alert.circuit_breaker(breaker.active, breaker.reason)
            self.log.error("CIRCUIT BREAKER halt: %s — flattened", breaker.reason)

        # --- 10. dashboard state
        return self._dashboard_state(state, breaker, regime_bars, signals,
                                     approved, rejected, modified)

    def _record_signal(self, now, symbol, ms, regime) -> None:
        self.recent_signals.append({
            "time": now.strftime("%H:%M"), "symbol": symbol,
            "change": f"{ms.position_size_pct:.0%} @ {ms.leverage:.2f}x",
            "reason": regime})
        self.recent_signals = self.recent_signals[-20:]

    def _dashboard_state(self, state, breaker, regime_bars, signals,
                         approved, rejected, modified) -> dict:
        day0 = self.state.day_start_equity or self.state.equity
        daily_pnl = self.state.equity - day0
        lead = signals[0] if signals else None
        positions = [{
            "symbol": s, "side": "LONG", "price": p.entry_price,
            "pnl_pct": 0.0, "stop": p.stop_loss, "held": "—",
        } for s, p in self.state.positions.items()]
        return {
            "status": "ok", "regime": state.label, "regime_prob": state.probability,
            "stability_bars": self.hmm.get_regime_stability(),
            "flicker": self.hmm.get_regime_flicker_rate(),
            "flicker_window": self.hmm.flicker_window,
            "uncertainty": self.state.regime_uncertain,
            "equity": self.state.equity, "cash": self.state.cash,
            "buying_power": self.state.buying_power,
            "daily_pnl": daily_pnl, "daily_pnl_pct": daily_pnl / day0 if day0 else 0.0,
            "allocation": lead.position_size_pct if lead else 0.0,
            "leverage": lead.leverage if lead else 1.0,
            "drawdown": self.state.drawdown, "breaker": breaker.active,
            "daily_dd": abs(min(0.0, daily_pnl / day0 if day0 else 0.0)),
            "daily_dd_limit": self.risk.breaker.daily_halt,
            "peak_dd": abs(min(0.0, self.state.drawdown)),
            "peak_dd_limit": self.risk.breaker.peak_halt,
            "open_positions": len(self.state.positions),
            "positions": positions, "recent_signals": self.recent_signals,
            "trades_today": self.state.trades_today,
            "approved": approved, "rejected": rejected, "modified": modified,
            "hmm_age": self._hmm_age_str(), "paper": self.client.paper,
            "data_ok": True, "api_ok": True, "api_ms": 0.0,
            "last_bar": str(regime_bars.index[-1].date()),
        }

    def _hmm_age_str(self) -> str:
        try:
            td = self.hmm.metadata.training_date
            days = (datetime.now(timezone.utc) - datetime.fromisoformat(td)).days
            return f"{days}d ago"
        except Exception:  # noqa: BLE001
            return "—"

    # --------------------------------------------------------------- main loop
    def run(self, poll_seconds: float | None = None, max_iterations: int | None = None) -> None:
        """Loop each bar close until interrupted. Saves state on exit."""
        self._install_signal_handlers()
        poll = poll_seconds if poll_seconds is not None else 60.0
        i = 0
        try:
            while not self._stop:
                if not self.dry_run and not self._market_open_safe():
                    self.log.info("market closed — waiting")
                    if max_iterations:  # tests/one-shot shouldn't block on the clock
                        break
                    time.sleep(min(poll, 60))
                    continue
                try:
                    summary = self.run_once()
                    self.dashboard.render_once(summary)
                except Exception:  # noqa: BLE001 - unhandled: log, snapshot, alert, continue
                    tb = traceback.format_exc()
                    self.log.error("unhandled loop error:\n%s", tb)
                    self._save_snapshot()
                    self.alert.alert("crash", "Unhandled loop error", tb)
                self._maybe_retrain()
                i += 1
                if max_iterations and i >= max_iterations:
                    break
                if not self._stop:
                    time.sleep(poll)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Graceful stop: keep positions (stops in place), snapshot, summarize."""
        self._save_snapshot()
        dur = (datetime.now(timezone.utc) - self._session_start).total_seconds() / 60
        self.log.info("session summary: %.1f min, %d orders, last regime %s",
                      dur, self._n_orders, self._last_regime)
        self.log.info("shutdown complete — positions left open (stops active)")

    # ------------------------------------------------------------- helpers
    def _annotate(self, sig, price: float) -> None:
        """Attach spread / sector / overnight metadata for the risk gate."""
        meta = dict(sig.metadata or {})
        meta.setdefault("sector", self.sector_map.get(sig.symbol, "UNKNOWN"))
        try:
            meta["spread_pct"] = self.market.get_latest_quote(sig.symbol)["spread_pct"]
        except Exception:  # noqa: BLE001 - quote optional; absence != rejection
            pass
        sig.metadata = meta

    def _update_stops(self, signals) -> None:
        for sig in signals:
            if sig.symbol in self.state.positions:
                try:
                    self.executor.modify_stop(sig.symbol, sig.stop_loss)
                except Exception as exc:  # noqa: BLE001
                    self.log.warning("stop update failed for %s: %s", sig.symbol, exc)

    def _refresh_portfolio_state(self, bars: dict | None = None) -> None:
        from core.risk_manager import Position

        acct = self.client.get_account()
        peak = max(self.state.peak_equity, acct.equity)
        self.state.equity = acct.equity
        self.state.cash = acct.cash
        self.state.buying_power = acct.buying_power
        self.state.peak_equity = peak
        if self.state.day_start_equity == 0:
            self.state.day_start_equity = acct.equity
        if self.state.week_start_equity == 0:
            self.state.week_start_equity = acct.equity
        self.state.positions = {
            s: Position(symbol=s, weight=(p.market_value / acct.equity if acct.equity else 0),
                        entry_price=p.avg_entry, stop_loss=0.0,
                        sector=self.sector_map.get(s, "UNKNOWN"))
            for s, p in acct.positions.items()}
        self.state.sector_map = self.sector_map
        if bars is not None:
            self.state.price_history = {s: df["close"] for s, df in bars.items()}

    def _market_open_safe(self) -> bool:
        try:
            return self.client.is_market_open()
        except Exception as exc:  # noqa: BLE001
            self.log.warning("clock check failed: %s", exc)
            return False

    def _maybe_retrain(self) -> None:
        if datetime.now(timezone.utc) - self._last_train >= timedelta(days=7):
            self.log.info("weekly HMM retrain")
            self.train_hmm(save_path=MODELS_DIR / f"hmm_{self.regime_symbol}.pkl")
            self._last_train = datetime.now(timezone.utc)

    def _install_signal_handlers(self) -> None:
        def handler(signum, _frame):
            self.log.info("signal %s received — shutting down", signum)
            self._stop = True
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except ValueError:  # not on main thread (tests)
                pass

    def _save_snapshot(self) -> None:
        try:
            SNAPSHOT_PATH.write_text(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "regime": self._last_regime, "equity": self.state.equity,
                "peak_equity": self.state.peak_equity,
                "day_start_equity": self.state.day_start_equity,
                "week_start_equity": self.state.week_start_equity,
                "trades_today": self.state.trades_today, "n_orders": self._n_orders,
            }, indent=2))
        except OSError as exc:
            self.log.warning("snapshot save failed: %s", exc)

    def _load_snapshot(self) -> None:
        if not SNAPSHOT_PATH.exists():
            return
        try:
            snap = json.loads(SNAPSHOT_PATH.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            self.log.warning("snapshot load failed: %s", exc)
            return
        self._last_regime = snap.get("regime")
        self.state.peak_equity = max(self.state.peak_equity, snap.get("peak_equity", 0) or 0)
        self.state.day_start_equity = snap.get("day_start_equity") or self.state.day_start_equity
        self.state.week_start_equity = snap.get("week_start_equity") or self.state.week_start_equity
        self.log.info("recovered state from %s (regime %s)",
                      snap.get("timestamp"), self._last_regime)


# --------------------------------------------------------------------- data (backtest)
def load_ohlcv(symbol: str, start: str | None, end: str | None, cfg: dict) -> pd.DataFrame:
    """OHLCV for ``symbol`` from ``data/<SYMBOL>.csv`` (falls back to Alpaca)."""
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

    from broker.alpaca_client import AlpacaClient
    from data.market_data import MarketData
    md = MarketData(cfg, AlpacaClient(cfg))
    return md.history(symbol, lookback_days=2000)


# --------------------------------------------------------------------- modes
def run_trade(cfg: dict, args: argparse.Namespace) -> None:
    """Live/paper trading loop (or --dry-run / --train-only)."""
    system = TradingSystem(cfg, dry_run=getattr(args, "dry_run", False))
    if getattr(args, "train_only", False):
        system.train_hmm(save_path=MODELS_DIR / f"hmm_{system.regime_symbol}.pkl")
        system.log.info("train-only complete")
        return
    system.startup()
    system.run(poll_seconds=getattr(args, "poll", None),
               max_iterations=getattr(args, "iterations", None))


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
            trades=res.meta["trade_log"], confidence=res.meta["confidence"])
        render_report(report, console)

        pre = OUTPUT_DIR / symbol
        res.equity_curve.to_csv(f"{pre}_equity_curve.csv")
        res.meta["trade_log"].to_csv(f"{pre}_trade_log.csv", index=False)
        res.regimes.to_csv(f"{pre}_regime_history.csv")

        if args.compare:
            prices = res.meta["price"].dropna()
            change_freq = res.trades / max(len(res.equity_curve), 1)
            bench = analyzer.compare_benchmarks(
                prices, report.total_return, report.sharpe, change_freq,
                capital=res.meta["initial_capital"])
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


def run_dashboard(cfg: dict, args: argparse.Namespace) -> None:
    """One-shot dashboard snapshot for a (dry) system state."""
    system = TradingSystem(cfg, dry_run=True)
    system.startup()
    system.dashboard.render_once(system.run_once())


# --------------------------------------------------------------------- cli
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="main.py")
    sub = p.add_subparsers(dest="mode", required=True)

    tr = sub.add_parser("trade", help="live/paper trading loop")
    tr.add_argument("--dry-run", action="store_true", help="full pipeline, no orders")
    tr.add_argument("--train-only", action="store_true", help="train the HMM and exit")
    tr.add_argument("--poll", type=float, default=None, help="seconds between bar checks")
    tr.add_argument("--iterations", type=int, default=None, help="stop after N iterations")

    bt = sub.add_parser("backtest", help="walk-forward backtest")
    bt.add_argument("--symbols", nargs="+", default=["SPY"])
    bt.add_argument("--start", default=None)
    bt.add_argument("--end", default=None)
    bt.add_argument("--compare", action="store_true", help="run benchmark comparisons")
    bt.add_argument("--stress-test", action="store_true", help="run stress tests instead")
    bt.add_argument("--sims", type=int, default=100, help="Monte-Carlo sims for stress")

    sub.add_parser("dashboard", help="one-shot dashboard snapshot")
    return p


def main(argv: list[str]) -> int:
    """Dispatch to trade / backtest / dashboard."""
    try:  # load .env credentials if python-dotenv is available
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    args = build_parser().parse_args(argv[1:])
    cfg = load_config()
    if args.mode == "trade":
        run_trade(cfg, args)
    elif args.mode == "backtest":
        run_stress(cfg, args) if args.stress_test else run_backtest(cfg, args)
    elif args.mode == "dashboard":
        run_dashboard(cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
