"""Performance metrics: Sharpe, Sortino, drawdown, regime & confidence
breakdowns, benchmark comparisons, worst-case tails.

All ratios are annualized on a 252-trading-day year. Trade-level stats treat
each allocation segment (rebalance to next rebalance) as one "trade".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class PerformanceReport:
    """Summary performance statistics for a backtest run."""

    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float          # negative fraction, e.g. -0.18
    calmar: float
    win_rate: float
    volatility: float
    max_dd_duration: int = 0     # trading days underwater at the worst stretch
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    total_trades: int = 0
    avg_hold: float = 0.0
    regime_breakdown: dict = field(default_factory=dict)
    confidence_breakdown: dict = field(default_factory=dict)
    worst_case: dict = field(default_factory=dict)
    vs_benchmark: dict = field(default_factory=dict)


class PerformanceAnalyzer:
    """Compute risk-adjusted metrics and regime/benchmark breakdowns."""

    def __init__(self, cfg: dict) -> None:
        self.rf: float = float(cfg["backtest"]["risk_free_rate"])

    # ---------------------------------------------------------------- core
    def analyze(
        self,
        equity: pd.Series,
        returns: pd.Series,
        regimes: pd.Series,
        benchmark: Optional[pd.Series] = None,
        trades: Optional[pd.DataFrame] = None,
        confidence: Optional[pd.Series] = None,
    ) -> PerformanceReport:
        """Full performance report."""
        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) else 0.0
        years = len(equity) / TRADING_DAYS if len(equity) else 0.0
        cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 else 0.0
        mdd, dd_dur = self._max_drawdown_dur(equity)
        vol = float(returns.std() * np.sqrt(TRADING_DAYS)) if len(returns) else 0.0

        ts = self._trade_stats(trades)
        report = PerformanceReport(
            total_return=total_return,
            cagr=cagr,
            sharpe=self.sharpe(returns),
            sortino=self.sortino(returns),
            max_drawdown=mdd,
            calmar=(cagr / abs(mdd)) if mdd < 0 else 0.0,
            win_rate=ts["win_rate"],
            volatility=vol,
            max_dd_duration=dd_dur,
            profit_factor=ts["profit_factor"],
            avg_win=ts["avg_win"],
            avg_loss=ts["avg_loss"],
            total_trades=ts["total_trades"],
            avg_hold=ts["avg_hold"],
            regime_breakdown=self.regime_breakdown(returns, regimes, trades),
            confidence_breakdown=self._confidence_breakdown(trades),
            worst_case=self._worst_case(returns, equity),
        )
        if benchmark is not None:
            report.vs_benchmark = {
                "strategy_return": total_return,
                "benchmark_return": float(benchmark.iloc[-1] / benchmark.iloc[0] - 1),
            }
        return report

    def sharpe(self, returns: pd.Series) -> float:
        """Annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        excess = returns - self.rf / TRADING_DAYS
        sd = excess.std()
        return float(excess.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0

    def sortino(self, returns: pd.Series) -> float:
        """Annualized Sortino ratio (downside deviation)."""
        if len(returns) < 2:
            return 0.0
        excess = returns - self.rf / TRADING_DAYS
        downside = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2))
        return float(excess.mean() / downside * np.sqrt(TRADING_DAYS)) if downside > 0 else 0.0

    def max_drawdown(self, equity: pd.Series) -> float:
        """Maximum peak-to-trough drawdown (negative fraction)."""
        return self._max_drawdown_dur(equity)[0]

    @staticmethod
    def _max_drawdown_dur(equity: pd.Series) -> tuple[float, int]:
        if len(equity) == 0:
            return 0.0, 0
        peak = equity.cummax()
        dd = equity / peak - 1
        mdd = float(dd.min())
        # longest underwater stretch (bars where equity < running peak)
        underwater = (dd < 0).to_numpy()
        longest = cur = 0
        for uw in underwater:
            cur = cur + 1 if uw else 0
            longest = max(longest, cur)
        return mdd, longest

    # ----------------------------------------------------------- breakdowns
    def regime_breakdown(
        self, returns: pd.Series, regimes: pd.Series,
        trades: Optional[pd.DataFrame] = None,
    ) -> dict:
        """Per-regime: % time in, return contribution, avg trade P&L, win rate,
        Sharpe. Proves each vol environment's strategy performs as expected."""
        out: dict = {}
        if len(regimes) == 0:
            return out
        reg = regimes.reindex(returns.index).dropna()
        rets = returns.reindex(reg.index)
        total = len(regimes)
        for label in sorted(regimes.unique()):
            mask = reg == label
            r = rets[mask]
            tr = trades[trades["regime"] == label] if trades is not None and len(trades) else None
            out[label] = {
                "pct_time": float((regimes == label).sum() / total),
                "return_contribution": float(r.sum()),
                "avg_trade_pnl": float(tr["pnl"].mean()) if tr is not None and len(tr) else 0.0,
                "win_rate": float((tr["pnl"] > 0).mean()) if tr is not None and len(tr) else 0.0,
                "sharpe": self.sharpe(r),
            }
        return out

    def _confidence_breakdown(self, trades: Optional[pd.DataFrame]) -> dict:
        """Bucket trades by regime confidence. If high-confidence trades
        outperform low-confidence, the HMM is adding value."""
        buckets = {"<50%": (0.0, 0.5), "50-60%": (0.5, 0.6),
                   "60-70%": (0.6, 0.7), "70%+": (0.7, 1.01)}
        out: dict = {}
        for name, (lo, hi) in buckets.items():
            if trades is None or not len(trades) or "confidence" not in trades:
                out[name] = {"trades": 0, "sharpe": 0.0, "win_rate": 0.0, "avg_pnl": 0.0}
                continue
            b = trades[(trades["confidence"] >= lo) & (trades["confidence"] < hi)]
            pnl = b["pnl_pct"] if len(b) else pd.Series(dtype=float)
            out[name] = {
                "trades": int(len(b)),
                "sharpe": float(pnl.mean() / pnl.std() * np.sqrt(TRADING_DAYS))
                if len(b) > 1 and pnl.std() > 0 else 0.0,
                "win_rate": float((b["pnl"] > 0).mean()) if len(b) else 0.0,
                "avg_pnl": float(b["pnl"].mean()) if len(b) else 0.0,
            }
        return out

    @staticmethod
    def _trade_stats(trades: Optional[pd.DataFrame]) -> dict:
        if trades is None or not len(trades):
            return {"win_rate": 0.0, "profit_factor": 0.0, "avg_win": 0.0,
                    "avg_loss": 0.0, "total_trades": 0, "avg_hold": 0.0}
        pnl = trades["pnl"]
        wins, losses = pnl[pnl > 0], pnl[pnl < 0]
        gross_loss = abs(losses.sum())
        return {
            "win_rate": float((pnl > 0).mean()),
            "profit_factor": float(wins.sum() / gross_loss) if gross_loss > 0 else float("inf"),
            "avg_win": float(wins.mean()) if len(wins) else 0.0,
            "avg_loss": float(losses.mean()) if len(losses) else 0.0,
            "total_trades": int(len(trades)),
            "avg_hold": float(trades["hold_bars"].mean()),
        }

    @staticmethod
    def _worst_case(returns: pd.Series, equity: pd.Series) -> dict:
        if len(returns) == 0:
            return {}
        # consecutive losing days
        neg = (returns < 0).to_numpy()
        longest = cur = 0
        for x in neg:
            cur = cur + 1 if x else 0
            longest = max(longest, cur)
        peak = equity.cummax()
        underwater = (equity < peak).to_numpy()
        uw_longest = uw_cur = 0
        for x in underwater:
            uw_cur = uw_cur + 1 if x else 0
            uw_longest = max(uw_longest, uw_cur)
        return {
            "worst_day": float(returns.min()),
            "worst_week": float(returns.rolling(5).sum().min()),
            "worst_month": float(returns.rolling(21).sum().min()),
            "max_consecutive_losses": int(longest),
            "longest_underwater_days": int(uw_longest),
        }

    # ----------------------------------------------------------- benchmarks
    def buy_and_hold(self, prices: pd.Series, capital: float = 100_000.0) -> pd.Series:
        """Hold the asset for the entire period."""
        return capital * prices / prices.iloc[0]

    def sma_trend(self, prices: pd.Series, window: int = 200,
                  capital: float = 100_000.0) -> pd.Series:
        """Long above the ``window``-SMA, cash below it."""
        sma = prices.rolling(window).mean()
        signal = (prices > sma).shift(1, fill_value=False).astype(float)  # act next bar
        ret = prices.pct_change().fillna(0.0) * signal
        return capital * (1 + ret).cumprod()

    def random_benchmark(
        self, prices: pd.Series, change_freq: float, seeds: int = 100,
        capital: float = 100_000.0, alloc_choices=(0.60, 0.95),
    ) -> dict:
        """Random allocation changes at the strategy's frequency, same sizing.
        Reports mean/std of total return and Sharpe over ``seeds`` runs."""
        rets = prices.pct_change().fillna(0.0).to_numpy()
        n = len(prices)
        totals, sharpes = [], []
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            alloc = np.empty(n)
            cur = rng.choice(alloc_choices)
            for i in range(n):
                if rng.random() < change_freq:
                    cur = rng.choice(alloc_choices)
                alloc[i] = cur
            strat_ret = np.roll(alloc, 1) * rets
            strat_ret[0] = 0.0
            eq = capital * np.cumprod(1 + strat_ret)
            totals.append(eq[-1] / capital - 1)
            sd = strat_ret.std()
            sharpes.append(strat_ret.mean() / sd * np.sqrt(TRADING_DAYS) if sd > 0 else 0.0)
        return {
            "mean_return": float(np.mean(totals)), "std_return": float(np.std(totals)),
            "mean_sharpe": float(np.mean(sharpes)), "std_sharpe": float(np.std(sharpes)),
        }

    def compare_benchmarks(
        self, prices: pd.Series, result_return: float, result_sharpe: float,
        change_freq: float, capital: float = 100_000.0,
    ) -> pd.DataFrame:
        """Build the benchmark comparison table (for terminal + CSV)."""
        bh = self.buy_and_hold(prices, capital)
        sma = self.sma_trend(prices, 200, capital)
        rnd = self.random_benchmark(prices, change_freq, 100, capital)
        rows = [
            {"strategy": "Regime (this)", "total_return": result_return, "sharpe": result_sharpe},
            {"strategy": "Buy & Hold", "total_return": float(bh.iloc[-1] / bh.iloc[0] - 1),
             "sharpe": self.sharpe(bh.pct_change().dropna())},
            {"strategy": "200-SMA Trend", "total_return": float(sma.iloc[-1] / sma.iloc[0] - 1),
             "sharpe": self.sharpe(sma.pct_change().dropna())},
            {"strategy": "Random (mean of 100)", "total_return": rnd["mean_return"],
             "sharpe": rnd["mean_sharpe"]},
        ]
        return pd.DataFrame(rows)


# --------------------------------------------------------------- rich output
def render_report(report: PerformanceReport, console=None) -> None:
    """Print the report as rich tables to the terminal."""
    from rich.console import Console
    from rich.table import Table

    console = console or Console()

    core = Table(title="Performance", show_header=False)
    core.add_column("metric", style="cyan")
    core.add_column("value", justify="right")
    for k, v in [
        ("Total Return", f"{report.total_return:.2%}"),
        ("CAGR", f"{report.cagr:.2%}"),
        ("Sharpe", f"{report.sharpe:.2f}"),
        ("Sortino", f"{report.sortino:.2f}"),
        ("Calmar", f"{report.calmar:.2f}"),
        ("Max Drawdown", f"{report.max_drawdown:.2%} ({report.max_dd_duration}d)"),
        ("Volatility", f"{report.volatility:.2%}"),
        ("Win Rate", f"{report.win_rate:.2%}"),
        ("Profit Factor", f"{report.profit_factor:.2f}"),
        ("Total Trades", str(report.total_trades)),
        ("Avg Hold (bars)", f"{report.avg_hold:.1f}"),
    ]:
        core.add_row(k, v)
    console.print(core)

    if report.regime_breakdown:
        rt = Table(title="By Regime")
        for c in ["Regime", "% Time", "Ret Contrib", "Avg Trade P&L", "Win Rate", "Sharpe"]:
            rt.add_column(c, justify="right")
        for label, m in report.regime_breakdown.items():
            rt.add_row(label, f"{m['pct_time']:.1%}", f"{m['return_contribution']:.3f}",
                       f"{m['avg_trade_pnl']:.0f}", f"{m['win_rate']:.1%}", f"{m['sharpe']:.2f}")
        console.print(rt)

    if report.confidence_breakdown:
        ct = Table(title="By Confidence")
        for c in ["Confidence", "Trades", "Sharpe", "Win Rate", "Avg P&L"]:
            ct.add_column(c, justify="right")
        for label, m in report.confidence_breakdown.items():
            ct.add_row(label, str(m["trades"]), f"{m['sharpe']:.2f}",
                       f"{m['win_rate']:.1%}", f"{m['avg_pnl']:.0f}")
        console.print(ct)

    if report.worst_case:
        wc = Table(title="Worst Case", show_header=False)
        wc.add_column("metric", style="magenta")
        wc.add_column("value", justify="right")
        for k, v in report.worst_case.items():
            fmt = f"{v:.2%}" if isinstance(v, float) else str(v)
            wc.add_row(k.replace("_", " ").title(), fmt)
        console.print(wc)
