"""Terminal-based live dashboard using rich.

Six panels — REGIME, PORTFOLIO, POSITIONS, RECENT SIGNALS, RISK STATUS, SYSTEM —
refreshing at ``monitoring.dashboard_refresh_seconds`` with color-coded risk
bars. The renderer is pure: it consumes a plain ``state`` dict, so it works the
same live or in a one-shot snapshot.
"""

from __future__ import annotations

import time
from typing import Callable


def _risk_bar(value: float, limit: float, width: int = 20) -> str:
    """Color-coded progress bar: green < 50%, yellow < 80%, red beyond."""
    frac = 0.0 if limit <= 0 else max(0.0, min(1.0, value / limit))
    filled = int(round(frac * width))
    color = "green" if frac < 0.5 else "yellow" if frac < 0.8 else "red"
    bar = "█" * filled + "░" * (width - filled)
    return f"[{color}]{bar}[/{color}] {value:.2%}/{limit:.0%}"


class Dashboard:
    """Rich-powered live terminal dashboard."""

    def __init__(self, cfg: dict) -> None:
        self.refresh = float(cfg.get("monitoring", {}).get("dashboard_refresh_seconds", 5))
        from rich.console import Console
        self.console = Console()

    # ------------------------------------------------------------- panels
    @staticmethod
    def _regime(s: dict):
        from rich.panel import Panel
        prob = s.get("regime_prob", 0.0) or 0.0
        txt = (f"[bold]{s.get('regime', '—')}[/bold] ({prob:.0%})   "
               f"Stability: {s.get('stability_bars', 0)} bars   "
               f"Flicker: {s.get('flicker', 0)}/{s.get('flicker_window', 20)}")
        if s.get("uncertainty"):
            txt += "   [yellow]UNCERTAIN[/yellow]"
        return Panel(txt, title="REGIME", border_style="cyan")

    @staticmethod
    def _portfolio(s: dict):
        from rich.panel import Panel
        pnl = s.get("daily_pnl", 0.0) or 0.0
        pct = s.get("daily_pnl_pct", 0.0) or 0.0
        color = "green" if pnl >= 0 else "red"
        txt = (f"Equity: [bold]${s.get('equity', 0):,.0f}[/bold]   "
               f"Daily: [{color}]{pnl:+,.0f} ({pct:+.2%})[/{color}]   "
               f"Allocation: {s.get('allocation', 0):.0%}   "
               f"Leverage: {s.get('leverage', 1.0):.2f}x")
        return Panel(txt, title="PORTFOLIO", border_style="cyan")

    @staticmethod
    def _positions(s: dict):
        from rich.panel import Panel
        from rich.table import Table
        t = Table.grid(padding=(0, 2))
        for _ in range(6):
            t.add_column()
        t.add_row("SYMBOL", "SIDE", "PRICE", "P&L", "STOP", "HELD")
        for p in s.get("positions", []) or []:
            pnl = p.get("pnl_pct", 0.0)
            color = "green" if pnl >= 0 else "red"
            t.add_row(p.get("symbol", ""), p.get("side", "LONG"),
                      f"${p.get('price', 0):,.2f}", f"[{color}]{pnl:+.1%}[/{color}]",
                      f"${p.get('stop', 0):,.2f}", p.get("held", "—"))
        return Panel(t, title="POSITIONS", border_style="cyan")

    @staticmethod
    def _signals(s: dict):
        from rich.panel import Panel
        from rich.table import Table
        t = Table.grid(padding=(0, 2))
        for _ in range(4):
            t.add_column()
        for sig in (s.get("recent_signals", []) or [])[-6:]:
            t.add_row(sig.get("time", ""), sig.get("symbol", ""),
                      sig.get("change", ""), sig.get("reason", ""))
        return Panel(t, title="RECENT SIGNALS", border_style="cyan")

    @staticmethod
    def _risk(s: dict):
        from rich.panel import Panel
        daily = (f"Daily DD:     {_risk_bar(s.get('daily_dd', 0), s.get('daily_dd_limit', 0.03))}")
        peak = (f"From Peak:    {_risk_bar(s.get('peak_dd', 0), s.get('peak_dd_limit', 0.10))}")
        breaker = s.get("breaker", "none")
        bstyle = "red" if breaker not in ("none", None) else "green"
        return Panel(f"{daily}\n{peak}\nBreaker: [{bstyle}]{breaker}[/{bstyle}]",
                     title="RISK STATUS", border_style="cyan")

    @staticmethod
    def _system(s: dict):
        from rich.panel import Panel

        def ok(flag):
            return "[green]OK[/green]" if flag else "[red]DOWN[/red]"
        txt = (f"Data: {ok(s.get('data_ok', True))}   "
               f"API: {ok(s.get('api_ok', True))} {s.get('api_ms', 0):.0f}ms   "
               f"HMM: {s.get('hmm_age', '—')}   "
               f"[bold]{'PAPER' if s.get('paper', True) else 'LIVE'}[/bold]")
        return Panel(txt, title="SYSTEM", border_style="cyan")

    def render(self, state: dict):
        """Build the full dashboard renderable from a state dict."""
        from rich.console import Group
        return Group(self._regime(state), self._portfolio(state), self._positions(state),
                     self._signals(state), self._risk(state), self._system(state))

    def render_once(self, state: dict) -> None:
        """Render a single frame to the console."""
        self.console.print(self.render(state))

    def run(self, state_provider: Callable[[], dict]) -> None:
        """Block, refreshing from ``state_provider()`` on the configured cadence."""
        from rich.live import Live
        with Live(self.render(state_provider()), console=self.console,
                  refresh_per_second=1) as live:
            while True:
                time.sleep(self.refresh)
                live.update(self.render(state_provider()))
