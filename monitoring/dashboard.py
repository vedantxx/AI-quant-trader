"""Terminal-based live dashboard using rich.

Renders account equity, current regime, open positions, and P&L, refreshing at
``monitoring.dashboard_refresh_seconds``.
"""

from __future__ import annotations

import time
from typing import Callable


class Dashboard:
    """Rich-powered live terminal dashboard."""

    def __init__(self, cfg: dict) -> None:
        self.refresh = float(cfg.get("monitoring", {}).get("dashboard_refresh_seconds", 5))
        from rich.console import Console
        self.console = Console()

    def render(self, state: dict) -> object:
        """Build a rich renderable from a dashboard state dict."""
        from rich.table import Table

        t = Table(title="AI-Quant-Trader", expand=True)
        t.add_column("field", style="cyan")
        t.add_column("value", justify="right")
        for k in ("status", "regime", "regime_prob", "uncertainty", "equity",
                  "cash", "buying_power", "drawdown", "breaker", "open_positions",
                  "trades_today", "last_bar"):
            if k in state:
                v = state[k]
                if isinstance(v, float):
                    v = f"{v:,.2f}"
                t.add_row(k.replace("_", " "), str(v))

        positions = state.get("positions") or {}
        if positions:
            pt = Table(title="Positions")
            for c in ("symbol", "qty", "entry", "price", "uPnL", "stop"):
                pt.add_column(c, justify="right")
            for sym, p in positions.items():
                pt.add_row(sym, f"{p.get('qty', 0):g}", f"{p.get('entry', 0):.2f}",
                           f"{p.get('price', 0):.2f}", f"{p.get('upnl', 0):.2f}",
                           f"{p.get('stop', 0):.2f}")
            from rich.console import Group
            return Group(t, pt)
        return t

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
