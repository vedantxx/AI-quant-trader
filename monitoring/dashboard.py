"""Terminal-based live dashboard using rich.

Renders account equity, current regime, open positions, and P&L, refreshing at
``monitoring.dashboard_refresh_seconds``.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
except ImportError:
    Console = Live = Table = Panel = Layout = None


@dataclass
class DashboardState:
    equity: float
    regime: str
    confidence: float
    positions: dict           # symbol -> {qty, market_value, unrealized_pl}
    halted: bool
    halt_reason: str


class Dashboard:
    def __init__(self, cfg: dict):
        self.refresh = cfg["monitoring"]["dashboard_refresh_seconds"]
        self.console = Console() if Console else None

    def _render(self, state: DashboardState):
        header = Panel(
            f"Equity ${state.equity:,.2f}   "
            f"Regime [bold]{state.regime}[/] ({state.confidence:.0%})   "
            + ("[red]HALTED[/]: " + state.halt_reason if state.halted else "[green]LIVE[/]"),
            title="AI-Quant-trader",
        )
        table = Table(expand=True)
        for col in ("Symbol", "Qty", "Value", "Unreal P&L"):
            table.add_column(col)
        for sym, p in state.positions.items():
            pl = p["unrealized_pl"]
            color = "green" if pl >= 0 else "red"
            table.add_row(
                sym,
                f"{p['qty']:g}",
                f"${p['market_value']:,.2f}",
                f"[{color}]${pl:,.2f}[/]",
            )
        layout = Layout()
        layout.split_column(Layout(header, size=3), Layout(table))
        return layout

    def run(self, state_provider):
        """Block, refreshing from ``state_provider()`` -> DashboardState."""
        if Live is None:
            raise ImportError("rich not installed")
        with Live(refresh_per_second=1, screen=True) as live:
            import time
            while True:
                live.update(self._render(state_provider()))
                time.sleep(self.refresh)

    def render_once(self, state: DashboardState):
        if self.console:
            self.console.print(self._render(state))
