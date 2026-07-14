"""Terminal-based live dashboard using rich.

Renders account equity, current regime, open positions, and P&L, refreshing at
``monitoring.dashboard_refresh_seconds``.

Stub scaffold: signatures, type hints, and docstrings only — no logic yet.
"""

from __future__ import annotations

from typing import Callable


class Dashboard:
    """Rich-powered live terminal dashboard."""

    def __init__(self, cfg: dict) -> None:
        """Read monitoring.dashboard_refresh_seconds; init the console."""
        ...

    def render(self, state: dict) -> object:
        """Build a rich renderable from a dashboard state dict."""
        ...

    def render_once(self, state: dict) -> None:
        """Render a single frame to the console."""
        ...

    def run(self, state_provider: Callable[[], dict]) -> None:
        """Block, refreshing from ``state_provider()`` on the configured cadence."""
        ...
