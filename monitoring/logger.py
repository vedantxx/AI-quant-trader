"""Structured logging.

JSON-line logs to file for machine parsing, plus a colorized console handler via
rich.

Stub scaffold: signatures, type hints, and docstrings only — no logic yet.
"""

from __future__ import annotations

import logging


class TradingLogger:
    """Structured JSON-line + rich console logger."""

    def __init__(self, name: str = "regime-trader", log_dir: str = "logs") -> None:
        """Configure file (JSON-line) and console (rich) handlers once."""
        ...

    def get(self) -> logging.Logger:
        """Return the configured stdlib logger."""
        ...

    def event(self, level: str, msg: str, **fields: object) -> None:
        """Log a structured event with extra key/value fields."""
        ...
