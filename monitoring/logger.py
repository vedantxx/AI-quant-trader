"""Structured logging.

JSON-line logs to file for machine parsing, plus a colorized console handler via
rich (falling back to a plain stream handler if rich is unavailable).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path


class _JsonLineFormatter(logging.Formatter):
    """One JSON object per log line; merges structured ``extra`` fields."""

    _RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
        "message", "asctime", "taskName"}

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k not in self._RESERVED and not k.startswith("_"):
                payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TradingLogger:
    """Structured JSON-line + rich console logger."""

    def __init__(self, name: str = "regime-trader", log_dir: str = "logs") -> None:
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        if self.logger.handlers:  # configure once
            return

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / "trading.jsonl")
        fh.setFormatter(_JsonLineFormatter())
        self.logger.addHandler(fh)

        try:
            from rich.logging import RichHandler
            ch: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False)
            ch.setFormatter(logging.Formatter("%(message)s"))
        except ImportError:  # pragma: no cover
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.addHandler(ch)

    def get(self) -> logging.Logger:
        """Return the configured stdlib logger."""
        return self.logger

    def event(self, level: str, msg: str, **fields: object) -> None:
        """Log a structured event with extra key/value fields."""
        self.logger.log(getattr(logging, level.upper(), logging.INFO), msg, extra=fields)
