"""Structured logging.

Four rotating JSON-line log files (10 MB per file, 30 backups) plus a colorized
console handler via rich:

    main.log     everything (also to console)
    trades.log   order / fill events
    alerts.log   alert deliveries
    regime.log   regime predictions and changes

Every JSON entry carries a rolling context — timestamp, regime, probability,
equity, positions, daily_pnl — injected by a logging filter so machine parsers
see the full state on each line.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

MAX_BYTES = 10 * 1024 * 1024   # 10 MB per file
BACKUPS = 30                    # keep 30 rotated files
_CONTEXT_KEYS = ("regime", "probability", "equity", "positions", "daily_pnl")


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


class _ContextFilter(logging.Filter):
    """Inject the current rolling context into every record (unless overridden)."""

    def __init__(self) -> None:
        super().__init__()
        self.context: dict = {k: None for k in _CONTEXT_KEYS}

    def filter(self, record: logging.LogRecord) -> bool:
        for k, v in self.context.items():
            if not hasattr(record, k):
                setattr(record, k, v)
        return True


class TradingLogger:
    """Structured JSON-line (rotating, 4 files) + rich console logger."""

    def __init__(self, name: str = "regime-trader", log_dir: str = "logs") -> None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self._ctx = _ContextFilter()

        self.logger = self._file_logger(name, Path(log_dir) / "main.log", console=True)
        self.trades = self._file_logger(f"{name}.trades", Path(log_dir) / "trades.log")
        self.regime_log = self._file_logger(f"{name}.regime", Path(log_dir) / "regime.log")
        self.alerts_log = self._file_logger(f"{name}.alerts", Path(log_dir) / "alerts.log")

    def _file_logger(self, name: str, path: Path, console: bool = False) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if logger.handlers:  # configure once
            return logger

        fh = RotatingFileHandler(path, maxBytes=MAX_BYTES, backupCount=BACKUPS)
        fh.setFormatter(_JsonLineFormatter())
        fh.addFilter(self._ctx)
        logger.addHandler(fh)

        if console:
            try:
                from rich.logging import RichHandler
                ch: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False)
                ch.setFormatter(logging.Formatter("%(message)s"))
            except ImportError:  # pragma: no cover
                ch = logging.StreamHandler()
                ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            ch.addFilter(self._ctx)
            logger.addHandler(ch)
        return logger

    # ------------------------------------------------------------- api
    def get(self) -> logging.Logger:
        """Return the main stdlib logger."""
        return self.logger

    def set_context(self, **fields: object) -> None:
        """Update the rolling context stamped onto every subsequent record."""
        self._ctx.context.update(fields)

    def event(self, level: str, msg: str, **fields: object) -> None:
        """Log a structured event to main.log with extra fields."""
        self.logger.log(getattr(logging, level.upper(), logging.INFO), msg, extra=fields)

    def trade(self, msg: str, **fields: object) -> None:
        """Log an order/fill event to trades.log."""
        self.trades.info(msg, extra=fields)

    def regime(self, msg: str, **fields: object) -> None:
        """Log a regime prediction/change to regime.log."""
        self.regime_log.info(msg, extra=fields)

    def alert_log(self, msg: str, **fields: object) -> None:
        """Log an alert delivery to alerts.log."""
        self.alerts_log.warning(msg, extra=fields)
