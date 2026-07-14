"""Structured logging.

JSON-line logs to file for machine parsing, plus a colorized console handler
via rich. One ``get_logger`` factory keeps handlers from duplicating.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

try:
    from rich.logging import RichHandler
except ImportError:
    RichHandler = None


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for k, v in getattr(record, "extra_fields", {}).items():
            payload[k] = v
        return json.dumps(payload)


def get_logger(name: str = "regime-trader", log_dir: str = "logs") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    Path(log_dir).mkdir(exist_ok=True)
    file_handler = logging.FileHandler(Path(log_dir) / "trader.jsonl")
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)

    if RichHandler is not None:
        console = RichHandler(rich_tracebacks=True, show_path=False)
    else:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(console)
    return logger
