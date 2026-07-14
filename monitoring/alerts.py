"""Email / webhook alerts for critical events, with rate limiting.

The same alert type won't fire more than once per
``monitoring.alert_rate_limit_minutes``.

Stub scaffold: signatures, type hints, and docstrings only — no logic yet.
"""

from __future__ import annotations

from typing import Optional


class AlertManager:
    """Rate-limited email + webhook alerting."""

    def __init__(self, cfg: dict, credentials: Optional[dict] = None) -> None:
        """Read rate limit from config; store alert credentials."""
        ...

    def alert(self, key: str, subject: str, body: str) -> bool:
        """Send an alert of type ``key``. Returns False if rate-limited."""
        ...

    def _rate_limited(self, key: str) -> bool:
        """True if an alert of this type fired within the rate-limit window."""
        ...

    def _send_email(self, subject: str, body: str) -> None:
        """Send an email alert if email delivery is enabled."""
        ...

    def _send_webhook(self, subject: str, body: str) -> None:
        """POST an alert to the configured webhook if enabled."""
        ...
