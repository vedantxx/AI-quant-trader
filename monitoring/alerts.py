"""Email / webhook alerts for critical events, with rate limiting.

The same alert type won't fire more than once per
``monitoring.alert_rate_limit_minutes``. Delivery channels are opt-in via
credentials; when neither is configured, alerts are logged only.
"""

from __future__ import annotations

import logging
import smtplib
import urllib.request
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Optional

logger = logging.getLogger("regime-trader.alerts")


class AlertManager:
    """Rate-limited email + webhook alerting."""

    def __init__(self, cfg: dict, credentials: Optional[dict] = None) -> None:
        mins = cfg.get("monitoring", {}).get("alert_rate_limit_minutes", 15)
        self.window = timedelta(minutes=mins)
        self.creds = credentials or {}
        self._last: dict[str, datetime] = {}

    def alert(self, key: str, subject: str, body: str) -> bool:
        """Send an alert of type ``key``. Returns False if rate-limited."""
        if self._rate_limited(key):
            logger.debug("alert %s rate-limited", key)
            return False
        self._last[key] = datetime.now(timezone.utc)
        logger.warning("ALERT[%s] %s — %s", key, subject, body)
        try:
            self._send_email(subject, body)
            self._send_webhook(subject, body)
        except Exception as exc:  # noqa: BLE001 - never let alerting crash the loop
            logger.error("alert delivery failed: %s", exc)
        return True

    def _rate_limited(self, key: str) -> bool:
        """True if an alert of this type fired within the rate-limit window."""
        last = self._last.get(key)
        return last is not None and datetime.now(timezone.utc) - last < self.window

    def _send_email(self, subject: str, body: str) -> None:
        """Send an email alert if email delivery is configured."""
        email = self.creds.get("email")
        if not email:
            return
        msg = EmailMessage()
        msg["Subject"], msg["From"], msg["To"] = subject, email["from"], email["to"]
        msg.set_content(body)
        with smtplib.SMTP(email["host"], int(email.get("port", 587))) as s:
            s.starttls()
            if email.get("username"):
                s.login(email["username"], email["password"])
            s.send_message(msg)

    def _send_webhook(self, subject: str, body: str) -> None:
        """POST an alert to the configured webhook if enabled."""
        url = self.creds.get("webhook_url")
        if not url:
            return
        import json
        data = json.dumps({"text": f"{subject}\n{body}"}).encode()
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5)  # noqa: S310 - operator-configured URL
