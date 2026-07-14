"""Email / webhook alerts for critical events, with rate limiting.

Same alert type won't fire more than once per
``monitoring.alert_rate_limit_minutes``.
"""

from __future__ import annotations

import json
import smtplib
import urllib.request
from datetime import datetime, timedelta
from email.mime.text import MIMEText


class AlertManager:
    def __init__(self, cfg: dict, credentials: dict | None = None):
        self.rate_limit = timedelta(
            minutes=cfg["monitoring"]["alert_rate_limit_minutes"]
        )
        self._last_sent: dict[str, datetime] = {}
        self.creds = (credentials or {}).get("alerts", {})

    def _rate_limited(self, key: str) -> bool:
        last = self._last_sent.get(key)
        if last and datetime.utcnow() - last < self.rate_limit:
            return True
        self._last_sent[key] = datetime.utcnow()
        return False

    def alert(self, key: str, subject: str, body: str) -> bool:
        """Send an alert of type ``key``. Returns False if rate-limited."""
        if self._rate_limited(key):
            return False
        self._send_email(subject, body)
        self._send_webhook(subject, body)
        return True

    def _send_email(self, subject: str, body: str) -> None:
        email = self.creds.get("email", {})
        if not email.get("enabled"):
            return
        msg = MIMEText(body)
        msg["Subject"] = f"[AI-Quant-trader] {subject}"
        msg["From"] = email["username"]
        msg["To"] = ", ".join(email["to"])
        try:
            with smtplib.SMTP(email["smtp_host"], email["smtp_port"]) as s:
                s.starttls()
                s.login(email["username"], email["password"])
                s.send_message(msg)
        except Exception:
            pass  # never let alerting crash the trading loop

    def _send_webhook(self, subject: str, body: str) -> None:
        hook = self.creds.get("webhook", {})
        if not hook.get("enabled"):
            return
        data = json.dumps({"text": f"*{subject}*\n{body}"}).encode()
        try:
            req = urllib.request.Request(
                hook["url"], data=data, headers={"Content-Type": "application/json"}
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass
