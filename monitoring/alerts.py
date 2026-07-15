"""Email / webhook alerts for critical events, with rate limiting.

Typed triggers (regime change, circuit breaker, large P&L, data feed down, API
lost, HMM retrained, flicker exceeded) fan out to console, an optional
``alerts.log`` sink, and optional email/webhook delivery. The same alert type
fires at most once per ``monitoring.alert_rate_limit_minutes``.
"""

from __future__ import annotations

import logging
import smtplib
import urllib.request
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Callable, Optional

logger = logging.getLogger("regime-trader.alerts")


class AlertManager:
    """Rate-limited email + webhook alerting with typed triggers."""

    def __init__(
        self, cfg: dict, credentials: Optional[dict] = None,
        log_sink: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        mins = cfg.get("monitoring", {}).get("alert_rate_limit_minutes", 15)
        self.window = timedelta(minutes=mins)
        self.creds = credentials or {}
        self.log_sink = log_sink            # e.g. TradingLogger.alert_log
        self._last: dict[str, datetime] = {}

    # ------------------------------------------------------------- core
    def alert(self, key: str, subject: str, body: str) -> bool:
        """Send an alert of type ``key``. Returns False if rate-limited."""
        if self._rate_limited(key):
            logger.debug("alert %s rate-limited", key)
            return False
        self._last[key] = datetime.now(timezone.utc)
        logger.warning("ALERT[%s] %s — %s", key, subject, body)
        if self.log_sink:
            try:
                self.log_sink(subject, key=key, body=body)
            except Exception:  # noqa: BLE001
                pass
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

    # ------------------------------------------------------------- triggers
    def regime_change(self, old: str, new: str, prob: float) -> bool:
        return self.alert("regime_change", "Regime change",
                          f"{old} -> {new} (p={prob:.0%})")

    def circuit_breaker(self, kind: str, detail: str) -> bool:
        return self.alert("breaker", f"Circuit breaker: {kind}", detail)

    def large_pnl(self, daily_pnl: float, pct: float) -> bool:
        return self.alert("large_pnl", "Large P&L move",
                          f"daily {daily_pnl:+,.0f} ({pct:+.2%})")

    def data_feed_down(self, detail: str) -> bool:
        return self.alert("data_down", "Data feed down", detail)

    def api_lost(self, detail: str) -> bool:
        return self.alert("api_lost", "Broker API lost", detail)

    def hmm_retrained(self, n_regimes: int, bic: float) -> bool:
        return self.alert("hmm_retrained", "HMM retrained",
                          f"{n_regimes} regimes, BIC={bic:.0f}")

    def flicker_exceeded(self, rate: int, threshold: int) -> bool:
        return self.alert("flicker", "Flicker rate exceeded",
                          f"{rate} changes > threshold {threshold}")

    # ------------------------------------------------------------- delivery
    def _send_email(self, subject: str, body: str) -> None:
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
        url = self.creds.get("webhook_url")
        if not url:
            return
        import json
        data = json.dumps({"text": f"{subject}\n{body}"}).encode()
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5)  # noqa: S310 - operator-configured URL
