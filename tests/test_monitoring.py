"""Tests for monitoring: logger, dashboard, alerts."""

import json

import pytest

from monitoring.alerts import AlertManager
from monitoring.dashboard import Dashboard, _risk_bar
from monitoring.logger import TradingLogger

CFG = {"monitoring": {"dashboard_refresh_seconds": 5, "alert_rate_limit_minutes": 15}}


# --------------------------------------------------------------------- logger
def test_logger_writes_four_json_files(tmp_path):
    tl = TradingLogger(name="test-rt", log_dir=str(tmp_path))
    tl.set_context(regime="BULL", equity=100000)
    tl.get().info("hello")
    tl.trade("order", symbol="SPY", qty=10)
    tl.regime("regime", probability=0.9)
    tl.alert_log("boom", key="breaker")
    for name in ("main.log", "trades.log", "regime.log", "alerts.log"):
        assert (tmp_path / name).exists()
    line = json.loads((tmp_path / "main.log").read_text().splitlines()[0])
    assert line["regime"] == "BULL" and line["equity"] == 100000 and "ts" in line


# --------------------------------------------------------------------- alerts
def test_alert_rate_limited():
    a = AlertManager(CFG)
    assert a.alert("k", "s", "b") is True
    assert a.alert("k", "s", "b") is False       # within window
    assert a.regime_change("BULL", "BEAR", 0.8) is True  # different key


def test_alert_log_sink_called():
    seen = []
    a = AlertManager(CFG, log_sink=lambda subj, **kw: seen.append((subj, kw["key"])))
    a.circuit_breaker("peak_dd", "down 11%")
    assert seen and seen[0][1] == "breaker"


# ------------------------------------------------------------------ dashboard
def test_risk_bar_color_thresholds():
    assert "green" in _risk_bar(0.001, 0.10)
    assert "yellow" in _risk_bar(0.06, 0.10)
    assert "red" in _risk_bar(0.09, 0.10)


def test_dashboard_render_no_crash():
    Dashboard(CFG).render({"regime": "BULL", "positions": [], "recent_signals": []})
