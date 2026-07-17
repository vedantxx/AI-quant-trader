"""Data layer for the dashboard: system lifecycle, secrets, snapshot access.

Keeps all TradingSystem interaction out of the presentation modules. The heavy
object lives in ``st.session_state``; the UI asks this module for a fresh
read-only snapshot.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

import main as app  # noqa: E402

load_dotenv(dotenv_path=str(ROOT / ".env"))

MAX_EQUITY_POINTS = 500


# ------------------------------------------------------------------- secrets
def app_password() -> str | None:
    try:
        pw = st.secrets.get("APP_PASSWORD")
    except Exception:
        pw = None
    return pw or os.getenv("APP_PASSWORD")


def load_secrets_into_env() -> None:
    for key in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
        if os.getenv(key):
            continue
        try:
            val = st.secrets.get(key)
        except Exception:
            val = None
        if val:
            os.environ[key] = str(val)


def creds_present() -> bool:
    load_secrets_into_env()
    return bool(os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"))


# ------------------------------------------------------------------- config
def base_config() -> dict:
    return app.load_config()


def build_config(base: dict, fast: bool, symbols: list[str], regime_symbol: str) -> dict:
    cfg = copy.deepcopy(base)
    cfg["broker"]["symbols"] = symbols
    cfg["broker"]["regime_symbol"] = regime_symbol
    if fast:
        cfg["hmm"]["n_candidates"] = [3, 4]
        cfg["hmm"]["n_init"] = 3
        cfg["hmm"]["covariance_type"] = "diag"
        cfg.setdefault("features", {})["zscore_window"] = 40
    return cfg


# ------------------------------------------------------------------- lifecycle
def init_system(cfg: dict, dry_run: bool):
    sysm = app.TradingSystem(cfg, dry_run=dry_run)
    sysm.startup()
    return sysm


def take_snapshot(system) -> dict:
    """Read-only snapshot; also append equity to the session history series."""
    snap = system.snapshot()
    hist = st.session_state.setdefault("equity_hist", [])
    hist.append({"t": snap["ts"], "equity": snap["topbar"]["equity"]})
    del hist[:-MAX_EQUITY_POINTS]
    snap["equity_hist"] = hist
    return snap


def run_iteration(system) -> dict:
    """Guarded: run one full loop pass (may place orders when dry_run is off)."""
    system.run_once()
    return take_snapshot(system)
