"""Streamlit UI for AI-Quant-trader.

Live control panel over the TradingSystem orchestration loop: regime, portfolio,
positions, signals, risk bars, and system health. Runs the same pipeline as
`python main.py trade` — data -> HMM -> allocation -> risk -> (paper) orders.

Run:
    streamlit run ui/streamlit_app.py

Safety: dry-run is ON by default (no orders placed). The account is paper unless
config/.env say otherwise; live orders still require the typed confirmation in
AlpacaClient.
"""

from __future__ import annotations

import copy
import os
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

import main as app  # noqa: E402

load_dotenv(dotenv_path=str(ROOT / ".env"))

st.set_page_config(page_title="AI-Quant-Trader", page_icon="📈", layout="wide")


def _app_password() -> str | None:
    """Password from Streamlit secrets or env (None -> open, local dev only)."""
    try:
        pw = st.secrets.get("APP_PASSWORD")
    except Exception:  # no secrets file configured
        pw = None
    return pw or os.getenv("APP_PASSWORD")


def _require_auth() -> bool:
    """Gate the app behind a password when one is configured (public hosting)."""
    pw = _app_password()
    if not pw:
        return True  # no password set: unrestricted (intended for local runs)
    if st.session_state.get("authed"):
        return True
    st.title("🔒 AI-Quant-Trader")
    with st.form("login"):
        entered = st.text_input("Password", type="password")
        if st.form_submit_button("Enter") and entered == pw:
            st.session_state.authed = True
            st.rerun()
    return bool(st.session_state.get("authed"))


if not _require_auth():
    st.stop()


def _fast_cfg(base: dict, fast: bool, symbols: list[str], regime_symbol: str) -> dict:
    cfg = copy.deepcopy(base)
    cfg["broker"]["symbols"] = symbols
    cfg["broker"]["regime_symbol"] = regime_symbol
    if fast:  # trim HMM search so the UI starts in seconds, not minutes
        cfg["hmm"]["n_candidates"] = [3, 4]
        cfg["hmm"]["n_init"] = 3
        cfg["hmm"]["covariance_type"] = "diag"
        cfg.setdefault("features", {})["zscore_window"] = 40
    return cfg


def _load_secrets_into_env() -> None:
    """Copy Alpaca keys from Streamlit secrets into the environment.

    On Streamlit Cloud, secrets are set in the dashboard (Settings -> Secrets);
    AlpacaClient reads them from os.environ. Locally, .env already populated it.
    """
    for key in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
        if os.getenv(key):
            continue
        try:
            val = st.secrets.get(key)
        except Exception:
            val = None
        if val:
            os.environ[key] = str(val)


def _creds_present() -> bool:
    _load_secrets_into_env()
    return bool(os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"))


def _init_system(cfg: dict, dry_run: bool):
    sysm = app.TradingSystem(cfg, dry_run=dry_run)
    sysm.startup()
    return sysm


# ------------------------------------------------------------------- sidebar
base_cfg = app.load_config()
st.sidebar.title("⚙️ Control")
all_syms = base_cfg["broker"]["symbols"]
symbols = st.sidebar.multiselect("Symbols", all_syms, default=all_syms[:3] or ["SPY"])
regime_symbol = st.sidebar.selectbox("Regime symbol", symbols or ["SPY"])
dry_run = st.sidebar.toggle("Dry-run (no orders)", value=True)
fast = st.sidebar.toggle("Fast HMM (quick start)", value=True)
auto = st.sidebar.toggle("Auto-refresh", value=False)
interval = st.sidebar.slider("Refresh seconds", 5, 60, 15)

if not dry_run:
    st.sidebar.warning("LIVE-ORDER mode: signals will be submitted to the broker.")

c1, c2 = st.sidebar.columns(2)
start = c1.button("▶ Start / Retrain", use_container_width=True)
step = c2.button("↻ Run once", use_container_width=True)

# ------------------------------------------------------------------- lifecycle
if not _creds_present():
    st.error(
        "Alpaca API keys not found. On Streamlit Cloud, open **Manage app → "
        "Settings → Secrets** and add:\n\n"
        "```toml\nALPACA_API_KEY = \"your_paper_key\"\n"
        "ALPACA_SECRET_KEY = \"your_paper_secret\"\n"
        "APP_PASSWORD = \"a-strong-password\"\n```\n\n"
        "Save, then rerun. Locally, put them in `.env` instead."
    )
    st.stop()

if start or "system" not in st.session_state:
    if not symbols:
        st.warning("Select at least one symbol.")
        st.stop()
    with st.spinner("Training HMM and connecting to broker…"):
        cfg = _fast_cfg(base_cfg, fast, symbols, regime_symbol)
        try:
            st.session_state.system = _init_system(cfg, dry_run)
            st.session_state.summary = st.session_state.system.run_once()
        except ImportError as exc:
            st.error(
                f"A required package failed to import: `{exc}`.\n\n"
                "`hmmlearn` ships wheels only for **Python 3.11 / 3.12**. In "
                "**Manage app → Settings**, set the Python version to **3.12** "
                "(or recreate the app choosing 3.12), then **Reboot**."
            )
            st.stop()
        except Exception as exc:  # noqa: BLE001 - surface any startup failure cleanly
            st.error(f"Startup failed: {exc}")
            st.stop()
    st.success("System online.")

system = st.session_state.system
system.dry_run = dry_run  # honor toggle without full restart

if step or auto:
    with st.spinner("Running pipeline iteration…"):
        try:
            st.session_state.summary = system.run_once()
        except Exception as exc:  # noqa: BLE001 - keep the loop alive on a transient error
            st.warning(f"Iteration error (will retry): {exc}")

s = st.session_state.get("summary", {})

# ------------------------------------------------------------------- header
mode = "PAPER" if s.get("paper", True) else "LIVE"
order_mode = "DRY-RUN" if dry_run else "LIVE-ORDER"
st.title("📈 AI-Quant-Trader")
st.caption(f"{mode} · {order_mode} · regime symbol {regime_symbol} · "
           f"last bar {s.get('last_bar', '—')} · updated {s.get('updated', '—')}")
if not auto:
    st.caption("↻ Auto-refresh is OFF — enable it in the sidebar for a live feed, "
               "or click **Run once**.")

if s.get("status") == "no_equity":
    st.error("Account equity is $0 — reset/refund the paper account. Holding (no sizing).")

# ------------------------------------------------------------------- regime
st.subheader("Regime")
r1, r2, r3, r4 = st.columns(4)
r1.metric("Regime", s.get("regime", "—"), f"{(s.get('regime_prob') or 0):.0%} conf")
r2.metric("Stability", f"{s.get('stability_bars', 0)} bars")
r3.metric("Flicker", f"{s.get('flicker', 0)}/{s.get('flicker_window', 20)}")
r4.metric("Uncertainty", "YES" if s.get("uncertainty") else "no")

# ------------------------------------------------------------------- portfolio
st.subheader("Portfolio")
p1, p2, p3, p4 = st.columns(4)
p1.metric("Equity", f"${s.get('equity', 0):,.0f}")
p2.metric("Daily P&L", f"${s.get('daily_pnl', 0):,.0f}", f"{(s.get('daily_pnl_pct') or 0):+.2%}")
p3.metric("Allocation", f"{(s.get('allocation') or 0):.0%}")
p4.metric("Leverage", f"{s.get('leverage', 1.0):.2f}x")

# ------------------------------------------------------------------- risk
st.subheader("Risk")
d_dd, d_lim = s.get("daily_dd", 0.0), s.get("daily_dd_limit", 0.03)
p_dd, p_lim = s.get("peak_dd", 0.0), s.get("peak_dd_limit", 0.10)
st.write(f"Daily drawdown: {d_dd:.2%} / {d_lim:.0%}")
st.progress(min(1.0, d_dd / d_lim if d_lim else 0.0))
st.write(f"From peak: {p_dd:.2%} / {p_lim:.0%}")
st.progress(min(1.0, p_dd / p_lim if p_lim else 0.0))
breaker = s.get("breaker", "none")
if breaker not in ("none", None):
    st.error(f"Circuit breaker: {breaker}")
else:
    st.success(f"Circuit breaker: {breaker}")

# ------------------------------------------------------------------- routing
o1, o2, o3 = st.columns(3)
o1.metric("Approved", s.get("approved", 0))
o2.metric("Modified", s.get("modified", 0))
o3.metric("Rejected", s.get("rejected", 0))

# ------------------------------------------------------------------- positions
st.subheader("Positions")
positions = s.get("positions") or []
if positions:
    st.dataframe(positions, use_container_width=True)
else:
    st.info("No open positions.")

# ------------------------------------------------------------------- signals
st.subheader("Recent signals")
sigs = s.get("recent_signals") or []
if sigs:
    st.dataframe(list(reversed(sigs)), use_container_width=True)
else:
    st.info("No signals yet.")

# ------------------------------------------------------------------- system
st.subheader("System")
y1, y2, y3, y4 = st.columns(4)
y1.metric("Data", "OK" if s.get("data_ok", True) else "DOWN")
y2.metric("API", "OK" if s.get("api_ok", True) else "DOWN")
y3.metric("HMM", s.get("hmm_age", "—"))
y4.metric("Open positions", s.get("open_positions", 0))

# ------------------------------------------------------------------- auto loop
if auto:
    time.sleep(interval)
    st.rerun()
