"""AI-Quant-Trader — portfolio operating console (Streamlit).

Componentized multi-tab dashboard over the TradingSystem. Read-only by default:
it renders live snapshots; placing orders is a guarded action (Run once with
Dry-run off). Data access lives in ui/system.py, primitives in ui/components.py,
tab renderers in ui/pages.py.

Run:  streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from ui import components as c  # noqa: E402
from ui import pages  # noqa: E402
from ui import system as sysmod  # noqa: E402

st.set_page_config(page_title="AI-Quant-Trader", page_icon="📈", layout="wide")
c.inject_css()


# ------------------------------------------------------------------- auth
def _require_auth() -> bool:
    pw = sysmod.app_password()
    if not pw:
        return True
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

base_cfg = sysmod.base_config()
all_syms = base_cfg["broker"]["symbols"]

# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### ⚙️ Control")
    symbols = st.multiselect("Universe", all_syms, default=all_syms)
    regime_symbol = st.selectbox("Regime symbol", symbols or ["SPY"])
    dry_run = st.toggle("Dry-run (no orders)", value=True)
    fast = st.toggle("Fast HMM", value=True)
    st.divider()
    auto = st.toggle("Auto-refresh", value=False)
    interval = st.slider("Refresh (s)", 5, 60, 15)
    st.divider()
    b1, b2 = st.columns(2)
    start = b1.button("▶ Start / Retrain", use_container_width=True)
    step = b2.button("↻ Run once", use_container_width=True,
                     help="Runs a full pipeline pass. Places orders if Dry-run is off.")
    refresh = st.button("⟳ Refresh data", use_container_width=True)
    if not dry_run:
        st.warning("LIVE-ORDER: Run once will submit to the broker.")

# ------------------------------------------------------------------- creds
if not sysmod.creds_present():
    st.error("Alpaca keys missing. Add `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` in "
             "**Settings → Secrets** (cloud) or `.env` (local).")
    st.stop()

# ------------------------------------------------------------------- lifecycle
if start or "system" not in st.session_state:
    if not symbols:
        st.warning("Select at least one symbol.")
        st.stop()
    with st.spinner("Training HMM and connecting to broker…"):
        try:
            cfg = sysmod.build_config(base_cfg, fast, symbols, regime_symbol)
            st.session_state.system = sysmod.init_system(cfg, dry_run)
            st.session_state.snap = sysmod.take_snapshot(st.session_state.system)
        except ImportError as exc:
            st.error(f"Import failed (`{exc}`). hmmlearn needs **Python 3.11/3.12** — "
                     "set it in Settings and Reboot.")
            st.stop()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Startup failed: {exc}")
            st.stop()

system = st.session_state.system
system.dry_run = dry_run

# refresh / guarded run
try:
    if step:
        with st.spinner("Running pipeline iteration…"):
            st.session_state.snap = sysmod.run_iteration(system)
    elif refresh or auto:
        with st.spinner("Refreshing…"):
            st.session_state.snap = sysmod.take_snapshot(system)
except Exception as exc:  # noqa: BLE001 - keep the console alive
    st.warning(f"Update error (will retry): {exc}")

snap = st.session_state.get("snap", {})

# ------------------------------------------------------------------- top bar
c.top_bar(snap["topbar"])
st.write("")

# ------------------------------------------------------------------- tabs
tabs = st.tabs(["Overview", "Portfolio", "Regimes", "Orders", "Risk", "Model",
                "Logs", "Settings"])
with tabs[0]:
    pages.overview(snap)
with tabs[1]:
    pages.portfolio(snap)
with tabs[2]:
    pages.regimes(snap)
with tabs[3]:
    pages.orders(snap)
with tabs[4]:
    pages.risk(snap)
with tabs[5]:
    pages.model(snap)
with tabs[6]:
    pages.logs(snap)
with tabs[7]:
    c.section("Session")
    st.write(f"Universe: **{', '.join(symbols)}**  ·  Regime symbol: **{regime_symbol}**")
    st.write(f"Mode: **{'DRY-RUN' if dry_run else 'LIVE-ORDER'}**  ·  "
             f"Fast HMM: **{fast}**  ·  Auto-refresh: **{auto}** ({interval}s)")
    st.write(f"Last update: **{snap['topbar']['updated']}**")
    st.caption("Controls live in the sidebar. The dashboard is read-only; only "
               "**Run once** with Dry-run off submits orders. Secrets are never shown.")

# ------------------------------------------------------------------- auto loop
if auto:
    time.sleep(interval)
    st.rerun()
