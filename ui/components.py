"""Reusable presentation primitives: CSS theme, top bar, cards, badges, tables.

Pure rendering — no system/broker access. Everything takes plain data.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

# ------------------------------------------------------------------- theme
_CSS = """
<style>
:root {
  --bg:#0b0e14; --panel:#141922; --panel2:#1b212c; --line:#232b39;
  --txt:#e6edf3; --muted:#8b98a9; --accent:#4cc9f0;
  --green:#2ecc71; --red:#ff5c5c; --yellow:#f2c94c;
}
.stApp { background: var(--bg); }
section[data-testid="stSidebar"] { background: var(--panel); border-right:1px solid var(--line); }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }
h1,h2,h3 { letter-spacing:-.01em; }
hr { border-color: var(--line); }

.kpi { background:var(--panel); border:1px solid var(--line); border-radius:14px;
       padding:14px 16px; height:100%; }
.kpi .lbl { color:var(--muted); font-size:.72rem; text-transform:uppercase;
            letter-spacing:.06em; margin-bottom:4px; }
.kpi .val { font-size:1.5rem; font-weight:650; line-height:1.1; color:var(--txt); }
.kpi .sub { font-size:.8rem; margin-top:3px; }
.up { color:var(--green); } .down { color:var(--red); } .mut { color:var(--muted); }

.badge { display:inline-block; padding:2px 10px; border-radius:999px;
         font-size:.72rem; font-weight:650; letter-spacing:.03em; }
.b-green{ background:rgba(46,204,113,.15); color:var(--green); }
.b-red  { background:rgba(255,92,92,.15);  color:var(--red); }
.b-yellow{background:rgba(242,201,76,.15); color:var(--yellow); }
.b-blue { background:rgba(76,201,240,.15); color:var(--accent); }
.b-gray { background:rgba(139,152,169,.15);color:var(--muted); }

.sect { font-size:.78rem; text-transform:uppercase; letter-spacing:.08em;
        color:var(--muted); margin:6px 0 8px; font-weight:650; }
.bar { height:8px; border-radius:6px; background:var(--panel2); overflow:hidden; }
.bar > span { display:block; height:100%; border-radius:6px; }
.stTabs [data-baseweb="tab-list"] { gap:2px; }
.stTabs [data-baseweb="tab"] { padding:8px 16px; }
[data-testid="stMetricValue"] { font-size:1.4rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------- format
def money(x: float) -> str:
    return f"${x:,.0f}" if abs(x) >= 100 else f"${x:,.2f}"


def pct(x: float, dp: int = 1) -> str:
    return f"{x*100:.{dp}f}%"


def badge(text: str, kind: str = "gray") -> str:
    return f'<span class="badge b-{kind}">{text}</span>'


def section(title: str) -> None:
    st.markdown(f'<div class="sect">{title}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------- cards
def kpi(col, label: str, value: str, sub: str = "", sub_kind: str = "mut") -> None:
    sub_html = f'<div class="sub {sub_kind}">{sub}</div>' if sub else ""
    col.markdown(
        f'<div class="kpi"><div class="lbl">{label}</div>'
        f'<div class="val">{value}</div>{sub_html}</div>',
        unsafe_allow_html=True)


def top_bar(tb: dict) -> None:
    """Nine-slot operating header."""
    mode_kind = "b-yellow" if tb["mode"] == "LIVE" else "b-blue"
    mkt_kind = "b-green" if tb["market_open"] else "b-gray"
    dpnl = tb["daily_pnl"]
    c = st.columns([1.1, 1.3, 1.2, 1.2, 1.3, 1.2, 1.4])
    kpi(c[0], "Mode", f'<span class="badge {mode_kind}">{tb["mode"]}</span>'
        + (" " + badge("DRY", "b-gray") if tb.get("dry_run") else ""))
    kpi(c[1], "Equity", money(tb["equity"]),
        f'{"↑" if dpnl>=0 else "↓"} {money(dpnl)} ({pct(tb["daily_pnl_pct"],2)})',
        "up" if dpnl >= 0 else "down")
    kpi(c[2], "Cash", money(tb["cash"]))
    kpi(c[3], "Buying Power", money(tb["buying_power"]))
    kpi(c[4], "Exposure", pct(tb["exposure"]), "gross / equity")
    kpi(c[5], "Regime", tb["regime"], pct(tb["regime_prob"], 0) + " conf")
    kpi(c[6], "Market",
        f'<span class="badge {mkt_kind}">{"OPEN" if tb["market_open"] else "CLOSED"}</span>',
        f'HMM {tb["hmm_age"]} · {tb["updated"]}')


# ------------------------------------------------------------------- bars
def risk_bar(label: str, value: float, limit: float) -> None:
    frac = 0.0 if limit <= 0 else max(0.0, min(1.0, value / limit))
    color = "var(--green)" if frac < 0.5 else "var(--yellow)" if frac < 0.8 else "var(--red)"
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;font-size:.8rem;'
        f'color:var(--muted);margin-bottom:3px"><span>{label}</span>'
        f'<span>{pct(value,2)} / {pct(limit,0)}</span></div>'
        f'<div class="bar"><span style="width:{frac*100:.0f}%;background:{color}"></span></div>',
        unsafe_allow_html=True)


# ------------------------------------------------------------------- tables
_ACTION_COLOR = {"BUY": "#2ecc71", "SELL": "#f2c94c", "HOLD": "#8b98a9", "REJECT": "#ff5c5c"}


def _smap(styler, func, **kw):
    """Styler.map added in pandas 2.1; fall back to applymap on older Cloud envs."""
    fn = getattr(styler, "map", None) or styler.applymap
    return fn(func, **kw)


def style_actions(df: pd.DataFrame, col: str = "action"):
    def _c(v):
        return f'color:{_ACTION_COLOR.get(v, "#e6edf3")};font-weight:600'
    return _smap(df.style, _c, subset=[col]) if col in df.columns else df.style


def heatmap(df: pd.DataFrame):
    """Correlation heatmap [-1,1] via CSS (no matplotlib dependency)."""
    def _bg(v):
        if pd.isna(v):
            return ""
        t = max(-1.0, min(1.0, float(v)))
        r, g = int(255 * (1 - (t + 1) / 2)), int(255 * ((t + 1) / 2))
        return f"background-color: rgba({r},{g},90,0.32)"
    return _smap(df.style, _bg).format("{:.2f}")


def mono_heat(df: pd.DataFrame, rgb: str = "76,201,240"):
    """Single-hue heatmap for 0..1 matrices (e.g. transition), no matplotlib."""
    def _bg(v):
        a = max(0.0, min(1.0, float(v))) * 0.5 if not pd.isna(v) else 0.0
        return f"background-color: rgba({rgb},{a})"
    return _smap(df.style, _bg).format("{:.2f}")


def empty(msg: str) -> None:
    st.markdown(f'<div class="mut" style="padding:14px 2px">{msg}</div>',
                unsafe_allow_html=True)
