"""Tab renderers for the dashboard. Each takes a read-only snapshot dict."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from . import components as c

ROOT = Path(__file__).resolve().parent.parent


def _symbols_df(snap: dict) -> pd.DataFrame:
    df = pd.DataFrame(snap["symbols"])
    if df.empty:
        return df
    return df[["symbol", "action", "target", "current", "drift", "price", "entry",
               "pnl_pct", "stop", "held", "leverage", "reason"]]


# ============================================================== OVERVIEW
def overview(snap: dict) -> None:
    left, right = st.columns([2, 1])

    with left:
        c.section("Equity curve")
        hist = snap.get("equity_hist", [])
        if len(hist) > 1:
            eq = pd.DataFrame(hist).set_index("t")["equity"]
            st.line_chart(eq, height=200)
        else:
            c.empty("Collecting equity points… refresh to build the curve.")

        c.section("Allocation by symbol")
        exp = snap["exposure_by_symbol"]
        if exp:
            st.bar_chart(pd.Series(exp, name="weight"), height=180)
        else:
            c.empty("No open positions.")

        c.section("Open positions")
        df = _symbols_df(snap)
        held = df[df["current"] > 0] if not df.empty else df
        if not held.empty:
            st.dataframe(c.style_actions(held), use_container_width=True, hide_index=True)
        else:
            c.empty("Flat — no open positions.")

    with right:
        _regime_card(snap["regime"])
        c.section("Risk status")
        dd = snap["drawdown"]
        rc = snap["risk_cfg"]
        c.risk_bar("Daily drawdown", dd["daily"], rc["daily_halt"])
        c.risk_bar("From peak", dd["peak"], rc["peak_halt"])
        bh = snap["breaker_history"]
        state = bh[-1]["breaker_type"] if bh else "all clear"
        c.section("Circuit breaker")
        st.markdown(c.badge(state, "red" if bh else "green"), unsafe_allow_html=True)

        c.section("Recent signals")
        sigs = snap.get("recent_signals", [])
        if sigs:
            st.dataframe(pd.DataFrame(list(reversed(sigs))), use_container_width=True,
                         hide_index=True, height=240)
        else:
            c.empty("No decisions yet — run an iteration.")


def _regime_card(r: dict) -> None:
    c.section("Regime")
    kind = "yellow" if r["uncertain"] else "green"
    st.markdown(
        f'<div class="kpi"><div class="val">{r["label"]} '
        f'{c.badge(c.pct(r["prob"],0), kind)}</div>'
        f'<div class="sub mut">Stability {r["stability"]} bars · '
        f'Flicker {r["flicker"]}/{r["flicker_window"]}</div></div>',
        unsafe_allow_html=True)
    probs = pd.Series(r["probs"], name="probability").sort_values(ascending=False)
    st.bar_chart(probs, height=150)


# ============================================================== SIGNALS
def signals(snap: dict) -> None:
    rows = snap["symbols"]
    counts = {"BUY": 0, "SELL": 0, "HOLD": 0, "REJECT": 0}
    for r in rows:
        counts[r["action"]] = counts.get(r["action"], 0) + 1

    m = st.columns(4)
    c.kpi(m[0], "Buy", str(counts["BUY"]), "actionable", "up")
    c.kpi(m[1], "Sell", str(counts["SELL"]), "actionable", "up")
    c.kpi(m[2], "Hold", str(counts["HOLD"]), "in band", "mut")
    c.kpi(m[3], "Reject", str(counts["REJECT"]), "risk veto", "down")

    top = st.columns([1, 1, 2])
    actionable_only = top[0].toggle("Actionable only", value=False)
    sort_by = top[1].selectbox("Sort", ["action", "confidence", "drift", "symbol"], key="sig_sort")

    c.section("Signals received  ·  live per-symbol decisions this bar")
    df = pd.DataFrame(rows)
    if df.empty:
        c.empty("No signals — run an iteration.")
    else:
        df = df[["symbol", "action", "target", "current", "drift", "confidence",
                 "price", "stop", "leverage", "reason"]]
        if actionable_only:
            df = df[df["action"].isin(["BUY", "SELL"])]
        df = df.sort_values(sort_by, ascending=(sort_by == "symbol"))
        if df.empty:
            c.empty("No actionable signals right now.")
        else:
            st.dataframe(
                c.style_actions(df).format({
                    "target": "{:.1%}", "current": "{:.1%}", "drift": "{:+.1%}",
                    "confidence": "{:.0%}", "price": "${:,.2f}", "stop": "${:,.2f}",
                    "leverage": "{:.2f}x"}),
                use_container_width=True, hide_index=True, height=430)

    c.section("Signal history  ·  rolling feed")
    hist = snap.get("recent_signals", [])
    if hist:
        st.dataframe(pd.DataFrame(list(reversed(hist))), use_container_width=True,
                     hide_index=True, height=260)
    else:
        c.empty("No history yet.")


# ============================================================== PORTFOLIO
def portfolio(snap: dict) -> None:
    df = _symbols_df(snap)
    if df.empty:
        c.empty("No symbols in the universe.")
        return
    top = st.columns([2, 1, 1])
    q = top[0].text_input("Search symbol", "").strip().upper()
    only_reb = top[1].toggle("Rebalance-needed only", value=False)
    sort_by = top[2].selectbox("Sort by", ["drift", "target", "current", "pnl_pct", "symbol"])

    view = df.copy()
    if q:
        view = view[view["symbol"].str.contains(q)]
    if only_reb:
        reb_syms = {r["symbol"] for r in snap["symbols"] if r["rebalance"]}
        view = view[view["symbol"].isin(reb_syms)]
    view = view.sort_values(sort_by, ascending=(sort_by == "symbol"))

    st.dataframe(
        c.style_actions(view).format({
            "target": "{:.1%}", "current": "{:.1%}", "drift": "{:+.1%}",
            "price": "${:,.2f}", "entry": "${:,.2f}", "pnl_pct": "{:+.1%}",
            "stop": "${:,.2f}", "leverage": "{:.2f}x"}),
        use_container_width=True, hide_index=True, height=460)
    c.empty(f"{len(view)} of {len(df)} symbols · target = strategy allocation after "
            "risk caps · drift = target − current")


# ============================================================== REGIMES
def regimes(snap: dict) -> None:
    r = snap["regime"]
    a, b = st.columns([1, 1])
    with a:
        c.section("Current regime probabilities")
        st.bar_chart(pd.Series(r["probs"], name="p").sort_values(ascending=False), height=200)
        c.section("Transition matrix  P(row → col)")
        tm = snap.get("transition")
        if tm is not None:
            st.dataframe(c.mono_heat(tm), use_container_width=True)
        else:
            c.empty("Model not loaded.")
    with b:
        c.section("Volatility-rank mapping  (drives allocation)")
        vr = pd.DataFrame(snap["vol_rank"])
        st.dataframe(vr.style.format({"expected_vol": "{:.4f}", "expected_return": "{:+.4f}",
                                      "vol_rank": "{:.2f}"}),
                     use_container_width=True, hide_index=True)
        st.info(
            "**Labels are descriptive, volatility rank drives allocation.** Regime "
            "labels (BULL/BEAR/…) are sorted by *return*. The strategy sorts the same "
            "regimes by *expected volatility* — lowest-vol → LowVolBull (95% @1.25x), "
            "highest-vol → HighVolDefensive (60%). A 'BULL' label does **not** imply low vol.")

    c.section("Regime history (recent decisions)")
    sigs = snap.get("recent_signals", [])
    if sigs:
        st.dataframe(pd.DataFrame(list(reversed(sigs))), use_container_width=True,
                     hide_index=True, height=200)
    else:
        c.empty("No history yet.")


# ============================================================== ORDERS
def orders(snap: dict) -> None:
    rows = snap["symbols"]
    orders_tbl = pd.DataFrame(snap["orders"])
    rejects = [r for r in rows if r["action"] == "REJECT"]
    actionable = [r for r in rows if r["action"] in ("BUY", "SELL")]

    m = st.columns(4)
    c.kpi(m[0], "Actionable", str(len(actionable)))
    c.kpi(m[1], "Rejected", str(len(rejects)))
    c.kpi(m[2], "Submitted (session)", str(len(orders_tbl)))
    c.kpi(m[3], "Slippage assumed", c.pct(snap["risk_cfg"].get("slippage", 0.0005), 3)
          if "slippage" in snap["risk_cfg"] else "5 bps")

    c.section("Submitted / pending orders")
    if not orders_tbl.empty:
        st.dataframe(orders_tbl, use_container_width=True, hide_index=True)
    else:
        c.empty("No orders submitted this session (dry-run or no rebalance).")

    c.section("Rejections & risk vetoes")
    if rejects:
        rj = pd.DataFrame(rejects)[["symbol", "current", "reason"]]
        st.dataframe(rj.style.format({"current": "{:.1%}"}),
                     use_container_width=True, hide_index=True)
    else:
        c.empty("No rejections.")


# ============================================================== RISK
def risk(snap: dict) -> None:
    dd, rc = snap["drawdown"], snap["risk_cfg"]
    a, b = st.columns([1, 1])
    with a:
        c.section("Drawdown")
        c.risk_bar("Daily", dd["daily"], rc["daily_halt"])
        c.risk_bar("From peak", dd["peak"], rc["peak_halt"])
        c.section("Exposure by symbol")
        es = snap["exposure_by_symbol"]
        if es:
            st.bar_chart(pd.Series(es, name="weight"), height=180)
        else:
            c.empty("Flat.")
        c.section("Exposure by sector")
        sec = snap["exposure_by_sector"]
        if sec:
            st.bar_chart(pd.Series(sec, name="weight"), height=160)
        else:
            c.empty("No sector exposure.")
    with b:
        c.section("Correlation (60-day returns)")
        corr = snap.get("correlation")
        if corr is not None:
            st.dataframe(c.heatmap(corr), use_container_width=True)
        else:
            c.empty("Need ≥2 symbols with history.")
        c.section("Limits")
        st.dataframe(pd.DataFrame([
            {"limit": "Max single position", "value": c.pct(rc["max_single"])},
            {"limit": "Max gross exposure", "value": c.pct(rc["max_exposure"])},
            {"limit": "Max sector", "value": c.pct(rc["max_sector"])},
            {"limit": "Max leverage", "value": f'{rc["max_leverage"]:.2f}x'},
        ]), use_container_width=True, hide_index=True)
        c.section("Circuit breaker history")
        bh = snap["breaker_history"]
        if bh:
            st.dataframe(pd.DataFrame(bh)[["timestamp", "breaker_type", "drawdown", "action"]],
                         use_container_width=True, hide_index=True)
        else:
            c.empty("No breakers fired.")


# ============================================================== MODEL
def model(snap: dict) -> None:
    m = snap["model"]
    if not m:
        c.empty("Model not loaded.")
        return
    cols = st.columns(4)
    c.kpi(cols[0], "Regimes (n)", str(m["n_regimes"]))
    c.kpi(cols[1], "BIC", f'{m["bic"]:,.0f}')
    c.kpi(cols[2], "Model age", f'{m.get("age_days","—")}d')
    c.kpi(cols[3], "Converged", "yes" if m["converged"] else "no")

    a, b = st.columns([1, 1])
    with a:
        c.section("BIC candidates (lower = better)")
        bic = pd.Series(m["all_bic"], name="BIC")
        bic.index = bic.index.map(lambda k: f"{k} regimes")
        st.bar_chart(bic, height=200)
    with b:
        c.section("Metadata")
        st.dataframe(pd.DataFrame([
            {"field": "Training date", "value": m["training_date"]},
            {"field": "Labels (return-sorted)", "value": ", ".join(m["labels"])},
            {"field": "Feature dim", "value": m["feature_dim"]},
            {"field": "Log-likelihood", "value": f'{m["log_likelihood"]:,.1f}'},
            {"field": "EM iterations", "value": m["iterations"]},
        ]), use_container_width=True, hide_index=True)
    c.section("Volatility-rank → strategy")
    st.dataframe(pd.DataFrame(snap["vol_rank"]).style.format(
        {"expected_vol": "{:.4f}", "expected_return": "{:+.4f}", "vol_rank": "{:.2f}"}),
        use_container_width=True, hide_index=True)


# ============================================================== LOGS
def logs(snap: dict) -> None:
    which = st.selectbox("Log file", ["main", "trades", "regime", "alerts"])
    n = st.slider("Lines", 20, 300, 80)
    path = ROOT / "logs" / f"{which}.log"
    if not path.exists():
        c.empty(f"No {path.name} yet.")
        return
    lines = path.read_text().splitlines()[-n:]
    parsed = []
    for ln in lines:
        try:
            parsed.append(json.loads(ln))
        except json.JSONDecodeError:
            parsed.append({"msg": ln})
    st.dataframe(pd.DataFrame(reversed(parsed)), use_container_width=True,
                 hide_index=True, height=460)
