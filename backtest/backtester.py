"""Walk-forward ALLOCATION backtester.

This backtester sets a TARGET PORTFOLIO ALLOCATION each bar from the detected
volatility regime and rebalances only when that target drifts meaningfully from
the current allocation. It does NOT track individual trade entries/exits — that
is how real systematic strategies run. Stops are a live-trading concern and are
deliberately absent here.

Walk-forward:
  * In-sample (IS)  — ``backtest.train_window`` bars: refit HMM (BIC selection).
  * Out-of-sample (OOS) — ``backtest.test_window`` bars: trade, no look-ahead.
  * Advance by ``backtest.step_size`` and repeat.

Causality: features are strictly causal (value at t depends only on t and
earlier), so the full-series feature matrix is sliced rather than recomputed per
bar — an exact equivalent that keeps the walk fast. Filtered (forward-algorithm)
HMM inference is used bar by bar; Viterbi is never used for live decisions.

Allocation math (kept exactly correct):
    equity        = cash + shares * price
    target_shares = int(equity * target_allocation / price)
    delta         = target_shares - current_shares
    cash         -= delta * fill_price      # fill_price includes slippage
    shares        = target_shares
With leverage > 1.0 the target allocation exceeds 1.0, so cash goes negative
(margin); equity stays correct because share value exceeds the margin debt.

Fill delay: a target computed from bar N's data is executed at bar N+1's open.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.hmm_engine import HMMEngine
from core.regime_strategies import StrategyOrchestrator
from data.feature_engineering import FeatureEngineer

logger = logging.getLogger("regime-trader.backtest")

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    """Output of a walk-forward backtest run."""

    equity_curve: pd.Series
    returns: pd.Series
    regimes: pd.Series
    trades: int
    folds: int
    meta: dict = field(default_factory=dict)


class Backtester:
    """Walk-forward allocation backtester (single asset)."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        b = cfg["backtest"]
        self.train_window: int = b["train_window"]
        self.test_window: int = b["test_window"]
        self.step_size: int = b["step_size"]
        self.initial_capital: float = float(b["initial_capital"])
        self.slippage_pct: float = float(b["slippage_pct"])
        self.rebalance_threshold: float = float(cfg["strategy"]["rebalance_threshold"])
        # peak-to-trough kill switch (drawdown circuit breaker)
        self.max_dd_from_peak: float = float(cfg["risk"]["max_dd_from_peak"])
        feat_cfg = cfg.get("features", {})
        self.fe = FeatureEngineer(**({"zscore_window": feat_cfg["zscore_window"]}
                                     if "zscore_window" in feat_cfg else {}))

    # ------------------------------------------------------------------- run
    def run(
        self,
        df: pd.DataFrame,
        symbol: str = "ASSET",
        shuffle_regime_seed: int | None = None,
    ) -> BacktestResult:
        """Single-asset walk-forward backtest over OHLCV ``df``.

        ``shuffle_regime_seed`` (stress-test hook) randomly permutes the
        regime -> strategy mapping so strategies are applied to the WRONG
        regimes, probing whether allocation caps still contain the damage.
        """
        df = df.sort_index()
        n = len(df)
        close = df["close"].to_numpy(dtype=float)
        open_ = df["open"].to_numpy(dtype=float)

        # Precompute causal features once; reindex to df so warm-up rows are NaN.
        feat_df = self.fe.build_features(df).reindex(df.index)
        feat_arr = feat_df.to_numpy(dtype=float)
        valid = ~np.isnan(feat_arr).any(axis=1)
        first_valid = int(np.argmax(valid)) if valid.any() else n

        # Portfolio state (carried across folds for a continuous equity curve).
        cash = self.initial_capital
        shares = 0
        peak_equity = self.initial_capital

        equity_idx: list = []
        equity_val: list[float] = []
        regime_idx: list = []
        regime_lbl: list[str] = []
        conf_val: list[float] = []
        alloc_val: list[float] = []
        trades: list[dict] = []

        n_rebalances = 0
        breaker_fired = False
        folds = 0

        # open-trade (allocation segment) tracker
        seg_open_ts = None
        seg_open_eq = None
        seg_regime = None
        seg_conf = None
        seg_alloc = None
        seg_bars = 0

        is_start = first_valid
        while is_start + self.train_window < n:
            is_end = is_start + self.train_window
            oos_end = min(is_end + self.test_window, n)
            if oos_end <= is_end:
                break

            # -------- train HMM on in-sample features (BIC model selection)
            is_rows = valid[is_start:is_end]
            X_is = feat_arr[is_start:is_end][is_rows]
            rets_is = df["close"].pct_change().to_numpy()[is_start:is_end][is_rows]
            engine = HMMEngine(self.cfg)
            if len(X_is) < engine.min_train_bars:
                logger.warning("fold skipped: %d IS bars < min", len(X_is))
                is_start += self.step_size
                continue
            try:
                engine.fit(X_is, returns=rets_is)
            except Exception as exc:
                logger.warning("HMM fit failed, skipping fold: %s", exc)
                is_start += self.step_size
                continue

            orch = StrategyOrchestrator(self.cfg, engine.regime_info)
            if shuffle_regime_seed is not None:
                self._shuffle_mapping(orch, engine, shuffle_regime_seed + folds)

            engine._reset_filter_state()
            folds += 1
            halted = False
            pending_target: float | None = None  # 1-bar fill delay

            for t in range(is_end, oos_end):
                price = close[t]

                # ---- execute the target computed on the PREVIOUS bar (delay)
                if pending_target is not None:
                    equity = cash + shares * price
                    cur_alloc = (shares * price) / equity if equity else 0.0
                    if self.needs_rebalance(cur_alloc, pending_target):
                        fill = open_[t]
                        target_shares = int(equity * pending_target / fill)
                        delta = target_shares - shares
                        if delta != 0:
                            slip = 1 + self.slippage_pct if delta > 0 else 1 - self.slippage_pct
                            cash -= delta * fill * slip
                            shares = target_shares
                            n_rebalances += 1
                            # close current segment, open a new one
                            eq_now = cash + shares * price
                            if seg_open_ts is not None:
                                trades.append(self._seg(
                                    seg_open_ts, df.index[t], seg_open_eq, eq_now,
                                    seg_regime, seg_conf, seg_alloc, seg_bars))
                            seg_open_ts, seg_open_eq = df.index[t], eq_now
                            seg_regime, seg_conf = regime_lbl[-1], conf_val[-1]
                            seg_alloc, seg_bars = pending_target, 0
                    pending_target = None

                # ---- filtered regime at bar t (forward over history up to t)
                state = engine.predict(feat_arr[first_valid:t + 1], df.index[t])
                bars_t = df.iloc[:t + 1]
                sigs = orch.generate_signals(
                    [symbol], {symbol: bars_t}, state,
                    is_flickering=state.is_flickering,
                )
                if sigs:
                    s = sigs[0]
                    target = s.position_size_pct * s.leverage
                else:
                    target = 0.0

                # ---- mark to market on the close
                equity = cash + shares * price
                peak_equity = max(peak_equity, equity)

                # ---- drawdown circuit breaker (kill switch -> flat, per fold)
                if not halted and peak_equity > 0 and equity / peak_equity - 1 <= -self.max_dd_from_peak:
                    halted = True
                    breaker_fired = True
                    logger.warning("circuit breaker fired at %s (dd from peak)", df.index[t])
                if halted:
                    target = 0.0

                pending_target = target

                equity_idx.append(df.index[t])
                equity_val.append(equity)
                regime_idx.append(df.index[t])
                regime_lbl.append(state.label)
                conf_val.append(state.probability)
                alloc_val.append((shares * price) / equity if equity else 0.0)
                if seg_open_ts is not None:
                    seg_bars += 1

            is_start += self.step_size

        # close the final open segment
        if seg_open_ts is not None and equity_val:
            trades.append(self._seg(
                seg_open_ts, equity_idx[-1], seg_open_eq, equity_val[-1],
                seg_regime, seg_conf, seg_alloc, seg_bars))

        equity_curve = pd.Series(equity_val, index=pd.Index(equity_idx), name="equity")
        returns = equity_curve.pct_change().dropna()
        regimes = pd.Series(regime_lbl, index=pd.Index(regime_idx), name="regime")
        confidence = pd.Series(conf_val, index=pd.Index(regime_idx), name="confidence")
        trade_log = pd.DataFrame(trades)

        return BacktestResult(
            equity_curve=equity_curve,
            returns=returns,
            regimes=regimes,
            trades=n_rebalances,
            folds=folds,
            meta={
                "confidence": confidence,
                "allocation": pd.Series(alloc_val, index=pd.Index(regime_idx), name="allocation"),
                "trade_log": trade_log,
                "price": df["close"].reindex(equity_curve.index),
                "breaker_fired": breaker_fired,
                "initial_capital": self.initial_capital,
                "symbol": symbol,
            },
        )

    # ------------------------------------------------------------- helpers
    def needs_rebalance(self, current_alloc: float, target_alloc: float) -> bool:
        """True if target allocation drifts from current by > threshold."""
        return abs(target_alloc - current_alloc) > self.rebalance_threshold

    @staticmethod
    def _seg(open_ts, close_ts, open_eq, close_eq, regime, conf, alloc, bars) -> dict:
        pnl = close_eq - open_eq
        return {
            "entry_time": open_ts,
            "exit_time": close_ts,
            "entry_equity": open_eq,
            "exit_equity": close_eq,
            "pnl": pnl,
            "pnl_pct": pnl / open_eq if open_eq else 0.0,
            "regime": regime,
            "confidence": conf,
            "target_alloc": alloc,
            "hold_bars": bars,
        }

    @staticmethod
    def _shuffle_mapping(orch: StrategyOrchestrator, engine: HMMEngine, seed: int) -> None:
        """Randomly reassign strategies across regimes (misclassification test)."""
        rng = np.random.default_rng(seed)
        ids = list(orch._by_regime.keys())
        strats = list(orch._by_regime.values())
        rng.shuffle(strats)
        orch._by_regime = dict(zip(ids, strats))
