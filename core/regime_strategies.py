"""Volatility-based allocation strategies.

The HMM classifies the market into VOLATILITY ENVIRONMENTS, not direction.
Stocks drift upward in calm periods; the worst drawdowns cluster in high-vol
spikes. The edge is AVOIDING BIG DRAWDOWNS via vol-based position sizing:

    low vol  -> fully invested (+ modest leverage); calm markets trend up
    mid vol  -> stay invested if trend intact, reduce if broken
    high vol -> reduce but stay partially invested; catch V-shaped rebounds

ALWAYS LONG. NEVER SHORT. Shorting was tested in walk-forward backtesting and
consistently destroyed returns: markets have upward drift, V-shaped recoveries
arrive faster than the HMM detects them (2-3 bars late), and shorts get wiped
out in the rebound. The correct response to high volatility is REDUCING
allocation, not reversing direction.

Strategy selection is by VOLATILITY RANK, independent of the return-sorted
regime labels: a "BULL" label does NOT imply low vol. The orchestrator sorts
regimes by ``expected_volatility`` and ignores labels entirely.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from data.feature_engineering import atr

if TYPE_CHECKING:  # avoid runtime dep on scipy/hmmlearn via hmm_engine
    from .hmm_engine import RegimeInfo, RegimeState


# --------------------------------------------------------------------- direction
class Direction:
    """Position directions. The strategy layer is long-only by design."""

    LONG = "LONG"
    FLAT = "FLAT"


# ------------------------------------------------------------------------ signal
@dataclass
class Signal:
    """A per-symbol trade intent emitted by a strategy for the current regime."""

    symbol: str
    direction: str                      # Direction.LONG or Direction.FLAT
    confidence: float                   # regime posterior probability
    entry_price: float
    stop_loss: float
    position_size_pct: float            # target exposure, 0.60 .. 0.95
    leverage: float                     # 1.0 or 1.25
    regime_id: int
    regime_name: str
    regime_probability: float
    strategy_name: str
    reasoning: str
    take_profit: Optional[float] = None
    timestamp: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class AllocationSignal:
    """Target allocation emitted by the legacy ``RegimeStrategies`` shim."""

    exposure: float         # target gross exposure as fraction of equity
    leverage: float         # allowed leverage multiplier
    size_mult: float        # sizing multiplier (uncertainty haircut)


# ------------------------------------------------------------------- indicators
ATR_WINDOW = 14
EMA_SPAN = 50
_MIN_BARS = ATR_WINDOW + 1  # need a previous close for the first true range


def _indicators(bars: pd.DataFrame) -> Optional[tuple[float, float, float]]:
    """Return (last_price, ATR, 50-EMA) or None if there is too little history."""
    if bars is None or len(bars) < _MIN_BARS:
        return None
    close = bars["close"]
    price = float(close.iloc[-1])
    atr_val = float(atr(bars["high"], bars["low"], close, ATR_WINDOW).iloc[-1])
    ema50 = float(close.ewm(span=EMA_SPAN, adjust=False).mean().iloc[-1])
    if not (np.isfinite(price) and np.isfinite(atr_val) and np.isfinite(ema50)):
        return None
    return price, atr_val, ema50


# ------------------------------------------------------------------- strategies
class BaseStrategy(ABC):
    """Maps the current regime to a long-only allocation for one symbol."""

    name: str = "base"

    def __init__(self, cfg: dict) -> None:
        self.scfg: dict = cfg["strategy"]
        self.min_confidence: float = cfg["hmm"]["min_confidence"]

    @abstractmethod
    def generate_signal(
        self, symbol: str, bars: pd.DataFrame, regime_state: "RegimeState"
    ) -> Optional[Signal]:
        """Emit a Signal for ``symbol`` given ``bars`` and the current regime."""
        ...

    def _signal(
        self,
        symbol: str,
        regime_state: "RegimeState",
        entry_price: float,
        stop_loss: float,
        position_size_pct: float,
        leverage: float,
        reasoning: str,
        metadata: dict,
    ) -> Signal:
        return Signal(
            symbol=symbol,
            direction=Direction.LONG,
            confidence=regime_state.probability,
            entry_price=entry_price,
            stop_loss=stop_loss,
            position_size_pct=position_size_pct,
            leverage=leverage,
            regime_id=regime_state.state_id,
            regime_name=regime_state.label,
            regime_probability=regime_state.probability,
            strategy_name=self.name,
            reasoning=reasoning,
            timestamp=regime_state.timestamp,
            metadata=metadata,
        )


class LowVolBullStrategy(BaseStrategy):
    """Lowest-vol third: fully invested with modest leverage. Most returns here."""

    name = "low_vol_bull"

    def generate_signal(
        self, symbol: str, bars: pd.DataFrame, regime_state: "RegimeState"
    ) -> Optional[Signal]:
        ind = _indicators(bars)
        if ind is None:
            return None
        price, atr_val, ema50 = ind
        stop = max(price - 3.0 * atr_val, ema50 - 0.5 * atr_val)
        alloc = self.scfg["low_vol_allocation"]
        lev = self.scfg["low_vol_leverage"]
        return self._signal(
            symbol, regime_state, price, stop, alloc, lev,
            reasoning=(
                f"Low-vol regime: fully invested {alloc:.0%} at {lev:.2f}x. "
                f"Calm markets trend up."
            ),
            metadata={"atr": atr_val, "ema50": ema50},
        )


class MidVolCautiousStrategy(BaseStrategy):
    """Middle third: stay invested if trend intact (price > 50 EMA), else reduce."""

    name = "mid_vol_cautious"

    def generate_signal(
        self, symbol: str, bars: pd.DataFrame, regime_state: "RegimeState"
    ) -> Optional[Signal]:
        ind = _indicators(bars)
        if ind is None:
            return None
        price, atr_val, ema50 = ind
        trend_up = price > ema50
        alloc = (
            self.scfg["mid_vol_allocation_trend"]
            if trend_up
            else self.scfg["mid_vol_allocation_no_trend"]
        )
        stop = ema50 - 0.5 * atr_val
        return self._signal(
            symbol, regime_state, price, stop, alloc, 1.0,
            reasoning=(
                f"Mid-vol regime: trend {'intact' if trend_up else 'broken'} "
                f"(price {'>' if trend_up else '<'} 50 EMA), allocate {alloc:.0%}."
            ),
            metadata={"atr": atr_val, "ema50": ema50, "trend_up": trend_up},
        )


class HighVolDefensiveStrategy(BaseStrategy):
    """Top vol third: reduce but stay partially invested to catch rebounds. LONG."""

    name = "high_vol_defensive"

    def generate_signal(
        self, symbol: str, bars: pd.DataFrame, regime_state: "RegimeState"
    ) -> Optional[Signal]:
        ind = _indicators(bars)
        if ind is None:
            return None
        price, atr_val, ema50 = ind
        alloc = self.scfg["high_vol_allocation"]
        stop = ema50 - 1.0 * atr_val  # wider stop for volatile conditions
        return self._signal(
            symbol, regime_state, price, stop, alloc, 1.0,
            reasoning=(
                f"High-vol regime: de-risk to {alloc:.0%}, stay LONG to catch "
                f"V-shaped rebounds (never short)."
            ),
            metadata={"atr": atr_val, "ema50": ema50},
        )


# ------------------------------------------------------------ backward-compat
# High-vol defensive covers every bearish/crash label; mid covers neutral; the
# calm-market bull strategy covers bullish labels. NOTE: labels are return-sorted
# and do NOT imply a volatility rank — this map is a legacy convenience only. The
# orchestrator ignores labels and selects by ``expected_volatility``.
CrashDefensiveStrategy = HighVolDefensiveStrategy
BearTrendStrategy = HighVolDefensiveStrategy
MeanReversionStrategy = MidVolCautiousStrategy
NeutralStrategy = MidVolCautiousStrategy
BullTrendStrategy = LowVolBullStrategy
EuphoriaCautiousStrategy = LowVolBullStrategy

LABEL_TO_STRATEGY: dict[str, type[BaseStrategy]] = {
    "CRASH": HighVolDefensiveStrategy,
    "STRONG_BEAR": HighVolDefensiveStrategy,
    "BEAR": HighVolDefensiveStrategy,
    "WEAK_BEAR": HighVolDefensiveStrategy,
    "NEUTRAL": MidVolCautiousStrategy,
    "WEAK_BULL": LowVolBullStrategy,
    "BULL": LowVolBullStrategy,
    "STRONG_BULL": LowVolBullStrategy,
    "EUPHORIA": LowVolBullStrategy,
}


# ----------------------------------------------------------------- orchestrator
class StrategyOrchestrator:
    """Selects a strategy per regime by VOLATILITY RANK and emits signals.

    Regimes are ranked ascending by ``expected_volatility`` (independent of the
    return-sorted labels). Each regime's normalized rank position picks a
    strategy::

        position = rank / (n_regimes - 1)   # 0.0 = lowest vol, 1.0 = highest
        position <= 0.33  -> LowVolBullStrategy
        position >= 0.67  -> HighVolDefensiveStrategy
        else              -> MidVolCautiousStrategy
    """

    def __init__(self, cfg: dict, regime_infos) -> None:
        self.cfg = cfg
        self.min_confidence: float = cfg["hmm"]["min_confidence"]
        self.rebalance_threshold: float = cfg["strategy"]["rebalance_threshold"]
        self.low = LowVolBullStrategy(cfg)
        self.mid = MidVolCautiousStrategy(cfg)
        self.high = HighVolDefensiveStrategy(cfg)
        self._by_regime: dict[int, BaseStrategy] = {}
        self.update_regime_infos(regime_infos)

    # ------------------------------------------------------------- vol mapping
    def update_regime_infos(self, regime_infos) -> None:
        """(Re)build the regime_id -> strategy map after an HMM (re)train."""
        infos = list(
            regime_infos.values() if isinstance(regime_infos, dict) else regime_infos
        )
        by_vol = sorted(infos, key=lambda r: r.expected_volatility)
        n = len(by_vol)
        self._by_regime = {}
        for rank, info in enumerate(by_vol):
            position = rank / (n - 1) if n > 1 else 0.0
            if position <= 0.33:
                strat = self.low
            elif position >= 0.67:
                strat = self.high
            else:
                strat = self.mid
            self._by_regime[info.regime_id] = strat

    def strategy_for(self, regime_id: int) -> Optional[BaseStrategy]:
        """The strategy mapped to ``regime_id`` (by vol rank), or None."""
        return self._by_regime.get(regime_id)

    def needs_rebalance(self, current_alloc: float, target_alloc: float) -> bool:
        """True if target exposure drifts from current by more than the
        configured threshold (default 10%). Prevents churn from minor
        probability fluctuations — fewer trades, less slippage."""
        return abs(target_alloc - current_alloc) > self.rebalance_threshold

    # ----------------------------------------------------------- signal output
    def generate_signals(
        self,
        symbols: list[str],
        bars,
        regime_state: "RegimeState",
        is_flickering: bool = False,
    ) -> list[Signal]:
        """One Signal per symbol for the current regime.

        ``bars`` is a ``{symbol: DataFrame}`` mapping (or a single DataFrame
        applied to every symbol). Uncertainty mode — triggered when the regime
        probability is below ``min_confidence`` or the regime is flickering —
        halves every position size and forces leverage to 1.0x.
        """
        strat = self._by_regime.get(regime_state.state_id)
        if strat is None:
            return []

        uncertain = (
            regime_state.probability < self.min_confidence
            or is_flickering
            or getattr(regime_state, "is_flickering", False)
        )

        signals: list[Signal] = []
        for sym in symbols:
            sym_bars = bars[sym] if isinstance(bars, dict) else bars
            sig = strat.generate_signal(sym, sym_bars, regime_state)
            if sig is None:
                continue
            if uncertain:
                sig.position_size_pct *= 0.5
                sig.leverage = 1.0
                sig.reasoning += " [UNCERTAINTY — size halved]"
                sig.metadata["uncertainty"] = True
            signals.append(sig)
        return signals


# ------------------------------------------------------- legacy allocation shim
class RegimeStrategies:
    """Label-keyed allocation shim kept for ``signal_generator`` compatibility.

    Prefer :class:`StrategyOrchestrator`, which selects by volatility rank. This
    shim resolves a strategy from a regime *label* via ``LABEL_TO_STRATEGY`` and
    returns its default exposure/leverage.
    """

    def __init__(self, cfg: dict) -> None:
        self.scfg: dict = cfg["strategy"]
        self.min_confidence: float = cfg["hmm"]["min_confidence"]
        self.rebalance_threshold: float = self.scfg["rebalance_threshold"]
        self.uncertainty_size_mult: float = self.scfg["uncertainty_size_mult"]

    def allocate(
        self, regime: str, trend_up: bool, confidence: float
    ) -> AllocationSignal:
        """Map regime label + trend + confidence to a target allocation."""
        strat_cls = LABEL_TO_STRATEGY.get(regime, MidVolCautiousStrategy)
        if strat_cls is LowVolBullStrategy:
            exposure = self.scfg["low_vol_allocation"]
            leverage = self.scfg["low_vol_leverage"]
        elif strat_cls is HighVolDefensiveStrategy:
            exposure = self.scfg["high_vol_allocation"]
            leverage = 1.0
        else:  # MidVolCautiousStrategy
            exposure = (
                self.scfg["mid_vol_allocation_trend"]
                if trend_up
                else self.scfg["mid_vol_allocation_no_trend"]
            )
            leverage = 1.0

        if confidence < self.min_confidence:
            size_mult = self.uncertainty_size_mult
            leverage = 1.0  # never lever an uncertain regime
        else:
            size_mult = 1.0
        return AllocationSignal(exposure=exposure, leverage=leverage, size_mult=size_mult)

    def needs_rebalance(self, current_weight: float, target_weight: float) -> bool:
        """True if drift from target exceeds the rebalance threshold."""
        return abs(target_weight - current_weight) > self.rebalance_threshold
