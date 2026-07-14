"""Tests for volatility-based allocation strategies and orchestrator."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from core.regime_strategies import (
    AllocationSignal,
    BullTrendStrategy,
    CrashDefensiveStrategy,
    Direction,
    HighVolDefensiveStrategy,
    LowVolBullStrategy,
    MeanReversionStrategy,
    MidVolCautiousStrategy,
    RegimeStrategies,
    Signal,
    StrategyOrchestrator,
)

CFG = {
    "strategy": {
        "low_vol_allocation": 0.95,
        "mid_vol_allocation_trend": 0.95,
        "mid_vol_allocation_no_trend": 0.60,
        "high_vol_allocation": 0.60,
        "low_vol_leverage": 1.25,
        "rebalance_threshold": 0.10,
        "uncertainty_size_mult": 0.50,
    },
    "hmm": {"min_confidence": 0.55},
}


# ------------------------------------------------------------------- fixtures
@dataclass
class FakeRegimeState:
    """Duck-typed stand-in for hmm_engine.RegimeState (no scipy import)."""

    label: str = "BULL"
    state_id: int = 0
    probability: float = 0.9
    is_flickering: bool = False
    timestamp: object = None


@dataclass
class FakeRegimeInfo:
    """Duck-typed stand-in for hmm_engine.RegimeInfo."""

    regime_id: int
    regime_name: str
    expected_volatility: float


def make_bars(trend: str = "up", n: int = 80) -> pd.DataFrame:
    """Synthetic OHLCV. ``trend`` up keeps price above the 50 EMA, down below."""
    if trend == "up":
        close = np.linspace(100.0, 160.0, n)
    else:  # sharp recent drop pushes price under the 50 EMA
        close = np.concatenate([np.linspace(150.0, 160.0, n - 10),
                                np.linspace(160.0, 120.0, 10)])
    close = pd.Series(close)
    high = close * 1.01
    low = close * 0.99
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close,
         "volume": np.full(n, 1_000_000.0)}
    )


# ---------------------------------------------------------------- strategies
def test_low_vol_uses_leverage() -> None:
    """Low-vol regime allocates full exposure with leverage, always LONG."""
    sig = LowVolBullStrategy(CFG).generate_signal("SPY", make_bars(), FakeRegimeState())
    assert sig is not None
    assert sig.direction == Direction.LONG
    assert sig.position_size_pct == 0.95
    assert sig.leverage == 1.25
    # stop = max(price - 3 ATR, 50 EMA - 0.5 ATR), below entry
    assert sig.stop_loss < sig.entry_price


def test_mid_vol_trend_vs_no_trend() -> None:
    """Mid-vol exposure depends on price vs 50 EMA; leverage stays 1.0."""
    strat = MidVolCautiousStrategy(CFG)
    up = strat.generate_signal("SPY", make_bars("up"), FakeRegimeState())
    down = strat.generate_signal("SPY", make_bars("down"), FakeRegimeState())
    assert up.position_size_pct == 0.95 and up.leverage == 1.0
    assert down.position_size_pct == 0.60 and down.leverage == 1.0


def test_high_vol_derisks() -> None:
    """High-vol regime cuts exposure to 0.60, drops leverage to 1.0, stays LONG."""
    sig = HighVolDefensiveStrategy(CFG).generate_signal(
        "SPY", make_bars(), FakeRegimeState()
    )
    assert sig.direction == Direction.LONG
    assert sig.position_size_pct == 0.60
    assert sig.leverage == 1.0


def test_never_short() -> None:
    """No strategy ever emits a short/FLAT-with-negative direction."""
    for cls in (LowVolBullStrategy, MidVolCautiousStrategy, HighVolDefensiveStrategy):
        sig = cls(CFG).generate_signal("SPY", make_bars("down"), FakeRegimeState())
        assert sig.direction == Direction.LONG


def test_insufficient_bars_returns_none() -> None:
    """Too little history yields no signal rather than a garbage stop."""
    assert LowVolBullStrategy(CFG).generate_signal(
        "SPY", make_bars(n=5), FakeRegimeState()
    ) is None


# --------------------------------------------------------------- orchestrator
def _infos() -> list[FakeRegimeInfo]:
    # regime_id order is NOT vol order: id 0 highest vol, id 2 lowest.
    return [
        FakeRegimeInfo(0, "BULL", expected_volatility=0.30),   # highest vol
        FakeRegimeInfo(1, "NEUTRAL", expected_volatility=0.15),
        FakeRegimeInfo(2, "BEAR", expected_volatility=0.05),   # lowest vol
    ]


def test_orchestrator_maps_by_vol_not_label() -> None:
    """Lowest-vol regime gets LowVolBull even if its label is BEAR."""
    orch = StrategyOrchestrator(CFG, _infos())
    assert isinstance(orch.strategy_for(2), LowVolBullStrategy)   # lowest vol
    assert isinstance(orch.strategy_for(1), MidVolCautiousStrategy)
    assert isinstance(orch.strategy_for(0), HighVolDefensiveStrategy)  # highest vol


def test_orchestrator_generates_one_signal_per_symbol() -> None:
    orch = StrategyOrchestrator(CFG, _infos())
    rs = FakeRegimeState(label="BEAR", state_id=2, probability=0.9)
    bars = {"SPY": make_bars(), "QQQ": make_bars()}
    sigs = orch.generate_signals(["SPY", "QQQ"], bars, rs)
    assert [s.symbol for s in sigs] == ["SPY", "QQQ"]
    assert all(s.leverage == 1.25 for s in sigs)  # lowest-vol regime -> leverage


def test_uncertainty_haircut() -> None:
    """Low confidence halves size and forces leverage to 1.0."""
    orch = StrategyOrchestrator(CFG, _infos())
    rs = FakeRegimeState(label="BEAR", state_id=2, probability=0.40)  # < 0.55
    sig = orch.generate_signals(["SPY"], {"SPY": make_bars()}, rs)[0]
    assert sig.position_size_pct == pytest.approx(0.95 * 0.5)
    assert sig.leverage == 1.0
    assert "UNCERTAINTY" in sig.reasoning


def test_flicker_forces_uncertainty() -> None:
    """A flickering regime triggers the haircut even at high confidence."""
    orch = StrategyOrchestrator(CFG, _infos())
    rs = FakeRegimeState(label="BEAR", state_id=2, probability=0.99)
    sig = orch.generate_signals(["SPY"], {"SPY": make_bars()}, rs, is_flickering=True)[0]
    assert sig.leverage == 1.0
    assert "UNCERTAINTY" in sig.reasoning


def test_update_regime_infos_rebuilds_map() -> None:
    """After retrain, a new vol ordering re-maps strategies."""
    orch = StrategyOrchestrator(CFG, _infos())
    flipped = [
        FakeRegimeInfo(0, "BULL", expected_volatility=0.05),   # now lowest vol
        FakeRegimeInfo(1, "NEUTRAL", expected_volatility=0.15),
        FakeRegimeInfo(2, "BEAR", expected_volatility=0.30),   # now highest vol
    ]
    orch.update_regime_infos(flipped)
    assert isinstance(orch.strategy_for(0), LowVolBullStrategy)
    assert isinstance(orch.strategy_for(2), HighVolDefensiveStrategy)


# ----------------------------------------------------------------- aliases
def test_backward_compatible_aliases() -> None:
    assert BullTrendStrategy is LowVolBullStrategy
    assert CrashDefensiveStrategy is HighVolDefensiveStrategy
    assert MeanReversionStrategy is MidVolCautiousStrategy


# --------------------------------------------------------- legacy allocation
def test_rebalance_threshold() -> None:
    """needs_rebalance() triggers only past the drift threshold."""
    rs = RegimeStrategies(CFG)
    assert rs.needs_rebalance(0.50, 0.65) is True     # 0.15 drift > 0.10
    assert rs.needs_rebalance(0.50, 0.55) is False    # 0.05 drift < 0.10


def test_legacy_allocate_low_vs_high_and_uncertainty() -> None:
    rs = RegimeStrategies(CFG)
    low = rs.allocate("BULL", trend_up=True, confidence=0.9)
    assert (low.exposure, low.leverage, low.size_mult) == (0.95, 1.25, 1.0)
    high = rs.allocate("CRASH", trend_up=False, confidence=0.9)
    assert (high.exposure, high.leverage) == (0.60, 1.0)
    unsure = rs.allocate("BULL", trend_up=True, confidence=0.3)
    assert unsure.size_mult == 0.50 and unsure.leverage == 1.0
