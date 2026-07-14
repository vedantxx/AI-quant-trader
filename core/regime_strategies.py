"""Volatility-based allocation strategies.

Maps a detected regime (plus trend flag and confidence) to a target equity
exposure and leverage, per ``strategy`` config.

Stub scaffold: signatures, type hints, and docstrings only — no logic yet.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AllocationSignal:
    """Target allocation emitted by the strategy for the current regime."""

    exposure: float         # target gross exposure as fraction of equity
    leverage: float         # allowed leverage multiplier
    size_mult: float        # sizing multiplier (uncertainty haircut)


class RegimeStrategies:
    """Volatility-based allocation strategy."""

    def __init__(self, cfg: dict) -> None:
        """Read strategy.* + hmm.min_confidence config."""
        ...

    def allocate(
        self, regime: str, trend_up: bool, confidence: float
    ) -> AllocationSignal:
        """Map regime + trend + confidence to a target allocation."""
        ...

    def needs_rebalance(self, current_weight: float, target_weight: float) -> bool:
        """True if drift from target exceeds the rebalance threshold."""
        ...
