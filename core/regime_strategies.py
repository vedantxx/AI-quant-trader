"""Volatility-based allocation strategies.

Maps a detected regime (plus a trend flag and regime confidence) to a target
equity exposure and leverage, per ``strategy`` config.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AllocationTarget:
    exposure: float        # target gross exposure as fraction of equity
    leverage: float        # allowed leverage multiplier
    size_mult: float       # extra sizing multiplier (uncertainty haircut)


class RegimeStrategy:
    def __init__(self, cfg: dict):
        s = cfg["strategy"]
        self.low_vol_allocation = s["low_vol_allocation"]
        self.mid_vol_allocation_trend = s["mid_vol_allocation_trend"]
        self.mid_vol_allocation_no_trend = s["mid_vol_allocation_no_trend"]
        self.high_vol_allocation = s["high_vol_allocation"]
        self.low_vol_leverage = s["low_vol_leverage"]
        self.rebalance_threshold = s["rebalance_threshold"]
        self.uncertainty_size_mult = s["uncertainty_size_mult"]
        self.min_confidence = cfg["hmm"]["min_confidence"]

    def target(
        self, regime: str, trend_up: bool, confidence: float
    ) -> AllocationTarget:
        """Compute allocation for the current regime."""
        if regime == "low":
            exposure = self.low_vol_allocation
            leverage = self.low_vol_leverage
        elif regime == "mid":
            exposure = (
                self.mid_vol_allocation_trend
                if trend_up
                else self.mid_vol_allocation_no_trend
            )
            leverage = 1.0
        else:  # high
            exposure = self.high_vol_allocation
            leverage = 1.0

        size_mult = (
            self.uncertainty_size_mult
            if confidence < self.min_confidence
            else 1.0
        )
        return AllocationTarget(exposure, leverage, size_mult)

    def needs_rebalance(self, current_weight: float, target_weight: float) -> bool:
        """True if drift exceeds the rebalance threshold."""
        return abs(current_weight - target_weight) >= self.rebalance_threshold
