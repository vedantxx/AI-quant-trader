"""Signal generator: combine HMM regime + strategy allocation into orders.

Produces target weights per symbol given the current regime, trend, and
risk state. Emits rebalance deltas the executor can act on.
"""

from __future__ import annotations

from dataclasses import dataclass

from .hmm_engine import RegimeResult
from .regime_strategies import RegimeStrategy
from .risk_manager import RiskManager, RiskState


@dataclass
class Signal:
    symbol: str
    target_weight: float
    current_weight: float
    action: str          # "buy" | "sell" | "hold"
    reason: str


class SignalGenerator:
    def __init__(self, cfg: dict, strategy: RegimeStrategy, risk: RiskManager):
        self.cfg = cfg
        self.strategy = strategy
        self.risk = risk
        self.symbols = cfg["broker"]["symbols"]

    def generate(
        self,
        regime: RegimeResult,
        trend_up: dict[str, bool],
        current_weights: dict[str, float],
        risk_state: RiskState,
    ) -> list[Signal]:
        """One signal per symbol. Equal-weight within target exposure."""
        alloc = self.strategy.target(
            regime.label, all(trend_up.values()), regime.confidence
        )

        if risk_state.halted:
            # flatten everything on halt
            return [
                Signal(sym, 0.0, current_weights.get(sym, 0.0), "sell", "risk halt")
                for sym in self.symbols
                if current_weights.get(sym, 0.0) > 0
            ]

        per_symbol = (
            alloc.exposure * alloc.size_mult * risk_state.size_scale
        ) / max(1, len(self.symbols))
        # clamp to single-position cap
        per_symbol = min(per_symbol, self.risk.max_single_position)

        signals: list[Signal] = []
        for sym in self.symbols:
            cur = current_weights.get(sym, 0.0)
            tgt = per_symbol if trend_up.get(sym, True) else 0.0
            if not self.strategy.needs_rebalance(cur, tgt):
                signals.append(Signal(sym, tgt, cur, "hold", "within threshold"))
                continue
            action = "buy" if tgt > cur else "sell"
            signals.append(
                Signal(sym, tgt, cur, action, f"regime={regime.label}")
            )
        return signals
