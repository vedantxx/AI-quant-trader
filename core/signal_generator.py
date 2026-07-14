"""Signal generator: combine HMM regime + strategy allocation into trades.

Produces target weights per symbol given the current regime, trend, and risk
state, then emits rebalance actions the executor can act on.

Stub scaffold: signatures, type hints, and docstrings only — no logic yet.
"""

from __future__ import annotations

from dataclasses import dataclass

from .hmm_engine import RegimeState
from .regime_strategies import RegimeStrategies
from .risk_manager import RiskManager, RiskState


@dataclass
class TradeSignal:
    """A per-symbol rebalance instruction."""

    symbol: str
    target_weight: float
    current_weight: float
    action: str          # "buy" | "sell" | "hold"
    reason: str


class SignalGenerator:
    """Combine regime + strategy + risk into per-symbol trade signals."""

    def __init__(
        self, cfg: dict, strategy: RegimeStrategies, risk: RiskManager
    ) -> None:
        """Store config and strategy/risk collaborators."""
        ...

    def generate(
        self,
        regime: RegimeState,
        trend_up: dict[str, bool],
        current_weights: dict[str, float],
        risk_state: RiskState,
    ) -> list[TradeSignal]:
        """One signal per symbol; flatten all on risk halt."""
        ...
