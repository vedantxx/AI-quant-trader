"""Track open positions and P&L."""

from __future__ import annotations

from dataclasses import dataclass

from .alpaca_client import AlpacaClient


@dataclass
class Position:
    symbol: str
    qty: float
    avg_entry: float
    market_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float

    @property
    def weight_of(self) -> float:
        return self.market_value


class PositionTracker:
    def __init__(self, client: AlpacaClient):
        self.client = client

    def snapshot(self) -> dict[str, Position]:
        out: dict[str, Position] = {}
        for p in self.client.get_positions():
            out[p.symbol] = Position(
                symbol=p.symbol,
                qty=float(p.qty),
                avg_entry=float(p.avg_entry_price),
                market_price=float(p.current_price),
                market_value=float(p.market_value),
                unrealized_pl=float(p.unrealized_pl),
                unrealized_plpc=float(p.unrealized_plpc),
            )
        return out

    def current_weights(self, equity: float) -> dict[str, float]:
        if equity <= 0:
            return {}
        return {
            sym: pos.market_value / equity
            for sym, pos in self.snapshot().items()
        }

    def total_unrealized(self) -> float:
        return sum(p.unrealized_pl for p in self.snapshot().values())
