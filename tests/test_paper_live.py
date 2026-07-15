"""Live paper-account smoke test (Phase 9d).

Skipped by default. Run explicitly against a funded Alpaca PAPER account:

    ALPACA_LIVE_TEST=1 pytest tests/test_paper_live.py -s

Places a bracket order for 1 share, tightens its stop, cancels everything, and
verifies no open orders remain. Never runs in the normal suite or on live money.
"""

import os
import time

import pytest

if os.getenv("ALPACA_LIVE_TEST") != "1":
    pytest.skip("live paper test disabled (set ALPACA_LIVE_TEST=1)", allow_module_level=True)

pytest.importorskip("alpaca")
from dotenv import load_dotenv

from broker.alpaca_client import AlpacaClient
from broker.order_executor import OrderExecutor
from core.regime_strategies import Direction, Signal

CFG = {"broker": {"paper_trading": True, "timeframe": "1Day"}}


def test_bracket_modify_cancel_clean_state():
    load_dotenv(dotenv_path=".env")
    client = AlpacaClient(CFG)
    assert client.paper is True
    ex = OrderExecutor(client)

    ref = float(client.get_available_margin())  # sanity: account reachable
    assert ref >= 0

    sig = Signal(symbol="SPY", direction=Direction.LONG, confidence=0.9,
                 entry_price=100.0, stop_loss=90.0, take_profit=130.0,
                 position_size_pct=0.0, leverage=1.0, regime_id=0, regime_name="BULL",
                 regime_probability=0.9, strategy_name="live", reasoning="", metadata={})
    res = ex.submit_bracket_order(sig, qty=1)
    assert res.trade_id in ex.trades and res.id
    print("bracket submitted:", res.id, res.status)

    ex.modify_stop("SPY", 92.0)            # tighten-only (no live order id -> local)
    assert ex._stops["SPY"] == 92.0

    time.sleep(1)
    ex.cancel_all()                        # cancel any working orders
    time.sleep(1)
    open_orders = client.get_order_history(status="open")
    assert len(open_orders) == 0, f"expected clean state, {len(open_orders)} open"
    print("clean state verified")
