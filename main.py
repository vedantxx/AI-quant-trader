"""AI-Quant-trader entry point.

Usage:
    python main.py backtest     walk-forward backtest over configured symbols
    python main.py trade        live/paper trading loop
    python main.py dashboard    terminal dashboard only

Config is read from config/settings.yaml; secrets from .env.

Stub scaffold: wiring is sketched; component methods are not implemented yet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent / "config" / "settings.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load settings.yaml into a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_credentials() -> dict:
    """Load optional config/credentials.yaml, or {} if absent."""
    path = Path(__file__).parent / "config" / "credentials.yaml"
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


# --------------------------------------------------------------------- modes
def run_backtest(cfg: dict) -> None:
    """Walk-forward backtest each configured symbol; log performance."""
    ...


def run_trade(cfg: dict) -> None:
    """Live/paper trading loop: data -> regime -> signals -> risk -> orders."""
    ...


def run_dashboard(cfg: dict) -> None:
    """Render the terminal dashboard."""
    ...


def main(argv: list[str]) -> int:
    """Dispatch to backtest / trade / dashboard."""
    if len(argv) < 2 or argv[1] not in {"backtest", "trade", "dashboard"}:
        print(__doc__)
        return 1
    cfg = load_config()
    {"backtest": run_backtest, "trade": run_trade, "dashboard": run_dashboard}[
        argv[1]
    ](cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
