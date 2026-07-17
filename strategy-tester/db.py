"""Local MySQL storage for the strategy tester.

Replaces the CSV files with a local MySQL database (tables: bars, sweep_results,
param_sensitivity, bootstrap_results). Everything degrades gracefully: if the
server or the driver is unavailable, callers fall back to CSV so the pipeline
never hard-fails.

Connection is read from repo-root .env (DB_HOST/DB_PORT/DB_USER/DB_PASSWORD/
DB_NAME) with local-dev defaults. Credentials are never logged.
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd

import layer1_data_strategies as L  # for _load_env

_AVAILABLE: Optional[bool] = None


def _cfg() -> dict:
    L._load_env()
    return dict(
        host=os.environ.get("DB_HOST", "127.0.0.1"),
        port=int(os.environ.get("DB_PORT", "3306")),
        user=os.environ.get("DB_USER", "root"),
        password=os.environ.get("DB_PASSWORD", ""),
        db=os.environ.get("DB_NAME", "strategy_tester"),
    )


def connect(create_db: bool = False):
    """Return a PyMySQL connection, or raise. If create_db, connect serverless
    first and CREATE DATABASE."""
    import pymysql

    c = _cfg()
    if create_db:
        root = pymysql.connect(host=c["host"], port=c["port"], user=c["user"],
                               password=c["password"], connect_timeout=5)
        with root.cursor() as cur:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{c['db']}` "
                        "CHARACTER SET utf8mb4")
        root.commit()
        root.close()
    return pymysql.connect(host=c["host"], port=c["port"], user=c["user"],
                           password=c["password"], database=c["db"],
                           connect_timeout=5, autocommit=False)


def available() -> bool:
    """True if the driver imports and the server accepts a connection."""
    global _AVAILABLE
    if _AVAILABLE is not None:
        return _AVAILABLE
    try:
        conn = connect(create_db=True)
        conn.close()
        _AVAILABLE = True
    except Exception:
        _AVAILABLE = False
    return _AVAILABLE


DDL = {
    "bars": """CREATE TABLE IF NOT EXISTS bars (
        ticker VARCHAR(32) NOT NULL, dt DATE NOT NULL,
        open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE,
        PRIMARY KEY (ticker, dt))""",
    "sweep_results": """CREATE TABLE IF NOT EXISTS sweep_results (
        config VARCHAR(191), strategy VARCHAR(64), category VARCHAR(32),
        asset VARCHAR(32), is_sharpe DOUBLE, oos_sharpe DOUBLE,
        oos_maxdd DOUBLE, trades INT,
        KEY k_strategy (strategy), KEY k_asset (asset))""",
    "param_sensitivity": """CREATE TABLE IF NOT EXISTS param_sensitivity (
        strategy VARCHAR(64), category VARCHAR(32), n_configs INT,
        mean_oos DOUBLE, std_oos DOUBLE, frac_pos DOUBLE)""",
    "bootstrap_results": """CREATE TABLE IF NOT EXISTS bootstrap_results (
        config VARCHAR(191), asset VARCHAR(32), category VARCHAR(32),
        oos_sharpe DOUBLE, p5_sharpe DOUBLE, p50_sharpe DOUBLE,
        p95_sharpe DOUBLE, worst_dd DOUBLE, verdict VARCHAR(16))""",
}


def init_schema() -> None:
    conn = connect(create_db=True)
    try:
        with conn.cursor() as cur:
            for ddl in DDL.values():
                cur.execute(ddl)
        conn.commit()
    finally:
        conn.close()


def _replace_table(table: str, df: pd.DataFrame, cols: list) -> None:
    conn = connect(create_db=True)
    try:
        with conn.cursor() as cur:
            cur.execute(DDL[table])
            cur.execute(f"TRUNCATE TABLE {table}")
            if len(df):
                ph = ",".join(["%s"] * len(cols))
                rows = [tuple(None if pd.isna(v) else v for v in r)
                        for r in df[cols].itertuples(index=False, name=None)]
                cur.executemany(
                    f"INSERT INTO {table} ({','.join(cols)}) VALUES ({ph})", rows)
        conn.commit()
    finally:
        conn.close()


# -- bars ------------------------------------------------------------------- #
def save_bars(ticker: str, df: pd.DataFrame) -> None:
    conn = connect(create_db=True)
    try:
        with conn.cursor() as cur:
            cur.execute(DDL["bars"])
            cur.execute("DELETE FROM bars WHERE ticker=%s", (ticker,))
            rows = [(ticker, idx.date() if hasattr(idx, "date") else idx,
                     float(r.Open), float(r.High), float(r.Low),
                     float(r.Close), float(r.Volume))
                    for idx, r in df.iterrows()]
            cur.executemany(
                "INSERT INTO bars (ticker,dt,open,high,low,close,volume) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s)", rows)
        conn.commit()
    finally:
        conn.close()


def load_bars(ticker: str) -> Optional[pd.DataFrame]:
    try:
        conn = connect()
    except Exception:
        return None
    try:
        df = pd.read_sql(
            "SELECT dt,open,high,low,close,volume FROM bars "
            "WHERE ticker=%s ORDER BY dt", conn, params=(ticker,))
    except Exception:
        return None
    finally:
        conn.close()
    if df.empty:
        return None
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "volume": "Volume"})
    df.index = pd.to_datetime(df.pop("dt"))
    return df


# -- results ---------------------------------------------------------------- #
def save_sweep(df: pd.DataFrame) -> None:
    _replace_table("sweep_results", df,
                   ["config", "strategy", "category", "asset",
                    "is_sharpe", "oos_sharpe", "oos_maxdd", "trades"])


def load_sweep() -> Optional[pd.DataFrame]:
    try:
        conn = connect()
        df = pd.read_sql("SELECT * FROM sweep_results", conn)
        conn.close()
        return df if len(df) else None
    except Exception:
        return None


def save_param_sensitivity(df: pd.DataFrame) -> None:
    _replace_table("param_sensitivity", df,
                   ["strategy", "category", "n_configs", "mean_oos",
                    "std_oos", "frac_pos"])


def save_bootstrap(df: pd.DataFrame) -> None:
    _replace_table("bootstrap_results", df,
                   ["config", "asset", "category", "oos_sharpe", "p5_sharpe",
                    "p50_sharpe", "p95_sharpe", "worst_dd", "verdict"])
