"""SQLite candle cache.

Two jobs:
  1. Survive an API outage - a skipped fetch degrades to slightly stale data
     instead of a crash.
  2. Keep history that free tiers will not serve twice (Twelve Data free only
     reaches back ~52 days on 15m, so what we saw once is worth keeping).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]


def save(conn: sqlite3.Connection, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
    """Upsert candles. Index must be a UTC DatetimeIndex of candle open times.

    Uses vectorized column extraction instead of iterrows() for speed.
    """
    if df is None or df.empty:
        return 0
    n = len(df)
    ts_ms = (df.index.astype("int64") // 1_000_000).tolist()
    rows = list(zip(
        [symbol] * n, [timeframe] * n, ts_ms,
        df["open"].tolist(), df["high"].tolist(), df["low"].tolist(),
        df["close"].tolist(), df["volume"].tolist(),
    ))
    conn.executemany(
        "INSERT INTO candles(symbol,timeframe,ts,open,high,low,close,volume) "
        "VALUES(?,?,?,?,?,?,?,?) "
        "ON CONFLICT(symbol,timeframe,ts) DO UPDATE SET "
        "open=excluded.open, high=excluded.high, low=excluded.low, "
        "close=excluded.close, volume=excluded.volume",
        rows,
    )
    conn.commit()
    return n


def load(
    conn: sqlite3.Connection, symbol: str, timeframe: str, limit: int = 500
) -> pd.DataFrame:
    """Return the most recent `limit` candles, oldest first."""
    cur = conn.execute(
        "SELECT ts, open, high, low, close, volume FROM candles "
        "WHERE symbol=? AND timeframe=? ORDER BY ts DESC LIMIT ?",
        (symbol, timeframe, int(limit)),
    )
    rows = cur.fetchall()
    if not rows:
        return empty()
    df = pd.DataFrame(
        [tuple(r) for r in rows], columns=["ts", *COLUMNS]
    ).iloc[::-1]
    df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
    df.index.name = "ts"
    return df.astype(float)


def last_timestamp(conn: sqlite3.Connection, symbol: str,
                   timeframe: str) -> datetime | None:
    """Return the most recent candle open time in cache, or None if empty.

    Used by the router for freshness-aware skipping: if the cache already has
    a candle within the current timeframe window, skip the HTTP fetch.
    """
    row = conn.execute(
        "SELECT MAX(ts) AS latest FROM candles "
        "WHERE symbol=? AND timeframe=?",
        (symbol, timeframe),
    ).fetchone()
    if not row or row["latest"] is None:
        return None
    return datetime.fromtimestamp(int(row["latest"]) / 1000, tz=timezone.utc)


def empty() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name="ts")
    return pd.DataFrame(columns=COLUMNS, index=idx, dtype=float)


def coverage(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Per symbol/timeframe row counts - used by the phase-1 report."""
    return conn.execute(
        "SELECT symbol, timeframe, COUNT(*) AS n, MIN(ts) AS first_ts, "
        "MAX(ts) AS last_ts FROM candles GROUP BY symbol, timeframe"
    ).fetchall()

