"""SQLite schema and connection helper.

One small file so the schema lives in exactly one place. WAL mode keeps a
long-running scanner and an ad-hoc query from blocking each other.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    symbol    TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    ts        INTEGER NOT NULL,          -- candle open time, ms UTC
    open      REAL NOT NULL,
    high      REAL NOT NULL,
    low       REAL NOT NULL,
    close     REAL NOT NULL,
    volume    REAL NOT NULL,
    PRIMARY KEY (symbol, timeframe, ts)
);

CREATE TABLE IF NOT EXISTS api_usage (
    day_utc  TEXT NOT NULL,
    provider TEXT NOT NULL,
    credits  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (day_utc, provider)
);

CREATE TABLE IF NOT EXISTS signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          INTEGER NOT NULL,
    symbol      TEXT NOT NULL,
    direction   TEXT NOT NULL,
    label       TEXT NOT NULL,
    score       REAL NOT NULL,
    confidence  REAL NOT NULL,
    entry_low   REAL, entry_high REAL,
    sl          REAL, tp1 REAL, tp2 REAL, tp3 REAL,
    rr          REAL, atr REAL,
    htf_bias    TEXT, mtf_bias TEXT, ltf_state TEXT,
    triggers    TEXT,
    narration   TEXT,
    narration_source TEXT,
    sent_ok     INTEGER NOT NULL DEFAULT 0,
    data_source TEXT,                     -- which source provided the candles
    status      TEXT NOT NULL DEFAULT 'open'  -- open | won | lost | expired | invalidated
);
CREATE INDEX IF NOT EXISTS idx_signals_symbol_ts ON signals(symbol, ts DESC);
CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status);

CREATE TABLE IF NOT EXISTS outcomes (
    signal_id    INTEGER PRIMARY KEY REFERENCES signals(id),
    checked_ts   INTEGER NOT NULL,
    hit          TEXT,                     -- sl | tp1 | tp2 | tp3 | open
    mfe_r        REAL,                    -- max favourable excursion in R
    mae_r        REAL,                    -- max adverse excursion in R
    bars_held    INTEGER,
    note         TEXT,
    tp1_hit_ts   INTEGER,                 -- exact timestamp TP1 was hit
    tp2_hit_ts   INTEGER,
    tp3_hit_ts   INTEGER,
    sl_hit_ts    INTEGER,
    price_at_check REAL,                  -- price when last checked
    entry_filled INTEGER NOT NULL DEFAULT 0  -- did price reach entry zone?
);

CREATE TABLE IF NOT EXISTS performance_daily (
    day_utc       TEXT PRIMARY KEY,
    total_signals INTEGER NOT NULL DEFAULT 0,
    wins          INTEGER NOT NULL DEFAULT 0,
    losses        INTEGER NOT NULL DEFAULT 0,
    open_signals  INTEGER NOT NULL DEFAULT 0,
    win_rate      REAL,
    avg_r         REAL,
    best_r        REAL,
    worst_r       REAL,
    by_symbol     TEXT,                   -- JSON: {"BTC": {"wins": 3, "losses": 1}, ...}
    by_label      TEXT                    -- JSON: {"BUY": {"wins": 5, "losses": 2}, ...}
);

CREATE TABLE IF NOT EXISTS mutes (
    symbol   TEXT PRIMARY KEY,
    until_ts INTEGER NOT NULL
);
"""


def connect(db_path: Path | str) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn
