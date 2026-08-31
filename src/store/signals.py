"""Signal store — save, load, query, cooldown, deduplication.

All signal persistence goes through this module so the rest of the bot
never touches raw SQL for signals.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import asdict
from typing import Any

log = logging.getLogger(__name__)

__all__ = [
    "save_signal", "get_open_signals", "get_last_signal",
    "update_status", "is_on_cooldown", "get_recent_signals",
]

# Default cooldown: don't re-signal the same symbol+direction within this window
COOLDOWN_S = 4 * 3600  # 4 hours


def save_signal(conn: sqlite3.Connection,
                signal: Any, plan: Any | None,
                narration: str, narration_source: str,
                sent_ok: bool = False,
                data_source: str = "") -> int:
    """Persist a signal to the database. Returns the signal ID.

    Args:
        signal: SignalResult from confluence engine
        plan: TradePlan or None
        narration: LLM/template explanation
        narration_source: "hf:model" or "template"
        sent_ok: whether Telegram delivery succeeded
        data_source: which data provider was used
    """
    now_ms = int(time.time() * 1000)

    # Extract timeframe biases
    htf_bias, mtf_bias, ltf_state = "", "", ""
    for tf, tfr in signal.tf_results.items():
        if not tfr.votes:
            continue
        avg = sum(v.value for v in tfr.votes) / len(tfr.votes)
        bias = "bullish" if avg > 0.2 else "bearish" if avg < -0.2 else "mixed"
        if tf in ("1d",):
            htf_bias = bias
        elif tf in ("4h", "1h") and not mtf_bias:
            mtf_bias = bias
        elif tf in ("15m",):
            ltf_state = bias

    # Key triggers
    triggers: list[str] = []
    for tf, tfr in signal.tf_results.items():
        for v in tfr.votes:
            if abs(v.value) >= 0.6 and v.detail:
                triggers.append(v.detail)

    dir_str = "long" if signal.direction > 0 else "short" if signal.direction < 0 else "neutral"

    row = {
        "ts": now_ms,
        "symbol": signal.symbol,
        "direction": dir_str,
        "label": signal.label,
        "score": round(signal.score, 2),
        "confidence": round(signal.confidence, 1),
        "entry_low": round(plan.entry_low, 6) if plan else None,
        "entry_high": round(plan.entry_high, 6) if plan else None,
        "sl": round(plan.sl, 6) if plan else None,
        "tp1": round(plan.tp1, 6) if plan else None,
        "tp2": round(plan.tp2, 6) if plan else None,
        "tp3": round(plan.tp3, 6) if plan else None,
        "rr": round(plan.rr, 2) if plan else None,
        "atr": round(plan.risk_atr, 2) if plan else None,
        "htf_bias": htf_bias,
        "mtf_bias": mtf_bias,
        "ltf_state": ltf_state,
        "triggers": json.dumps(triggers[:8]),
        "narration": narration,
        "narration_source": narration_source,
        "sent_ok": 1 if sent_ok else 0,
        "data_source": data_source,
        "status": "open",
    }

    cols = ", ".join(row.keys())
    placeholders = ", ".join(["?"] * len(row))
    sql = f"INSERT INTO signals ({cols}) VALUES ({placeholders})"

    cur = conn.execute(sql, list(row.values()))
    conn.commit()
    signal_id = cur.lastrowid
    log.info("Saved signal #%s: %s %s %s (score=%+.1f)",
             str(signal_id), signal.symbol, dir_str, signal.label, signal.score)
    return int(signal_id) if signal_id is not None else 0


def get_open_signals(conn: sqlite3.Connection) -> list[dict]:
    """Get all signals with status='open'."""
    rows = conn.execute(
        "SELECT * FROM signals WHERE status = 'open' ORDER BY ts DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def get_last_signal(conn: sqlite3.Connection,
                    symbol: str | None = None,
                    direction: str | None = None) -> dict | None:
    """Get the most recent signal, optionally filtered by symbol and/or direction."""
    conditions: list[str] = []
    params: list[Any] = []
    if symbol:
        conditions.append("symbol = ?")
        params.append(symbol)
    if direction:
        conditions.append("direction = ?")
        params.append(direction)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    row = conn.execute(
        f"SELECT * FROM signals {where} ORDER BY ts DESC LIMIT 1",
        params
    ).fetchone()
    return dict(row) if row else None


def get_recent_signals(conn: sqlite3.Connection,
                       hours: int = 24,
                       symbol: str | None = None) -> list[dict]:
    """Get signals from the last N hours."""
    cutoff_ms = int((time.time() - hours * 3600) * 1000)
    if symbol:
        rows = conn.execute(
            "SELECT * FROM signals WHERE ts >= ? AND symbol = ? ORDER BY ts DESC",
            (cutoff_ms, symbol)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM signals WHERE ts >= ? ORDER BY ts DESC",
            (cutoff_ms,)
        ).fetchall()
    return [dict(r) for r in rows]


def update_status(conn: sqlite3.Connection, signal_id: int,
                  status: str) -> None:
    """Update a signal's status (open → won/lost/expired/invalidated)."""
    valid = {"open", "won", "lost", "expired", "invalidated"}
    if status not in valid:
        raise ValueError(f"Invalid status {status!r}, must be one of {valid}")
    conn.execute("UPDATE signals SET status = ? WHERE id = ?",
                 (status, signal_id))
    conn.commit()
    log.info("Signal #%d status → %s", signal_id, status)


def is_on_cooldown(conn: sqlite3.Connection, symbol: str,
                   direction: int, cooldown_s: int = COOLDOWN_S) -> bool:
    """Check if a recent signal exists for the same symbol+direction.

    Prevents duplicate alerts within the cooldown window.
    """
    cutoff_ms = int((time.time() - cooldown_s) * 1000)
    dir_str = "long" if direction > 0 else "short"
    row = conn.execute(
        "SELECT id FROM signals "
        "WHERE symbol = ? AND direction = ? AND ts >= ? "
        "ORDER BY ts DESC LIMIT 1",
        (symbol, dir_str, cutoff_ms)
    ).fetchone()
    if row:
        log.debug("Cooldown active for %s %s (signal #%d)",
                  symbol, dir_str, row["id"])
        return True
    return False


def has_overlapping_signal(conn: sqlite3.Connection, symbol: str,
                           direction: int, entry_low: float,
                           entry_high: float, atr: float) -> bool:
    """Check if an open signal already covers this price zone.

    Returns True if there is an open (status='open') signal for the same
    symbol + direction whose entry zone overlaps the new one within 1 ATR
    of tolerance.  This prevents firing near-identical signals when price
    oscillates around the same level.
    """
    if atr <= 0:
        return False
    dir_str = "long" if direction > 0 else "short"
    rows = conn.execute(
        "SELECT id, entry_low, entry_high FROM signals "
        "WHERE symbol = ? AND direction = ? AND status = 'open' "
        "AND entry_low IS NOT NULL AND entry_high IS NOT NULL",
        (symbol, dir_str)
    ).fetchall()
    for row in rows:
        existing_lo = float(row["entry_low"])
        existing_hi = float(row["entry_high"])
        # Overlap check with ATR tolerance band
        # New zone expanded by 1 ATR on each side
        new_lo = entry_low - atr
        new_hi = entry_high + atr
        if new_lo <= existing_hi and new_hi >= existing_lo:
            log.debug("Overlap: %s %s signal #%d entry [%.2f–%.2f] "
                      "overlaps new [%.2f–%.2f] (±%.2f ATR)",
                      symbol, dir_str, row["id"],
                      existing_lo, existing_hi,
                      entry_low, entry_high, atr)
            return True
    return False

