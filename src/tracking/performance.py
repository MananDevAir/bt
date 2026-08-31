"""Performance tracker — computes win rate, avg R, and statistics.

Reads from the signals + outcomes tables and produces aggregated stats
for daily performance snapshots and on-demand queries.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from typing import Any

log = logging.getLogger(__name__)

__all__ = ["compute_stats", "save_daily_snapshot", "get_daily_snapshot"]


def compute_stats(conn: sqlite3.Connection,
                  hours: int | None = None,
                  day_utc: str | None = None) -> dict[str, Any]:
    """Compute performance statistics.

    Args:
        hours: look back N hours from now (e.g. 24 for last day)
        day_utc: specific day "YYYY-MM-DD" (overrides hours)

    Returns a dict with:
        total, wins, losses, expired, open,
        win_rate, avg_mfe_r, avg_mae_r, best_r, worst_r,
        by_symbol, by_label
    """
    if day_utc:
        # Signals from a specific UTC day
        # Day boundaries in epoch ms
        from datetime import datetime, timezone
        start = int(datetime.strptime(day_utc, "%Y-%m-%d").replace(
            tzinfo=timezone.utc).timestamp() * 1000)
        end = start + 86400 * 1000
        signals = conn.execute(
            "SELECT s.*, o.mfe_r, o.mae_r, o.hit, o.bars_held "
            "FROM signals s LEFT JOIN outcomes o ON s.id = o.signal_id "
            "WHERE s.ts >= ? AND s.ts < ? ORDER BY s.ts",
            (start, end)
        ).fetchall()
    elif hours:
        cutoff_ms = int((time.time() - hours * 3600) * 1000)
        signals = conn.execute(
            "SELECT s.*, o.mfe_r, o.mae_r, o.hit, o.bars_held "
            "FROM signals s LEFT JOIN outcomes o ON s.id = o.signal_id "
            "WHERE s.ts >= ? ORDER BY s.ts",
            (cutoff_ms,)
        ).fetchall()
    else:
        signals = conn.execute(
            "SELECT s.*, o.mfe_r, o.mae_r, o.hit, o.bars_held "
            "FROM signals s LEFT JOIN outcomes o ON s.id = o.signal_id "
            "ORDER BY s.ts"
        ).fetchall()

    total = len(signals)
    if total == 0:
        return _empty_stats()

    wins = sum(1 for s in signals if s["status"] == "won")
    losses = sum(1 for s in signals if s["status"] == "lost")
    expired = sum(1 for s in signals if s["status"] == "expired")
    still_open = sum(1 for s in signals if s["status"] == "open")
    closed = wins + losses

    win_rate = (wins / closed * 100) if closed > 0 else 0.0

    # R metrics
    mfe_vals = [s["mfe_r"] for s in signals if s["mfe_r"] is not None]
    mae_vals = [s["mae_r"] for s in signals if s["mae_r"] is not None]
    avg_mfe = sum(mfe_vals) / len(mfe_vals) if mfe_vals else 0
    avg_mae = sum(mae_vals) / len(mae_vals) if mae_vals else 0
    best_r = max(mfe_vals) if mfe_vals else 0
    worst_r = max(mae_vals) if mae_vals else 0  # worst adverse excursion

    # By symbol breakdown
    by_symbol: dict[str, dict] = {}
    for s in signals:
        sym = s["symbol"]
        if sym not in by_symbol:
            by_symbol[sym] = {"wins": 0, "losses": 0, "total": 0}
        by_symbol[sym]["total"] += 1
        if s["status"] == "won":
            by_symbol[sym]["wins"] += 1
        elif s["status"] == "lost":
            by_symbol[sym]["losses"] += 1

    # By label breakdown
    by_label: dict[str, dict] = {}
    for s in signals:
        label = s["label"]
        if label not in by_label:
            by_label[label] = {"wins": 0, "losses": 0, "total": 0}
        by_label[label]["total"] += 1
        if s["status"] == "won":
            by_label[label]["wins"] += 1
        elif s["status"] == "lost":
            by_label[label]["losses"] += 1

    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "expired": expired,
        "open": still_open,
        "win_rate": round(win_rate, 1),
        "avg_mfe_r": round(avg_mfe, 2),
        "avg_mae_r": round(avg_mae, 2),
        "best_r": round(best_r, 2),
        "worst_r": round(worst_r, 2),
        "by_symbol": by_symbol,
        "by_label": by_label,
    }


def save_daily_snapshot(conn: sqlite3.Connection, day_utc: str) -> dict:
    """Compute and save a daily performance snapshot."""
    stats = compute_stats(conn, day_utc=day_utc)

    conn.execute(
        "INSERT OR REPLACE INTO performance_daily "
        "(day_utc, total_signals, wins, losses, open_signals, "
        " win_rate, avg_r, best_r, worst_r, by_symbol, by_label) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            day_utc,
            stats["total"],
            stats["wins"],
            stats["losses"],
            stats["open"],
            stats["win_rate"],
            stats["avg_mfe_r"],
            stats["best_r"],
            stats["worst_r"],
            json.dumps(stats["by_symbol"]),
            json.dumps(stats["by_label"]),
        )
    )
    conn.commit()
    log.info("Daily snapshot saved for %s: %d signals, %.1f%% win rate",
             day_utc, stats["total"], stats["win_rate"])
    return stats


def get_daily_snapshot(conn: sqlite3.Connection,
                      day_utc: str) -> dict | None:
    """Load a saved daily snapshot."""
    row = conn.execute(
        "SELECT * FROM performance_daily WHERE day_utc = ?", (day_utc,)
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["by_symbol"] = json.loads(d.get("by_symbol") or "{}")
    d["by_label"] = json.loads(d.get("by_label") or "{}")
    return d


def _empty_stats() -> dict[str, Any]:
    return {
        "total": 0, "wins": 0, "losses": 0, "expired": 0, "open": 0,
        "win_rate": 0.0, "avg_mfe_r": 0.0, "avg_mae_r": 0.0,
        "best_r": 0.0, "worst_r": 0.0,
        "by_symbol": {}, "by_label": {},
    }
