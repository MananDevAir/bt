"""Win/loss streak tracking with automatic threshold adjustment.

Tracks consecutive wins and losses. After 3 consecutive losses,
automatically raises the 'watch' threshold by +5 to require stronger
confluence before trading. Resets after 2 consecutive wins.

State is persisted in the bot_state table so it survives GitHub Actions restarts.
"""
from __future__ import annotations

import json
import logging
import sqlite3

log = logging.getLogger(__name__)

__all__ = ["update_streak", "get_streak", "get_threshold_adjustment"]

# After this many consecutive losses, raise thresholds
LOSS_STREAK_TRIGGER = 3
# After this many consecutive wins, reset the penalty
WIN_STREAK_RESET = 2
# How much to raise the watch threshold per trigger
THRESHOLD_BUMP = 5
# Maximum threshold bump (don't raise more than this total)
MAX_BUMP = 15


def _get_state(conn: sqlite3.Connection, key: str, default: str = "") -> str:
    """Read a value from bot_state."""
    row = conn.execute(
        "SELECT value FROM bot_state WHERE key = ?", (key,)
    ).fetchone()
    return row["value"] if row else default


def _set_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Write a value to bot_state."""
    conn.execute(
        "INSERT OR REPLACE INTO bot_state (key, value) VALUES (?, ?)",
        (key, value)
    )
    conn.commit()


def get_streak(conn: sqlite3.Connection) -> dict:
    """Get the current streak state."""
    return {
        "loss_streak": int(_get_state(conn, "loss_streak", "0")),
        "win_streak": int(_get_state(conn, "win_streak", "0")),
        "threshold_bump": int(_get_state(conn, "threshold_bump", "0")),
    }


def update_streak(conn: sqlite3.Connection, outcome: str) -> dict:
    """Update the streak after a trade closes.

    Args:
        outcome: "won" or "lost"

    Returns:
        Updated streak state dict with any threshold adjustment.
    """
    loss_streak = int(_get_state(conn, "loss_streak", "0"))
    win_streak = int(_get_state(conn, "win_streak", "0"))
    threshold_bump = int(_get_state(conn, "threshold_bump", "0"))

    if outcome == "won":
        win_streak += 1
        loss_streak = 0  # reset loss streak

        # After enough consecutive wins, remove the threshold penalty
        if win_streak >= WIN_STREAK_RESET and threshold_bump > 0:
            threshold_bump = max(0, threshold_bump - THRESHOLD_BUMP)
            log.info("Win streak %d: removing threshold bump (now +%d)",
                     win_streak, threshold_bump)

    elif outcome == "lost":
        loss_streak += 1
        win_streak = 0  # reset win streak

        # After enough consecutive losses, raise thresholds
        if loss_streak >= LOSS_STREAK_TRIGGER:
            new_bump = min(MAX_BUMP, threshold_bump + THRESHOLD_BUMP)
            if new_bump != threshold_bump:
                threshold_bump = new_bump
                log.warning("Loss streak %d: raising threshold bump to +%d",
                            loss_streak, threshold_bump)

    _set_state(conn, "loss_streak", str(loss_streak))
    _set_state(conn, "win_streak", str(win_streak))
    _set_state(conn, "threshold_bump", str(threshold_bump))

    return {
        "loss_streak": loss_streak,
        "win_streak": win_streak,
        "threshold_bump": threshold_bump,
    }


def get_threshold_adjustment(conn: sqlite3.Connection) -> int:
    """Get the current threshold adjustment due to streaks.

    Returns a non-negative integer that should be ADDED to the 'watch'
    threshold. 0 means no adjustment.
    """
    return int(_get_state(conn, "threshold_bump", "0"))
