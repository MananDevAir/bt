"""Twelve Data credit budget: daily ledger + per-minute rate limiter.

The free plan allows 8 credits/minute and 800/day (reset 00:00 UTC). Blowing
through either one means the whole TradFi side goes dark, so spending is
metered here and nowhere else. Crypto never touches this module.

Fix 5: The per-minute deque is persisted to bot_state so it survives restarts.
       A restart mid-minute no longer resets the rate counter to zero.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime, timezone

log = logging.getLogger(__name__)

PROVIDER = "twelvedata"

# Key used to store per-minute call timestamps in bot_state
_RATE_STATE_KEY = "td_rate_ts"


class BudgetExceeded(RuntimeError):
    """Raised when the daily credit cap has been reached."""


class Budget:
    def __init__(
        self,
        conn: sqlite3.Connection,
        daily_cap: int = 750,
        per_min: int = 7,
        provider: str = PROVIDER,
    ) -> None:
        self.conn = conn
        self.daily_cap = int(daily_cap)
        self.per_min = int(per_min)
        self.provider = provider
        self._lock = threading.Lock()
        self._recent: deque[float] = deque()
        self._load_recent()  # Fix 5: restore in-flight rate window from DB

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # ── Fix 5: persistence helpers ───────────────────────────────────────── #

    def _load_recent(self) -> None:
        """Load persisted per-minute timestamps from bot_state on startup."""
        try:
            row = self.conn.execute(
                "SELECT value FROM bot_state WHERE key = ?", (_RATE_STATE_KEY,)
            ).fetchone()
            if not row:
                return
            now = time.time()
            # Filter to only timestamps within the last 60 seconds
            timestamps = [ts for ts in json.loads(row["value"]) if now - ts < 60.0]
            self._recent = deque(timestamps)
        except Exception:
            pass  # If corrupt or missing, start fresh — safe default

    def _persist_recent(self) -> None:
        """Save current per-minute timestamps to bot_state."""
        try:
            now = time.time()
            # Prune stale entries before persisting
            timestamps = [ts for ts in self._recent if now - ts < 60.0]
            self.conn.execute(
                "INSERT INTO bot_state(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (_RATE_STATE_KEY, json.dumps(timestamps)),
            )
            self.conn.commit()
        except Exception as exc:
            log.warning("Budget _persist_recent failed: %s", exc)

    # ── Core budget logic ─────────────────────────────────────────────────── #

    def used(self) -> int:
        row = self.conn.execute(
            "SELECT credits FROM api_usage WHERE day_utc=? AND provider=?",
            (self._today(), self.provider),
        ).fetchone()
        return int(row["credits"]) if row else 0

    def remaining(self) -> int:
        return max(0, self.daily_cap - self.used())

    def can_spend(self, credits: int = 1) -> bool:
        return self.used() + credits <= self.daily_cap

    def spend(self, credits: int = 1) -> None:
        """Record credits after a successful call."""
        with self._lock:
            self.conn.execute(
                "INSERT INTO api_usage(day_utc, provider, credits) VALUES(?,?,?) "
                "ON CONFLICT(day_utc, provider) DO UPDATE SET credits = credits + ?",
                (self._today(), self.provider, credits, credits),
            )
            self.conn.commit()

    def throttle(self) -> None:
        """Block until another request fits inside the per-minute allowance."""
        with self._lock:
            now = time.time()  # wall-clock so it persists across restarts
            while self._recent and now - self._recent[0] >= 60.0:
                self._recent.popleft()
            if len(self._recent) >= self.per_min:
                sleep_for = 60.0 - (now - self._recent[0]) + 0.25
                if sleep_for > 0:
                    time.sleep(sleep_for)
                now = time.time()
                while self._recent and now - self._recent[0] >= 60.0:
                    self._recent.popleft()
            self._recent.append(time.time())
            self._persist_recent()  # Fix 5: save after every call

    def acquire(self, credits: int = 1) -> None:
        """Gate a call: enforce the daily cap, then the per-minute pace."""
        if not self.can_spend(credits):
            raise BudgetExceeded(
                f"{self.provider} daily cap reached "
                f"({self.used()}/{self.daily_cap} credits)"
            )
        self.throttle()

    def report(self) -> str:
        return f"{self.used()}/{self.daily_cap} credits used today"
