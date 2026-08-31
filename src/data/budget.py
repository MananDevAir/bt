"""Twelve Data credit budget: daily ledger + per-minute rate limiter.

The free plan allows 8 credits/minute and 800/day (reset 00:00 UTC). Blowing
through either one means the whole TradFi side goes dark, so spending is
metered here and nowhere else. Crypto never touches this module.
"""
from __future__ import annotations

import sqlite3
import time
from collections import deque
from datetime import datetime, timezone

PROVIDER = "twelvedata"


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
        self._recent: deque[float] = deque()

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

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
        self.conn.execute(
            "INSERT INTO api_usage(day_utc, provider, credits) VALUES(?,?,?) "
            "ON CONFLICT(day_utc, provider) DO UPDATE SET credits = credits + ?",
            (self._today(), self.provider, credits, credits),
        )
        self.conn.commit()

    def throttle(self) -> None:
        """Block until another request fits inside the per-minute allowance."""
        now = time.monotonic()
        while self._recent and now - self._recent[0] >= 60.0:
            self._recent.popleft()
        if len(self._recent) >= self.per_min:
            sleep_for = 60.0 - (now - self._recent[0]) + 0.25
            if sleep_for > 0:
                time.sleep(sleep_for)
            now = time.monotonic()
            while self._recent and now - self._recent[0] >= 60.0:
                self._recent.popleft()
        self._recent.append(time.monotonic())

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
