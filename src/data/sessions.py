"""Market session calendar.

Decides whether a symbol is tradeable right now, so we do not burn Twelve Data
credits fetching a closed index at 3am. DST is handled by zoneinfo rather than
hardcoded UTC offsets, which is why US cash hours stay correct in both summer
and winter.

Known simplification: exchange holidays are not modelled. A holiday costs a few
wasted credits and produces a stale-data scan, not a wrong signal, because the
analysis layer only ever reads closed candles.
"""
from __future__ import annotations

from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")

US_CASH_OPEN = time(9, 30)
US_CASH_CLOSE = time(16, 0)

# FX trades continuously from Sunday 17:00 NY to Friday 17:00 NY.
FX_WEEK_BOUNDARY = time(17, 0)


def _now_utc(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def is_us_cash_open(now: datetime | None = None) -> bool:
    ny = _now_utc(now).astimezone(NY)
    if ny.weekday() > 4:  # Saturday, Sunday
        return False
    return US_CASH_OPEN <= ny.time() < US_CASH_CLOSE


def is_fx_week_open(now: datetime | None = None) -> bool:
    ny = _now_utc(now).astimezone(NY)
    weekday, clock = ny.weekday(), ny.time()
    if weekday == 6:  # Sunday - opens at 17:00
        return clock >= FX_WEEK_BOUNDARY
    if weekday == 5:  # Saturday - closed all day
        return False
    if weekday == 4:  # Friday - closes at 17:00
        return clock < FX_WEEK_BOUNDARY
    return True  # Mon-Thu


def is_open(session: str, now: datetime | None = None) -> bool:
    if session == "always":
        return True
    if session == "us_cash":
        return is_us_cash_open(now)
    if session == "fx_week":
        return is_fx_week_open(now)
    raise ValueError(f"unknown session {session!r}")


def describe(session: str, now: datetime | None = None) -> str:
    """Short human label used in the scan report."""
    if session == "always":
        return "24/7"
    return "OPEN" if is_open(session, now) else "closed"
