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


def get_active_killzone(now: datetime | None = None) -> str | None:
    """Check if current time is within high-probability ICT/liquidity killzones.
    
    Anchored to America/New_York to correctly observe DST shifts.
    """
    ny = _now_utc(now).astimezone(NY)
    if ny.weekday() > 4:  # Weekend
        return None
    t = ny.time()
    
    # Asian Killzone: 20:00 - 00:00 NY
    if time(20, 0) <= t or t < time(0, 0):
        return "Asian Open"
    # London Killzone: 02:00 - 05:00 NY
    if time(2, 0) <= t < time(5, 0):
        return "London Open"
    # New York Killzone: 07:00 - 10:00 NY
    if time(7, 0) <= t < time(10, 0):
        return "New York Open"
    # London Close: 10:00 - 12:00 NY
    if time(10, 0) <= t < time(12, 0):
        return "London Close"
    
    return None


def get_market_session(now: datetime | None = None) -> str:
    """Return the active global session name (lowercase, stable identifiers).

    Returns one of: 'asian_dead', 'asian', 'london', 'overlap', 'new_york',
    'new_york_close', 'weekend'.
    
    Anchored to America/New_York to correctly observe DST shifts.
    """
    ny = _now_utc(now).astimezone(NY)
    if ny.weekday() > 4:          # Saturday or Sunday
        return "weekend"
    
    t = ny.time()
    # Asian Dead Zone: 19:00 - 00:00 NY
    if time(19, 0) <= t or t < time(0, 0):
        return "asian_dead"       # pre-London dead zone — high fakeout risk
    # Asian / Pre-London: 00:00 - 02:00 NY
    if time(0, 0) <= t < time(2, 0):
        return "asian"            # late Asian / early Europe pre-open
    # London cash session: 02:00 - 07:30 NY
    if time(2, 0) <= t < time(7, 30):
        return "london"           
    # London / New York overlap: 07:30 - 12:00 NY
    if time(7, 30) <= t < time(12, 0):
        return "overlap"          # highest volume
    # New York afternoon: 12:00 - 16:00 NY
    if time(12, 0) <= t < time(16, 0):
        return "new_york"         
    # NY Close / early Asian build-up: 16:00 - 19:00 NY
    return "new_york_close"
