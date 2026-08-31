"""Indices / forex candles from Twelve Data (free plan: 8/min, 800/day).

Budget strategy that makes the free tier work (see PLAN.md section 5):
  * pull ONE deep 15m series per symbol per scan (outputsize up to 5000,
    ~52 days) and resample it up to 1h/4h locally  -> 1 credit per symbol
  * pull the 1d series at most once per day for 200-EMA depth -> 1 credit

Every call goes through Budget.acquire() which enforces the daily cap and paces
requests below 8/min.
"""
from __future__ import annotations

import time

import pandas as pd
import requests

from .budget import Budget, BudgetExceeded

URL = "https://api.twelvedata.com/time_series"
TIMEOUT = 25
MAX_OUTPUTSIZE = 5000

COLUMNS = ["open", "high", "low", "close", "volume"]

# Twelve Data interval names differ from ours for the daily bar.
INTERVAL_MAP = {"15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1day"}


class FetchError(RuntimeError):
    pass


class MissingKey(FetchError):
    pass


def fetch(
    ticker: str,
    timeframe: str,
    limit: int,
    api_key: str | None,
    budget: Budget,
) -> pd.DataFrame:
    """Fetch closed candles for one symbol/timeframe, oldest first."""
    if not api_key:
        raise MissingKey("TWELVEDATA_KEY is not set")

    interval = INTERVAL_MAP.get(timeframe)
    if interval is None:
        raise FetchError(f"timeframe {timeframe!r} not supported by twelvedata")

    params = {
        "symbol": ticker,
        "interval": interval,
        "outputsize": min(MAX_OUTPUTSIZE, int(limit)),
        "apikey": api_key,
        "format": "JSON",
        "timezone": "UTC",
        "order": "ASC",
    }

    budget.acquire(1)  # raises BudgetExceeded before any network call
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.get(URL, params=params, timeout=TIMEOUT)
            if resp.status_code == 429:
                time.sleep(2 ** (attempt + 1))
                continue
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("status") == "error":
                message = str(payload.get("message", "unknown error"))
                # a rejected call still counts against nothing - do not spend
                raise FetchError(f"twelvedata: {message}")
            budget.spend(1)
            return _to_frame(payload.get("values") or [])
        except FetchError:
            raise
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise FetchError(f"twelvedata fetch failed for {ticker} {timeframe}: {last_error}")


def _to_frame(values: list[dict]) -> pd.DataFrame:
    if not values:
        return empty()
    rows, index = [], []
    for v in values:
        try:
            index.append(pd.Timestamp(v["datetime"], tz="UTC"))
            rows.append(
                [
                    float(v["open"]),
                    float(v["high"]),
                    float(v["low"]),
                    float(v["close"]),
                    float(v.get("volume") or 0.0),
                ]
            )
        except (KeyError, TypeError, ValueError):
            continue  # skip malformed bar rather than fail the whole series
    if not rows:
        return empty()
    df = pd.DataFrame(rows, columns=COLUMNS, dtype=float)
    df.index = pd.DatetimeIndex(index, name="ts")
    return df.sort_index()


def empty() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name="ts")
    return pd.DataFrame(columns=COLUMNS, index=idx, dtype=float)


__all__ = ["fetch", "empty", "FetchError", "MissingKey", "BudgetExceeded"]
