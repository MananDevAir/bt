"""Hyperliquid candle fetcher — keyless public API.

Hyperliquid serves market data via a single POST endpoint with no API key.
Supports crypto perps (BTC, ETH, PAXG …) and HIP-3 builder markets for
indices (xyz:SP500, xyz:XYZ100) and forex (xyz:EUR, xyz:GBP, xyz:JPY).

Rate limit: 1200 requests/min — far more than this bot needs.
"""
from __future__ import annotations

import time

import pandas as pd
import requests

URL = "https://api.hyperliquid.xyz/info"
HEADERS = {"Content-Type": "application/json"}
TIMEOUT = 20
COLUMNS = ["open", "high", "low", "close", "volume"]

# Map our standard interval strings to Hyperliquid's names.
# Hyperliquid uses the same names for most, but let's be explicit.
INTERVAL_MAP = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "2h": "2h", "4h": "4h", "8h": "8h", "12h": "12h",
    "1d": "1d", "3d": "3d", "1w": "1w", "1M": "1M",
}

# Approximate bar duration in milliseconds, for computing startTime.
BAR_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000,
    "4h": 14_400_000, "8h": 28_800_000, "12h": 43_200_000,
    "1d": 86_400_000, "3d": 259_200_000, "1w": 604_800_000,
    "1M": 2_592_000_000,
}


class FetchError(RuntimeError):
    pass


def fetch(ticker: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    """Fetch closed candles for one symbol/timeframe, oldest first.

    ticker: e.g. "BTC", "ETH", "PAXG", "xyz:SP500", "xyz:XYZ100", "xyz:EUR"
    timeframe: e.g. "15m", "1h", "4h", "1d"
    limit: approximate number of bars desired
    """
    interval = INTERVAL_MAP.get(timeframe)
    if interval is None:
        raise FetchError(f"unsupported timeframe {timeframe!r} for Hyperliquid")

    now_ms = int(time.time() * 1000)
    bar_dur = BAR_MS.get(timeframe, 3_600_000)
    # Request a bit more than needed so we can drop the live candle
    start_ms = now_ms - (int(limit) + 5) * bar_dur

    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": ticker,
            "interval": interval,
            "startTime": start_ms,
            "endTime": now_ms,
        },
    }

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.post(URL, json=payload, headers=HEADERS, timeout=TIMEOUT)
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                raise FetchError(f"unexpected response for {ticker}: {str(data)[:120]}")
            return _to_frame(data, limit)
        except FetchError:
            raise
        except Exception as exc:
            last_error = exc
            time.sleep(1 * (attempt + 1))

    raise FetchError(f"hyperliquid fetch failed for {ticker} {timeframe}: {last_error}")


def _to_frame(raw: list[dict], limit: int) -> pd.DataFrame:
    """Convert Hyperliquid candle JSON to a pandas DataFrame.

    Each candle: {"t": open_ms, "T": close_ms, "s": symbol,
                  "i": interval, "o": open, "c": close,
                  "h": high, "l": low, "v": volume, "n": num_trades}
    """
    if not raw:
        return _empty()

    now_ms = int(time.time() * 1000)
    rows, index = [], []

    for candle in raw:
        close_time = int(candle.get("T", 0))
        if close_time >= now_ms:
            continue  # candle still forming — never analyse it
        open_time = int(candle.get("t", 0))
        index.append(open_time)
        rows.append([
            float(candle["o"]),
            float(candle["h"]),
            float(candle["l"]),
            float(candle["c"]),
            float(candle.get("v", 0)),
        ])

    if not rows:
        return _empty()

    df = pd.DataFrame(rows, columns=COLUMNS, dtype=float)
    df.index = pd.to_datetime(index, unit="ms", utc=True)
    df.index.name = "ts"
    # Trim to requested limit (oldest are dropped)
    return df.tail(limit)


def _empty() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name="ts")
    return pd.DataFrame(columns=COLUMNS, index=idx, dtype=float)
