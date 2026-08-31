"""Crypto / tokenized-asset candles from Binance public REST (no API key).

Binance serves klines on a keyless public endpoint with a 6,000 request-weight
per-minute IP allowance; a full scan of a handful of symbols costs a few dozen,
so there is no practical limit for this bot.

Only CLOSED candles are returned. Binance includes the in-progress candle as the
last row and using it would make signals repaint, so it is dropped here once,
centrally, rather than trusted to every caller.
"""
from __future__ import annotations

import time

import pandas as pd
import requests

BASE_URLS = (
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api-gcp.binance.com",
)
MAX_LIMIT = 1000
TIMEOUT = 20

COLUMNS = ["open", "high", "low", "close", "volume"]


class FetchError(RuntimeError):
    pass


def _to_binance_symbol(ticker: str) -> str:
    return ticker.replace("/", "").replace("-", "").upper()


def fetch(ticker: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    """Fetch closed candles for one symbol/timeframe, oldest first."""
    symbol = _to_binance_symbol(ticker)
    # ask for one extra so dropping the live candle still leaves `limit`
    want = min(MAX_LIMIT, int(limit) + 1)
    params = {"symbol": symbol, "interval": timeframe, "limit": want}

    last_error: Exception | None = None
    for base in BASE_URLS:
        for attempt in range(3):
            try:
                resp = requests.get(
                    f"{base}/api/v3/klines", params=params, timeout=TIMEOUT
                )
                if resp.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                return _to_frame(resp.json())
            except Exception as exc:  # network, HTTP, or parse
                last_error = exc
                time.sleep(0.5 * (attempt + 1))
    raise FetchError(f"binance fetch failed for {ticker} {timeframe}: {last_error}")


def _to_frame(raw: list) -> pd.DataFrame:
    if not raw:
        return _empty()
    now_ms = int(time.time() * 1000)
    rows, index = [], []
    for k in raw:
        close_time = int(k[6])
        if close_time >= now_ms:
            continue  # candle still forming - never analyse it
        index.append(int(k[0]))
        rows.append([float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])])
    if not rows:
        return _empty()
    df = pd.DataFrame(rows, columns=COLUMNS, dtype=float)
    df.index = pd.to_datetime(index, unit="ms", utc=True)
    df.index.name = "ts"
    return df


def _empty() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name="ts")
    return pd.DataFrame(columns=COLUMNS, index=idx, dtype=float)


def fetch_via_ccxt(ticker: str, timeframe: str, limit: int, exchanges: list[str]) -> pd.DataFrame:
    """Fallback for hosts where Binance is geo-blocked (HTTP 451).

    ccxt is imported lazily so the bot still runs if it is not installed.
    Exchange instances are cached at module level to avoid repeated market
    metadata loading.
    """
    try:
        import ccxt  # noqa: PLC0415 - optional dependency
    except ImportError as exc:
        raise FetchError(f"ccxt not installed, cannot fall back: {exc}") from exc

    # Module-level cache for exchange instances
    if not hasattr(fetch_via_ccxt, "_exchanges"):
        fetch_via_ccxt._exchanges = {}  # type: ignore[attr-defined]
    ex_cache = fetch_via_ccxt._exchanges  # type: ignore[attr-defined]

    last_error: Exception | None = None
    for name in exchanges:
        try:
            if name not in ex_cache:
                ex_cache[name] = getattr(ccxt, name)({"enableRateLimit": True})
            ex = ex_cache[name]
            raw = ex.fetch_ohlcv(ticker, timeframe=timeframe, limit=int(limit) + 1)
            if not raw:
                continue
            df = pd.DataFrame(
                [r[1:6] for r in raw], columns=COLUMNS, dtype=float
            )
            df.index = pd.to_datetime([int(r[0]) for r in raw], unit="ms", utc=True)
            df.index.name = "ts"
            # ccxt also returns the live candle last
            return df.iloc[:-1] if len(df) > 1 else df
        except Exception as exc:
            last_error = exc
    raise FetchError(f"all ccxt fallbacks failed for {ticker}: {last_error}")
