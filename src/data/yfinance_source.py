"""Yahoo Finance candle fetcher via yfinance — no API key needed.

Primary source for Dow Jones (^DJI) and fallback for indices/forex.
yfinance is an unofficial Yahoo Finance scraper; it has no guaranteed SLA
but is generous in practice and needs zero signup.

Limitations:
- 15m data: max ~60 days of history
- 1h data: max ~730 days
- 1d data: max ~20 years (more than enough)
- No official rate limit, but we throttle to be polite.
"""
from __future__ import annotations

import time

import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]

# Cached Ticker objects — avoids repeated metadata lookups across calls
_ticker_cache: dict[str, object] = {}

# Map our timeframes to yfinance (period, interval) pairs.
# period controls how far back to fetch; interval is the bar size.
TF_MAP = {
    "15m": ("60d", "15m"),     # max 60 days for 15m
    "1h":  ("730d", "1h"),     # max ~2 years for 1h
    "4h":  ("730d", "1h"),     # yfinance has no 4h — fetch 1h and resample
    "1d":  ("5y", "1d"),       # plenty for 300-bar daily
    "1w":  ("10y", "1wk"),     # ~520 bars, yfinance uses "1wk" not "1w"
}


class FetchError(RuntimeError):
    pass


def fetch(ticker: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    """Fetch closed candles for one symbol/timeframe, oldest first.

    ticker: e.g. "^DJI", "^GSPC", "^NDX", "EURUSD=X"
    timeframe: e.g. "15m", "1h", "4h", "1d"
    limit: approximate number of bars desired
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise FetchError(f"yfinance not installed: {exc}") from exc

    mapping = TF_MAP.get(timeframe)
    if mapping is None:
        raise FetchError(f"unsupported timeframe {timeframe!r} for yfinance")

    period, interval = mapping
    needs_resample = (timeframe == "4h")


    last_error: Exception | None = None
    for attempt in range(3):
        try:
            if ticker not in _ticker_cache:
                _ticker_cache[ticker] = yf.Ticker(ticker)
            t = _ticker_cache[ticker]
            data = t.history(period=period, interval=interval)
            if data is None or data.empty:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                return _empty()

            df = _normalize(data)

            if needs_resample:
                df = _resample_4h(df)

            # Trim to requested limit
            return df.tail(limit)

        except FetchError:
            raise
        except Exception as exc:
            last_error = exc
            time.sleep(2 ** attempt)

    raise FetchError(f"yfinance fetch failed for {ticker} {timeframe}: {last_error}")


def _normalize(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance output to our standard format."""
    df = data.copy()

    # yfinance column names are capitalized
    df.columns = [c.lower() for c in df.columns]

    # Keep only OHLCV
    for col in COLUMNS:
        if col not in df.columns:
            if col == "volume":
                df["volume"] = 0.0
            else:
                raise FetchError(f"missing column {col!r} in yfinance data")

    df = df[COLUMNS].copy()
    df = df.astype(float)

    # Ensure UTC timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "ts"

    # Drop rows with NaN closes (can happen on Yahoo gaps)
    df = df.dropna(subset=["close"])
    return df


def _resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1h data to 4h candles."""
    if df.empty:
        return df
    resampled = df.resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["close"])
    return resampled


def _empty() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name="ts")
    return pd.DataFrame(columns=COLUMNS, index=idx, dtype=float)
