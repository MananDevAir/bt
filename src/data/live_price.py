"""Lightweight live price fetcher for outcome checking.

Fetches the current price for a symbol without relying on the candle cache.
Uses free, no-auth APIs:
  - Binance for crypto (BTC, ETH, XAUUSDT)
  - Yahoo Finance for stocks/forex (US30, US500, US100, EURUSD, etc.)

This is intentionally minimal — one HTTP call per symbol, returns a dict
with high, low, close from the most recent completed candle.
"""
from __future__ import annotations

import logging
import requests
from typing import Any

log = logging.getLogger(__name__)

__all__ = ["get_live_price"]

# Timeout for API calls (seconds)
_TIMEOUT = 8

# Symbol → Binance ticker mapping
_BINANCE_MAP: dict[str, str] = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "XAUUSDT": "XAUTUSDT",
}

# Symbol → Yahoo Finance ticker mapping
_YAHOO_MAP: dict[str, str] = {
    "US100": "^NDX",
    "US500": "^GSPC",
    "US30": "^DJI",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
}


def get_live_price(symbol: str) -> dict[str, float] | None:
    """Fetch the latest price for a symbol.

    Returns {"high": float, "low": float, "close": float} or None on failure.
    Tries Binance first (for crypto), then Yahoo Finance (for everything else).
    """
    # Try Binance
    binance_ticker = _BINANCE_MAP.get(symbol)
    if binance_ticker:
        result = _fetch_binance(binance_ticker)
        if result:
            return result

    # Try Yahoo Finance
    yahoo_ticker = _YAHOO_MAP.get(symbol)
    if yahoo_ticker:
        result = _fetch_yahoo(yahoo_ticker)
        if result:
            return result

    # Fallback: try Binance with symbol + USDT
    result = _fetch_binance(f"{symbol}USDT")
    if result:
        return result

    log.warning("Could not fetch live price for %s", symbol)
    return None


def _fetch_binance(ticker: str) -> dict[str, float] | None:
    """Fetch the latest 15m kline from Binance (free, no key)."""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": ticker,
        "interval": "15m",
        "limit": 2,  # Get last 2 candles (second-to-last is the closed one)
    }
    try:
        resp = requests.get(url, params=params, timeout=_TIMEOUT)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not data or len(data) < 2:
            return None
        # Use the second-to-last candle (most recently closed)
        candle = data[-2]
        return {
            "high": float(candle[2]),
            "low": float(candle[3]),
            "close": float(candle[4]),
        }
    except Exception as exc:
        log.debug("Binance fetch failed for %s: %s", ticker, exc)
        return None


def _fetch_yahoo(ticker: str) -> dict[str, float] | None:
    """Fetch the latest price from Yahoo Finance (free, no key)."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "interval": "15m",
        "range": "5d",  # 5d ensures weekend queries on forex/stocks still find the last closed candle
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=_TIMEOUT)
        if resp.status_code != 200:
            return None
        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None
        
        quote = result[0].get("indicators", {}).get("quote", [{}])[0]
        highs = quote.get("high", [])
        lows = quote.get("low", [])
        closes = quote.get("close", [])

        # Find the last valid non-None candle
        valid_indices = [i for i, c in enumerate(closes) if c is not None]
        if valid_indices:
            idx = valid_indices[-2] if len(valid_indices) >= 2 else valid_indices[-1]
            h = highs[idx] if idx < len(highs) and highs[idx] is not None else closes[idx]
            l = lows[idx] if idx < len(lows) and lows[idx] is not None else closes[idx]
            c = closes[idx]
            return {
                "high": float(h),
                "low": float(l),
                "close": float(c),
            }

        # Fallback to meta market price if candle list is empty
        meta_price = result[0].get("meta", {}).get("regularMarketPrice")
        if meta_price is not None:
            mp = float(meta_price)
            return {"high": mp, "low": mp, "close": mp}

        return None
    except Exception as exc:
        log.debug("Yahoo fetch failed for %s: %s", ticker, exc)
        return None
