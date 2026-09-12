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
    "SOL": "SOLUSDT",
    "ZEC": "ZECUSDT",
    # XAUUSDT: Binance does not have gold spot. Falls through to Hyperliquid PAXG.
}

# Symbol → Hyperliquid builder perps (real-time 24/7/365, zero delay)
_HYPERLIQUID_MAP: dict[str, str] = {
    "US100": "xyz:XYZ100",
    "US500": "xyz:SP500",
    "BTC": "BTC",
    "ETH": "ETH",
    "SOL": "SOL",
    "ZEC": "ZEC",
    "XAUUSDT": "PAXG",
}


# Symbol → Yahoo Finance ticker mapping (fallback)
_YAHOO_MAP: dict[str, str] = {
    "US100": "NQ=F",
    "US500": "ES=F",
    "US30": "YM=F",
    "XAUUSDT": "GC=F",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
}



def get_live_price(symbol: str) -> dict[str, float] | None:
    """Fetch the latest price for a symbol.

    Returns {"high": float, "low": float, "close": float, "source": str} or None on failure.
    Priority order:
      1. Binance (for crypto / metals)
      2. Hyperliquid (for 24/7 real-time perp indices US100, US500, EURUSD)
      3. Yahoo Finance (fallback for traditional equities / forex)
    """
    # 1. Try Binance
    binance_ticker = _BINANCE_MAP.get(symbol)
    if binance_ticker:
        result = _fetch_binance(binance_ticker)
        if result:
            result["source"] = "Binance"
            return result

    # 2. Try Hyperliquid (24/7 real-time sub-second perp market)
    hl_ticker = _HYPERLIQUID_MAP.get(symbol)
    if hl_ticker:
        result = _fetch_hyperliquid(hl_ticker)
        if result:
            result["source"] = "Hyperliquid"
            return result

    # 3. Try Yahoo Finance (fallback)
    yahoo_ticker = _YAHOO_MAP.get(symbol)
    if yahoo_ticker:
        result = _fetch_yahoo(yahoo_ticker)
        if result:
            result["source"] = "Yahoo Finance"
            return result

    # Fallback: try Binance with symbol + USDT
    result = _fetch_binance(f"{symbol}USDT")
    if result:
        result["source"] = "Binance"
        return result

    log.warning("Could not fetch live price for %s", symbol)
    return None


def _fetch_hyperliquid(ticker: str) -> dict[str, float] | None:
    """Fetch the latest 15m candle from Hyperliquid (24/7 real-time, keyless)."""
    import time
    now_ms = int(time.time() * 1000)
    url = "https://api.hyperliquid.xyz/info"
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": ticker,
            "interval": "15m",
            "startTime": now_ms - 3600 * 1000,
            "endTime": now_ms,
        },
    }
    try:
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=_TIMEOUT)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not isinstance(data, list) or not data:
            return None
        candle = data[-1]
        return {
            "high": float(candle["h"]),
            "low": float(candle["l"]),
            "close": float(candle["c"]),
        }
    except Exception as exc:
        log.debug("Hyperliquid fetch failed for %s: %s", ticker, exc)
        return None


def _fetch_binance(ticker: str) -> dict[str, float] | None:
    """Fetch the latest 15m kline from Binance Futures / Spot (free, no key)."""
    endpoints = [
        "https://fapi.binance.com/fapi/v1/klines",  # Binance USDT-M Perpetual Futures
        "https://api.binance.com/api/v3/klines",     # Binance Spot fallback
    ]
    params = {
        "symbol": ticker,
        "interval": "15m",
        "limit": 2,  # Get last 2 candles (second-to-last is the closed one)
    }
    for url in endpoints:
        try:
            resp = requests.get(url, params=params, timeout=_TIMEOUT)
            if resp.status_code != 200:
                continue
            data = resp.json()
            if not data:
                continue
            # Use the last COMPLETED candle — consistent with close-based SL check in outcome_checker.
            # Using data[-1] (the live bar) could trigger TP on an intra-bar wick that fully reverses.
            candle = data[-2] if len(data) >= 2 else data[0]
            return {
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
            }
        except Exception as exc:
            log.debug("Binance fetch failed for %s on %s: %s", ticker, url, exc)
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
            idx = valid_indices[-1]
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
