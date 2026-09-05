"""Price action analysis — candlestick patterns, key levels, support/resistance.

Deterministic detection of price action signals. Every function takes an OHLCV
DataFrame and returns structured results. No randomness, no forward-looking.

Depends on: indicators.atr, pivots.pivots
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .indicators import atr as calc_atr, sma
from .pivots import Pivot, pivots

__all__ = [
    "CandlePattern", "candle_patterns",
    "KeyLevel", "key_levels",
    "SRZone", "support_resistance",
    "FibLevel", "fibonacci_retracement",
    "price_action_scan",
]


# =========================================================================== #
# Candlestick patterns
# =========================================================================== #
@dataclass(frozen=True, slots=True)
class CandlePattern:
    name: str           # engulfing, pin_bar, doji, inside_bar, marubozu, shooting_star
    bias: int           # +1 bullish, -1 bearish, 0 neutral
    idx: int            # positional index in the frame
    ts: pd.Timestamp
    strength: float     # 0-1 quality score


def candle_patterns(df: pd.DataFrame, atr_series: pd.Series | None = None,
                    lookback: int = 5) -> list[CandlePattern]:
    """Detect classic patterns in the last `lookback` bars."""
    if len(df) < 3:
        return []
    if atr_series is None:
        atr_series = calc_atr(df["high"], df["low"], df["close"], 14)

    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    a = atr_series.to_numpy(dtype=float)
    found: list[CandlePattern] = []
    start = max(1, len(df) - lookback)

    for i in range(start, len(df)):
        if np.isnan(a[i]) or a[i] <= 0:
            continue
        body = abs(c[i] - o[i])
        rng = h[i] - l[i]
        if rng <= 0:
            continue
        upper_wick = h[i] - max(o[i], c[i])
        lower_wick = min(o[i], c[i]) - l[i]
        bull = c[i] > o[i]
        ts = df.index[i]

        # --- Doji: body < 10% of range ---
        if body < 0.10 * rng:
            found.append(CandlePattern("doji", 0, i, ts, 0.5))
            continue

        # --- Pin bar / Hammer (bullish): lower wick > 2x body, small upper wick ---
        if lower_wick > 2.0 * body and upper_wick < 0.3 * rng:
            found.append(CandlePattern("pin_bar", +1, i, ts,
                                       min(1.0, lower_wick / (2.5 * body))))

        # --- Shooting star (bearish): upper wick > 2x body, small lower wick ---
        if upper_wick > 2.0 * body and lower_wick < 0.3 * rng:
            found.append(CandlePattern("shooting_star", -1, i, ts,
                                       min(1.0, upper_wick / (2.5 * body))))

        # --- Marubozu: body > 85% of range ---
        if body > 0.85 * rng:
            bias = +1 if bull else -1
            found.append(CandlePattern("marubozu", bias, i, ts,
                                       min(1.0, body / rng)))

        # --- Engulfing: this body fully contains previous body ---
        if i >= 1:
            prev_body_hi = max(o[i - 1], c[i - 1])
            prev_body_lo = min(o[i - 1], c[i - 1])
            curr_body_hi = max(o[i], c[i])
            curr_body_lo = min(o[i], c[i])
            prev_bull = c[i - 1] > o[i - 1]
            if (curr_body_hi > prev_body_hi and curr_body_lo < prev_body_lo
                    and bull != prev_bull and body > 0.3 * a[i]):
                bias = +1 if bull else -1
                strength = min(1.0, body / a[i])
                found.append(CandlePattern("engulfing", bias, i, ts, strength))

        # --- Inside bar: this bar entirely inside previous bar ---
        if i >= 1 and h[i] <= h[i - 1] and l[i] >= l[i - 1]:
            found.append(CandlePattern("inside_bar", 0, i, ts, 0.6))

    return found


# =========================================================================== #
# Key levels — prev day/week H/L/C, round numbers
# =========================================================================== #
@dataclass(frozen=True, slots=True)
class KeyLevel:
    price: float
    label: str          # prev_day_high, prev_week_low, round_number, etc.
    importance: float   # 0-1


def key_levels(df: pd.DataFrame, current_price: float | None = None) -> list[KeyLevel]:
    """Static key levels from the current frame's data."""
    if len(df) < 2:
        return []
    levels: list[KeyLevel] = []
    price = current_price if current_price is not None else float(df["close"].iloc[-1])

    # Prev-day high/low/close (using the last 2 daily-equivalent bars)
    if len(df) >= 2:
        levels.append(KeyLevel(float(df["high"].iloc[-2]), "prev_bar_high", 0.7))
        levels.append(KeyLevel(float(df["low"].iloc[-2]), "prev_bar_low", 0.7))
        levels.append(KeyLevel(float(df["close"].iloc[-2]), "prev_bar_close", 0.5))

    # Round numbers — find the nearest ones above/below
    if price > 0:
        # Determine magnitude: use 1/10th of ATR or fallback to price-based
        magnitude = _round_number_step(price)
        base = (price // magnitude) * magnitude
        for m in range(-3, 5):
            rn = base + m * magnitude
            if rn > 0:
                levels.append(KeyLevel(rn, "round_number", 0.4))

    return levels


def _round_number_step(price: float) -> float:
    """Choose the right round-number spacing for the asset's price scale."""
    if price > 10000:
        return 1000.0       # BTC: 70000, 71000, ...
    if price > 1000:
        return 100.0        # Gold, indices: 2500, 2600, ...
    if price > 100:
        return 10.0         # ETH: 2400, 2410, ...
    if price > 10:
        return 1.0
    if price > 1:
        return 0.1          # Forex majors: 1.10, 1.20, ...
    return 0.01


# =========================================================================== #
# Support / Resistance zones — clustered pivots + volume profile
# =========================================================================== #
@dataclass(frozen=True, slots=True)
class SRZone:
    lo: float
    hi: float
    mid: float
    kind: str           # support, resistance
    touches: int
    source: str         # pivots, volume_profile, merged
    strength: float     # 0-1


def support_resistance(df: pd.DataFrame, left: int = 3, right: int = 3,
                       merge_atr_mult: float = 0.3) -> list[SRZone]:
    """Cluster swing pivots into S/R zones, merging those within merge_atr_mult × ATR."""
    if len(df) < 20:
        return []

    atr_val = calc_atr(df["high"], df["low"], df["close"], 14)
    current_atr = float(atr_val.iloc[-1]) if not np.isnan(atr_val.iloc[-1]) else 0
    if current_atr <= 0:
        return []
    merge_dist = merge_atr_mult * current_atr

    pv = pivots(df, left, right)
    if not pv:
        return []

    current_price = float(df["close"].iloc[-1])

    # Collect all pivot prices
    pivot_prices = [(p.price, p.kind) for p in pv]

    # Sort by price
    pivot_prices.sort(key=lambda x: x[0])

    # Cluster nearby pivots
    clusters: list[list[tuple[float, str]]] = []
    for pp in pivot_prices:
        merged = False
        for cluster in clusters:
            cluster_mid = sum(p[0] for p in cluster) / len(cluster)
            if abs(pp[0] - cluster_mid) <= merge_dist:
                cluster.append(pp)
                merged = True
                break
        if not merged:
            clusters.append([pp])

    zones: list[SRZone] = []
    for cluster in clusters:
        prices = [p[0] for p in cluster]
        lo, hi = min(prices), max(prices)
        mid = sum(prices) / len(prices)
        touches = len(cluster)
        kind = "support" if mid < current_price else "resistance"
        # Strength: more touches = stronger, recency bonus
        strength = min(1.0, 0.3 + 0.15 * touches)
        zones.append(SRZone(lo, hi, mid, kind, touches, "pivots", strength))

    # Sort by distance from current price (closest first)
    zones.sort(key=lambda z: abs(z.mid - current_price))
    return zones


# =========================================================================== #
# Fibonacci retracement / extension
# =========================================================================== #
FIB_LEVELS = (0.236, 0.382, 0.5, 0.618, 0.705, 0.786)
FIB_EXT_LEVELS = (-0.272, -0.618, -1.0)


@dataclass(frozen=True, slots=True)
class FibLevel:
    ratio: float
    price: float
    kind: str           # retracement, extension
    direction: int = 1  # +1 upswing (bullish retracement), -1 downswing (bearish retracement)


def fibonacci_retracement(swing_lo: float, swing_hi: float,
                          direction: int = 1) -> list[FibLevel]:
    """Fibonacci levels from a measured swing.

    direction=+1: upswing (retracements are below current, extensions above)
    direction=-1: downswing (retracements are above current, extensions below)
    """
    if swing_hi <= swing_lo:
        return []
    span = swing_hi - swing_lo
    levels: list[FibLevel] = []

    for r in FIB_LEVELS:
        if direction > 0:
            price = swing_hi - r * span   # pull-back from top
        else:
            price = swing_lo + r * span   # pull-back from bottom
        levels.append(FibLevel(r, price, "retracement", direction=direction))

    for r in FIB_EXT_LEVELS:
        if direction > 0:
            price = swing_hi - r * span   # extend above top
        else:
            price = swing_lo + r * span   # extend below bottom
        levels.append(FibLevel(r, price, "extension", direction=direction))

    return levels


# =========================================================================== #
# Bundle: run everything on a frame
# =========================================================================== #
def price_action_scan(df: pd.DataFrame) -> dict:
    """Run all price-action detectors on one OHLCV frame.

    Returns a dict with keys: patterns, key_levels, sr_zones, fibonacci.
    """
    atr_s = calc_atr(df["high"], df["low"], df["close"], 14)

    patterns = candle_patterns(df, atr_s)
    kl = key_levels(df)
    sr = support_resistance(df)

    # Fibonacci from the most recent swing
    pv = pivots(df, 3, 3)
    fibs: list[FibLevel] = []
    if len(pv) >= 2:
        # Find last significant swing
        last_high = next((p for p in reversed(pv) if p.kind == "high"), None)
        last_low = next((p for p in reversed(pv) if p.kind == "low"), None)
        if last_high and last_low:
            if last_high.idx > last_low.idx:
                # Upswing: low -> high
                fibs = fibonacci_retracement(last_low.price, last_high.price, +1)
            else:
                # Downswing: high -> low
                fibs = fibonacci_retracement(last_low.price, last_high.price, -1)

    return {
        "patterns": patterns,
        "key_levels": kl,
        "sr_zones": sr,
        "fibonacci": fibs,
    }
