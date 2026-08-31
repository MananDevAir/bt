"""Smart Money Concepts — market structure, BOS, CHoCH, order blocks, FVGs,
liquidity sweeps, premium/discount, supply/demand zones.

Every concept is fully deterministic: same candles → same labels. The detection
rules follow ICT methodology distilled into code. No randomness, no
forward-looking.

Depends on: indicators.atr, pivots.pivots
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from .indicators import atr as calc_atr
from .pivots import Pivot, pivots

__all__ = [
    "StructureLabel", "StructurePoint", "market_structure",
    "BOS", "detect_bos",
    "OrderBlock", "detect_order_blocks",
    "FVG", "detect_fvg",
    "LiquiditySweep", "detect_liquidity_sweeps",
    "EqualLevel", "detect_equal_levels",
    "PremiumDiscount", "premium_discount",
    "smc_scan",
]


# =========================================================================== #
# Market structure: HH, HL, LH, LL labelling
# =========================================================================== #
class StructureLabel(str, Enum):
    HH = "HH"   # higher high
    HL = "HL"   # higher low
    LH = "LH"   # lower high
    LL = "LL"   # lower low


@dataclass(frozen=True, slots=True)
class StructurePoint:
    pivot: Pivot
    label: StructureLabel
    bias: int             # +1 bullish, -1 bearish


def market_structure(df: pd.DataFrame, left: int = 3, right: int = 3) -> list[StructurePoint]:
    """Label each swing pivot as HH/HL/LH/LL and derive the structural bias.

    Bias is +1 when the last label is HH or HL (bullish structure),
    -1 when it is LH or LL (bearish structure).
    """
    pv = pivots(df, left, right)
    if len(pv) < 2:
        return []

    result: list[StructurePoint] = []
    prev_highs: list[Pivot] = []
    prev_lows: list[Pivot] = []
    bias = 0

    for p in pv:
        if p.kind == "high":
            if prev_highs:
                label = StructureLabel.HH if p.price > prev_highs[-1].price else StructureLabel.LH
            else:
                label = StructureLabel.HH  # first high, assume HH
            prev_highs.append(p)
            if label == StructureLabel.HH:
                bias = +1
            elif label == StructureLabel.LH:
                bias = -1
        else:  # low
            if prev_lows:
                label = StructureLabel.HL if p.price > prev_lows[-1].price else StructureLabel.LL
            else:
                label = StructureLabel.HL  # first low, assume HL
            prev_lows.append(p)
            if label == StructureLabel.HL:
                bias = +1
            elif label == StructureLabel.LL:
                bias = -1

        result.append(StructurePoint(p, label, bias))

    return result


# =========================================================================== #
# BOS (Break of Structure) and CHoCH (Change of Character)
# =========================================================================== #
@dataclass(frozen=True, slots=True)
class BOS:
    idx: int
    ts: pd.Timestamp
    direction: int        # +1 bullish break, -1 bearish break
    kind: str             # "bos" or "choch"
    broken_level: float   # the swing level that was broken
    close_price: float    # the closing price that broke it


def detect_bos(df: pd.DataFrame, structure: list[StructurePoint] | None = None,
               left: int = 3, right: int = 3) -> list[BOS]:
    """Detect Break of Structure and Change of Character.

    BOS: candle body closes beyond the last swing high/low in the direction of
         the prevailing trend.
    CHoCH: BOS in the opposite direction of the prevailing trend — early reversal.

    We use body close (not wick) to filter fakeouts.
    """
    if structure is None:
        structure = market_structure(df, left, right)
    if not structure:
        return []

    o = df["open"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    result: list[BOS] = []

    # Track the last significant swing high and low
    last_swing_high: Pivot | None = None
    last_swing_low: Pivot | None = None
    prev_bias = 0

    for sp in structure:
        if sp.pivot.kind == "high":
            last_swing_high = sp.pivot
        else:
            last_swing_low = sp.pivot
        prev_bias = sp.bias

    if last_swing_high is None or last_swing_low is None:
        return []

    # Scan bars after the last confirmed pivot for breaks
    last_pivot_idx = max(last_swing_high.idx, last_swing_low.idx)
    swing_highs = [sp.pivot for sp in structure if sp.pivot.kind == "high"]
    swing_lows = [sp.pivot for sp in structure if sp.pivot.kind == "low"]

    # Check each swing for breaks by subsequent candles
    broken_highs: set[int] = set()
    broken_lows: set[int] = set()

    for sh in swing_highs[-5:]:  # check last 5 swing highs
        for i in range(sh.idx + 1, len(df)):
            body_close = c[i]
            if body_close > sh.price and sh.idx not in broken_highs:
                broken_highs.add(sh.idx)
                # Determine if this is BOS or CHoCH
                # If bias was already bullish, this is BOS; if bearish, this is CHoCH
                bias_at_break = _bias_at(structure, sh.idx)
                kind = "bos" if bias_at_break >= 0 else "choch"
                result.append(BOS(i, df.index[i], +1, kind, sh.price, body_close))
                break

    for sl in swing_lows[-5:]:  # check last 5 swing lows
        for i in range(sl.idx + 1, len(df)):
            body_close = c[i]
            if body_close < sl.price and sl.idx not in broken_lows:
                broken_lows.add(sl.idx)
                bias_at_break = _bias_at(structure, sl.idx)
                kind = "bos" if bias_at_break <= 0 else "choch"
                result.append(BOS(i, df.index[i], -1, kind, sl.price, body_close))
                break

    # Sort by index and return only the most recent ones
    result.sort(key=lambda b: b.idx)
    return result[-10:]  # keep last 10 at most


def _bias_at(structure: list[StructurePoint], before_idx: int) -> int:
    """Get the structural bias at a given point in time."""
    bias = 0
    for sp in structure:
        if sp.pivot.idx > before_idx:
            break
        bias = sp.bias
    return bias


# =========================================================================== #
# Order Blocks
# =========================================================================== #
@dataclass(frozen=True, slots=True)
class OrderBlock:
    idx: int
    ts: pd.Timestamp
    hi: float
    lo: float
    direction: int        # +1 bullish OB (demand), -1 bearish OB (supply)
    mitigated: bool       # True if price has already returned to this OB
    displacement: float   # size of the move that followed (in ATR units)


def detect_order_blocks(df: pd.DataFrame, bos_list: list[BOS] | None = None,
                        atr_series: pd.Series | None = None) -> list[OrderBlock]:
    """Detect order blocks: the last opposite-colour candle before a displacement
    leg that caused a BOS. Zone = that candle's high-low range.

    An OB is marked mitigated if price has returned to its zone after formation.
    """
    if len(df) < 10:
        return []
    if atr_series is None:
        atr_series = calc_atr(df["high"], df["low"], df["close"], 14)
    if bos_list is None:
        bos_list = detect_bos(df)
    if not bos_list:
        return []

    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    a = atr_series.to_numpy(dtype=float)

    obs: list[OrderBlock] = []

    for bos in bos_list:
        if bos.idx < 2:
            continue
        atr_val = a[bos.idx] if not np.isnan(a[bos.idx]) else 0
        if atr_val <= 0:
            continue

        # Walk backwards from the BOS candle to find the last opposite-colour candle
        ob_idx = None
        for j in range(bos.idx - 1, max(bos.idx - 10, 0), -1):
            is_bull = c[j] > o[j]
            if bos.direction > 0 and not is_bull:  # bullish BOS, find last bearish candle
                ob_idx = j
                break
            elif bos.direction < 0 and is_bull:  # bearish BOS, find last bullish candle
                ob_idx = j
                break

        if ob_idx is None:
            continue

        ob_hi = h[ob_idx]
        ob_lo = l[ob_idx]
        displacement = abs(bos.close_price - (ob_hi if bos.direction > 0 else ob_lo))
        disp_atr = displacement / atr_val if atr_val > 0 else 0

        # Only count if displacement is meaningful (> 0.5 ATR)
        if disp_atr < 0.5:
            continue

        # Check mitigation: has price returned to the OB zone after formation?
        mitigated = False
        for k in range(ob_idx + 1, len(df)):
            if bos.direction > 0:  # bullish OB: demand zone
                if l[k] <= ob_hi:  # price dipped back into the zone
                    mitigated = True
                    break
            else:  # bearish OB: supply zone
                if h[k] >= ob_lo:  # price rallied back into the zone
                    mitigated = True
                    break

        obs.append(OrderBlock(
            ob_idx, df.index[ob_idx], ob_hi, ob_lo,
            bos.direction, mitigated, round(disp_atr, 2),
        ))

    return obs


# =========================================================================== #
# Fair Value Gaps (FVGs) — 3-candle imbalance
# =========================================================================== #
@dataclass(frozen=True, slots=True)
class FVG:
    idx: int              # index of the middle candle
    ts: pd.Timestamp
    hi: float             # top of the gap
    lo: float             # bottom of the gap
    direction: int        # +1 bullish, -1 bearish
    filled_pct: float     # 0.0 to 1.0, how much has been filled
    still_open: bool      # True if < 50% filled


def detect_fvg(df: pd.DataFrame, min_gap_atr: float = 0.1) -> list[FVG]:
    """Detect Fair Value Gaps (3-candle imbalance).

    Bullish FVG: candle3.low > candle1.high (gap up)
    Bearish FVG: candle1.low > candle3.high (gap down)

    Track how much the gap has been filled by subsequent price action.
    """
    if len(df) < 3:
        return []

    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    atr_s = calc_atr(df["high"], df["low"], df["close"], 14).to_numpy(dtype=float)

    fvgs: list[FVG] = []

    for i in range(2, len(df)):
        if np.isnan(atr_s[i]) or atr_s[i] <= 0:
            continue

        # Bullish FVG: candle 3 low > candle 1 high
        if l[i] > h[i - 2]:
            gap_size = l[i] - h[i - 2]
            if gap_size < min_gap_atr * atr_s[i]:
                continue
            gap_hi = l[i]
            gap_lo = h[i - 2]
            # Check fill: how far has price come down into the gap?
            filled = 0.0
            for k in range(i + 1, len(df)):
                penetration = gap_hi - l[k]
                if penetration > 0:
                    filled = max(filled, penetration / gap_size)
            still_open = filled < 0.5
            fvgs.append(FVG(i - 1, df.index[i - 1], gap_hi, gap_lo,
                            +1, min(1.0, filled), still_open))

        # Bearish FVG: candle 1 low > candle 3 high
        if l[i - 2] > h[i]:
            gap_size = l[i - 2] - h[i]
            if gap_size < min_gap_atr * atr_s[i]:
                continue
            gap_hi = l[i - 2]
            gap_lo = h[i]
            filled = 0.0
            for k in range(i + 1, len(df)):
                penetration = h[k] - gap_lo
                if penetration > 0:
                    filled = max(filled, penetration / gap_size)
            still_open = filled < 0.5
            fvgs.append(FVG(i - 1, df.index[i - 1], gap_hi, gap_lo,
                            -1, min(1.0, filled), still_open))

    # Return only recent, still-open FVGs (within last 100 bars)
    cutoff = len(df) - 100
    return [f for f in fvgs if f.idx >= cutoff or f.still_open]


# =========================================================================== #
# Liquidity sweeps — wick beyond swing, close back inside
# =========================================================================== #
@dataclass(frozen=True, slots=True)
class LiquiditySweep:
    idx: int
    ts: pd.Timestamp
    swept_level: float
    direction: int        # +1 swept lows (bullish reversal signal), -1 swept highs (bearish)
    wick_beyond: float    # how far the wick went past the level


def detect_liquidity_sweeps(df: pd.DataFrame, left: int = 3,
                            right: int = 3) -> list[LiquiditySweep]:
    """Detect liquidity sweeps: wick pierces a prior swing then closes back inside.

    A sweep of lows is a bullish signal (stops were hunted, smart money loaded).
    A sweep of highs is bearish.
    """
    pv = pivots(df, left, right)
    if not pv:
        return []

    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    o = df["open"].to_numpy(dtype=float)

    sweeps: list[LiquiditySweep] = []

    swing_highs = [p for p in pv if p.kind == "high"]
    swing_lows = [p for p in pv if p.kind == "low"]

    # Check recent bars for sweeps
    lookback = min(30, len(df) - 1)
    for i in range(len(df) - lookback, len(df)):
        body_hi = max(o[i], c[i])
        body_lo = min(o[i], c[i])

        # Sweep of highs: wick goes above swing high, body closes below
        for sh in swing_highs:
            if sh.idx >= i:
                continue
            if h[i] > sh.price and body_hi < sh.price:
                beyond = h[i] - sh.price
                sweeps.append(LiquiditySweep(
                    i, df.index[i], sh.price, -1, beyond))

        # Sweep of lows: wick goes below swing low, body closes above
        for sl in swing_lows:
            if sl.idx >= i:
                continue
            if l[i] < sl.price and body_lo > sl.price:
                beyond = sl.price - l[i]
                sweeps.append(LiquiditySweep(
                    i, df.index[i], sl.price, +1, beyond))

    # Deduplicate: keep the most recent sweep per level
    seen: set[float] = set()
    unique: list[LiquiditySweep] = []
    for s in reversed(sweeps):
        key = round(s.swept_level, 8)
        if key not in seen:
            seen.add(key)
            unique.append(s)
    unique.reverse()
    return unique[-10:]


# =========================================================================== #
# Equal highs / lows — liquidity pools
# =========================================================================== #
@dataclass(frozen=True, slots=True)
class EqualLevel:
    price: float
    kind: str             # "equal_highs" or "equal_lows"
    count: int            # how many swings cluster here
    first_ts: pd.Timestamp
    last_ts: pd.Timestamp


def detect_equal_levels(df: pd.DataFrame, atr_mult: float = 0.1,
                        left: int = 3, right: int = 3) -> list[EqualLevel]:
    """Two or more swings within atr_mult × ATR = liquidity pool."""
    pv = pivots(df, left, right)
    if len(pv) < 2:
        return []

    atr_s = calc_atr(df["high"], df["low"], df["close"], 14)
    atr_val = float(atr_s.iloc[-1]) if not np.isnan(atr_s.iloc[-1]) else 0
    if atr_val <= 0:
        return []
    threshold = atr_mult * atr_val

    result: list[EqualLevel] = []
    for kind in ("high", "low"):
        swings = [p for p in pv if p.kind == kind]
        if len(swings) < 2:
            continue
        # Cluster swings by proximity
        clusters: list[list[Pivot]] = []
        for s in swings:
            merged = False
            for cluster in clusters:
                avg = sum(p.price for p in cluster) / len(cluster)
                if abs(s.price - avg) <= threshold:
                    cluster.append(s)
                    merged = True
                    break
            if not merged:
                clusters.append([s])

        for cluster in clusters:
            if len(cluster) >= 2:
                avg_price = sum(p.price for p in cluster) / len(cluster)
                label = "equal_highs" if kind == "high" else "equal_lows"
                result.append(EqualLevel(
                    avg_price, label, len(cluster),
                    cluster[0].ts, cluster[-1].ts))

    return result


# =========================================================================== #
# Premium / Discount zones
# =========================================================================== #
@dataclass(frozen=True, slots=True)
class PremiumDiscount:
    zone: str             # "premium", "discount", "equilibrium"
    pct: float            # position within dealing range (0=low, 100=high)
    range_high: float
    range_low: float
    equilibrium: float


def premium_discount(df: pd.DataFrame, lookback: int = 50) -> PremiumDiscount:
    """Where current price sits within the recent dealing range.

    >70% = premium (sellers' zone), <30% = discount (buyers' zone),
    50% = equilibrium.
    """
    window = df.tail(lookback)
    range_high = float(window["high"].max())
    range_low = float(window["low"].min())
    span = range_high - range_low
    current = float(df["close"].iloc[-1])

    if span <= 0:
        return PremiumDiscount("equilibrium", 50.0, range_high, range_low,
                               (range_high + range_low) / 2)

    pct = 100.0 * (current - range_low) / span
    eq = (range_high + range_low) / 2.0

    if pct > 70:
        zone = "premium"
    elif pct < 30:
        zone = "discount"
    else:
        zone = "equilibrium"

    return PremiumDiscount(zone, round(pct, 1), range_high, range_low, eq)


# =========================================================================== #
# Bundle: run all SMC detectors
# =========================================================================== #
def smc_scan(df: pd.DataFrame, left: int = 3, right: int = 3) -> dict:
    """Run all Smart Money Concept detectors on one OHLCV frame.

    Returns a dict with keys: structure, bos, order_blocks, fvg,
    liquidity_sweeps, equal_levels, premium_discount.
    """
    struct = market_structure(df, left, right)
    bos_list = detect_bos(df, struct, left, right)
    obs = detect_order_blocks(df, bos_list)
    fvgs = detect_fvg(df)
    sweeps = detect_liquidity_sweeps(df, left, right)
    eq = detect_equal_levels(df, 0.1, left, right)
    pd_zone = premium_discount(df)

    # Derive overall bias from structure
    bias_label = "neutral"
    if struct:
        last_bias = struct[-1].bias
        if last_bias > 0:
            bias_label = "bullish"
        elif last_bias < 0:
            bias_label = "bearish"

    return {
        "structure": struct,
        "structure_bias": bias_label,
        "bos": bos_list,
        "order_blocks": obs,
        "order_blocks_unmitigated": [ob for ob in obs if not ob.mitigated],
        "fvg": fvgs,
        "fvg_open": [f for f in fvgs if f.still_open],
        "liquidity_sweeps": sweeps,
        "equal_levels": eq,
        "premium_discount": pd_zone,
    }
