"""Trade plan generator — entry zone, stop loss, take profits, R:R.

All values are **suggestions for the user** — nothing is executed.
Levels are derived from structure (order blocks, FVGs, swing points, S/R,
Fibonacci) and bounded by ATR-based sanity checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from .confluence import SignalResult
from .indicators import atr as calc_atr
from .pivots import Pivot, pivots

__all__ = ["TradePlan", "generate_plan"]


@dataclass
class TradePlan:
    """A complete trade suggestion — entry, stop, targets, risk metrics."""
    direction: int            # +1 long, -1 short
    entry_low: float
    entry_high: float
    entry_mid: float
    sl: float
    tp1: float
    tp2: float
    tp3: float
    rr: float                 # R:R to TP2
    risk_pct: float           # distance to SL as % of entry
    risk_atr: float           # distance to SL in ATR units
    tp_allocation: list[int]  # e.g. [50, 30, 20]
    invalidation: str         # condition that invalidates the trade
    holding_horizon: str      # expected hold time from trigger TF
    source: str               # what determined entry: ob, fvg, fib_ote, market
    trade_type: str = ""      # Intraday, Swing, Short-term, Positional
    brief_reason: str = ""    # 1-line human-readable reason for the signal
    session: str = ""         # e.g. London Session, New York Session
    killzone: str = ""        # e.g. London Open, New York Open (if in killzone)


def generate_plan(signal: SignalResult, cfg: Config) -> TradePlan | None:
    """Build a trade plan from a scored signal.

    Returns None if the plan fails hard gates (R:R too low, stop too wide).
    """
    if signal.label == "NEUTRAL" or signal.direction == 0:
        return None

    direction = signal.direction
    risk_cfg = cfg.get("risk", default={}) or {}
    gates_cfg = cfg.get("gates", default={}) or {}
    min_rr = float(gates_cfg.get("min_rr", 1.5))
    max_stop_atr = float(gates_cfg.get("max_stop_atr", 3.0))
    atr_stop_mult = float(risk_cfg.get("atr_stop_mult", 1.5))
    struct_buffer = float(risk_cfg.get("structure_stop_buffer_atr", 0.20))
    tp_r_mults = risk_cfg.get("tp_r_multiples", [1.0, 2.0, 3.0])
    tp_alloc = risk_cfg.get("tp_allocation", [50, 30, 20])
    snap_tol = float(risk_cfg.get("snap_tolerance_atr", 0.30))

    # Use LTF for entry precision, HTF for structural context
    ltf = cfg.ltf
    htf = cfg.htf
    ltf_result = signal.tf_results.get(ltf)
    htf_result = signal.tf_results.get(htf)

    if ltf_result is None or ltf_result.classic is None:
        return None

    atr_val = float(ltf_result.classic["atr"].iloc[-1])
    if np.isnan(atr_val) or atr_val <= 0:
        return None

    # Get actual close price from the raw OHLCV frame
    if ltf_result.raw_df is not None and not ltf_result.raw_df.empty:
        current_close = float(ltf_result.raw_df["close"].iloc[-1])
    else:
        # Fallback: approximate from ema20
        current_close = float(ltf_result.classic["ema20"].iloc[-1])
    if np.isnan(current_close) or current_close <= 0:
        return None

    # Use HTF ATR for wider context
    htf_atr = atr_val
    if htf_result and htf_result.classic is not None:
        htf_a = float(htf_result.classic["atr"].iloc[-1])
        if not np.isnan(htf_a) and htf_a > 0:
            htf_atr = htf_a

    # ----- Entry zone -----
    entry_mid, entry_lo, entry_hi, source = _find_entry(
        signal, direction, current_close, atr_val)

    # ----- Stop loss -----
    sl = _find_stop(signal, direction, entry_mid, atr_val,
                    atr_stop_mult, struct_buffer)

    risk = abs(entry_mid - sl)
    if risk <= 0:
        signal.gates["plan_risk_zero"] = {
            "action": "drop", "detail": "risk distance is zero"}
        return None

    # Stop distance gate
    risk_atr_units = risk / atr_val if atr_val > 0 else 0
    if risk > max_stop_atr * atr_val:
        signal.gates["stop_too_wide"] = {
            "action": "drop",
            "detail": f"stop distance {risk_atr_units:.1f} ATR > max {max_stop_atr:.1f} ATR",
        }
        return None  # stop too wide

    # Price location gate: is price already past the stop loss?
    if direction > 0 and current_close < sl:
        signal.gates["already_stopped"] = {
            "action": "drop", "detail": f"price {current_close} already below SL {sl}"
        }
        return None
    if direction < 0 and current_close > sl:
        signal.gates["already_stopped"] = {
            "action": "drop", "detail": f"price {current_close} already above SL {sl}"
        }
        return None

    # ----- Take profits -----
    tp1 = entry_mid + direction * tp_r_mults[0] * risk
    tp2 = entry_mid + direction * tp_r_mults[1] * risk
    tp3 = entry_mid + direction * tp_r_mults[2] * risk

    # Snap TPs to nearby structure levels
    tp3 = _snap_to_structure(tp3, signal, direction, atr_val, snap_tol)

    # R:R (measured to TP2)
    rr = abs(tp2 - entry_mid) / risk if risk > 0 else 0
    if rr < min_rr:
        signal.gates["rr_too_low"] = {
            "action": "drop",
            "detail": f"R:R {rr:.2f} < minimum {min_rr:.1f}",
        }
        return None  # R:R too low

    # Risk metrics
    risk_pct = 100.0 * risk / entry_mid if entry_mid > 0 else 0
    risk_atr_units = risk / atr_val if atr_val > 0 else 0

    # Holding horizon based on LTF
    horizon_map = {"15m": "hours", "1h": "1-2 days", "4h": "2-5 days", "1d": "1-2 weeks"}
    horizon = horizon_map.get(ltf, "unknown")

    # Trade type classification
    trade_type = _classify_trade_type(signal, horizon)

    # Brief reason: why this signal was captured
    brief = _build_brief_reason(signal, direction, source)

    # Session & Killzone detection
    try:
        from ..data.sessions import get_active_killzone, get_market_session
        sess_name = get_market_session()
        kz_name = get_active_killzone() or ""
    except Exception:
        sess_name = ""
        kz_name = ""

    # Invalidation
    if direction > 0:
        inv = f"HTF/MTF close below SL ({sl:,.2f})"
    else:
        inv = f"HTF/MTF close above SL ({sl:,.2f})"

    return TradePlan(
        direction=direction,
        entry_low=round(entry_lo, _decimals(entry_mid)),
        entry_high=round(entry_hi, _decimals(entry_mid)),
        entry_mid=round(entry_mid, _decimals(entry_mid)),
        sl=round(sl, _decimals(sl)),
        tp1=round(tp1, _decimals(tp1)),
        tp2=round(tp2, _decimals(tp2)),
        tp3=round(tp3, _decimals(tp3)),
        rr=round(rr, 2),
        risk_pct=round(risk_pct, 2),
        risk_atr=round(risk_atr_units, 2),
        tp_allocation=list(tp_alloc),
        invalidation=inv,
        holding_horizon=horizon,
        source=source,
        trade_type=trade_type,
        brief_reason=brief,
        session=sess_name,
        killzone=kz_name,
    )


# =========================================================================== #
# Entry zone finder
# =========================================================================== #
def _find_entry(signal: SignalResult, direction: int, close: float,
                atr_val: float) -> tuple[float, float, float, str]:
    """Find the best entry zone from OBs, FVGs, fib OTE, or market."""

    # Priority 1: nearest unmitigated order block in signal direction
    for tf, tfr in signal.tf_results.items():
        if tfr.smc:
            for ob in tfr.smc.get("order_blocks_unmitigated", []):
                if ob.direction == direction:
                    dist = abs(close - (ob.hi + ob.lo) / 2) / atr_val
                    if dist < 3.0:  # within 3 ATR
                        return (ob.hi + ob.lo) / 2, ob.lo, ob.hi, "order_block"

    # Priority 2: nearest open FVG in signal direction
    for tf, tfr in signal.tf_results.items():
        if tfr.smc:
            for fvg in tfr.smc.get("fvg_open", []):
                if fvg.direction == direction:
                    dist = abs(close - (fvg.hi + fvg.lo) / 2) / atr_val
                    if dist < 3.0:
                        return (fvg.hi + fvg.lo) / 2, fvg.lo, fvg.hi, "fvg"

    # Priority 3: Fibonacci OTE (0.618-0.705)
    for tf, tfr in signal.tf_results.items():
        if tfr.pa:
            for fib in tfr.pa.get("fibonacci", []):
                if fib.kind == "retracement" and 0.600 <= fib.ratio <= 0.720:
                    dist = abs(close - fib.price) / atr_val
                    if dist < 2.0:
                        band = 0.25 * atr_val
                        return fib.price, fib.price - band, fib.price + band, "fib_ote"

    # Fallback: market entry (current close ± 0.25 ATR)
    band = 0.25 * atr_val
    return close, close - band, close + band, "market"


# =========================================================================== #
# Stop loss finder
# =========================================================================== #
def _find_stop(signal: SignalResult, direction: int, entry: float,
               atr_val: float, atr_mult: float, buffer: float) -> float:
    """Find stop loss: structure first, ATR as sanity bound."""

    # ATR-based stop
    sl_atr = entry - direction * atr_mult * atr_val

    # Structure-based stop: last swing in the opposite direction
    sl_struct = sl_atr  # fallback
    for tf in ("15m", "1h", "4h", "1d"):
        tfr = signal.tf_results.get(tf)
        if not tfr or not tfr.smc:
            continue
        structure = tfr.smc.get("structure", [])
        for sp in reversed(structure):
            if direction > 0 and sp.pivot.kind == "low":
                candidate = sp.pivot.price - buffer * atr_val
                if candidate < entry:
                    sl_struct = candidate
                    break
            elif direction < 0 and sp.pivot.kind == "high":
                candidate = sp.pivot.price + buffer * atr_val
                if candidate > entry:
                    sl_struct = candidate
                    break
        if sl_struct != sl_atr:
            break

    # If structure stop is too tight (< 0.8 ATR from entry), use ATR stop
    struct_dist = abs(entry - sl_struct)
    if struct_dist < 0.8 * atr_val:
        return sl_atr

    return sl_struct


# =========================================================================== #
# TP snapping to structure
# =========================================================================== #
def _snap_to_structure(tp: float, signal: SignalResult, direction: int,
                       atr_val: float, tolerance: float) -> float:
    """If a real level is near a raw TP, snap to just before it."""
    for tf, tfr in signal.tf_results.items():
        if not tfr.pa:
            continue
        # Check S/R zones
        for sr in tfr.pa.get("sr_zones", []):
            if abs(sr.mid - tp) <= tolerance * atr_val:
                # Snap to just inside the level (0.05 ATR)
                return sr.mid - direction * 0.05 * atr_val
        # Check key levels
        for kl in tfr.pa.get("key_levels", []):
            if abs(kl.price - tp) <= tolerance * atr_val:
                return kl.price - direction * 0.05 * atr_val
    return tp


# =========================================================================== #
# Helpers
# =========================================================================== #
def _get_close(ci: pd.DataFrame) -> float | None:
    """Extract the current close price from a classic indicator frame."""
    # The classic frame has ema20 which is close to the close; but we need
    # the actual close. It's indexed the same as the original OHLCV.
    # We can approximate from ema20 for level computation.
    if ci is None or ci.empty:
        return None
    # Use ema20 as a proxy — it's very close to close on the last bar
    ema20 = ci["ema20"].iloc[-1]
    if not np.isnan(ema20):
        return float(ema20)
    return None


def _decimals(price: float) -> int:
    """Auto-detect reasonable decimal places for a price."""
    if price >= 1000:
        return 2
    if price >= 10:
        return 2
    if price >= 1:
        return 4
    return 5


# =========================================================================== #
# Trade type classification
# =========================================================================== #
def _classify_trade_type(signal: 'SignalResult', horizon: str) -> str:
    """Classify the trade type based on timeframe dominance and score.

    Returns one of: Intraday, Swing, Short-term, Positional
    """
    # Calculate which timeframes contributed most to the score
    tf_strength: dict[str, float] = {}
    for tf, tfr in signal.tf_results.items():
        if tfr.votes:
            avg = sum(abs(v.value) for v in tfr.votes) / len(tfr.votes)
            tf_strength[tf] = avg

    # Check if higher timeframes dominate
    macro_htf = sum(tf_strength.get(tf, 0) for tf in ("1w", "1d"))
    mtf = sum(tf_strength.get(tf, 0) for tf in ("4h", "1h"))
    ltf = tf_strength.get("15m", 0)

    score_abs = abs(signal.score)

    # Strong signal across all timeframes with weekly/daily alignment
    if score_abs >= 40 and macro_htf > mtf:
        return "Positional"  # days to weeks
    # Daily/4h alignment with good score
    elif score_abs >= 30 and macro_htf >= 0.5:
        return "Short-term"  # 1-5 days
    # 4h/1h structure with 15m trigger
    elif mtf > ltf and horizon in ("hours", "1-2 days"):
        return "Swing"  # hours to 2 days
    # Primarily 15m/1h driven
    else:
        return "Intraday"  # hours


def _build_brief_reason(signal: 'SignalResult', direction: int,
                        entry_source: str) -> str:
    """Build a 1-line human-readable reason for the signal.

    Examples:
      "Daily uptrend + 4h BOS + 15m pullback to order block"
      "Weekly bearish + 1h CHoCH + FVG retest"
    """
    parts: list[str] = []

    # Macro/HTF bias
    for tf in ("1w", "1d"):
        tfr = signal.tf_results.get(tf)
        if not tfr or not tfr.votes:
            continue
        avg = sum(v.value for v in tfr.votes) / len(tfr.votes)
        if abs(avg) >= 0.3:
            tf_label = "Weekly" if tf == "1w" else "Daily"
            bias = "uptrend" if avg > 0 else "downtrend"
            parts.append(f"{tf_label} {bias}")
        break  # only the highest available

    # MTF structure (BOS/CHoCH)
    for tf in ("4h", "1h"):
        tfr = signal.tf_results.get(tf)
        if not tfr or not tfr.votes:
            continue
        for v in tfr.votes:
            if v.name in ("bos", "choch") and abs(v.value) >= 0.6:
                parts.append(f"{tf} {v.detail}")
                break
        if len(parts) >= 2:
            break

    # If no BOS/CHoCH found, use strongest MTF trigger
    if len(parts) < 2:
        for tf in ("4h", "1h"):
            tfr = signal.tf_results.get(tf)
            if not tfr or not tfr.votes:
                continue
            best = max(tfr.votes, key=lambda v: abs(v.value), default=None)
            if best and abs(best.value) >= 0.5:
                parts.append(f"{tf} {best.detail}")
                break

    # LTF trigger / entry method
    source_labels = {
        "order_block": "entry at order block",
        "fvg": "entry at FVG",
        "fib_ote": "pullback to fib OTE",
        "market": "market entry",
    }
    entry_label = source_labels.get(entry_source, entry_source)
    parts.append(entry_label)

    return " + ".join(parts[:3]) if parts else "confluence signal"
