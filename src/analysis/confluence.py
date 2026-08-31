"""Confluence engine — the deterministic decision maker.

Every check returns a vote in -1..+1. Votes are grouped into 5 categories,
each weighted. Each timeframe has a multiplier. The final score is normalised
to -100..+100 and mapped to a label (STRONG BUY → STRONG SELL).

Golden rule: same candles → same score, always. No randomness, no LLM input.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from .indicators import classic_frame
from .modern import modern_frame, mfi, volume_profile, divergences, volatility_regime
from .price_action import price_action_scan
from .smc import smc_scan


__all__ = ["Vote", "TFResult", "SignalResult", "score_symbol"]


# =========================================================================== #
# Category weights (sum = 100)
# =========================================================================== #
DEFAULT_WEIGHTS = {"trend": 30, "structure": 30, "momentum": 20, "zones": 12, "volume": 8}

LABELS = [
    (-100, -65, "STRONG SELL"),
    (-65,  -40, "SELL"),
    (-40,  -18, "WATCH SHORT"),
    (-18,   18, "NEUTRAL"),
    (18,    40, "WATCH LONG"),
    (40,    65, "BUY"),
    (65,   100, "STRONG BUY"),
]


# =========================================================================== #
# Data structures
# =========================================================================== #
@dataclass
class Vote:
    """One check's contribution to the score."""
    name: str
    category: str         # trend, structure, momentum, zones, volume
    value: float          # -1 .. +1
    detail: str = ""      # human-readable note


@dataclass
class TFResult:
    """All votes + computed indicators for one timeframe."""
    timeframe: str
    votes: list[Vote] = field(default_factory=list)
    raw_df: pd.DataFrame | None = None    # original OHLCV
    classic: pd.DataFrame | None = None
    modern: pd.DataFrame | None = None
    pa: dict | None = None
    smc: dict | None = None
    regime: dict | None = None


@dataclass
class SignalResult:
    """Final output of the confluence engine for one symbol."""
    symbol: str
    direction: int        # +1 long, -1 short, 0 neutral
    score: float          # -100 .. +100
    label: str            # STRONG BUY, BUY, WATCH LONG, NEUTRAL, etc.
    confidence: float     # 55-95%
    tf_results: dict[str, TFResult] = field(default_factory=dict)
    gates: dict[str, Any] = field(default_factory=dict)
    gate_passed: bool = True
    raw_score: float = 0.0
    max_possible: float = 1.0


# =========================================================================== #
# Voting functions — each returns a list of Vote objects
# =========================================================================== #
def _trend_votes(ci: pd.DataFrame, mi: pd.DataFrame, last: pd.Series,
                 mlast: pd.Series) -> list[Vote]:
    """Trend category: EMA stack, SuperTrend, Ichimoku, ADX/DI, SMA cross."""
    votes: list[Vote] = []
    c = last

    # EMA stack: 20 > 50 > 200 = strong bull, reverse = strong bear
    e20, e50, e200 = c["ema20"], c["ema50"], c["ema200"]
    if e20 > e50 > e200:
        votes.append(Vote("ema_stack", "trend", +1.0, "EMA 20>50>200 aligned bull"))
    elif e20 < e50 < e200:
        votes.append(Vote("ema_stack", "trend", -1.0, "EMA 20<50<200 aligned bear"))
    elif e20 > e200:
        votes.append(Vote("ema_stack", "trend", +0.4, "price above EMA200"))
    elif e20 < e200:
        votes.append(Vote("ema_stack", "trend", -0.4, "price below EMA200"))
    else:
        votes.append(Vote("ema_stack", "trend", 0.0, "EMAs mixed"))

    # SMA 50/200 golden/death cross
    if c["sma50"] > c["sma200"]:
        votes.append(Vote("sma_cross", "trend", +0.6, "golden cross (SMA50>200)"))
    elif c["sma50"] < c["sma200"]:
        votes.append(Vote("sma_cross", "trend", -0.6, "death cross (SMA50<200)"))

    # SuperTrend direction
    if not np.isnan(mlast["st_dir"]):
        val = float(mlast["st_dir"])  # +1 or -1
        votes.append(Vote("supertrend", "trend", val,
                          f"SuperTrend {'UP' if val > 0 else 'DOWN'}"))

    # Ichimoku: price vs cloud
    if not np.isnan(mlast.get("cloud_top", np.nan)):
        close = ci["close"].iloc[-1] if "close" in ci.columns else e20
        cloud_top = mlast["cloud_top"]
        cloud_bot = mlast["cloud_bot"]
        if close > cloud_top:
            votes.append(Vote("ichimoku_cloud", "trend", +0.8, "price above cloud"))
        elif close < cloud_bot:
            votes.append(Vote("ichimoku_cloud", "trend", -0.8, "price below cloud"))
        else:
            votes.append(Vote("ichimoku_cloud", "trend", 0.0, "price inside cloud"))

        # TK cross
        if not np.isnan(mlast.get("tenkan", np.nan)):
            if mlast["tenkan"] > mlast["kijun"]:
                votes.append(Vote("tk_cross", "trend", +0.5, "tenkan > kijun"))
            else:
                votes.append(Vote("tk_cross", "trend", -0.5, "tenkan < kijun"))

    # ADX trend strength — not directional, just how strong
    if not np.isnan(c["adx"]):
        if c["adx"] >= 25 and c["plus_di"] > c["minus_di"]:
            votes.append(Vote("adx_trend", "trend", +0.7, f"ADX={c['adx']:.0f} +DI leading"))
        elif c["adx"] >= 25 and c["minus_di"] > c["plus_di"]:
            votes.append(Vote("adx_trend", "trend", -0.7, f"ADX={c['adx']:.0f} -DI leading"))
        else:
            votes.append(Vote("adx_trend", "trend", 0.0, f"ADX={c['adx']:.0f} weak trend"))

    return votes


def _structure_votes(smc_data: dict) -> list[Vote]:
    """Structure category: HH-HL labels, BOS, CHoCH, premium/discount."""
    votes: list[Vote] = []

    # Market structure bias
    bias = smc_data.get("structure_bias", "neutral")
    if bias == "bullish":
        votes.append(Vote("structure_bias", "structure", +1.0, "HH-HL bullish structure"))
    elif bias == "bearish":
        votes.append(Vote("structure_bias", "structure", -1.0, "LH-LL bearish structure"))
    else:
        votes.append(Vote("structure_bias", "structure", 0.0, "neutral structure"))

    # BOS / CHoCH — most recent
    bos_list = smc_data.get("bos", [])
    if bos_list:
        last_bos = bos_list[-1]
        val = 0.9 * last_bos.direction
        votes.append(Vote("bos", "structure", val,
                          f"{last_bos.kind.upper()} {'bull' if last_bos.direction > 0 else 'bear'}"))

    # Premium / Discount
    pd_zone = smc_data.get("premium_discount")
    if pd_zone:
        if pd_zone.zone == "premium":
            votes.append(Vote("premium_discount", "structure", -0.5,
                              f"premium zone ({pd_zone.pct:.0f}%)"))
        elif pd_zone.zone == "discount":
            votes.append(Vote("premium_discount", "structure", +0.5,
                              f"discount zone ({pd_zone.pct:.0f}%)"))
        else:
            votes.append(Vote("premium_discount", "structure", 0.0, "equilibrium"))

    # Equal levels (liquidity pools)
    eq = smc_data.get("equal_levels", [])
    if eq:
        # Equal lows below = bullish target, equal highs above = bearish target
        for el in eq[-2:]:
            if el.kind == "equal_lows":
                votes.append(Vote("equal_levels", "structure", +0.3,
                                  f"equal lows @ {el.price:,.0f} (liq pool)"))
            else:
                votes.append(Vote("equal_levels", "structure", -0.3,
                                  f"equal highs @ {el.price:,.0f} (liq pool)"))

    return votes


def _momentum_votes(last: pd.Series, mi_last: pd.Series,
                    divs: list) -> list[Vote]:
    """Momentum category: RSI, MACD, Stochastic, divergences."""
    votes: list[Vote] = []

    # RSI
    rsi_val = last["rsi"]
    if not np.isnan(rsi_val):
        if rsi_val > 70:
            votes.append(Vote("rsi", "momentum", -0.6, f"RSI={rsi_val:.0f} overbought"))
        elif rsi_val < 30:
            votes.append(Vote("rsi", "momentum", +0.6, f"RSI={rsi_val:.0f} oversold"))
        elif rsi_val > 50:
            votes.append(Vote("rsi", "momentum", +0.3, f"RSI={rsi_val:.0f} above 50"))
        else:
            votes.append(Vote("rsi", "momentum", -0.3, f"RSI={rsi_val:.0f} below 50"))

    # MACD histogram
    hist = last["macd_hist"]
    if not np.isnan(hist):
        if hist > 0:
            votes.append(Vote("macd", "momentum", +0.7, "MACD histogram positive"))
        else:
            votes.append(Vote("macd", "momentum", -0.7, "MACD histogram negative"))

    # Stochastic
    k, d = last["stoch_k"], last["stoch_d"]
    if not np.isnan(k):
        if k > 80 and d > 80:
            votes.append(Vote("stoch", "momentum", -0.5, f"Stoch K={k:.0f} overbought"))
        elif k < 20 and d < 20:
            votes.append(Vote("stoch", "momentum", +0.5, f"Stoch K={k:.0f} oversold"))
        elif k > d:
            votes.append(Vote("stoch", "momentum", +0.3, "Stoch K > D"))
        else:
            votes.append(Vote("stoch", "momentum", -0.3, "Stoch K < D"))

    # Divergences
    for div in divs:
        val = 0.8 * div.bias
        votes.append(Vote("divergence", "momentum", val,
                          f"{div.kind} div on {div.osc} ({div.bars_ago} bars ago)"))

    # MFI
    mfi_val = mi_last.get("mfi", np.nan)
    if not np.isnan(mfi_val):
        if mfi_val > 80:
            votes.append(Vote("mfi", "momentum", -0.4, f"MFI={mfi_val:.0f} overbought"))
        elif mfi_val < 20:
            votes.append(Vote("mfi", "momentum", +0.4, f"MFI={mfi_val:.0f} oversold"))

    return votes


def _zone_votes(pa_data: dict, smc_data: dict, close: float,
                atr_val: float) -> list[Vote]:
    """Zones category: OB, FVG, S/R proximity, fib OTE, liquidity sweep."""
    votes: list[Vote] = []
    if atr_val <= 0:
        return votes

    # Unmitigated order blocks near current price
    for ob in smc_data.get("order_blocks_unmitigated", [])[:3]:
        dist = abs(close - (ob.hi + ob.lo) / 2) / atr_val
        if dist < 2.0:  # within 2 ATR
            votes.append(Vote("order_block", "zones", 0.8 * ob.direction,
                              f"{'bull' if ob.direction > 0 else 'bear'} OB nearby ({dist:.1f} ATR)"))

    # Open FVGs near current price
    for fvg in smc_data.get("fvg_open", [])[:3]:
        mid = (fvg.hi + fvg.lo) / 2
        dist = abs(close - mid) / atr_val
        if dist < 3.0:
            votes.append(Vote("fvg", "zones", 0.6 * fvg.direction,
                              f"{'bull' if fvg.direction > 0 else 'bear'} FVG ({fvg.filled_pct:.0%} filled)"))

    # S/R proximity — nearest support is bullish (bounce), nearest resistance bearish
    for sr in pa_data.get("sr_zones", [])[:4]:
        dist = abs(close - sr.mid) / atr_val
        if dist < 1.5:
            val = +0.5 if sr.kind == "support" else -0.5
            votes.append(Vote("sr_zone", "zones", val,
                              f"{sr.kind} @ {sr.mid:,.0f} ({dist:.1f} ATR away)"))

    # Fibonacci OTE (0.618-0.705 zone)
    for fib in pa_data.get("fibonacci", []):
        if fib.kind == "retracement" and 0.600 <= fib.ratio <= 0.720:
            dist = abs(close - fib.price) / atr_val
            if dist < 1.0:
                votes.append(Vote("fib_ote", "zones", +0.7,
                                  f"price in fib OTE ({fib.ratio:.3f} @ {fib.price:,.0f})"))
                break

    # Liquidity sweeps — recent sweep is a reversal hint
    for sweep in smc_data.get("liquidity_sweeps", [])[-2:]:
        votes.append(Vote("liq_sweep", "zones", 0.6 * sweep.direction,
                          f"{'bull' if sweep.direction > 0 else 'bear'} liq sweep"))

    return votes


def _volume_votes(last: pd.Series, vp, mi_last: pd.Series,
                  close: float) -> list[Vote]:
    """Volume category: OBV slope, MFI, volume vs average, POC relation."""
    votes: list[Vote] = []

    # OBV vs its EMA
    obv, obv_ema = last.get("obv", np.nan), last.get("obv_ema", np.nan)
    if not np.isnan(obv) and not np.isnan(obv_ema):
        if obv > obv_ema:
            votes.append(Vote("obv_slope", "volume", +0.7, "OBV above EMA (accumulation)"))
        else:
            votes.append(Vote("obv_slope", "volume", -0.7, "OBV below EMA (distribution)"))

    # Volume ratio
    vol_ratio = last.get("vol_ratio", np.nan)
    if not np.isnan(vol_ratio):
        if vol_ratio > 1.5:
            votes.append(Vote("vol_spike", "volume", +0.5, f"volume {vol_ratio:.1f}x avg"))
        elif vol_ratio < 0.5:
            votes.append(Vote("vol_spike", "volume", -0.3, "low volume"))

    # Volume Profile POC relation
    if vp is not None:
        dist_to_poc = (close - vp.poc) / abs(vp.poc) * 100 if vp.poc > 0 else 0
        if dist_to_poc > 2:
            votes.append(Vote("vp_poc", "volume", +0.4, f"price above POC ({vp.poc:,.0f})"))
        elif dist_to_poc < -2:
            votes.append(Vote("vp_poc", "volume", -0.4, f"price below POC ({vp.poc:,.0f})"))
        else:
            votes.append(Vote("vp_poc", "volume", 0.0, "price at POC (fair value)"))

    return votes


# =========================================================================== #
# Core scoring engine
# =========================================================================== #
def _compute_tf(df: pd.DataFrame, tf: str) -> TFResult:
    """Compute all indicators and votes for one timeframe."""
    result = TFResult(timeframe=tf, raw_df=df)

    # Run indicators
    ci = classic_frame(df)
    mi = modern_frame(df)
    result.classic = ci
    result.modern = mi

    last = ci.iloc[-1]
    mlast = mi.iloc[-1]

    # Run PA + SMC
    pa = price_action_scan(df)
    smc = smc_scan(df)
    result.pa = pa
    result.smc = smc

    # Compute regime
    regime = volatility_regime(df)
    result.regime = regime

    # Volume profile
    vp = volume_profile(df)

    # Divergences
    divs = divergences(df)

    close = float(df["close"].iloc[-1])
    atr_val = float(last["atr"]) if not np.isnan(last["atr"]) else 0

    # Collect all votes
    result.votes.extend(_trend_votes(ci, mi, last, mlast))
    result.votes.extend(_structure_votes(smc))
    result.votes.extend(_momentum_votes(last, mlast, divs))
    result.votes.extend(_zone_votes(pa, smc, close, atr_val))
    result.votes.extend(_volume_votes(last, vp, mlast, close))

    return result


def score_symbol(frames: dict[str, pd.DataFrame], symbol_name: str,
                 cfg: Config) -> SignalResult:
    """Score one symbol across all available timeframes.

    This is the main entry point for the confluence engine.
    """
    weights = cfg.get("weights", default=DEFAULT_WEIGHTS) or DEFAULT_WEIGHTS
    tf_mult = cfg.get("tf_multiplier", default={}) or {}
    thresholds = cfg.get("thresholds", default={}) or {}

    result = SignalResult(symbol=symbol_name, direction=0, score=0.0,
                          label="NEUTRAL", confidence=50.0)

    # Process each timeframe
    raw_score = 0.0
    max_possible = 0.0

    for tf, df in frames.items():
        if df is None or df.empty or len(df) < 30:
            continue

        tf_result = _compute_tf(df, tf)
        result.tf_results[tf] = tf_result

        mult = float(tf_mult.get(tf, 1.0))

        # Aggregate votes per category
        cat_scores: dict[str, float] = {}
        cat_counts: dict[str, int] = {}
        for vote in tf_result.votes:
            cat_scores[vote.category] = cat_scores.get(vote.category, 0.0) + vote.value
            cat_counts[vote.category] = cat_counts.get(vote.category, 0) + 1

        # Normalise each category to -1..+1, then apply weight
        for cat, w in weights.items():
            n = cat_counts.get(cat, 0)
            if n > 0:
                avg = cat_scores.get(cat, 0.0) / n  # average vote in -1..+1
                raw_score += mult * w * avg
            max_possible += mult * w  # theoretical max if all votes = +1

    if max_possible <= 0:
        return result

    # Normalise to -100..+100
    score = 100.0 * raw_score / max_possible
    score = max(-100.0, min(100.0, score))

    result.raw_score = raw_score
    result.max_possible = max_possible
    result.score = round(score, 1)
    result.direction = +1 if score > 0 else (-1 if score < 0 else 0)

    # Map to label
    strong = int(thresholds.get("strong", 65))
    signal = int(thresholds.get("signal", 40))
    watch = int(thresholds.get("watch", 18))

    # Symbol-specific threshold overrides (backtest-tuned)
    sym_overrides = cfg.get("symbol_overrides", default={}) or {}
    if symbol_name in sym_overrides:
        ov = sym_overrides[symbol_name]
        if isinstance(ov, dict):
            watch = int(ov.get("watch", watch))
            signal = int(ov.get("signal", signal))
            strong = int(ov.get("strong", strong))

    abs_score = abs(score)
    if abs_score >= strong:
        base_label = "STRONG BUY" if score > 0 else "STRONG SELL"
    elif abs_score >= signal:
        base_label = "BUY" if score > 0 else "SELL"
    elif abs_score >= watch:
        base_label = "WATCH LONG" if score > 0 else "WATCH SHORT"
    else:
        base_label = "NEUTRAL"
    result.label = base_label

    # Apply hard gates
    result.gates = _apply_gates(result, cfg)

    # Compute confidence
    confidence = 50.0 + abs(score) / 2.0
    gate_penalties = sum(1 for g, v in result.gates.items()
                         if v.get("action") == "downgrade")
    confidence -= gate_penalties * 10
    result.confidence = max(55.0, min(95.0, confidence))

    return result


# =========================================================================== #
# Hard gates
# =========================================================================== #
def _apply_gates(result: SignalResult, cfg: Config) -> dict[str, Any]:
    """Apply hard gates that can downgrade or drop a signal."""
    gates_cfg = cfg.get("gates", default={}) or {}
    gates: dict[str, Any] = {}

    # 1. HTF conflict gate
    htf_tf = cfg.htf
    htf_result = result.tf_results.get(htf_tf)
    if htf_result and htf_result.smc:
        htf_bias_str = htf_result.smc.get("structure_bias", "neutral")
        htf_bias = +1 if htf_bias_str == "bullish" else (-1 if htf_bias_str == "bearish" else 0)
        if htf_bias != 0 and htf_bias != result.direction:
            gates["htf_conflict"] = {
                "action": "downgrade", "detail": f"HTF bias={htf_bias_str} opposes signal",
                "max_label": "WATCH"
            }
            if result.label in ("BUY", "STRONG BUY", "SELL", "STRONG SELL"):
                result.label = "WATCH LONG" if result.direction > 0 else "WATCH SHORT"

    # 2. ADX gate — need ADX >= 20 on MTF for trend signals
    min_adx = float(gates_cfg.get("min_adx", 20))
    for mtf in cfg.mtf:
        mtf_result = result.tf_results.get(mtf)
        if mtf_result and mtf_result.classic is not None:
            adx_val = float(mtf_result.classic["adx"].iloc[-1])
            if not np.isnan(adx_val) and adx_val < min_adx:
                gates["adx_weak"] = {
                    "action": "note", "detail": f"ADX={adx_val:.0f} < {min_adx} on {mtf} (range mode)"
                }
                break

    # 3. Volatility gate
    atr_min = float(gates_cfg.get("atr_pct_min", 15))
    atr_max = float(gates_cfg.get("atr_pct_max", 95))
    for tf, tfr in result.tf_results.items():
        if tfr.regime:
            pct = tfr.regime.get("percentile", 50)
            if isinstance(pct, (int, float)) and not np.isnan(pct):
                pct100 = pct * 100 if pct <= 1.0 else pct
                if pct100 < atr_min:
                    gates["vol_dead"] = {
                        "action": "downgrade",
                        "detail": f"ATR pctl={pct100:.0f}% < {atr_min}% (dead tape)"
                    }
                    if result.label in ("BUY", "STRONG BUY", "SELL", "STRONG SELL"):
                        result.label = "WATCH LONG" if result.direction > 0 else "WATCH SHORT"
                    break
                if pct100 > atr_max:
                    gates["vol_extreme"] = {
                        "action": "downgrade",
                        "detail": f"ATR pctl={pct100:.0f}% > {atr_max}% (extreme vol)"
                    }
                    if result.label in ("BUY", "STRONG BUY", "SELL", "STRONG SELL"):
                        result.label = "WATCH LONG" if result.direction > 0 else "WATCH SHORT"
                    break

    # Macro (1w) conflict gate — if weekly opposes, cap at WATCH
    macro_tf = cfg.macro
    if macro_tf:
        macro_result = result.tf_results.get(macro_tf)
        if macro_result and macro_result.smc:
            macro_bias_str = macro_result.smc.get("structure_bias", "neutral")
            macro_bias = +1 if macro_bias_str == "bullish" else (-1 if macro_bias_str == "bearish" else 0)
            if macro_bias != 0 and macro_bias != result.direction:
                gates["macro_conflict"] = {
                    "action": "downgrade",
                    "detail": f"Macro(1w) bias={macro_bias_str} opposes signal",
                    "max_label": "WATCH"
                }
                if result.label in ("BUY", "STRONG BUY", "SELL", "STRONG SELL"):
                    result.label = "WATCH LONG" if result.direction > 0 else "WATCH SHORT"

    result.gate_passed = not any(
        g.get("action") == "drop" for g in gates.values()
    )

    return gates
