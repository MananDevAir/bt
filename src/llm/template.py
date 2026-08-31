"""Deterministic narration template — always works, no network call.

When the HF Inference API is down, out of credits, or returns a bad reply,
this module builds a readable explanation from the same fact sheet the LLM
would receive.  It is intentionally plain — the value is reliability, not
creativity.
"""
from __future__ import annotations

from typing import Any


def narrate(facts: dict[str, Any]) -> str:
    """Build a 2-4 sentence explanation from a signal fact sheet.

    ``facts`` keys (all optional, graceful fallbacks):
      symbol, direction, label, score, confidence,
      entry_low, entry_high, sl, tp1, tp2, tp3, rr,
      htf_bias, mtf_bias, ltf_state,
      triggers (list[str]), gates (dict), source (entry method)
    """
    sym = facts.get("symbol", "?")
    label = facts.get("label", "NEUTRAL")
    score = facts.get("score", 0)
    conf = facts.get("confidence", 55)
    direction = facts.get("direction", 0)
    rr = facts.get("rr")

    dir_word = "long" if direction > 0 else "short" if direction < 0 else "neutral"

    # Sentence 1: headline
    parts = [f"{sym} scores {score:+.1f} ({label}, {conf:.0f}% confidence) "
             f"with a {dir_word} bias."]

    # Sentence 2: timeframe alignment
    htf = facts.get("htf_bias", "")
    mtf = facts.get("mtf_bias", "")
    ltf = facts.get("ltf_state", "")
    tf_parts = []
    if htf:
        tf_parts.append(f"daily {htf}")
    if mtf:
        tf_parts.append(f"4h/1h {mtf}")
    if ltf:
        tf_parts.append(f"15m {ltf}")
    if tf_parts:
        parts.append(f"Timeframes align: {', '.join(tf_parts)}.")

    # Sentence 3: key triggers
    triggers = facts.get("triggers", [])
    if triggers:
        top = triggers[:4]  # max 4 triggers in the narration
        parts.append(f"Key triggers: {', '.join(top)}.")

    # Sentence 4: risk-reward
    if rr is not None:
        source = facts.get("source", "market")
        parts.append(f"Risk-reward {rr:.1f}:1 via {source} entry.")

    # Gate warnings
    gates = facts.get("gates", {})
    if gates:
        gate_names = list(gates.keys())[:3]
        parts.append(f"Note: {', '.join(gate_names)} gate(s) flagged.")

    return " ".join(parts)


def build_fact_sheet(signal: Any, plan: Any | None = None) -> dict[str, Any]:
    """Extract a flat dict from SignalResult + optional TradePlan.

    This is shared by both the LLM prompt builder and the template.
    """
    facts: dict[str, Any] = {
        "symbol": signal.symbol,
        "direction": signal.direction,
        "score": round(signal.score, 1),
        "label": signal.label,
        "confidence": round(signal.confidence, 0),
    }

    # Per-timeframe summaries
    for tf, tfr in signal.tf_results.items():
        votes = tfr.votes
        if not votes:
            continue
        avg = sum(v.value for v in votes) / len(votes)
        if avg > 0.2:
            bias = "bullish"
        elif avg < -0.2:
            bias = "bearish"
        else:
            bias = "mixed"

        if tf in ("1d",):
            facts["htf_bias"] = bias
        elif tf in ("4h", "1h"):
            facts.setdefault("mtf_bias", bias)
        elif tf in ("15m",):
            facts["ltf_state"] = bias
        elif tf in ("1w",):
            facts["macro_bias"] = bias

    # Key triggers from votes with |value| >= 0.6
    triggers: list[str] = []
    for tf, tfr in signal.tf_results.items():
        for v in tfr.votes:
            if abs(v.value) >= 0.6 and v.detail:
                triggers.append(v.detail)
    facts["triggers"] = triggers[:8]  # cap for prompt size

    # Gates
    if signal.gates:
        facts["gates"] = dict(signal.gates)

    # Trade plan levels
    if plan is not None:
        facts.update({
            "entry_low": round(plan.entry_low, 2),
            "entry_high": round(plan.entry_high, 2),
            "sl": round(plan.sl, 2),
            "tp1": round(plan.tp1, 2),
            "tp2": round(plan.tp2, 2),
            "tp3": round(plan.tp3, 2),
            "rr": round(plan.rr, 2),
            "risk_pct": round(plan.risk_pct, 2),
            "risk_atr": round(plan.risk_atr, 1),
            "source": plan.source,
            "invalidation": plan.invalidation,
            "holding_horizon": plan.holding_horizon,
            "trade_type": getattr(plan, "trade_type", ""),
            "brief_reason": getattr(plan, "brief_reason", ""),
        })

    return facts
