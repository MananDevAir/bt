"""Telegram message formatter — clean, scannable signal alerts.

Design goals:
  - Glanceable: know the direction + symbol in 1 second
  - Clean sections with visual separation
  - IST timestamp so user knows when to check
  - Only show what matters for decision-making
  - Minimal clutter, no walls of text
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

# IST offset: UTC + 5:30
IST = timezone(timedelta(hours=5, minutes=30))


def _fmt_price(val: float, symbol: str = "") -> str:
    """Format a price with sensible precision."""
    sym = symbol.upper()
    if sym in ("EURUSD", "GBPUSD"):
        return f"{val:.5f}"
    if sym in ("USDJPY",):
        return f"{val:.3f}"
    if val >= 1000:
        return f"{val:,.2f}"
    if val >= 1:
        return f"{val:.2f}"
    return f"{val:.4f}"


def _ist_now() -> str:
    """Current time in IST, formatted for display."""
    return datetime.now(IST).strftime("%d %b %Y, %I:%M %p IST")


def format_signal(signal: Any, plan: Any | None,
                  narration: str, narration_source: str) -> str:
    """Build a clean Telegram HTML message for a signal alert."""
    sym = signal.symbol
    fp = lambda v: _fmt_price(v, sym)  # noqa: E731

    # Direction header
    if signal.direction > 0:
        arrow = "\U0001f7e2"   # 🟢
        dir_word = "LONG"
    elif signal.direction < 0:
        arrow = "\U0001f534"   # 🔴
        dir_word = "SHORT"
    else:
        arrow = "\u26aa"       # ⚪
        dir_word = "NEUTRAL"

    lines: list[str] = []

    # Trade type emoji mapping
    type_emoji = {
        "Intraday": "\u23f1",      # ⏱
        "Swing": "\U0001f504",     # 🔄
        "Short-term": "\U0001f4c5", # 📅
        "Positional": "\U0001f4c8", # 📈
    }

    # ── Header ──────────────────────────────────────
    lines.append(f"{arrow} <b>{sym}  \u2022  {signal.label}</b>")
    lines.append(f"\U0001f4ca Score: {signal.score:+.1f}  |  Confidence: {signal.confidence:.0f}%")

    # ── Trade Type (immediately visible) ────────────
    if plan and plan.trade_type:
        te = type_emoji.get(plan.trade_type, "\U0001f4cb")
        type_desc = {
            "Intraday": "close within hours",
            "Swing": "hold hours to 1-2 days",
            "Short-term": "hold 1-5 days",
            "Positional": "hold days to weeks",
        }
        desc = type_desc.get(plan.trade_type, "")
        lines.append(f"{te} <b>Type: {plan.trade_type}</b>  \u2014  {desc}")
    lines.append("")

    # ── Session / Killzone ──────────────────────────
    if plan and (getattr(plan, "session", "") or getattr(plan, "killzone", "")):
        kz = getattr(plan, "killzone", "")
        sess = getattr(plan, "session", "")
        if kz:
            lines.append(f"\u26a1 <b>Session:</b> {sess} ({kz} Killzone)")
        elif sess:
            lines.append(f"\U0001f310 <b>Session:</b> {sess}")
        lines.append("")

    # ── Brief reason (WHY this signal) ──────────────
    if plan and plan.brief_reason:
        lines.append(f"\U0001f50d {plan.brief_reason}")
        lines.append("")

    # ── Timeframes (one clean line) ─────────────────
    tf_parts: list[str] = []
    for tf, tfr in signal.tf_results.items():
        votes = tfr.votes
        if not votes:
            continue
        avg = sum(v.value for v in votes) / len(votes)
        if avg > 0.2:
            tf_parts.append(f"{tf}\u2191")
        elif avg < -0.2:
            tf_parts.append(f"{tf}\u2193")
        else:
            tf_parts.append(f"{tf}\u2194")
    if tf_parts:
        lines.append(f"\U0001f4c8 {' \u2022 '.join(tf_parts)}")
        lines.append("")

    # ── Trade Plan ──────────────────────────────────
    if plan:
        lines.append("\u2500" * 25)
        lines.append(f"\U0001f3af <b>TRADE PLAN</b>  ({dir_word})")
        lines.append("")
        lines.append(f"  Entry     {fp(plan.entry_low)} \u2013 {fp(plan.entry_high)}")
        lines.append(f"  Stop       {fp(plan.sl)}")
        lines.append(f"  TP1        {fp(plan.tp1)}  ({plan.tp_allocation[0]}%)")
        lines.append(f"  TP2        {fp(plan.tp2)}  ({plan.tp_allocation[1]}%)")
        lines.append(f"  TP3        {fp(plan.tp3)}  ({plan.tp_allocation[2]}%)")
        lines.append("")
        lines.append(f"  R:R  <b>{plan.rr:.1f}</b>  |  Risk  {plan.risk_pct:.1f}%")
        lines.append("\u2500" * 25)
        lines.append("")

        if plan.invalidation:
            lines.append(f"\u26a0\ufe0f Cancel if: {plan.invalidation}")
            lines.append("")

    # ── AI Insight (narration — short) ──────────────
    if narration:
        short = narration[:200].rstrip()
        if len(narration) > 200:
            short += "..."
        lines.append(f"\U0001f4ac {short}")
        lines.append("")

    # ── Footer ──────────────────────────────────────
    lines.append(f"\U0001f552 {_ist_now()}")
    lines.append("<i>Analysis only \u2014 not financial advice.</i>")

    return "\n".join(lines)


def format_status(symbols_status: list[dict]) -> str:
    """Format a /status overview message — clean table."""
    lines = [
        f"\U0001f4ca <b>Market Overview</b>",
        f"\U0001f552 {_ist_now()}",
        "",
    ]

    for s in symbols_status:
        score = s.get("score", 0)
        label = s.get("label", "?")
        if score > 18:
            dot = "\U0001f7e2"     # 🟢
        elif score < -18:
            dot = "\U0001f534"     # 🔴
        else:
            dot = "\u26aa"         # ⚪
        lines.append(f"{dot} <b>{s['symbol']:8s}</b>  {score:+6.1f}  {label}")

    lines.append("")
    lines.append(f"<i>{len(symbols_status)} symbols</i>")
    return "\n".join(lines)


def format_watchlist(symbols: list[str]) -> str:
    """Format a /watchlist response."""
    lines = [f"\U0001f4cb <b>Watchlist</b>  ({len(symbols)} symbols)", ""]
    for sym in symbols:
        lines.append(f"  \u2022 {sym}")
    return "\n".join(lines)
