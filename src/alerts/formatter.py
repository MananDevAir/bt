"""Alert message formatter — Discord and Telegram compatible.

Design goals:
  - Glanceable: know the direction + symbol in 1 second
  - Clean sections with visual separation
  - IST timestamp so user knows when to check
  - Only show what matters for decision-making
  - Minimal clutter, no walls of text
"""
from __future__ import annotations

import html
import re
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


def _format_narration_html(text: str) -> str:
    """Safely format LLM narration into clean Telegram HTML without markdown artifacts."""
    # 1. Clean markdown headers (#, ##, ###)
    cleaned = re.sub(r"(?m)^\s*#+\s*", "", text.strip())

    # 2. Escape HTML special chars (<, >, &)
    escaped = html.escape(cleaned)

    # 3. Convert **bold** to <b>bold</b> and *italic* to <i>italic</i>
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"\*(.+?)\*", r"<i>\1</i>", escaped)

    # 4. Normalize bullets (convert -, +, • at line starts to •)
    escaped = re.sub(r"(?m)^[\s\-\+]+\s+", "• ", escaped)

    # 5. Clean up any leftover stray # characters
    escaped = escaped.replace("#", "")

    # 6. Emphasize standard bullet headers (• Entry:, • Stop-Loss:, • Targets:)
    escaped = re.sub(r"•\s*(Entry|Stop-Loss|Stop|Targets|Target):\s*", r"• <b>\1:</b> ", escaped, flags=re.IGNORECASE)

    # 7. If bullet points exist, filter out non-bullet clutter/preamble headers
    lines = [line.strip() for line in escaped.splitlines() if line.strip()]
    bullet_lines = [line for line in lines if line.startswith("•")]
    if len(bullet_lines) >= 2:
        return "\n".join(bullet_lines)

    return "\n".join(lines)


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
        "Intraday": "\u23f1",       # ⏱
        "Swing": "\U0001f504",      # 🔄
        "Positional": "\U0001f4c8", # 📈
    }

    # Grade badge
    grade = getattr(plan, "grade", "") if plan else ""
    if grade == "A+":
        badge = "  \u2022  \U0001f3c6 <b>GRADE A+</b>"
    elif grade == "A":
        badge = "  \u2022  \U0001f3af <b>GRADE A</b>"
    elif grade == "B":
        badge = "  \u2022  \U0001f441\ufe0f <b>GRADE B</b>"
    else:
        badge = ""

    # ── Header ──────────────────────────────────────
    lines.append(f"{arrow} <b>{sym}  \u2022  {signal.label}</b>{badge}")
    lines.append(f"\U0001f4ca Score: {signal.score:+.1f}  |  Confidence: {signal.confidence:.0f}%")

    # ── Trade Type (immediately visible) ────────────
    if plan and plan.trade_type:
        te = type_emoji.get(plan.trade_type, "\U0001f4cb")
        type_desc = {
            "Intraday": "Hold: 2 to 8 hours (Day Trade)",
            "Swing": "Hold: 1 to 3 days (Swing Trade)",
            "Positional": "Hold: Days to Weeks (Position Trade)",
            "Short-term": "Hold: 1 to 5 days",
        }
        desc = type_desc.get(plan.trade_type, getattr(plan, "holding_horizon", ""))
        lines.append(f"{te} <b>Type: {plan.trade_type.upper()}</b>  \u2014  <i>{desc}</i>")
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
        trade_lane_tag = f" \u2022 {plan.trade_type.upper()}" if plan.trade_type else ""
        lines.append(f"\U0001f3af <b>TRADE PLAN</b>  ({dir_word}{trade_lane_tag})")
        lines.append("")
        lines.append(f"  Entry     {fp(plan.entry_low)} \u2013 {fp(plan.entry_high)}")
        lines.append(f"  Stop       {fp(plan.sl)}")
        lines.append(f"  TP1        {fp(plan.tp1)}  ({plan.tp_allocation[0]}%)")
        lines.append(f"  TP2        {fp(plan.tp2)}  ({plan.tp_allocation[1]}%)")
        lines.append(f"  TP3        {fp(plan.tp3)}  ({plan.tp_allocation[2]}%)")
        lines.append("")
        risk_atr_str = f" ({plan.risk_atr:.1f} ATR)" if hasattr(plan, "risk_atr") and plan.risk_atr > 0 else ""
        risk_dist = abs(plan.entry_mid - plan.sl)
        
        risk_str = f"{plan.risk_pct:.1f}%"
        if sym in ("EURUSD", "GBPUSD"):
            pips = risk_dist / 0.0001
            risk_str = f"{pips:.1f} pips"
        elif sym in ("USDJPY",):
            pips = risk_dist / 0.01
            risk_str = f"{pips:.1f} pips"
            
        lines.append(f"  R:R  <b>{plan.rr:.1f}</b>  |  Risk  {risk_str}{risk_atr_str}")

        # Position sizing guide (1% risk on $10k reference equity)
        if risk_dist > 0:
            risk_usd = 100.0
            if sym in ("EURUSD", "GBPUSD"):
                pips = risk_dist / 0.0001
                lots = risk_usd / (pips * 10.0) if pips > 0 else 0
                lines.append(f"  \U0001f4bc Size (1% on $10k): <b>{lots:.2f} lots</b> ({pips:.1f} pips)")
            elif sym in ("USDJPY",):
                pips = risk_dist / 0.01
                lots = risk_usd / (pips * 9.0) if pips > 0 else 0
                lines.append(f"  \U0001f4bc Size (1% on $10k): <b>{lots:.2f} lots</b> ({pips:.1f} pips)")
            elif sym in ("BTC", "ETH", "SOL", "ZEC"):
                units = risk_usd / risk_dist
                lines.append(f"  \U0001f4bc Size (1% on $10k): <b>{units:.3f} {sym}</b>")
            elif sym in ("XAUUSDT",):
                oz = risk_usd / risk_dist
                lines.append(f"  \U0001f4bc Size (1% on $10k): <b>{oz:.2f} oz</b>")
            elif sym in ("US100", "US500", "US30"):
                contracts = risk_usd / risk_dist
                lines.append(f"  \U0001f4bc Size (1% on $10k): <b>{contracts:.2f} contracts</b> ({risk_dist:.1f} pts)")

        lines.append("\u2500" * 25)
        lines.append("")

        if plan.invalidation:
            lines.append(f"\u26a0\ufe0f Cancel if: {plan.invalidation}")
            lines.append("")

    # ── Price Action Logic & Analysis ──────────────
    if narration:
        lines.append("🧠 <b>Price Action Logic:</b>")
        lines.append(_format_narration_html(narration))
        lines.append("")

    # ── Footer ──────────────────────────────────────
    source_badge = ""
    if narration_source:
        if narration_source.startswith("hf:") or narration_source.startswith("groq:"):
            source_badge = " \u2022 <i>AI</i>"
        elif narration_source == "template":
            source_badge = " \u2022 <i>Rules</i>"
    lines.append(f"\U0001f552 {_ist_now()}{source_badge}")
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


def format_tp_update(sig: dict[str, Any], tp_level: str,
                     r_mult: str, exit_price: float,
                     sl_action: str, profit_banked: str) -> str:
    """Format a rich Take Profit update card for Telegram."""
    sym = sig.get("symbol", "")
    sig_id = sig.get("id", "?")
    direction = sig.get("direction", "long").upper()
    fp = lambda v: _fmt_price(v, sym)

    progress_map = {
        "tp1": "[🎯 TP1 (50%)] ➔ [⏳ TP2] ➔ [⏳ TP3]",
        "tp2": "[✅ TP1] ➔ [🎯 TP2 (30%)] ➔ [⏳ TP3]",
        "tp3": "[✅ TP1] ➔ [✅ TP2] ➔ [🎉 TP3 (20%) \u2014 FULL WIN]",
    }
    progress = progress_map.get(tp_level, "")

    lines = [
        f"\U0001f3af <b>TAKE PROFIT {tp_level.upper()} HIT (+{r_mult}R)</b>",
        "\u2500" * 25,
        f"\U0001f4cc <b>Trade:</b> #{sig_id} \u2022 <b>{sym} {direction}</b>",
        f"\U0001f3af <b>Target Level:</b> {fp(exit_price)}",
    ]
    if progress:
        lines.append(f"\U0001f4ca <b>Status:</b> {progress}")
    if sl_action:
        lines.append(f"\U0001f6e1\ufe0f <b>Action:</b> {sl_action}")
    if profit_banked:
        lines.append(f"\U0001f4b0 <b>Banked:</b> {profit_banked}")

    lines.append("\u2500" * 25)
    lines.append(f"\U0001f552 {_ist_now()}")
    return "\n".join(lines)


def format_sl_update(sig: dict[str, Any], exit_price: float,
                     sl_type: str, pnl_r: str, mfe_r: float = 0.0) -> str:
    """Format a rich Stop Loss / Breakeven update card for Telegram."""
    sym = sig.get("symbol", "")
    sig_id = sig.get("id", "?")
    direction = sig.get("direction", "long").upper()
    fp = lambda v: _fmt_price(v, sym)

    if sl_type == "breakeven":
        header = "\U0001f6e1\ufe0f <b>BREAKEVEN STOP HIT (Risk-Free Exit)</b>"
        res_str = f"Partial Win (TP1 secured at +1.0R, remaining at BE)"
        pnl_badge = f"\U0001f4b0 <b>Net Gain:</b> +{pnl_r}R"
    elif sl_type == "trailed_tp1":
        header = "\U0001f7e2 <b>TRAILED STOP HIT (TP1 Locked Level)</b>"
        res_str = f"Solid Win (80% profit banked at TP1 & TP2)"
        pnl_badge = f"\U0001f4b0 <b>Net Gain:</b> +{pnl_r}R"
    else:
        header = "\U0001f534 <b>STOP LOSS HIT</b>"
        res_str = f"Closed as Loss"
        pnl_badge = f"\U0001f4c9 <b>Loss:</b> -1.0R (Max Adverse: {mfe_r:.1f}R)"

    lines = [
        header,
        "\u2500" * 25,
        f"\U0001f4cc <b>Trade:</b> #{sig_id} \u2022 <b>{sym} {direction}</b>",
        f"\U0001f6e1\ufe0f <b>Exit Level:</b> {fp(exit_price)}",
        f"\U0001f4ca <b>Result:</b> {res_str}",
        pnl_badge,
        "\u2500" * 25,
        f"\U0001f552 {_ist_now()}",
    ]
    return "\n".join(lines)


def format_expiry_update(sig: dict[str, Any], exit_price: float,
                         current_r: float, max_age_days: float) -> str:
    """Format a signal expiration update card for Telegram."""
    sym = sig.get("symbol", "")
    sig_id = sig.get("id", "?")
    direction = sig.get("direction", "long").upper()
    fp = lambda v: _fmt_price(v, sym)

    lines = [
        f"\u231b <b>SIGNAL #{sig_id} EXPIRED ({max_age_days:.0f}d Limit)</b>",
        "\u2500" * 25,
        f"\U0001f4cc <b>Trade:</b> #{sig_id} \u2022 <b>{sym} {direction}</b>",
        f"\U0001f4b5 <b>Current Price:</b> {fp(exit_price)}",
        f"\U0001f4ca <b>PnL at Close:</b> {current_r:+.1f}R",
        "\u2500" * 25,
        f"\U0001f552 {_ist_now()}",
    ]
    return "\n".join(lines)
