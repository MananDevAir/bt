"""Discord Webhook integration — Rich Embeds + plain text alerts."""
from __future__ import annotations

import logging
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from ..config import Config

log = logging.getLogger(__name__)

# Embed sidebar colours
_COLOR_LONG    = 0x00E676  # bright green
_COLOR_SHORT   = 0xFF1744  # bright red
_COLOR_NEUTRAL = 0x607D8B  # blue-grey
_COLOR_INFO    = 0x5865F2  # Discord blurple (for reports / startup)
_COLOR_WIN     = 0x00E676  # green (TP hit / trailing win)
_COLOR_LOSS    = 0xFF1744  # red (SL hit)
_COLOR_EXPIRY  = 0x607D8B  # grey (expired)


# ══════════════════════════════════════════════════════════════════════ #
# HTTP helpers
# ══════════════════════════════════════════════════════════════════════ #

def _get_webhook_url(cfg: Config | None = None) -> str:
    """Get webhook URL — env var DISCORD_WEBHOOK_URL takes priority over config.yaml."""
    import os
    env_url = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if env_url:
        return env_url
    if cfg is None:
        from .. import config as config_mod
        cfg = config_mod.load()
    return str(cfg.get("alerts", "discord_webhook_url", default="")).strip()


def _build_url(webhook_url: str, thread_name: str | None = None,
               thread_id: int | None = None) -> str:
    """Append the ?wait=true and optional thread params to the webhook URL."""
    params = ["wait=true"]
    if thread_name:
        params.append(f"thread_name={urllib.parse.quote(thread_name)}")
    if thread_id:
        params.append(f"thread_id={thread_id}")
    sep = "&" if "?" in webhook_url else "?"
    return f"{webhook_url}{sep}{'&'.join(params)}"


def _post(url: str, payload: dict, timeout: int = 10) -> int | None:
    """POST JSON payload to Discord. Returns message_id or None.

    Fix 3: Logs the full Discord error body on 4xx/5xx so we can see exactly
    why a thread reply failed (e.g. 'Unknown Channel', 'Missing Access').
    """
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        if not resp.ok:
            # Log the full Discord error body for debugging
            try:
                err_body = resp.json()
            except Exception:
                err_body = resp.text[:500]
            # Extra warning for thread-reply failures (wrong channel type etc.)
            if "thread_id" in url:
                log.error(
                    "Discord thread reply FAILED (HTTP %d) — is your channel a Forum "
                    "channel? Error: %s", resp.status_code, err_body
                )
            else:
                log.error("Discord post failed (HTTP %d): %s", resp.status_code, err_body)
            return None
        data = resp.json()
        if "id" in data:
            return int(data["id"])
    except requests.exceptions.RequestException as e:
        log.error("Discord post network error: %s", e)
    except (ValueError, KeyError):
        pass
    return None


def _post_with_fallback(url_with_thread: str, url_plain: str,
                        payload: dict) -> int | None:
    """Try posting with thread_id first; fall back to plain channel post if rejected.

    Fix 3: Guarantees the message reaches Discord even if the thread ID is wrong.
    """
    msg_id = _post(url_with_thread, payload)
    if msg_id is None and "thread_id" in url_with_thread:
        log.warning("Thread reply failed — falling back to plain channel post")
        msg_id = _post(url_plain, payload)
    return msg_id


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


def _discord_ts(unix_ms: float | None = None) -> str:
    """Return a Discord dynamic timestamp tag. Shows in the user's local timezone."""
    ts = int((unix_ms / 1000) if unix_ms else time.time())
    return f"<t:{ts}:F>"


# ══════════════════════════════════════════════════════════════════════ #
# Signal embed (initial alert card)
# ══════════════════════════════════════════════════════════════════════ #

def build_signal_embed(signal: Any, plan: Any | None, narration: str) -> dict:
    """Build a Discord Embed payload dict for a signal alert.

    Returns the full `embeds` list ready to pass as JSON to the webhook.
    """
    sym = signal.symbol
    fp = lambda v: _fmt_price(v, sym)  # noqa: E731

    # Direction
    if signal.direction > 0:
        color = _COLOR_LONG
        arrow = "🟢"
        dir_word = "LONG"
    elif signal.direction < 0:
        color = _COLOR_SHORT
        arrow = "🔴"
        dir_word = "SHORT"
    else:
        color = _COLOR_NEUTRAL
        arrow = "⚪"
        dir_word = "NEUTRAL"

    # Grade badge
    grade = getattr(plan, "grade", "") if plan else ""
    grade_str = f"🏆 GRADE {grade}" if grade else ""

    # Title
    title = f"{arrow}  {sym}  •  {signal.label}  •  {grade_str}" if grade_str else f"{arrow}  {sym}  •  {signal.label}"

    desc_parts = []
    trade_type = getattr(plan, "trade_type", "") if plan else "—"
    desc_parts.append(f"📊 **Score:** `{signal.score:+.1f}`  |  **Conf:** `{signal.confidence:.0f}%`  |  **Type:** `{trade_type}`")
    
    session = getattr(plan, "session", "") if plan else ""
    if session:
        kz = getattr(plan, "killzone", "") if plan else ""
        desc_parts.append(f"🌐 **Session:** `{session}`" + (f" ({kz} Killzone)" if kz else ""))
    
    desc_parts.append("")

    if plan and plan.brief_reason:
        desc_parts.append(f"📍 {plan.brief_reason}")

    # Timeframe trend arrows
    tf_parts = []
    for tf, tfr in signal.tf_results.items():
        if not tfr.votes:
            continue
        avg = sum(v.value for v in tfr.votes) / len(tfr.votes)
        if avg > 0.2:
            tf_parts.append(f"{tf}↑")
        elif avg < -0.2:
            tf_parts.append(f"{tf}↓")
        else:
            tf_parts.append(f"{tf}↔")
    if tf_parts:
        desc_parts.append("📈 **Trends:** " + " • ".join(tf_parts))

    if plan:
        desc_parts.append("")
        alloc = getattr(plan, "tp_allocation", [50, 30, 20])
        
        # Risk text
        risk_dist = abs(plan.entry_mid - plan.sl) if hasattr(plan, "entry_mid") else 0
        risk_str = ""
        if risk_dist > 0:
            if sym in ("EURUSD", "GBPUSD"):
                risk_str = f" (Risk: {risk_dist / 0.0001:.1f} pips)"
            elif sym in ("USDJPY",):
                risk_str = f" (Risk: {risk_dist / 0.01:.1f} pips)"
            else:
                risk_str = f" (Risk: {plan.risk_pct:.1f}%)"
                
        desc_parts.append(f"🎯 **Entry:** `{fp(plan.entry_low)}` — `{fp(plan.entry_high)}`")
        desc_parts.append(f"🛑 **Stop:** `{fp(plan.sl)}`{risk_str}")
        desc_parts.append(f"🏁 **TP1:** `{fp(plan.tp1)}` ({alloc[0]}%)")
        desc_parts.append(f"🏁 **TP2:** `{fp(plan.tp2)}` ({alloc[1]}%)")
        desc_parts.append(f"🏁 **TP3:** `{fp(plan.tp3)}` ({alloc[2]}%)")

    description = "\n".join(desc_parts) if desc_parts else ""
    fields: list[dict] = []

    # Footer with dynamic Discord timestamp
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    footer_text = f"Score {signal.score:+.1f}  •  Analysis only — not financial advice"

    embed = {
        "color": color,
        "title": title,
        "description": description,
        "fields": fields,
        "footer": {"text": footer_text},
        "timestamp": now_iso,   # Discord renders this in the user's local timezone
    }
    return embed


# ══════════════════════════════════════════════════════════════════════ #
# Outcome embeds (Fix 8) — TP / SL / Expiry update cards
# ══════════════════════════════════════════════════════════════════════ #

def build_tp_embed(sig: dict, tp_level: str, r_mult: str,
                   exit_price: float, sl_action: str,
                   profit_banked: str) -> dict:
    """Build a rich embed for a Take Profit hit update."""
    sym = sig.get("symbol", "")
    sig_id = sig.get("id", "?")
    direction = sig.get("direction", "long").upper()
    fp = lambda v: _fmt_price(v, sym)  # noqa: E731

    progress_map = {
        "tp1": "[🎯 TP1 (50%)] ➔ [⏳ TP2] ➔ [⏳ TP3]",
        "tp2": "[✅ TP1] ➔ [🎯 TP2 (30%)] ➔ [⏳ TP3]",
        "tp3": "[✅ TP1] ➔ [✅ TP2] ➔ [🎉 TP3 (20%) — FULL WIN]",
    }
    progress = progress_map.get(tp_level, "")

    desc_parts = [
        f"📌 **#{sig_id}** • **{sym} {direction}**",
        f"💰 **Exit:** `{fp(exit_price)}`  |  **Banked:** {profit_banked}"
    ]
    if progress:
        desc_parts.append(f"📊 **Progress:** {progress}")
    if sl_action:
        desc_parts.append(f"🛡️ **Action:** {sl_action}")

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    return {
        "color": _COLOR_WIN,
        "title": f"🎯  TAKE PROFIT {tp_level.upper()} HIT  (+{r_mult}R)",
        "description": "\n".join(desc_parts),
        "fields": [],
        "footer": {"text": "Analysis only — not financial advice"},
        "timestamp": now_iso,
    }


def build_sl_embed(sig: dict, exit_price: float,
                   sl_type: str, pnl_r: str, mfe_r: float = 0.0) -> dict:
    """Build a rich embed for a Stop Loss / Breakeven update."""
    sym = sig.get("symbol", "")
    sig_id = sig.get("id", "?")
    direction = sig.get("direction", "long").upper()
    fp = lambda v: _fmt_price(v, sym)  # noqa: E731

    if sl_type == "breakeven":
        title = "🛡️  BREAKEVEN STOP HIT  (Risk-Free Exit)"
        result = "Partial Win — TP1 secured at +1.0R"
        color = _COLOR_WIN
    elif sl_type == "trailed_tp1":
        title = "🟢  TRAILED STOP HIT  (TP1 Locked Level)"
        result = "Solid Win — 80% profit banked at TP1 & TP2"
        color = _COLOR_WIN
    else:
        title = "🔴  STOP LOSS HIT"
        result = f"Closed as Loss  •  MFE was +{mfe_r:.1f}R"
        color = _COLOR_LOSS

    desc_parts = [
        f"📌 **#{sig_id}** • **{sym} {direction}**",
        f"🛑 **Exit Level:** `{fp(exit_price)}`",
        f"📈 **PnL:** `{pnl_r}R`",
        f"📊 **Result:** {result}"
    ]

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    return {
        "color": color,
        "title": title,
        "description": "\n".join(desc_parts),
        "fields": [],
        "footer": {"text": "Analysis only — not financial advice"},
        "timestamp": now_iso,
    }


def build_expiry_embed(sig: dict, exit_price: float,
                       current_r: float, max_age_days: float) -> dict:
    """Build a rich embed for a signal expiry notification."""
    sym = sig.get("symbol", "")
    sig_id = sig.get("id", "?")
    direction = sig.get("direction", "long").upper()
    fp = lambda v: _fmt_price(v, sym)  # noqa: E731

    desc_parts = [
        f"📌 **#{sig_id}** • **{sym} {direction}**",
        f"💵 **Price at Expiry:** `{fp(exit_price)}`",
        f"📈 **PnL at Close:** `{current_r:+.1f}R`",
        f"ℹ️ **Reason:** Signal exceeded {max_age_days:.0f}-day hold limit"
    ]

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    return {
        "color": _COLOR_EXPIRY,
        "title": f"⌛  SIGNAL #{sig_id} EXPIRED  ({max_age_days:.0f}d Limit)",
        "description": "\n".join(desc_parts),
        "fields": [],
        "footer": {"text": "Analysis only — not financial advice"},
        "timestamp": now_iso,
    }


# ══════════════════════════════════════════════════════════════════════ #
# Public send functions
# ══════════════════════════════════════════════════════════════════════ #

def send_signal_embed(signal: Any, plan: Any | None, narration: str,
                      cfg: Config | None = None,
                      thread_name: str | None = None) -> int | None:
    """Build and send a rich embed signal card to Discord. Returns message_id."""
    webhook_url = _get_webhook_url(cfg)
    if not webhook_url:
        log.warning("Discord webhook URL not configured, skipping send_signal_embed.")
        return None

    embed = build_signal_embed(signal, plan, narration)
    url = _build_url(webhook_url, thread_name=thread_name)
    msg_id = _post(url, {"embeds": [embed]})
    return msg_id


def send_outcome_embed(embed: dict, cfg: Config | None = None,
                       reply_to_message_id: int | None = None) -> int | None:
    """Send a TP/SL/expiry embed as a thread reply.

    Fix 3 + Fix 8: Uses _post_with_fallback so a wrong thread_id
    gracefully falls back to a plain channel post rather than silently dropping.
    """
    webhook_url = _get_webhook_url(cfg)
    if not webhook_url:
        log.warning("Discord webhook URL not configured, skipping send_outcome_embed.")
        return None

    url_plain = _build_url(webhook_url)
    if reply_to_message_id:
        url_threaded = _build_url(webhook_url, thread_id=reply_to_message_id)
    else:
        url_threaded = url_plain

    return _post_with_fallback(url_threaded, url_plain, {"embeds": [embed]})


def send_text(text: str, cfg: Config | None = None, **kwargs) -> int | None:
    """Send a plain text message to the Discord Webhook.

    Kwargs:
        thread_name (str): Creates a new Forum thread with this name.
        reply_to_message_id (int): Posts inside an existing thread.
    """
    webhook_url = _get_webhook_url(cfg)
    if not webhook_url:
        log.warning("Discord webhook URL not configured, skipping send_text.")
        return None

    thread_name = kwargs.get("thread_name")
    reply_id = kwargs.get("reply_to_message_id")
    url_with_thread = _build_url(webhook_url, thread_name=thread_name, thread_id=reply_id)
    url_plain = _build_url(webhook_url)

    # Convert basic HTML tags (used by Telegram) to Discord Markdown
    text = text.replace("<b>", "**").replace("</b>", "**")
    text = text.replace("<i>", "*").replace("</i>", "*")
    text = text.replace("<code>", "`").replace("</code>", "`")
    text = text.replace("<pre>", "```").replace("</pre>", "```")
    text = text.replace("<u>", "__").replace("</u>", "__")

    # Fix 3: fallback if thread reply fails
    if reply_id:
        return _post_with_fallback(url_with_thread, url_plain, {"content": text})
    return _post(url_with_thread, {"content": text})


def poll_commands(cfg: Config, conn: Any, **kwargs) -> None:
    """Discord Webhooks cannot receive commands, so this is a no-op."""
