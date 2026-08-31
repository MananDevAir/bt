"""Telegram Bot API sender — delivers signal alerts and handles commands.

Features:
  - send_signal():   format + narrate + deliver a signal alert
  - send_text():     send arbitrary text (status, reports)
  - Retry logic:     3 attempts with backoff on 429/5xx
  - dry_run mode:    logs the message without sending
  - Bot commands:    /status, /watchlist, /last
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests

from ..config import Config
from .formatter import format_signal, format_status, format_watchlist

log = logging.getLogger(__name__)

__all__ = ["send_signal", "send_text", "TelegramError"]

API_BASE = "https://api.telegram.org/bot{token}"


class TelegramError(RuntimeError):
    """Raised when Telegram delivery fails after all retries."""


def _get_credentials() -> tuple[str, str]:
    """Load bot token and chat ID from environment."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    return token, chat_id


def send_text(text: str, token: str | None = None,
              chat_id: str | None = None,
              parse_mode: str = "HTML",
              silent: bool = False) -> dict | None:
    """Send a text message via Telegram Bot API.

    Args:
        text: message content (HTML or plain)
        token: bot token (reads from env if None)
        chat_id: target chat (reads from env if None)
        parse_mode: "HTML" or "MarkdownV2"
        silent: if True, send without notification sound

    Returns:
        Telegram API response dict, or None on failure.
    """
    if not token or not chat_id:
        env_token, env_chat = _get_credentials()
        token = token or env_token
        chat_id = chat_id or env_chat

    if not token or not chat_id:
        log.error("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set")
        return None

    url = f"{API_BASE.format(token=token)}/sendMessage"
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }
    if silent:
        payload["disable_notification"] = True

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=15)
            data = resp.json()

            if data.get("ok"):
                msg_id = data.get("result", {}).get("message_id", "?")
                log.info("Telegram message sent (id=%s, len=%d)", msg_id, len(text))
                return data

            error_code = data.get("error_code", 0)
            description = data.get("description", "unknown error")

            if error_code == 429:
                # Rate limited — respect retry_after
                retry_after = data.get("parameters", {}).get("retry_after", 5)
                log.warning("Telegram 429, waiting %ds", retry_after)
                time.sleep(retry_after)
                continue
            elif error_code >= 500:
                log.warning("Telegram %d: %s (attempt %d)",
                            error_code, description, attempt + 1)
                time.sleep(2 ** (attempt + 1))
                continue
            else:
                # Client error (400, 401, 403) — don't retry
                log.error("Telegram %d: %s", error_code, description)
                return None

        except requests.exceptions.Timeout:
            log.warning("Telegram timeout (attempt %d)", attempt + 1)
            last_error = TimeoutError("Telegram API timeout")
            time.sleep(2)
        except Exception as exc:
            log.error("Telegram error: %s", exc)
            last_error = exc
            time.sleep(2)

    log.error("Telegram: all 3 attempts failed. Last: %s", last_error)
    return None


def send_signal(signal: Any, plan: Any | None,
                narration: str, narration_source: str,
                cfg: Config) -> bool:
    """Format and send a signal alert to Telegram.

    Args:
        signal: SignalResult from confluence engine
        plan: TradePlan or None
        narration: explanation text
        narration_source: "hf:model" or "template"
        cfg: bot configuration

    Returns:
        True if sent successfully (or dry_run), False otherwise.
    """
    # Build the message
    msg = format_signal(signal, plan, narration, narration_source)

    # Dry run mode — log but don't send
    dry_run = cfg.get("dry_run", default=True)
    if dry_run:
        log.info("DRY RUN — would send signal for %s (%s):\n%s",
                 signal.symbol, signal.label, msg)
        print(f"\n  [DRY RUN] Signal for {signal.symbol} ({signal.label}):")
        print(f"  Message length: {len(msg)} chars")
        print("  ---")
        # Print without HTML tags for console readability
        import re
        clean = re.sub(r"<[^>]+>", "", msg)
        for line in clean.split("\n"):
            print(f"  {line}")
        print("  ---")
        return True

    # Send for real
    result = send_text(msg)
    if result:
        log.info("Signal sent: %s %s (score=%+.1f)",
                 signal.symbol, signal.label, signal.score)
        return True
    else:
        log.error("Failed to send signal for %s", signal.symbol)
        return False


def handle_command(command: str, cfg: Config,
                   conn: Any = None,
                   symbols_status: list[dict] | None = None) -> str | None:
    """Process a Telegram bot command and return the response text.

    Supported commands:
        /status    — current scores for all symbols
        /watchlist — list of watched symbols
        /last      — last emitted signal details
        /report    — today's performance summary
        /help      — list commands
    """
    cmd = command.strip().lower().split()[0] if command.strip() else ""

    if cmd == "/status":
        if symbols_status:
            return format_status(symbols_status)
        return "\U0001f4ca No scan data available yet. Wait for next scan cycle."

    elif cmd == "/watchlist":
        names = [s.name for s in cfg.symbols]
        return format_watchlist(names)

    elif cmd == "/last":
        if conn is None:
            return "\U0001f4ac Signal store not available."
        try:
            from ..store.signals import get_last_signal
            last = get_last_signal(conn)
            if not last:
                return "\U0001f4ac No signals emitted yet."

            from datetime import datetime, timezone, timedelta
            IST = timezone(timedelta(hours=5, minutes=30))
            ts = datetime.fromtimestamp(last["ts"] / 1000, tz=IST)

            arrow = "\U0001f7e2" if last["direction"] == "long" else "\U0001f534"
            lines = [
                f"{arrow} <b>Last Signal: {last['symbol']}</b>",
                f"  {last['label']}  |  Score: {last['score']:+.1f}",
                f"  Status: <b>{last['status']}</b>",
            ]
            if last.get("entry_low") and last.get("entry_high"):
                lines.append(f"  Entry: {last['entry_low']:.2f} – {last['entry_high']:.2f}")
            if last.get("sl"):
                lines.append(f"  SL: {last['sl']:.2f}")
            if last.get("rr"):
                lines.append(f"  R:R: {last['rr']:.1f}")
            lines.append(f"\U0001f552 {ts:%d %b %Y, %I:%M %p IST}")
            return "\n".join(lines)
        except Exception as exc:
            log.warning("Error in /last: %s", exc)
            return f"\u26a0 Error: {exc}"

    elif cmd == "/report":
        if conn is None:
            return "\U0001f4ca Signal store not available."
        try:
            from ..tracking.performance import compute_stats
            from ..tracking.report import format_report
            from datetime import datetime, timezone
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            stats = compute_stats(conn, hours=24)
            return format_report(stats, "Today's Report", today)
        except Exception as exc:
            log.warning("Error in /report: %s", exc)
            return f"\u26a0 Error: {exc}"

    elif cmd == "/help":
        return (
            "<b>Signal Bot Commands</b>\n"
            "/status  \u2014  Current market scores\n"
            "/last  \u2014  Last emitted signal\n"
            "/report  \u2014  Today's performance\n"
            "/watchlist  \u2014  Active symbols\n"
            "/help  \u2014  This message"
        )

    return None


# ── Telegram update polling ─────────────────────────────────────────
_last_update_id = 0


def poll_commands(cfg: Config, conn: Any = None,
                  symbols_status: list[dict] | None = None) -> int:
    """Check for new Telegram commands and respond.

    Uses getUpdates long-polling with a short timeout.
    Returns the number of commands processed.
    """
    global _last_update_id

    token, chat_id = _get_credentials()
    if not token:
        return 0

    try:
        url = f"{API_BASE.format(token=token)}/getUpdates"
        params = {"timeout": 1, "allowed_updates": ["message"]}
        if _last_update_id:
            params["offset"] = _last_update_id + 1

        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code != 200:
            return 0

        data = resp.json()
        if not data.get("ok"):
            return 0

        processed = 0
        for update in data.get("result", []):
            _last_update_id = update["update_id"]
            msg = update.get("message", {})
            text = msg.get("text", "")
            msg_chat_id = str(msg.get("chat", {}).get("id", ""))

            # Only respond to our chat
            if msg_chat_id != chat_id:
                continue

            if not text.startswith("/"):
                continue

            response = handle_command(text, cfg, conn=conn,
                                      symbols_status=symbols_status)
            if response:
                send_text(response, token=token, chat_id=chat_id)
                processed += 1
                log.info("Command: %s → responded (%d chars)",
                         text.split()[0], len(response))

        return processed

    except Exception as exc:
        log.debug("Poll error: %s", exc)
        return 0
