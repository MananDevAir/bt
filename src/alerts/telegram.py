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
            "/streak  \u2014  Win/loss streak status\n"
            "/config  \u2014  Current settings\n"
            "/set &lt;key&gt; &lt;value&gt;  \u2014  Change a setting\n"
            "/reset [key|all]  \u2014  Reset overrides to default\n"
            "/help  \u2014  This message\n"
            "\n"
            "<b>Settable keys:</b>\n"
            "  watch, cooldown, min_rr, max_stop_atr, risk_pct"
        )

    elif cmd == "/streak":
        if conn is None:
            return "\U0001f4ca Signal store not available."
        try:
            from ..tracking.streaks import get_streak
            s = get_streak(conn)
            lines = [
                "\U0001f4ca <b>Streak Status</b>",
                f"  Win streak:  {s['win_streak']}",
                f"  Loss streak: {s['loss_streak']}",
            ]
            if s["threshold_bump"] > 0:
                lines.append(f"  \u26a0\ufe0f Threshold raised by +{s['threshold_bump']} (losing streak)")
            else:
                lines.append("  \u2705 No threshold adjustment")
            return "\n".join(lines)
        except Exception as exc:
            log.warning("Error in /streak: %s", exc)
            return f"\u26a0 Error: {exc}"

    elif cmd == "/config":
        try:
            watch = cfg.get("thresholds", "watch", default=18)
            cooldown = cfg.get("gates", "cooldown_hours", default=4)
            min_rr = cfg.get("gates", "min_rr", default=1.5)
            max_stop = cfg.get("gates", "max_stop_atr", default=3.0)
            risk_pct = cfg.get("risk", "default_risk_pct", default=1.0)
            dry_run = cfg.get("dry_run", default=True)

            # Check for runtime overrides
            overrides = []
            if conn:
                rows = conn.execute(
                    "SELECT key, value FROM bot_state WHERE key LIKE 'cfg_%'"
                ).fetchall()
                overrides = [(r["key"].replace("cfg_", ""), r["value"]) for r in rows]

            lines = [
                "\u2699\ufe0f <b>Current Config</b>",
                f"  Watch threshold: {watch}",
                f"  Cooldown: {cooldown}h",
                f"  Min R:R: {min_rr}",
                f"  Max stop: {max_stop} ATR",
                f"  Risk per trade: {risk_pct}%",
                f"  Mode: {'DRY RUN' if dry_run else 'LIVE'}",
            ]
            if overrides:
                lines.append("")
                lines.append("<b>Runtime Overrides</b>")
                for k, v in overrides:
                    lines.append(f"  {k}: {v}")
            return "\n".join(lines)
        except Exception as exc:
            log.warning("Error in /config: %s", exc)
            return f"\u26a0 Error: {exc}"

    elif cmd == "/set":
        if conn is None:
            return "\u26a0 Database not available."
        try:
            parts = command.strip().split()
            if len(parts) < 3:
                return (
                    "\u26a0 Usage: /set &lt;key&gt; &lt;value&gt;\n\n"
                    "<b>Available keys:</b>\n"
                    "  watch — signal threshold (default 18)\n"
                    "  cooldown — hours between signals (default 4)\n"
                    "  min_rr — minimum R:R ratio (default 1.5)\n"
                    "  max_stop_atr — max stop in ATR (default 3.0)\n"
                    "  risk_pct — risk per trade % (default 1.0)\n\n"
                    "<i>To remove an override: /set &lt;key&gt; reset</i>"
                )

            key = parts[1].lower()
            value = parts[2]

            allowed = {
                "watch": ("thresholds", "watch", int, 10, 50),
                "cooldown": ("gates", "cooldown_hours", int, 1, 24),
                "min_rr": ("gates", "min_rr", float, 0.5, 5.0),
                "max_stop_atr": ("gates", "max_stop_atr", float, 1.0, 10.0),
                "risk_pct": ("risk", "default_risk_pct", float, 0.1, 5.0),
            }

            if key not in allowed:
                return f"\u26a0 Unknown key: {key}\nAllowed: {', '.join(allowed.keys())}"

            if value.lower() in ("reset", "default", "none", "clear"):
                conn.execute("DELETE FROM bot_state WHERE key = ?", (f"cfg_{key}",))
                conn.commit()
                return f"\u2705 Reset <b>{key}</b> to default."

            section, cfg_key, cast, min_val, max_val = allowed[key]
            try:
                parsed = cast(value)
            except ValueError:
                return f"\u26a0 Invalid value: {value} (expected {cast.__name__})"

            if parsed < min_val or parsed > max_val:
                return f"\u26a0 Value out of range: {min_val} – {max_val}"

            # Store override in bot_state
            conn.execute(
                "INSERT OR REPLACE INTO bot_state (key, value) VALUES (?, ?)",
                (f"cfg_{key}", str(parsed))
            )
            conn.commit()

            return (
                f"\u2705 <b>{key}</b> set to <b>{parsed}</b>\n"
                f"This override persists across restarts.\n"
                f"Use /config to see current settings."
            )
        except Exception as exc:
            log.warning("Error in /set: %s", exc)
            return f"\u26a0 Error: {exc}"

    elif cmd == "/reset":
        if conn is None:
            return "\u26a0 Database not available."
        try:
            parts = command.strip().split()
            if len(parts) > 1 and parts[1].lower() not in ("all", "*"):
                key = parts[1].lower()
                conn.execute("DELETE FROM bot_state WHERE key = ?", (f"cfg_{key}",))
                conn.commit()
                return f"\u2705 Reset <b>{key}</b> to default."
            else:
                conn.execute("DELETE FROM bot_state WHERE key LIKE 'cfg_%'")
                conn.commit()
                return "\u2705 All runtime overrides cleared. Defaults active."
        except Exception as exc:
            log.warning("Error in /reset: %s", exc)
            return f"\u26a0 Error: {exc}"

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
