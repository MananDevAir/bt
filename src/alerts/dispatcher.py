"""Alert dispatcher — routes alerts to Telegram or Discord based on config.

Switch provider by setting alerts.provider in config.yaml:
  alerts:
    provider: discord   # or 'telegram'
    discord_webhook_url: ""  # or set DISCORD_WEBHOOK_URL env var

All original Telegram code is still intact — just set provider: telegram to switch back.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..config import Config
from . import telegram, discord

log = logging.getLogger(__name__)


def _get_provider(cfg: Config | None = None) -> str:
    if cfg is None:
        from .. import config as config_mod
        cfg = config_mod.load()
    return str(cfg.get("alerts", "provider", default="telegram")).lower().strip()


def send_text(text: str, cfg: Config | None = None, **kwargs) -> int | None:
    """Route a plain text alert to Telegram or Discord."""
    provider = _get_provider(cfg)
    if provider == "discord":
        return discord.send_text(text, cfg, **kwargs)
    else:
        return telegram.send_text(text, **kwargs)


def send_signal(
    signal: Any,
    plan: Any | None,
    narration: str,
    narration_source: str,
    cfg: Config,
    return_msg_id: bool = False,
) -> bool | tuple[bool, int | None]:
    """Route a signal alert to Telegram or Discord.

    Signature mirrors telegram.send_signal exactly so scanner.py
    needs no changes when the provider switches.
    """
    provider = _get_provider(cfg)

    if provider == "discord":
        # Dry-run guard — same behaviour as Telegram
        dry_run = cfg.get("dry_run", default=True)
        if dry_run:
            log.info("DRY RUN — would send Discord signal for %s (%s)",
                     signal.symbol, signal.label)
            return (True, None) if return_msg_id else True

        # Discord threading: use the symbol and label as the thread name
        thread_name = f"{signal.symbol} {signal.label}"
        msg_id = discord.send_signal_embed(signal, plan, narration, cfg, thread_name=thread_name)
        
        log.info("Discord signal sent: %s %s (score=%+.1f)",
                 signal.symbol, signal.label, signal.score)
        return (True, msg_id) if return_msg_id else True

    else:
        # Full Telegram path — unchanged
        return telegram.send_signal(
            signal, plan, narration, narration_source, cfg,
            return_msg_id=return_msg_id,
        )


def poll_commands(cfg: Config, conn: Any, **kwargs) -> None:
    """Poll Telegram commands if using Telegram; no-op for Discord."""
    provider = _get_provider(cfg)
    if provider == "telegram":
        telegram.poll_commands(cfg, conn, **kwargs)
    # Discord webhooks are one-way — commands not supported


def send_outcome_embed(embed: dict, cfg: Config | None = None,
                       reply_to_message_id: int | None = None) -> int | None:
    """Route an outcome embed (TP/SL/expiry) to Discord, or fall back to plain text.

    Fix 8: Outcome updates now use rich embeds on Discord instead of raw text.
    On Telegram, this is a no-op (outcome_checker sends text directly via send_text).
    """
    provider = _get_provider(cfg)
    if provider == "discord":
        return discord.send_outcome_embed(embed, cfg, reply_to_message_id=reply_to_message_id)
    # Telegram outcomes are sent as text by outcome_checker directly via send_text
    return None

