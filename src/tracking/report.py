"""Daily/weekly performance reports — formatted for Telegram delivery.

Generates human-readable summaries with win rate, R stats,
per-symbol breakdown, and sends them via Telegram.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any

from ..config import Config
from ..alerts.telegram import send_text
from .performance import compute_stats, save_daily_snapshot

log = logging.getLogger(__name__)

__all__ = ["daily_report", "weekly_report", "format_report"]

IST = timezone(timedelta(hours=5, minutes=30))


def format_report(stats: dict[str, Any], title: str = "Daily Report",
                  period: str = "") -> str:
    """Build a Telegram HTML report from performance stats."""
    lines: list[str] = []

    ist_now = datetime.now(IST).strftime("%d %b %Y, %I:%M %p IST")

    lines.append(f"\U0001f4ca <b>{title}</b>")
    if period:
        lines.append(f"\U0001f4c5 {period}")
    lines.append("")

    total = stats["total"]
    if total == 0:
        lines.append("No signals emitted in this period.")
        lines.append("")
        lines.append(f"\U0001f552 {ist_now}")
        return "\n".join(lines)

    wins = stats["wins"]
    losses = stats["losses"]
    expired = stats["expired"]
    still_open = stats["open"]
    win_rate = stats["win_rate"]

    # Overall stats
    if wins + losses > 0:
        wr_emoji = "\U0001f7e2" if win_rate >= 60 else "\U0001f7e1" if win_rate >= 45 else "\U0001f534"
        lines.append(f"{wr_emoji} <b>Win Rate: {win_rate:.0f}%</b>  ({wins}W / {losses}L)")
    else:
        lines.append(f"\u26aa No closed trades yet")

    lines.append(f"\U0001f4cb Total: {total}  |  Open: {still_open}  |  Expired: {expired}")
    lines.append("")

    # R metrics
    if stats["avg_mfe_r"] > 0 or stats["avg_mae_r"] > 0:
        lines.append("<b>Risk Metrics</b>")
        lines.append(f"  Avg MFE: {stats['avg_mfe_r']:.1f}R  |  Avg MAE: {stats['avg_mae_r']:.1f}R")
        lines.append(f"  Best: {stats['best_r']:.1f}R  |  Worst: -{stats['worst_r']:.1f}R")
        lines.append("")

    # Per-symbol breakdown (only symbols with activity)
    by_sym = stats.get("by_symbol", {})
    if by_sym:
        lines.append("<b>By Symbol</b>")
        for sym, data in sorted(by_sym.items()):
            w = data.get("wins", 0)
            l = data.get("losses", 0)
            t = data.get("total", 0)
            if w + l > 0:
                wr = w / (w + l) * 100
                dot = "\U0001f7e2" if wr >= 60 else "\U0001f534"
                lines.append(f"  {dot} {sym}: {w}W/{l}L ({wr:.0f}%)")
            else:
                lines.append(f"  \u26aa {sym}: {t} signals (no closes)")
        lines.append("")

    # Per-label breakdown
    by_label = stats.get("by_label", {})
    if by_label:
        lines.append("<b>By Signal Type</b>")
        for label, data in sorted(by_label.items()):
            t = data.get("total", 0)
            w = data.get("wins", 0)
            l = data.get("losses", 0)
            lines.append(f"  {label}: {t} total ({w}W/{l}L)")
        lines.append("")

    lines.append(f"\U0001f552 {ist_now}")

    return "\n".join(lines)


def daily_report(conn: Any, cfg: Config,
                 send: bool = True) -> dict[str, Any]:
    """Generate and optionally send a daily performance report.

    Returns the stats dict.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    stats = save_daily_snapshot(conn, today)

    msg = format_report(stats, title="Daily Report", period=today)

    if send and not cfg.get("dry_run", default=True):
        result = send_text(msg)
        if result:
            log.info("Daily report sent for %s", today)
        else:
            log.warning("Failed to send daily report for %s", today)
    else:
        log.info("Daily report (dry_run):\n%s", msg)

    return stats


def weekly_report(conn: Any, cfg: Config,
                  send: bool = True) -> dict[str, Any]:
    """Generate and optionally send a weekly performance report.

    Returns the stats dict.
    """
    stats = compute_stats(conn, hours=168)  # 7 days

    now = datetime.now(timezone.utc)
    week_start = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    week_end = now.strftime("%Y-%m-%d")
    period = f"{week_start} to {week_end}"

    msg = format_report(stats, title="Weekly Report", period=period)

    if send and not cfg.get("dry_run", default=True):
        result = send_text(msg)
        if result:
            log.info("Weekly report sent for %s", period)
        else:
            log.warning("Failed to send weekly report for %s", period)
    else:
        log.info("Weekly report (dry_run):\n%s", msg)

    return stats
