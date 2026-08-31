"""Scheduler — runs scan cycles at configured intervals.

Features:
  - Configurable scan interval (default: 15 min)
  - Outcome checks on a separate slower interval (default: 1h)
  - Daily report at configured hour (default: 00:00 UTC)
  - Weekly report on configured day (default: Sunday)
  - Telegram bot command polling between scans
  - Graceful shutdown on Ctrl+C
"""
from __future__ import annotations

import logging
import signal as os_signal
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from .config import Config
from .data.budget import Budget
from .data.router import Router
from .scanner import run_scan
from .tracking.outcome_checker import check_outcomes
from .tracking.report import daily_report, weekly_report
from .maintenance import cleanup
from .alerts.telegram import send_text, poll_commands

log = logging.getLogger(__name__)

__all__ = ["run_loop"]

IST = timezone(timedelta(hours=5, minutes=30))


class _GracefulExit:
    """Catches SIGINT/SIGTERM for graceful shutdown."""
    _stop = False

    def __init__(self):
        os_signal.signal(os_signal.SIGINT, self._handler)
        os_signal.signal(os_signal.SIGTERM, self._handler)

    def _handler(self, *_):
        log.info("Shutdown signal received — finishing current cycle")
        self._stop = True

    @property
    def should_stop(self) -> bool:
        return self._stop


def run_loop(cfg: Config, conn: sqlite3.Connection) -> None:
    """Main event loop — runs until interrupted.

    Cycle:
      1. Run a full scan
      2. Check outcomes for open signals
      3. Send daily/weekly reports at scheduled times
      4. Sleep until next interval
    """
    stopper = _GracefulExit()

    # Setup
    budget = Budget(conn, 750, 7)
    router = Router(cfg, conn, budget)
    data_dir = cfg.db_path.parent

    scan_interval = int(cfg.get("scan_interval_minutes", default=15) or 15) * 60
    outcome_interval = int(
        cfg.get("tracking", "outcome_check_interval_hours", default=1) or 1
    ) * 3600
    report_hour = int(cfg.get("tracking", "report_hour_utc", default=0) or 0)
    weekly_day = str(cfg.get("tracking", "weekly_report_day", default="sunday") or "sunday").lower()

    dry_run = cfg.get("dry_run", default=True)

    # Tracking state
    last_outcome_check = 0.0
    last_daily_report = ""
    last_weekly_report = ""
    last_maintenance = ""
    cycle = 0

    # Startup message
    ist_now = datetime.now(IST).strftime("%d %b %Y, %I:%M %p IST")
    startup_msg = (
        f"\U0001f680 <b>Signal Bot Started</b>\n"
        f"\U0001f552 {ist_now}\n"
        f"\U0001f4cb {len(cfg.symbols)} symbols\n"
        f"\u23f0 Scan every {scan_interval // 60} min\n"
        f"\U0001f4ca Mode: {'DRY RUN' if dry_run else 'LIVE'}"
    )
    if not dry_run:
        send_text(startup_msg)
    log.info("Bot started: %d symbols, %d min interval, dry_run=%s",
             len(cfg.symbols), scan_interval // 60, dry_run)

    try:
        while not stopper.should_stop:
            cycle += 1
            cycle_start = time.time()
            now = datetime.now(timezone.utc)

            log.info("=== Cycle %d  %s ===", cycle, now.strftime("%H:%M UTC"))

            # Sleep Window Check (IST)
            sleep_cfg = cfg.get("sleep_window", default={}) or {}
            is_sleeping = False
            if sleep_cfg.get("enabled", False):
                ist_now = datetime.now(IST)
                start_h = int(sleep_cfg.get("start_hour_ist", 0))
                end_h = int(sleep_cfg.get("end_hour_ist", 5))
                if start_h <= ist_now.hour < end_h:
                    is_sleeping = True

            # ── 1. Scan ─────────────────────────────────────
            if is_sleeping:
                log.info("Night mode active (%02d:00-%02d:00 IST) - skipping scan", start_h, end_h)
                summary = {"symbols_scanned": 0, "signals_emitted": 0}
                summary["scores"] = {}
            else:
                try:
                    summary = run_scan(cfg, conn, budget, router, data_dir)
                except Exception as exc:
                    log.error("Scan failed: %s", exc, exc_info=True)
                    summary = {"symbols_scanned": 0, "signals_emitted": 0}
                    summary["scores"] = {}

            # ── 2. Outcome check ─────────────────────────────
            if time.time() - last_outcome_check >= outcome_interval:
                try:
                    outcomes = check_outcomes(conn, cfg, data_dir=data_dir)
                    last_outcome_check = time.time()
                    if outcomes.get("won", 0) or outcomes.get("lost", 0):
                        log.info("Outcomes: %d won, %d lost",
                                 outcomes["won"], outcomes["lost"])
                except Exception as exc:
                    log.error("Outcome check failed: %s", exc)

            # ── 3. Daily report ──────────────────────────────
            today = now.strftime("%Y-%m-%d")
            if now.hour == report_hour and today != last_daily_report:
                try:
                    daily_report(conn, cfg, send=not dry_run)
                    last_daily_report = today
                except Exception as exc:
                    log.error("Daily report failed: %s", exc)

            # ── 4. Weekly report ─────────────────────────────
            day_names = ["monday", "tuesday", "wednesday", "thursday",
                         "friday", "saturday", "sunday"]
            current_day = day_names[now.weekday()]
            week_key = f"{today}-weekly"
            if (current_day == weekly_day and now.hour == report_hour
                    and week_key != last_weekly_report):
                try:
                    weekly_report(conn, cfg, send=not dry_run)
                    last_weekly_report = week_key
                except Exception as exc:
                    log.error("Weekly report failed: %s", exc)

            # ── 5. Daily maintenance ─────────────────────────
            if now.hour == report_hour and today != last_maintenance:
                try:
                    result = cleanup(conn, data_dir, execute=True)
                    last_maintenance = today
                    if result["candles_deleted"] > 0:
                        log.info("Maintenance: cleaned %d candles, %d signals",
                                 result["candles_deleted"],
                                 result["signals_archived"])
                except Exception as exc:
                    log.error("Maintenance failed: %s", exc)

            # ── 6. Sleep + poll commands ────────────────────
            elapsed = time.time() - cycle_start
            sleep_time = max(10, scan_interval - elapsed)

            # Build status for /status command
            scores = summary.get("scores", {})
            status_list = [
                {"symbol": sym, "score": sc,
                 "label": "LONG" if sc > 18 else "SHORT" if sc < -18 else "NEUTRAL"}
                for sym, sc in scores.items()
            ]

            log.info("Cycle %d done in %.1fs. Next scan in %.0fs",
                     cycle, elapsed, sleep_time)

            # Sleep in small chunks, polling Telegram between naps
            sleep_end = time.time() + sleep_time
            while time.time() < sleep_end and not stopper.should_stop:
                # Poll for commands every 5 seconds
                if not dry_run:
                    try:
                        poll_commands(cfg, conn=conn,
                                     symbols_status=status_list)
                    except Exception:
                        pass
                time.sleep(min(5, sleep_end - time.time()))

    except KeyboardInterrupt:
        pass

    # Shutdown
    ist_now = datetime.now(IST).strftime("%d %b %Y, %I:%M %p IST")
    shutdown_msg = (
        f"\U0001f6d1 <b>Signal Bot Stopped</b>\n"
        f"\U0001f552 {ist_now}\n"
        f"Cycles completed: {cycle}"
    )
    if not dry_run:
        send_text(shutdown_msg)
    log.info("Bot stopped after %d cycles", cycle)
