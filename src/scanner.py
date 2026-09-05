"""Scanner — one complete scan cycle.

A single scan:
  1. Iterates all symbols (respecting session hours)
  2. Fetches data → computes confluence score → generates trade plan
  3. Checks cooldown → generates narration → sends Telegram alert
  4. Saves signal to DB + JSONL log
  5. Returns a scan summary

This is the core loop body, called by the scheduler on each interval.
"""
from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone, timedelta
from typing import Any

from .config import Config
from .data.budget import Budget
from .data.router import Router
from .store import db, signals as sig_store
from .analysis.confluence import score_symbol
from .analysis.levels import generate_plan
from .llm.explain import explain
from .llm.template import build_fact_sheet, narrate as template_narrate
from .alerts.telegram import send_signal
from .logging_util import log_signal, log_scan

log = logging.getLogger(__name__)

__all__ = ["run_scan"]

# Session windows (UTC hours). None = always active.
SESSION_WINDOWS: dict[str, tuple[int, int] | None] = {
    "always": None,
    "us_cash": (13, 21),     # NYSE: 9:30-16:00 ET = 13:30-20:00 UTC (approx 13-21)
    "fx_week": (21, 21),     # Sun 21:00 - Fri 21:00 UTC (handled separately)
}


def _is_session_active(session: str, now: datetime) -> bool:
    """Check if a symbol's market session is currently active."""
    if session == "always":
        return True

    if session == "fx_week":
        # Active Sun 21:00 UTC through Fri 21:00 UTC
        wd = now.weekday()  # Mon=0 .. Sun=6
        hour = now.hour
        if wd == 6 and hour >= 21:  # Sunday after 21:00
            return True
        if wd == 5:  # Saturday
            return False
        if wd == 4 and hour >= 21:  # Friday after 21:00
            return False
        if 0 <= wd <= 4:  # Mon-Fri
            return True
        return False

    if session == "us_cash":
        # NYSE is closed on weekends
        if now.weekday() >= 5:  # 5=Saturday, 6=Sunday
            return False
        # Weekdays fall through to the hour check below

    window = SESSION_WINDOWS.get(session)
    if window is None:
        return True

    start_h, end_h = window
    hour = now.hour
    if start_h <= end_h:
        return start_h <= hour < end_h
    else:  # wraps midnight
        return hour >= start_h or hour < end_h


def run_scan(cfg: Config, conn: Any, budget: Budget,
             router: Router, data_dir: Any) -> dict[str, Any]:
    """Execute one complete scan cycle.

    Returns a summary dict:
        symbols_scanned, signals_emitted, duration_s,
        scores, signals_sent, errors
    """
    now = datetime.now(timezone.utc)
    scan_start = time.time()

    # Apply runtime overrides from Telegram /set commands
    try:
        if conn:
            rows = conn.execute("SELECT key, value FROM bot_state WHERE key LIKE 'cfg_%'").fetchall()
            mapping = {
                "cfg_watch": ("thresholds", "watch", int),
                "cfg_cooldown": ("gates", "cooldown_hours", int),
                "cfg_min_rr": ("gates", "min_rr", float),
                "cfg_max_stop_atr": ("gates", "max_stop_atr", float),
                "cfg_risk_pct": ("risk", "default_risk_pct", float),
            }
            for row in rows:
                k, v = row["key"], row["value"]
                if k in mapping:
                    sec, field, cast = mapping[k]
                    if sec not in cfg.raw or not isinstance(cfg.raw[sec], dict):
                        cfg.raw[sec] = {}
                    cfg.raw[sec][field] = cast(v)
                    log.info("Applied runtime override %s.%s = %s", sec, field, v)
    except Exception as exc:
        log.debug("Could not load runtime overrides: %s", exc)

    dry_run = cfg.get("dry_run", default=True)
    cooldown_h = int(cfg.get("gates", "cooldown_hours", default=4) or 4)
    cooldown_s = cooldown_h * 3600
    cooldown_override = float(cfg.get("gates", "cooldown_score_override", default=15) or 15)

    # Freshness limits: reject data older than 2× the timeframe duration
    tf_max_age_s = {
        "15m": 1800, "1h": 7200, "4h": 28800, "1d": 172800, "1w": 1209600,
    }

    summary: dict[str, Any] = {
        "symbols_scanned": 0,
        "signals_emitted": 0,
        "signals_sent": 0,
        "scores": {},
        "errors": [],
    }

    for sym in cfg.symbols:
        # Session check
        if not _is_session_active(sym.session, now):
            log.debug("Skipping %s — session %s not active", sym.name, sym.session)
            continue

        try:
            res = router.fetch_symbol(sym, now)
            if not res.ok:
                log.warning("Data incomplete for %s: %s", sym.name, "; ".join(res.notes))
                summary["errors"].append(f"{sym.name}: data incomplete")
                continue

            # Freshness check — reject stale data
            stale = False
            for tf, df in res.frames.items():
                if df is None or df.empty:
                    continue
                last_ts = df.index[-1]
                if hasattr(last_ts, 'timestamp'):
                    age_s = (now - last_ts.to_pydatetime().replace(
                        tzinfo=timezone.utc)).total_seconds()
                else:
                    age_s = 0
                max_age = tf_max_age_s.get(tf, 7200)
                if age_s > max_age:
                    log.warning("Stale data for %s/%s: %.0fs old (max %ds)",
                                sym.name, tf, age_s, max_age)
                    stale = True
                    break
            if stale:
                summary["errors"].append(f"{sym.name}: stale data")
                continue

            summary["symbols_scanned"] += 1

            # Score
            signal = score_symbol(res.frames, sym.name, cfg)
            summary["scores"][sym.name] = round(signal.score, 1)

            # Apply streak-based threshold adjustment
            from src.tracking.streaks import get_threshold_adjustment
            streak_bump = get_threshold_adjustment(conn)
            if streak_bump > 0:
                effective_watch = int(cfg.get("thresholds", "watch", default=18) or 18) + streak_bump
                if abs(signal.score) < effective_watch:
                    signal.label = "NEUTRAL"
                    signal.direction = 0
                    log.debug("%s: streak bump +%d raised threshold to %d, score %.1f → NEUTRAL",
                              sym.name, streak_bump, effective_watch, signal.score)

            # Skip neutrals
            if signal.label == "NEUTRAL" or signal.direction == 0:
                continue

            # Generate plan
            plan = generate_plan(signal, cfg)
            if plan is None:
                log.debug("%s: plan failed gates (label=%s)", sym.name, signal.label)
                continue

            # Cooldown check (with score override for strong signals)
            on_cooldown = sig_store.is_on_cooldown(conn, sym.name,
                                                    signal.direction, cooldown_s)
            if on_cooldown:
                # Check if new score is significantly higher → override cooldown
                dir_str = "long" if signal.direction > 0 else "short"
                last = sig_store.get_last_signal(conn, sym.name, direction=dir_str)
                if last and abs(signal.score) - abs(last["score"]) > cooldown_override:
                    log.info("%s: cooldown override (score jump +%.1f)",
                             sym.name, abs(signal.score) - abs(last["score"]))
                else:
                    log.debug("%s: on cooldown, skipping", sym.name)
                    continue

            # Price-level deduplication: skip if an open signal already
            # covers this entry zone (within 1 ATR tolerance)
            ltf_result = signal.tf_results.get(cfg.ltf)
            plan_atr = 0.0
            if ltf_result and ltf_result.classic is not None:
                a = float(ltf_result.classic["atr"].iloc[-1])
                if not math.isnan(a):
                    plan_atr = a
            if plan_atr > 0 and sig_store.has_overlapping_signal(
                    conn, sym.name, signal.direction,
                    plan.entry_low, plan.entry_high, plan_atr):
                log.debug("%s: overlapping open signal, skipping", sym.name)
                continue

            # Narration
            narration, narr_source = explain(signal, plan, cfg)

            # Send alert
            sent_ok = send_signal(signal, plan, narration, narr_source, cfg)
            if sent_ok:
                summary["signals_sent"] += 1

            # Save to DB
            signal_id = sig_store.save_signal(
                conn, signal, plan, narration, narr_source,
                sent_ok=sent_ok, data_source=res.source,
            )

            # JSONL log
            plan_dict = {
                "entry_low": plan.entry_low, "entry_high": plan.entry_high,
                "sl": plan.sl, "tp1": plan.tp1, "tp2": plan.tp2, "tp3": plan.tp3,
                "rr": plan.rr, "trade_type": plan.trade_type,
                "brief_reason": plan.brief_reason,
            }
            log_signal(data_dir, signal_id, sym.name,
                       "long" if signal.direction > 0 else "short",
                       signal.label, signal.score,
                       plan=plan_dict, narration_source=narr_source,
                       sent_ok=sent_ok)

            summary["signals_emitted"] += 1
            log.info("SIGNAL: %s %s %s (score=%+.1f, type=%s)",
                     sym.name, signal.label,
                     "long" if signal.direction > 0 else "short",
                     signal.score, plan.trade_type)

        except Exception as exc:
            log.error("Error scanning %s: %s", sym.name, exc, exc_info=True)
            summary["errors"].append(f"{sym.name}: {exc}")

    duration = time.time() - scan_start
    summary["duration_s"] = round(duration, 1)

    # Log scan summary
    log_scan(data_dir, summary["symbols_scanned"],
             summary["signals_emitted"], duration,
             summary["scores"])

    log.info("Scan complete: %d symbols, %d signals, %d sent, %.1fs",
             summary["symbols_scanned"], summary["signals_emitted"],
             summary["signals_sent"], duration)

    return summary
