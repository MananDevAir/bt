"""Outcome checker — tracks open signals and updates their status.

On each check cycle:
  1. Load all open signals from DB
  2. Replay all 15m candles from DB that closed since last check (Fix 1)
     — instead of fetching a single live price snapshot once per hour,
     we replay every candle bar-by-bar so intra-hour TP/SL hits are never missed
  3. For still-open signals, fetch one live price snapshot to update MFE/MAE
  4. Expire signals that are too old
  5. Write outcome records to DB + JSONL log
  6. Send TP/SL/expiry alerts as rich Discord embeds (Fix 8)
  7. Track discord_notified flag so alerts survive bot crashes (Fix 4)

Fix 7: All DB writes are batched into a single commit at end of loop.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from typing import Any

from ..config import Config
from ..store import signals as sig_store
from ..logging_util import log_outcome
from ..alerts.formatter import (
    format_tp_update,
    format_sl_update,
    format_expiry_update,
    _fmt_price,
)
from ..alerts.discord import (
    build_tp_embed,
    build_sl_embed,
    build_expiry_embed,
)
from ..alerts.dispatcher import send_text, send_outcome_embed
from ..data.live_price import get_live_price
from .streaks import update_streak

log = logging.getLogger(__name__)

__all__ = ["check_outcomes", "replay_missed_notifications"]

# Expiry: signals older than this (hours) are auto-expired
DEFAULT_EXPIRY_H = 48


# ══════════════════════════════════════════════════════════════════════ #
# DB helpers
# ══════════════════════════════════════════════════════════════════════ #

def _get_latest_candle(conn: sqlite3.Connection, symbol: str) -> dict[str, float] | None:
    """Get the latest high, low, and close prices for a symbol from cached candles."""
    row = conn.execute(
        "SELECT high, low, close FROM candles WHERE symbol = ? "
        "ORDER BY ts DESC LIMIT 1",
        (symbol,)
    ).fetchone()
    if not row:
        return None
    return {"high": float(row["high"]), "low": float(row["low"]), "close": float(row["close"])}


def _get_candles_since(conn: sqlite3.Connection, symbol: str,
                       since_ts: int) -> list[dict]:
    """Fix 1: Return all completed 15m candles for symbol with ts > since_ts, oldest first."""
    rows = conn.execute(
        "SELECT ts, high, low, close FROM candles "
        "WHERE symbol = ? AND timeframe = '15m' AND ts > ? "
        "ORDER BY ts ASC",
        (symbol, since_ts)
    ).fetchall()
    return [{"ts": r["ts"], "high": float(r["high"]),
             "low": float(r["low"]), "close": float(r["close"])} for r in rows]


def _ensure_outcome_row(conn: sqlite3.Connection, signal_id: int) -> None:
    """Create an outcome row if it doesn't exist yet."""
    exists = conn.execute(
        "SELECT 1 FROM outcomes WHERE signal_id = ?", (signal_id,)
    ).fetchone()
    if not exists:
        conn.execute(
            "INSERT INTO outcomes (signal_id, checked_ts, hit, entry_filled, discord_notified) "
            "VALUES (?, ?, 'open', 0, 0)",
            (signal_id, int(time.time() * 1000))
        )


def _set_discord_notified(conn: sqlite3.Connection, signal_id: int) -> None:
    """Fix 4: Mark a signal's outcome alert as successfully delivered."""
    conn.execute(
        "UPDATE outcomes SET discord_notified = 1 WHERE signal_id = ?",
        (signal_id,)
    )


# ══════════════════════════════════════════════════════════════════════ #
# Fix 4: Crash-safe notification replay
# ══════════════════════════════════════════════════════════════════════ #

def replay_missed_notifications(conn: sqlite3.Connection, cfg: Config) -> None:
    """On startup, resend any TP/SL/expiry alerts that were recorded in DB
    but never delivered to Discord (discord_notified = 0).

    This handles the crash window: process died after DB write but before send.
    """
    rows = conn.execute(
        "SELECT s.*, o.hit, o.price_at_check, o.mfe_r "
        "FROM signals s JOIN outcomes o ON o.signal_id = s.id "
        "WHERE o.discord_notified = 0 AND o.hit NOT IN ('open') AND o.hit IS NOT NULL "
        "AND s.status != 'open'"
    ).fetchall()

    if not rows:
        return

    log.info("Replaying %d missed Discord notifications from before crash", len(rows))
    for row in rows:
        sig = dict(row)
        hit = sig.get("hit", "")
        price = sig.get("price_at_check") or 0.0
        mfe_r = sig.get("mfe_r") or 0.0
        tg_msg_id = sig.get("tg_msg_id")

        try:
            if hit in ("tp1", "tp2", "tp3"):
                r_mult = {"tp1": "1.0", "tp2": "2.0", "tp3": "3.0"}.get(hit, "?")
                tp_val = sig.get(hit) or price
                profit_map = {
                    "tp1": "50% Banked (Risk-Free Trade)",
                    "tp2": "80% Locked (Holding 20% for TP3)",
                    "tp3": "100% Locked (+3.0R) 🎉",
                }
                sl_action_map = {
                    "tp1": f"SL Moved to Breakeven ({_fmt_price(sig.get('entry_low', 0) + (sig.get('entry_high', 0) - sig.get('entry_low', 0)) / 2, sig.get('symbol', ''))})",
                    "tp2": f"SL Trailed to TP1 level ({_fmt_price(sig.get('tp1') or 0, sig.get('symbol', ''))})",
                    "tp3": "Trade Closed (Target Met)",
                }
                embed = build_tp_embed(sig, hit, r_mult, tp_val,
                                       sl_action_map.get(hit, ""),
                                       profit_map.get(hit, ""))
            elif hit in ("sl", "sl_after_tp1", "sl_after_tp2", "breakeven"):
                sl_type = "breakeven" if hit == "sl_after_tp1" else \
                          "trailed_tp1" if hit == "sl_after_tp2" else "loss"
                pnl_r = "+1.60" if sl_type == "trailed_tp1" else \
                        "+0.50" if sl_type == "breakeven" else "-1.00"
                embed = build_sl_embed(sig, price, sl_type, pnl_r, mfe_r)
            else:
                max_age = float(cfg.get("tracking", "max_signal_age_days", default=7) or 7)
                embed = build_expiry_embed(sig, price, mfe_r, max_age)

            send_outcome_embed(embed, cfg, reply_to_message_id=tg_msg_id)
            _set_discord_notified(conn, sig["id"])
            conn.commit()
        except Exception as exc:
            log.warning("Failed to replay missed notification for signal #%d: %s", sig["id"], exc)


# ══════════════════════════════════════════════════════════════════════ #
# Alert sender helper (Fix 4 + Fix 8)
# ══════════════════════════════════════════════════════════════════════ #

def _send_outcome_alert(conn: sqlite3.Connection, cfg: Config,
                        embed: dict, fallback_text: str,
                        signal_id: int, tg_msg_id: int | None) -> None:
    """Send an outcome alert and mark discord_notified=1 on success.

    Tries rich embed first (Discord), then plain text fallback (Telegram).
    Fix 4: Sets discord_notified=1 only after confirmed delivery.
    """
    try:
        # Try embed path first (Discord)
        sent_id = send_outcome_embed(embed, cfg, reply_to_message_id=tg_msg_id)
        if sent_id is None:
            # Telegram path — send as plain text
            send_text(fallback_text, cfg, reply_to_message_id=tg_msg_id)
        _set_discord_notified(conn, signal_id)
    except Exception as exc:
        log.warning("Failed to send outcome alert for signal #%d: %s", signal_id, exc)


# ══════════════════════════════════════════════════════════════════════ #
# Main outcome checker
# ══════════════════════════════════════════════════════════════════════ #

def check_outcomes(conn: sqlite3.Connection, cfg: Config,
                   data_dir: Any = None) -> dict[str, Any]:
    """Check all open signals against bar-by-bar candle history.

    Fix 1: Instead of one live price snapshot per hour, we replay every
    15m candle from the DB that arrived since the last check. This means
    intra-hour TP/SL hits are never missed.

    Returns a summary dict with counts.
    """
    open_signals = sig_store.get_open_signals(conn)
    if not open_signals:
        return {"checked": 0, "won": 0, "lost": 0, "expired": 0}

    now_ms = int(time.time() * 1000)
    max_age_days = float(cfg.get("tracking", "max_signal_age_days", default=7) or 7)
    summary = {"checked": 0, "won": 0, "lost": 0, "expired": 0, "still_open": 0}

    for sig in open_signals:
        signal_id = sig["id"]
        symbol = sig["symbol"]
        direction = 1 if sig["direction"] == "long" else -1
        tg_msg_id = sig.get("tg_msg_id")

        summary["checked"] += 1
        _ensure_outcome_row(conn, signal_id)

        entry_mid = (sig["entry_low"] + sig["entry_high"]) / 2 \
            if sig["entry_low"] and sig["entry_high"] else None
        sl = sig["sl"]
        tp1 = sig["tp1"]
        tp2 = sig["tp2"]
        tp3 = sig["tp3"]

        if not entry_mid or not sl:
            summary["still_open"] += 1
            continue

        risk = abs(entry_mid - sl)
        if risk <= 0:
            summary["still_open"] += 1
            continue

        # Load current outcome row
        _row = conn.execute(
            "SELECT * FROM outcomes WHERE signal_id = ?", (signal_id,)
        ).fetchone()
        outcome = dict(_row) if _row is not None else {}
        entry_filled = outcome.get("entry_filled", 0)

        # ── Fix 1: Replay candles since last check ────────────────────── #
        last_checked_ts = outcome.get("last_outcome_checked_ts") or sig["ts"]
        candles = _get_candles_since(conn, symbol, last_checked_ts)

        # If no cached 15m candles exist yet, fall back to single live price
        if not candles:
            live = get_live_price(symbol)
            if live is None:
                live = _get_latest_candle(conn, symbol)
            if live:
                candles = [{"ts": now_ms, "high": live["high"],
                            "low": live["low"], "close": live["close"]}]

        if not candles:
            summary["still_open"] += 1
            continue

        # Track running MFE/MAE across all replayed candles
        mfe = float(outcome.get("mfe_r") or 0)
        mae = float(outcome.get("mae_r") or 0)
        latest_ts = last_checked_ts
        trade_resolved = False

        tp2_was_hit = outcome.get("tp2_hit_ts")
        tp1_was_hit = outcome.get("tp1_hit_ts")

        for candle in candles:
            c_high = candle["high"]
            c_low  = candle["low"]
            c_close = candle["close"]
            latest_ts = candle["ts"]

            # ── Entry fill check ─────────────────────────────── #
            if not entry_filled:
                if direction > 0 and c_low <= entry_mid:
                    entry_filled = 1
                elif direction < 0 and c_high >= entry_mid:
                    entry_filled = 1
                if entry_filled:
                    conn.execute(
                        "UPDATE outcomes SET entry_filled = 1 WHERE signal_id = ?",
                        (signal_id,)
                    )
                    log.info("Signal #%d entry filled (candle ts=%d)", signal_id, latest_ts)

            if not entry_filled:
                # Check expiry for unfilled signals
                age_ms = now_ms - sig["ts"]
                if age_ms > max_age_days * 86400 * 1000:
                    sig_store.update_status(conn, signal_id, "expired")
                    _update_outcome(conn, signal_id, "expired_unfilled", now_ms,
                                    price=c_close)
                    if data_dir:
                        log_outcome(data_dir, signal_id, symbol, "expired", hit="unfilled")
                    summary["expired"] += 1
                    log.info("Signal #%d expired unfilled", signal_id)

                    exp_embed = build_expiry_embed(sig, c_close, 0.0, max_age_days)
                    exp_text = format_expiry_update(sig, c_close, 0.0, max_age_days)
                    _send_outcome_alert(conn, cfg, exp_embed, exp_text,
                                        signal_id, tg_msg_id)
                    trade_resolved = True
                continue

            # ── MFE / MAE update ──────────────────────────────── #
            if direction > 0:
                mfe = max(mfe, (c_high - entry_mid) / risk)
                mae = max(mae, (entry_mid - c_low) / risk)
            else:
                mfe = max(mfe, (entry_mid - c_low) / risk)
                mae = max(mae, (c_high - entry_mid) / risk)

            # ── Trailing SL level ─────────────────────────────── #
            if tp2_was_hit and tp1 is not None:
                active_sl = tp1
            elif tp1_was_hit:
                active_sl = entry_mid
            else:
                active_sl = sl

            # ── SL check (candle wick, accurate to real-world limits) ──── #
            sl_hit = (direction > 0 and c_low <= active_sl) or \
                     (direction < 0 and c_high >= active_sl)

            if sl_hit:
                if tp2_was_hit:
                    sig_store.update_status(conn, signal_id, "won")
                    _update_outcome(conn, signal_id, "sl_after_tp2", now_ms,
                                    mfe_r=mfe, mae_r=mae, price=active_sl,
                                    sl_hit_ts=latest_ts)
                    if data_dir:
                        log_outcome(data_dir, signal_id, symbol, "won",
                                    hit="sl_after_tp2", mfe_r=mfe, mae_r=mae)
                    summary["won"] += 1
                    update_streak(conn, "won")
                    embed = build_sl_embed(sig, active_sl,
                                          "trailed_tp1", "+1.60", mfe)
                    text = format_sl_update(sig, active_sl,
                                            "trailed_tp1", "+1.60", mfe)
                elif tp1_was_hit:
                    sig_store.update_status(conn, signal_id, "won")
                    _update_outcome(conn, signal_id, "sl_after_tp1", now_ms,
                                    mfe_r=mfe, mae_r=mae, price=active_sl,
                                    sl_hit_ts=latest_ts)
                    if data_dir:
                        log_outcome(data_dir, signal_id, symbol, "won",
                                    hit="sl_after_tp1", mfe_r=mfe, mae_r=mae)
                    summary["won"] += 1
                    update_streak(conn, "won")
                    embed = build_sl_embed(sig, entry_mid, "breakeven", "+0.50", mfe)
                    text = format_sl_update(sig, entry_mid, "breakeven", "+0.50", mfe)
                else:
                    sig_store.update_status(conn, signal_id, "lost")
                    _update_outcome(conn, signal_id, "sl", now_ms,
                                    mfe_r=mfe, mae_r=mae, price=active_sl,
                                    sl_hit_ts=latest_ts)
                    if data_dir:
                        log_outcome(data_dir, signal_id, symbol, "lost",
                                    hit="sl", mfe_r=mfe, mae_r=mae)
                    summary["lost"] += 1
                    update_streak(conn, "lost")
                    embed = build_sl_embed(sig, active_sl, "loss", "-1.00", mfe)
                    text = format_sl_update(sig, active_sl, "loss", "-1.00", mfe)

                _send_outcome_alert(conn, cfg, embed, text, signal_id, tg_msg_id)
                log.info("Signal #%d SL hit at %.4f (ts=%d)", signal_id, active_sl, latest_ts)
                trade_resolved = True
                break  # Stop replaying — trade is closed

            # ── TP checks (candle HIGH/LOW for long/short) ────────── #
            tp_hit = None
            if tp3 and ((direction > 0 and c_high >= tp3) or (direction < 0 and c_low <= tp3)):
                tp_hit = "tp3"
            elif tp2 and ((direction > 0 and c_high >= tp2) or (direction < 0 and c_low <= tp2)):
                tp_hit = "tp2"
            elif tp1 and ((direction > 0 and c_high >= tp1) or (direction < 0 and c_low <= tp1)):
                tp_hit = "tp1"

            if tp_hit:
                tp_ts_col = f"{tp_hit}_hit_ts"
                already_hit = outcome.get(tp_ts_col)

                if not already_hit:
                    conn.execute(
                        f"UPDATE outcomes SET {tp_ts_col} = ? WHERE signal_id = ?",
                        (latest_ts, signal_id)
                    )
                    # Update local state for rest of replay
                    if tp_hit == "tp1":
                        tp1_was_hit = latest_ts
                    elif tp_hit == "tp2":
                        tp2_was_hit = latest_ts

                    r_mult = {"tp1": "1.0", "tp2": "2.0", "tp3": "3.0"}.get(tp_hit, "?")

                    if tp_hit == "tp3":
                        sig_store.update_status(conn, signal_id, "won")
                        _update_outcome(conn, signal_id, tp_hit, now_ms,
                                        mfe_r=mfe, mae_r=mae, price=tp3)
                        if data_dir:
                            log_outcome(data_dir, signal_id, symbol, "won",
                                        hit=tp_hit, mfe_r=mfe, mae_r=mae)
                        summary["won"] += 1
                        update_streak(conn, "won")
                        embed = build_tp_embed(sig, "tp3", "3.0", tp3,
                                               "Trade Closed (Target Met)",
                                               "100% Locked (+3.0R) 🎉")
                        text = format_tp_update(sig, "tp3", "3.0", tp3,
                                                "Trade Closed (Target Met)",
                                                "100% Locked (+3.0R) \U0001f389")
                        _send_outcome_alert(conn, cfg, embed, text, signal_id, tg_msg_id)
                        log.info("Signal #%d TP3 hit — FULL WIN (ts=%d)", signal_id, latest_ts)
                        trade_resolved = True
                        break

                    elif tp_hit == "tp2":
                        _update_outcome(conn, signal_id, "open", now_ms,
                                        mfe_r=mfe, mae_r=mae, price=tp2)
                        sl_action = f"SL Trailed to TP1 level ({_fmt_price(tp1, symbol)})"
                        embed = build_tp_embed(sig, "tp2", "2.0", tp2, sl_action,
                                               "80% Locked (Holding 20% for TP3)")
                        text = format_tp_update(sig, "tp2", "2.0", tp2, sl_action,
                                                "80% Locked (Holding 20% for TP3)")
                        _send_outcome_alert(conn, cfg, embed, text, signal_id, tg_msg_id)
                        log.info("Signal #%d TP2 hit (ts=%d)", signal_id, latest_ts)

                    elif tp_hit == "tp1":
                        _update_outcome(conn, signal_id, "open", now_ms,
                                        mfe_r=mfe, mae_r=mae, price=tp1)
                        sl_action = f"SL Moved to Breakeven ({_fmt_price(entry_mid, symbol)})"
                        embed = build_tp_embed(sig, "tp1", "1.0", tp1, sl_action,
                                               "50% Banked (Risk-Free Trade)")
                        text = format_tp_update(sig, "tp1", "1.0", tp1, sl_action,
                                                "50% Banked (Risk-Free Trade)")
                        _send_outcome_alert(conn, cfg, embed, text, signal_id, tg_msg_id)
                        log.info("Signal #%d TP1 hit (ts=%d)", signal_id, latest_ts)

            # End of candle loop — continue to next candle

        if trade_resolved:
            continue

        # ── Expiry check for filled signals ──────────────────────── #
        age_ms = now_ms - sig["ts"]
        if entry_filled and age_ms > max_age_days * 86400 * 1000:
            # Use last replayed candle price
            c_close = candles[-1]["close"] if candles else 0.0
            current_r = (c_close - entry_mid) / risk if direction > 0 \
                        else (entry_mid - c_close) / risk
            status_str = "won" if current_r >= 0.5 else \
                         "lost" if current_r <= -0.5 else "expired"
            sig_store.update_status(conn, signal_id, status_str)
            _update_outcome(conn, signal_id, "expired", now_ms,
                            mfe_r=mfe, mae_r=mae, price=c_close)
            if data_dir:
                log_outcome(data_dir, signal_id, symbol, status_str,
                            hit="expired", mfe_r=mfe, mae_r=mae)
            if status_str == "won":
                summary["won"] += 1
                update_streak(conn, "won")
            elif status_str == "lost":
                summary["lost"] += 1
                update_streak(conn, "lost")
            else:
                summary["expired"] += 1

            exp_embed = build_expiry_embed(sig, c_close, current_r, max_age_days)
            exp_text = format_expiry_update(sig, c_close, current_r, max_age_days)
            _send_outcome_alert(conn, cfg, exp_embed, exp_text, signal_id, tg_msg_id)
            log.info("Signal #%d expired after %.1f days (%.1fR)", signal_id, max_age_days, current_r)
            continue

        # ── Still open — update tracking ─────────────────────────── #
        c_close = candles[-1]["close"] if candles else 0.0
        _update_outcome(conn, signal_id, "open", now_ms,
                        mfe_r=mfe, mae_r=mae, price=c_close,
                        last_checked_ts=latest_ts)
        summary["still_open"] += 1

    # Fix 7: Single commit covers the full batch of all outcome updates
    conn.commit()

    log.info("Outcome check: %d checked, %d won, %d lost, %d expired, %d open",
             summary["checked"], summary["won"], summary["lost"],
             summary["expired"], summary["still_open"])
    return summary


# ══════════════════════════════════════════════════════════════════════ #
# DB write helper
# ══════════════════════════════════════════════════════════════════════ #

def _update_outcome(conn: sqlite3.Connection, signal_id: int,
                    hit: str, checked_ts: int,
                    mfe_r: float = 0, mae_r: float = 0,
                    price: float | None = None,
                    sl_hit_ts: int | None = None,
                    tp1_hit_ts: int | None = None,
                    tp2_hit_ts: int | None = None,
                    tp3_hit_ts: int | None = None,
                    last_checked_ts: int | None = None) -> None:
    """Update the outcome tracking row.

    Fix 7: Does NOT call conn.commit() — caller batches commits.
    """
    _ensure_outcome_row(conn, signal_id)

    parts = ["checked_ts = ?", "hit = ?"]
    vals: list[Any] = [checked_ts, hit]

    if mfe_r:
        parts.append("mfe_r = MAX(COALESCE(mfe_r, 0), ?)")
        vals.append(mfe_r)
    if mae_r:
        parts.append("mae_r = MAX(COALESCE(mae_r, 0), ?)")
        vals.append(mae_r)
    if price is not None:
        parts.append("price_at_check = ?")
        vals.append(price)
    if sl_hit_ts is not None:
        parts.append("sl_hit_ts = ?")
        vals.append(sl_hit_ts)
    if tp1_hit_ts is not None:
        parts.append("tp1_hit_ts = ?")
        vals.append(tp1_hit_ts)
    if tp2_hit_ts is not None:
        parts.append("tp2_hit_ts = ?")
        vals.append(tp2_hit_ts)
    if tp3_hit_ts is not None:
        parts.append("tp3_hit_ts = ?")
        vals.append(tp3_hit_ts)
    if last_checked_ts is not None:
        parts.append("last_outcome_checked_ts = ?")
        vals.append(last_checked_ts)

    vals.append(signal_id)
    sql = f"UPDATE outcomes SET {', '.join(parts)} WHERE signal_id = ?"
    conn.execute(sql, vals)
    # NOTE: caller is responsible for committing — do NOT commit here.
    # A single conn.commit() at the end of check_outcomes() covers the full batch.
