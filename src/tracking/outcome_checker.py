"""Outcome checker — tracks open signals and updates their status.

On each check cycle:
  1. Load all open signals from DB
  2. Fetch current price for each symbol
  3. Determine if entry was filled, then check SL/TP hits
  4. Update MFE/MAE (max favorable/adverse excursion in R)
  5. Expire signals that are too old
  6. Write outcome records to DB + JSONL log
"""
from __future__ import annotations

import logging
import sqlite3
import time
from typing import Any

from ..config import Config
from ..store import signals as sig_store
from ..logging_util import log_outcome
from ..alerts.telegram import send_text
from ..data.live_price import get_live_price
from .streaks import update_streak

log = logging.getLogger(__name__)

__all__ = ["check_outcomes"]

# Expiry: signals older than this (hours) are auto-expired
DEFAULT_EXPIRY_H = 48


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


def _ensure_outcome_row(conn: sqlite3.Connection, signal_id: int) -> None:
    """Create an outcome row if it doesn't exist yet."""
    exists = conn.execute(
        "SELECT 1 FROM outcomes WHERE signal_id = ?", (signal_id,)
    ).fetchone()
    if not exists:
        conn.execute(
            "INSERT INTO outcomes (signal_id, checked_ts, hit, entry_filled) "
            "VALUES (?, ?, 'open', 0)",
            (signal_id, int(time.time() * 1000))
        )
        conn.commit()


def check_outcomes(conn: sqlite3.Connection, cfg: Config,
                   data_dir: Any = None) -> dict[str, Any]:
    """Check all open signals against current prices.

    Returns a summary dict with counts.
    """
    open_signals = sig_store.get_open_signals(conn)
    if not open_signals:
        return {"checked": 0, "won": 0, "lost": 0, "expired": 0}

    now_ms = int(time.time() * 1000)
    expiry_ms = DEFAULT_EXPIRY_H * 3600 * 1000
    summary = {"checked": 0, "won": 0, "lost": 0, "expired": 0, "still_open": 0}
    hit_alerts: list[str] = []  # Telegram follow-up messages
    streak_updates: list[str] = []  # track streak changes

    for sig in open_signals:
        signal_id = sig["id"]
        symbol = sig["symbol"]
        direction = 1 if sig["direction"] == "long" else -1

        # Check expiry
        age_ms = now_ms - sig["ts"]
        if age_ms > expiry_ms:
            sig_store.update_status(conn, signal_id, "expired")
            _update_outcome(conn, signal_id, "expired", now_ms)
            if data_dir:
                log_outcome(data_dir, signal_id, symbol, "expired")
            summary["expired"] += 1
            log.info("Signal #%d expired (%s)", signal_id, symbol)
            continue

        # Get current price — prefer live API over stale cache
        candle = get_live_price(symbol)
        if candle is None:
            # Fallback to cached candle if live fetch fails
            candle = _get_latest_candle(conn, symbol)
        if candle is None:
            log.debug("No price data for %s, skipping signal #%d", symbol, signal_id)
            summary["still_open"] += 1
            continue

        price = candle["close"]
        high = candle["high"]
        low = candle["low"]

        summary["checked"] += 1
        _ensure_outcome_row(conn, signal_id)

        entry_mid = (sig["entry_low"] + sig["entry_high"]) / 2 if sig["entry_low"] and sig["entry_high"] else None
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

        # Check if entry was filled
        outcome = conn.execute(
            "SELECT * FROM outcomes WHERE signal_id = ?", (signal_id,)
        ).fetchone()
        entry_filled = outcome["entry_filled"] if outcome else 0

        if not entry_filled:
            # Check if price reached entry zone
            if direction > 0 and low <= sig["entry_high"]:
                entry_filled = 1
            elif direction < 0 and high >= sig["entry_low"]:
                entry_filled = 1

            if entry_filled:
                conn.execute(
                    "UPDATE outcomes SET entry_filled = 1 WHERE signal_id = ?",
                    (signal_id,)
                )
                conn.commit()
                log.info("Signal #%d entry filled at %.2f", signal_id, price)

        if not entry_filled:
            # Price hasn't reached entry zone yet
            summary["still_open"] += 1
            _update_outcome(conn, signal_id, "open", now_ms, price=price)
            continue

        # Calculate MFE/MAE using high/low for true excursion
        if direction > 0:
            mfe = max(0, (high - entry_mid) / risk)   # best case: candle high
            mae = max(0, (entry_mid - low) / risk)    # worst case: candle low
        else:
            mfe = max(0, (entry_mid - low) / risk)    # best case: candle low
            mae = max(0, (high - entry_mid) / risk)   # worst case: candle high

        # Update running MFE/MAE
        old_mfe = outcome["mfe_r"] if outcome and outcome["mfe_r"] else 0
        old_mae = outcome["mae_r"] if outcome and outcome["mae_r"] else 0
        new_mfe = max(old_mfe, mfe)
        new_mae = max(old_mae, mae)

        # Check SL hit
        sl_hit_long = direction > 0 and low <= sl
        sl_hit_short = direction < 0 and high >= sl
        
        if sl_hit_long or sl_hit_short:
            # Check if TP2 or TP1 was already hit
            tp2_was_hit = outcome.get("tp2_hit_ts") if outcome else None
            tp1_was_hit = outcome.get("tp1_hit_ts") if outcome else None

            if tp2_was_hit:
                # TP2 was hit, SL was trailed to TP1 — this is a solid win
                sig_store.update_status(conn, signal_id, "won")
                _update_outcome(conn, signal_id, "sl_after_tp2", now_ms,
                                mfe_r=new_mfe, mae_r=new_mae, price=price,
                                sl_hit_ts=now_ms)
                if data_dir:
                    log_outcome(data_dir, signal_id, symbol, "won",
                                hit="sl_after_tp2", mfe_r=new_mfe, mae_r=new_mae)
                summary["won"] += 1
                update_streak(conn, "won")
                hit_alerts.append(
                    f"\U0001f7e2 <b>{symbol} TRAILED SL HIT (TP1 level)</b>\n"
                    f"  Price: {price:,.2f}  |  MFE: {new_mfe:.1f}R\n"
                    f"  Signal #{signal_id} closed as <b>WIN</b> (80% profit locked)"
                )
                log.info("Signal #%d SL hit at TP1 level after TP2 (%s)", signal_id, symbol)
            elif tp1_was_hit:
                # TP1 was hit, SL moved to breakeven — this is a partial win
                sig_store.update_status(conn, signal_id, "won")
                _update_outcome(conn, signal_id, "sl_after_tp1", now_ms,
                                mfe_r=new_mfe, mae_r=new_mae, price=price,
                                sl_hit_ts=now_ms)
                if data_dir:
                    log_outcome(data_dir, signal_id, symbol, "won",
                                hit="sl_after_tp1", mfe_r=new_mfe, mae_r=new_mae)
                summary["won"] += 1
                update_streak(conn, "won")
                hit_alerts.append(
                    f"\U0001f7e1 <b>{symbol} SL HIT (breakeven)</b>\n"
                    f"  Price: {price:,.2f}  |  MFE: {new_mfe:.1f}R\n"
                    f"  Signal #{signal_id} closed as <b>PARTIAL WIN</b> (TP1 banked)"
                )
                log.info("Signal #%d SL hit at breakeven after TP1 (%s)", signal_id, symbol)
            else:
                # Normal loss — TP1 was never reached
                sig_store.update_status(conn, signal_id, "lost")
                _update_outcome(conn, signal_id, "sl", now_ms,
                                mfe_r=new_mfe, mae_r=new_mae, price=price,
                                sl_hit_ts=now_ms)
                if data_dir:
                    log_outcome(data_dir, signal_id, symbol, "lost",
                                hit="sl", mfe_r=new_mfe, mae_r=new_mae)
                summary["lost"] += 1
                update_streak(conn, "lost")
                hit_alerts.append(
                    f"\U0001f534 <b>{symbol} SL HIT</b>\n"
                    f"  Price: {price:,.2f}  |  MFE: {new_mfe:.1f}R\n"
                    f"  Signal #{signal_id} closed as <b>LOSS</b>"
                )
                log.info("Signal #%d SL hit at %.2f (%s)", signal_id, price, symbol)
            continue

        # Check TP hits (track individually with partial management)
        tp_hit = None
        if tp3 and ((direction > 0 and high >= tp3) or (direction < 0 and low <= tp3)):
            tp_hit = "tp3"
        elif tp2 and ((direction > 0 and high >= tp2) or (direction < 0 and low <= tp2)):
            tp_hit = "tp2"
        elif tp1 and ((direction > 0 and high >= tp1) or (direction < 0 and low <= tp1)):
            tp_hit = "tp1"

        if tp_hit:
            # Record individual TP timestamps
            tp_ts_col = f"{tp_hit}_hit_ts"
            already_hit = outcome.get(tp_ts_col) if outcome else None
            if not already_hit:
                conn.execute(
                    f"UPDATE outcomes SET {tp_ts_col} = ? WHERE signal_id = ?",
                    (now_ms, signal_id)
                )

            r_mult = {"tp1": "1.0", "tp2": "2.0", "tp3": "3.0"}.get(tp_hit, "?")

            # Check which TPs were already hit before
            tp1_already = outcome.get("tp1_hit_ts") if outcome else None
            tp2_already = outcome.get("tp2_hit_ts") if outcome else None

            if tp_hit == "tp3":
                # TP3 = FULL WIN — close the trade
                sig_store.update_status(conn, signal_id, "won")
                _update_outcome(conn, signal_id, tp_hit, now_ms,
                                mfe_r=new_mfe, mae_r=new_mae, price=price)
                if data_dir:
                    log_outcome(data_dir, signal_id, symbol, "won",
                                hit=tp_hit, mfe_r=new_mfe, mae_r=new_mae)
                summary["won"] += 1
                update_streak(conn, "won")
                hit_alerts.append(
                    f"\U0001f7e2 <b>{symbol} TP3 HIT \u2014 FULL WIN!</b>\n"
                    f"  Price: {price:,.2f}  |  +{r_mult}R\n"
                    f"  Signal #{signal_id} closed as <b>WIN</b> \U0001f389"
                )
                log.info("Signal #%d TP3 hit at %.2f (%s) \u2014 FULL WIN",
                         signal_id, price, symbol)
                continue

            elif tp_hit == "tp2" and not tp2_already:
                # TP2 hit for the first time — trail SL to TP1 level and keep tracking for TP3
                if tp1:
                    conn.execute(
                        "UPDATE signals SET sl = ? WHERE id = ?",
                        (tp1, signal_id)
                    )
                _update_outcome(conn, signal_id, "open", now_ms,
                                mfe_r=new_mfe, mae_r=new_mae, price=price)
                hit_alerts.append(
                    f"\U0001f7e2 <b>{symbol} TP2 HIT \u2014 +{r_mult}R</b>\n"
                    f"  Price: {price:,.2f}  |  SL \u2192 TP1 level ({tp1:,.2f})\n"
                    f"  Holding 20% for TP3 (80% banked) \u2705"
                )
                log.info("Signal #%d TP2 hit at %.2f (%s) \u2014 SL trailed to TP1, tracking TP3",
                         signal_id, price, symbol)
                summary["still_open"] += 1
                continue

            elif tp_hit == "tp1" and not tp1_already:
                # TP1 hit for the first time — move SL to breakeven
                conn.execute(
                    "UPDATE signals SET sl = ? WHERE id = ?",
                    (entry_mid, signal_id)
                )
                _update_outcome(conn, signal_id, "open", now_ms,
                                mfe_r=new_mfe, mae_r=new_mae, price=price)
                hit_alerts.append(
                    f"\U0001f7e2 <b>{symbol} TP1 HIT \u2014 +{r_mult}R</b>\n"
                    f"  Price: {price:,.2f}  |  SL \u2192 breakeven ({entry_mid:,.2f})\n"
                    f"  Signal #{signal_id} still <b>OPEN</b> (50% closed)"
                )
                log.info("Signal #%d TP1 hit at %.2f (%s) \u2014 SL moved to breakeven",
                         signal_id, price, symbol)
                summary["still_open"] += 1
                continue

            else:
                # Already alerted for this TP level, just update tracking
                _update_outcome(conn, signal_id, "open", now_ms,
                                mfe_r=new_mfe, mae_r=new_mae, price=price)
                summary["still_open"] += 1
                continue

        # Still open — update tracking
        _update_outcome(conn, signal_id, "open", now_ms,
                        mfe_r=new_mfe, mae_r=new_mae, price=price)
        summary["still_open"] += 1

    conn.commit()

    # Send follow-up alerts to Telegram
    if hit_alerts:
        from datetime import datetime, timezone, timedelta
        IST = timezone(timedelta(hours=5, minutes=30))
        ist_now = datetime.now(IST).strftime("%d %b, %I:%M %p IST")
        combined = "\n\n".join(hit_alerts)
        msg = f"{combined}\n\n\U0001f552 {ist_now}"
        try:
            send_text(msg)
            log.info("Sent %d outcome alert(s)", len(hit_alerts))
        except Exception as exc:
            log.warning("Failed to send outcome alerts: %s", exc)

    log.info("Outcome check: %d checked, %d won, %d lost, %d expired, %d open",
             summary["checked"], summary["won"], summary["lost"],
             summary["expired"], summary["still_open"])
    return summary


def _update_outcome(conn: sqlite3.Connection, signal_id: int,
                    hit: str, checked_ts: int,
                    mfe_r: float = 0, mae_r: float = 0,
                    price: float | None = None,
                    sl_hit_ts: int | None = None) -> None:
    """Update the outcome tracking row."""
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

    vals.append(signal_id)
    sql = f"UPDATE outcomes SET {', '.join(parts)} WHERE signal_id = ?"
    conn.execute(sql, vals)
