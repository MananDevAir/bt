"""Walk-forward backtest engine.

Replays historical candles bar-by-bar, using ONLY closed bars visible
at each step (no lookahead). Generates the same signals the live bot
would have produced, then scores them against what price actually did.

Usage:
    python -m src.backtest.replay                    # all symbols, last 60 days
    python -m src.backtest.replay --symbol BTC       # one symbol
    python -m src.backtest.replay --days 30          # last 30 days
    python -m src.backtest.replay --step 4h          # step every 4h (default)
"""
from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import Config
from ..analysis.confluence import score_symbol
from ..analysis.levels import generate_plan
from ..scanner import _is_session_active

log = logging.getLogger(__name__)

__all__ = ["replay_symbol", "replay_all", "BacktestResult"]

# Step durations in milliseconds
STEP_MS = {
    "15m": 15 * 60_000,
    "1h": 3600_000,
    "4h": 4 * 3600_000,
    "1d": 86400_000,
}

# Minimum bars needed per timeframe for indicators to warm up
MIN_WARMUP = {
    "1w": 30,
    "1d": 100,
    "4h": 100,
    "1h": 100,
    "15m": 100,
}


@dataclass
class BacktestSignal:
    """One signal emitted during backtest."""
    ts: int                   # emission timestamp (ms)
    symbol: str
    direction: int            # +1 long, -1 short
    label: str
    score: float
    trade_type: str
    entry_low: float
    entry_high: float
    sl: float
    tp1: float
    tp2: float | None
    tp3: float | None
    rr: float
    brief_reason: str

    # Filled by outcome evaluation
    outcome: str = "pending"  # won / lost / expired / no_entry
    hit: str = ""             # tp1 / tp2 / tp3 / sl
    mfe_r: float = 0.0       # max favorable excursion in R
    mae_r: float = 0.0       # max adverse excursion in R
    entry_filled: bool = False
    bars_to_entry: int = 0
    bars_to_exit: int = 0


@dataclass
class BacktestResult:
    """Aggregate backtest result for one or many symbols."""
    symbol: str               # or "ALL"
    signals: list[BacktestSignal] = field(default_factory=list)
    start_date: str = ""
    end_date: str = ""
    step: str = "4h"
    bars_walked: int = 0
    duration_s: float = 0.0

    @property
    def total(self) -> int:
        return len(self.signals)

    @property
    def entered(self) -> list[BacktestSignal]:
        return [s for s in self.signals if s.entry_filled]

    @property
    def won(self) -> int:
        return sum(1 for s in self.signals if s.outcome == "won")

    @property
    def lost(self) -> int:
        return sum(1 for s in self.signals if s.outcome == "lost")

    @property
    def expired(self) -> int:
        return sum(1 for s in self.signals if s.outcome == "expired")

    @property
    def no_entry(self) -> int:
        return sum(1 for s in self.signals if s.outcome == "no_entry")

    @property
    def win_rate(self) -> float:
        closed = self.won + self.lost
        return (self.won / closed * 100) if closed > 0 else 0.0

    @property
    def avg_mfe_r(self) -> float:
        entered = self.entered
        return sum(s.mfe_r for s in entered) / len(entered) if entered else 0.0

    @property
    def avg_mae_r(self) -> float:
        entered = self.entered
        return sum(s.mae_r for s in entered) / len(entered) if entered else 0.0

    @property
    def best_r(self) -> float:
        return max((s.mfe_r for s in self.entered), default=0)

    @property
    def worst_r(self) -> float:
        return max((s.mae_r for s in self.entered), default=0)

    @property
    def by_label(self) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for s in self.signals:
            d = result.setdefault(s.label, {"total": 0, "won": 0, "lost": 0})
            d["total"] += 1
            if s.outcome == "won":
                d["won"] += 1
            elif s.outcome == "lost":
                d["lost"] += 1
        return result

    @property
    def by_trade_type(self) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for s in self.signals:
            d = result.setdefault(s.trade_type, {"total": 0, "won": 0, "lost": 0})
            d["total"] += 1
            if s.outcome == "won":
                d["won"] += 1
            elif s.outcome == "lost":
                d["lost"] += 1
        return result


def _load_candles(conn: sqlite3.Connection, symbol: str,
                  timeframe: str) -> pd.DataFrame:
    """Load all candles for a symbol+timeframe from DB."""
    rows = conn.execute(
        "SELECT ts, open, high, low, close, volume FROM candles "
        "WHERE symbol = ? AND timeframe = ? ORDER BY ts ASC",
        (symbol, timeframe)
    ).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    df = df.astype(float)
    return df


def _evaluate_outcome(sig: BacktestSignal, future_candles: pd.DataFrame,
                      max_bars: int = 96) -> None:
    """Check if a signal would have been a win/loss using future price bars.

    Uses the smallest available timeframe bars for outcome evaluation.
    max_bars: maximum bars to look ahead (default 96 = 96×4h = 16 days)
    """
    if future_candles.empty:
        sig.outcome = "expired"
        return

    entry_mid = (sig.entry_low + sig.entry_high) / 2
    risk = abs(entry_mid - sig.sl)
    if risk <= 0:
        sig.outcome = "expired"
        return

    direction = sig.direction

    # Phase 1: Check if entry zone was reached
    for i, (_, bar) in enumerate(future_candles.iloc[:max_bars].iterrows()):
        # Entry fill check
        if not sig.entry_filled:
            if direction > 0:
                if bar["low"] <= sig.entry_high:
                    sig.entry_filled = True
                    sig.bars_to_entry = i + 1
            else:
                if bar["high"] >= sig.entry_low:
                    sig.entry_filled = True
                    sig.bars_to_entry = i + 1

        if not sig.entry_filled:
            continue

        # Track MFE / MAE
        if direction > 0:
            bar_mfe = max(0, (bar["high"] - entry_mid) / risk)
            bar_mae = max(0, (entry_mid - bar["low"]) / risk)
        else:
            bar_mfe = max(0, (entry_mid - bar["low"]) / risk)
            bar_mae = max(0, (bar["high"] - entry_mid) / risk)

        sig.mfe_r = max(sig.mfe_r, bar_mfe)
        sig.mae_r = max(sig.mae_r, bar_mae)

        # SL check (always first — SL checked before TP on same bar)
        if direction > 0 and bar["low"] <= sig.sl:
            sig.outcome = "lost"
            sig.hit = "sl"
            sig.bars_to_exit = i + 1
            return
        elif direction < 0 and bar["high"] >= sig.sl:
            sig.outcome = "lost"
            sig.hit = "sl"
            sig.bars_to_exit = i + 1
            return

        # TP checks (best to worst)
        if sig.tp3 and direction > 0 and bar["high"] >= sig.tp3:
            sig.outcome = "won"
            sig.hit = "tp3"
            sig.bars_to_exit = i + 1
            return
        if sig.tp3 and direction < 0 and bar["low"] <= sig.tp3:
            sig.outcome = "won"
            sig.hit = "tp3"
            sig.bars_to_exit = i + 1
            return
        if sig.tp2 and direction > 0 and bar["high"] >= sig.tp2:
            sig.outcome = "won"
            sig.hit = "tp2"
            sig.bars_to_exit = i + 1
            return
        if sig.tp2 and direction < 0 and bar["low"] <= sig.tp2:
            sig.outcome = "won"
            sig.hit = "tp2"
            sig.bars_to_exit = i + 1
            return
        if sig.tp1 and direction > 0 and bar["high"] >= sig.tp1:
            sig.outcome = "won"
            sig.hit = "tp1"
            sig.bars_to_exit = i + 1
            return
        if sig.tp1 and direction < 0 and bar["low"] <= sig.tp1:
            sig.outcome = "won"
            sig.hit = "tp1"
            sig.bars_to_exit = i + 1
            return

    # Ran out of bars
    if sig.entry_filled:
        sig.outcome = "expired"
    else:
        sig.outcome = "no_entry"


def replay_symbol(conn: sqlite3.Connection, cfg: Config,
                  symbol: str, days: int = 60,
                  step: str = "4h") -> BacktestResult:
    """Walk-forward replay for one symbol.

    Steps through time at `step` intervals, computing scores using
    only candles closed BEFORE the current step (no lookahead).
    """
    timeframes = list(cfg.get("timeframes", default={}).values())
    # Flatten any lists
    flat_tfs = []
    for tf in timeframes:
        if isinstance(tf, list):
            flat_tfs.extend(tf)
        else:
            flat_tfs.append(tf)

    # Load all candles
    all_candles: dict[str, pd.DataFrame] = {}
    for tf in flat_tfs:
        df = _load_candles(conn, symbol, tf)
        if not df.empty:
            all_candles[tf] = df

    if not all_candles:
        log.warning("No candles for %s", symbol)
        return BacktestResult(symbol=symbol)

    # Determine replay window
    step_ms = STEP_MS.get(step, STEP_MS["4h"])

    # Use the step timeframe to determine time range
    step_df = all_candles.get(step)
    if step_df is None or step_df.empty:
        # Fall back to 4h
        step_df = all_candles.get("4h")
    if step_df is None or step_df.empty:
        log.warning("No step-timeframe data for %s", symbol)
        return BacktestResult(symbol=symbol)

    # Replay the last `days` of data, leaving warmup before
    end_ts = step_df.index[-1]
    start_ts = end_ts - timedelta(days=days)

    # Ensure warmup: we need enough bars before start_ts for indicators
    warmup_bars = MIN_WARMUP.get(step, 100)
    replay_mask = step_df.index >= start_ts
    if replay_mask.sum() < 10:
        log.warning("Not enough data for %s (only %d bars in window)",
                     symbol, replay_mask.sum())
        return BacktestResult(symbol=symbol)

    replay_timestamps = step_df.index[replay_mask]

    result = BacktestResult(
        symbol=symbol,
        start_date=start_ts.strftime("%Y-%m-%d"),
        end_date=end_ts.strftime("%Y-%m-%d"),
        step=step,
    )

    t0 = time.time()
    cooldown_until: dict[int, pd.Timestamp] = {}  # direction -> cooldown_end
    cooldown_hours = int(cfg.get("gates", "cooldown_hours", default=4) or 4)

    sym_obj = next((s for s in cfg.symbols if s.name == symbol), None)
    session = sym_obj.session if sym_obj else "always"

    for step_ts in replay_timestamps:
        if not _is_session_active(session, step_ts):
            continue

        result.bars_walked += 1

        # Slice candles: only bars with index < step_ts (closed candles)
        frames: dict[str, pd.DataFrame] = {}
        for tf, df in all_candles.items():
            visible = df[df.index < step_ts]
            if len(visible) >= 30:  # minimum for indicators
                frames[tf] = visible

        if len(frames) < 2:
            continue

        # Score
        try:
            signal = score_symbol(frames, symbol, cfg)
        except Exception as exc:
            log.debug("Score error at %s for %s: %s", step_ts, symbol, exc)
            continue

        # Skip neutrals
        if signal.label == "NEUTRAL" or signal.direction == 0:
            continue

        # Cooldown check
        cd_end = cooldown_until.get(signal.direction)
        if cd_end and step_ts < cd_end:
            continue

        # Generate plan
        plan = generate_plan(signal, cfg)
        if plan is None:
            continue

        # Record signal
        bt_sig = BacktestSignal(
            ts=int(step_ts.timestamp() * 1000),
            symbol=symbol,
            direction=signal.direction,
            label=signal.label,
            score=signal.score,
            trade_type=plan.trade_type,
            entry_low=plan.entry_low,
            entry_high=plan.entry_high,
            sl=plan.sl,
            tp1=plan.tp1,
            tp2=plan.tp2,
            tp3=plan.tp3,
            rr=plan.rr,
            brief_reason=plan.brief_reason,
        )

        # Evaluate outcome against future bars
        # Use the smallest available TF for precision
        eval_tf = "1h" if "1h" in all_candles else "4h" if "4h" in all_candles else step
        eval_df = all_candles.get(eval_tf, step_df)
        future = eval_df[eval_df.index >= step_ts]
        _evaluate_outcome(bt_sig, future)

        result.signals.append(bt_sig)

        # Set cooldown
        cooldown_until[signal.direction] = step_ts + timedelta(hours=cooldown_hours)

    result.duration_s = round(time.time() - t0, 1)
    return result


def replay_all(conn: sqlite3.Connection, cfg: Config,
               days: int = 60, step: str = "4h",
               symbols: list[str] | None = None) -> BacktestResult:
    """Replay all symbols and merge into one result."""
    sym_names = symbols or [s.name for s in cfg.symbols]
    combined = BacktestResult(symbol="ALL", step=step)

    t0 = time.time()
    for i, sym_name in enumerate(sym_names, 1):
        sym_start = time.time()
        print(f"  [{i}/{len(sym_names)}] {sym_name}...", end="", flush=True)

        r = replay_symbol(conn, cfg, sym_name, days=days, step=step)
        combined.signals.extend(r.signals)
        combined.bars_walked += r.bars_walked

        # Progress
        elapsed = time.time() - sym_start
        won = sum(1 for s in r.signals if s.outcome == "won")
        lost = sum(1 for s in r.signals if s.outcome == "lost")
        print(f"  {r.total} signals ({won}W/{lost}L)  [{elapsed:.0f}s]", flush=True)

        if r.start_date:
            if not combined.start_date or r.start_date < combined.start_date:
                combined.start_date = r.start_date
            if not combined.end_date or r.end_date > combined.end_date:
                combined.end_date = r.end_date

    combined.duration_s = round(time.time() - t0, 1)
    return combined
