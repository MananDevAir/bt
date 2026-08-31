"""Database maintenance — smart cleanup that preserves backtest data.

PRINCIPLE: Never delete signals, outcomes, or candles.
           These ARE the backtest dataset. Deleting them = losing history.

What we DO clean:
    - Expired mutes
    - Old api_usage counters (just accounting, not analysis data)
    - WAL checkpoint (compact the WAL file)
    - VACUUM (reclaim space from natural SQLite fragmentation)

What we NEVER clean:
    - Candles     → needed for backtesting + indicator computation
    - Signals     → the core performance record
    - Outcomes    → the core performance record
    - JSONL logs  → immutable audit trail (compress, never delete)

Run:
    python -m src.maintenance               # show DB stats
    python -m src.maintenance --compact      # checkpoint WAL + VACUUM
    python -m src.maintenance --archive-logs # compress old JSONL files
"""
from __future__ import annotations

import gzip
import logging
import os
import shutil
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

__all__ = ["cleanup", "vacuum", "archive_logs", "db_stats"]


def db_stats(conn: sqlite3.Connection, data_dir: Path) -> dict[str, Any]:
    """Gather comprehensive DB statistics for monitoring."""
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    db_size = os.path.getsize(db_path) / 1024 / 1024

    candle_count = conn.execute("SELECT COUNT(*) FROM candles").fetchone()[0]
    signal_count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    outcome_count = conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
    open_signals = conn.execute(
        "SELECT COUNT(*) FROM signals WHERE status = 'open'"
    ).fetchone()[0]
    won = conn.execute(
        "SELECT COUNT(*) FROM signals WHERE status = 'won'"
    ).fetchone()[0]
    lost = conn.execute(
        "SELECT COUNT(*) FROM signals WHERE status = 'lost'"
    ).fetchone()[0]

    # Candle date range
    earliest = conn.execute(
        "SELECT MIN(ts) FROM candles"
    ).fetchone()[0]
    latest = conn.execute(
        "SELECT MAX(ts) FROM candles"
    ).fetchone()[0]

    earliest_dt = datetime.fromtimestamp(earliest / 1000, tz=timezone.utc) if earliest else None
    latest_dt = datetime.fromtimestamp(latest / 1000, tz=timezone.utc) if latest else None

    # Per-symbol candle counts
    per_symbol: dict[str, int] = {}
    rows = conn.execute(
        "SELECT symbol, COUNT(*) as cnt FROM candles GROUP BY symbol"
    ).fetchall()
    for r in rows:
        per_symbol[r["symbol"]] = r["cnt"]

    # JSONL sizes
    jsonl_stats: dict[str, dict] = {}
    for fname in ("signals.jsonl", "outcomes.jsonl", "scans.jsonl"):
        fpath = data_dir / fname
        if fpath.exists():
            sz = fpath.stat().st_size
            lines = sum(1 for _ in open(fpath, encoding="utf-8"))
            jsonl_stats[fname] = {"size_kb": sz / 1024, "records": lines}

    return {
        "db_size_mb": round(db_size, 2),
        "candles": candle_count,
        "signals": signal_count,
        "outcomes": outcome_count,
        "open_signals": open_signals,
        "won": won,
        "lost": lost,
        "candle_range": (earliest_dt, latest_dt),
        "per_symbol": per_symbol,
        "jsonl": jsonl_stats,
    }


def cleanup(conn: sqlite3.Connection, data_dir: Path,
            execute: bool = True) -> dict[str, Any]:
    """Lightweight cleanup — only removes truly ephemeral data.

    SAFE: never touches candles, signals, or outcomes.
    """
    summary: dict[str, Any] = {
        "mutes_expired": 0,
        "api_usage_cleaned": 0,
        "wal_checkpointed": False,
    }

    # 1. Expire old mutes
    now_ms = int(time.time() * 1000)
    expired_mutes = conn.execute(
        "SELECT COUNT(*) FROM mutes WHERE until_ts < ?", (now_ms,)
    ).fetchone()[0]
    if expired_mutes > 0 and execute:
        conn.execute("DELETE FROM mutes WHERE until_ts < ?", (now_ms,))
        conn.commit()
    summary["mutes_expired"] = expired_mutes

    # 2. Clean api_usage older than 30 days (just accounting data)
    cutoff_date = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
    old_usage = conn.execute(
        "SELECT COUNT(*) FROM api_usage WHERE day_utc < ?", (cutoff_date,)
    ).fetchone()[0]
    if old_usage > 0 and execute:
        conn.execute("DELETE FROM api_usage WHERE day_utc < ?", (cutoff_date,))
        conn.commit()
    summary["api_usage_cleaned"] = old_usage

    # 3. WAL checkpoint (merge WAL back into main DB)
    if execute:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        summary["wal_checkpointed"] = True

    return summary


def vacuum(conn: sqlite3.Connection) -> float:
    """Run VACUUM to compact the database file. Returns size saved in MB."""
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    size_before = os.path.getsize(db_path) / 1024 / 1024
    conn.execute("VACUUM")
    size_after = os.path.getsize(db_path) / 1024 / 1024
    saved = size_before - size_after
    log.info("VACUUM: %.2f MB → %.2f MB (saved %.2f MB)",
             size_before, size_after, saved)
    return saved


def archive_logs(data_dir: Path, max_size_mb: float = 10.0) -> list[str]:
    """Compress JSONL files that exceed max_size_mb.

    Archives to data/archive/ as .jsonl.gz — NEVER deletes originals
    until the compressed version is verified.
    """
    archived = []
    archive_dir = data_dir / "archive"

    for fname in ("signals.jsonl", "outcomes.jsonl", "scans.jsonl"):
        fpath = data_dir / fname
        if not fpath.exists():
            continue

        size_mb = fpath.stat().st_size / 1024 / 1024
        if size_mb < max_size_mb:
            continue

        # Compress
        archive_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        gz_name = f"{fpath.stem}_{ts}.jsonl.gz"
        gz_path = archive_dir / gz_name

        with open(fpath, "rb") as f_in:
            with gzip.open(gz_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Verify compressed file
        if gz_path.exists() and gz_path.stat().st_size > 0:
            # Truncate the original (start fresh, don't delete)
            with open(fpath, "w") as f:
                pass  # empty file
            archived.append(f"{fname} ({size_mb:.1f} MB → {gz_path.name})")
            log.info("Archived %s (%.1f MB) → %s", fname, size_mb, gz_name)

    return archived


def main():
    """CLI entry point."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from src import config as config_mod
    from src.store import db

    logging.basicConfig(level=logging.INFO,
                        format="  %(name)-25s %(levelname)-7s %(message)s")

    do_compact = "--compact" in sys.argv
    do_archive = "--archive-logs" in sys.argv

    cfg = config_mod.load()
    conn = db.connect(cfg.db_path)
    data_dir = cfg.db_path.parent

    # Always show stats
    stats = db_stats(conn, data_dir)
    r = stats["candle_range"]

    print(f"\n  Signal Bot — Database Stats")
    print("  " + "=" * 55)
    print(f"  DB size:      {stats['db_size_mb']:.2f} MB")
    print(f"  Candles:      {stats['candles']:,} rows")
    if r[0] and r[1]:
        print(f"  Date range:   {r[0]:%Y-%m-%d} → {r[1]:%Y-%m-%d} "
              f"({(r[1] - r[0]).days} days)")
    print()

    # Per-symbol
    print("  Per symbol:")
    for sym, cnt in sorted(stats["per_symbol"].items()):
        print(f"    {sym:10s}  {cnt:>6,} candles")
    print()

    # Signals
    print(f"  Signals:      {stats['signals']}")
    print(f"    Open:       {stats['open_signals']}")
    print(f"    Won:        {stats['won']}")
    print(f"    Lost:       {stats['lost']}")
    print()

    # JSONL
    if stats["jsonl"]:
        print("  JSONL logs:")
        for name, info in stats["jsonl"].items():
            print(f"    {name:20s}  {info['records']:>5} records  "
                  f"({info['size_kb']:.1f} KB)")
        print()

    # Size estimate
    growth_per_day = stats["candles"] / max((r[1] - r[0]).days, 1) if r[0] and r[1] else 0
    if growth_per_day > 0:
        days_to_100mb = (100 * 1024 * 1024 / stats["db_size_mb"]) / growth_per_day if stats["db_size_mb"] > 0 else 999
        print(f"  Growth rate:  ~{growth_per_day:.0f} candles/day")
        print(f"  Est. 100 MB:  ~{days_to_100mb:.0f} days")
        print()

    # Compact
    if do_compact:
        print("  Running cleanup + VACUUM...")
        cleanup(conn, data_dir, execute=True)
        saved = vacuum(conn)
        new_size = os.path.getsize(str(cfg.db_path)) / 1024 / 1024
        print(f"  Done: {new_size:.2f} MB (saved {saved:.2f} MB)")
        print()

    # Archive
    if do_archive:
        print("  Checking JSONL for archival...")
        archived = archive_logs(data_dir)
        if archived:
            for a in archived:
                print(f"    {a}")
        else:
            print("    All logs under 10 MB — nothing to archive")
        print()

    if not do_compact and not do_archive:
        print("  Options:")
        print("    --compact       WAL checkpoint + VACUUM")
        print("    --archive-logs  Compress JSONL files > 10 MB")
        print()

    conn.close()


if __name__ == "__main__":
    main()
