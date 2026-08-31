"""JSONL append-only logging — immutable audit trail.

Three log files, all in the data/ directory:
  signals.jsonl   — every emitted signal
  outcomes.jsonl  — every outcome check
  scans.jsonl     — every scan cycle summary

Each line is a self-contained JSON object with a UTC timestamp.
Files are never truncated or edited — append-only by design.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

__all__ = ["log_signal", "log_outcome", "log_scan"]


def _append(path: Path, record: dict[str, Any]) -> None:
    """Append a JSON record as one line to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record["_ts_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record["_ts_epoch"] = int(time.time())
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
    except Exception as exc:
        log.warning("Failed to write JSONL to %s: %s", path, exc)


def log_signal(data_dir: Path, signal_id: int, symbol: str,
               direction: str, label: str, score: float,
               plan: dict | None = None,
               narration_source: str = "",
               sent_ok: bool = False) -> None:
    """Log an emitted signal to signals.jsonl."""
    record = {
        "event": "signal",
        "signal_id": signal_id,
        "symbol": symbol,
        "direction": direction,
        "label": label,
        "score": round(score, 2),
        "sent_ok": sent_ok,
        "narration_source": narration_source,
    }
    if plan:
        record["plan"] = plan
    _append(data_dir / "signals.jsonl", record)


def log_outcome(data_dir: Path, signal_id: int, symbol: str,
                status: str, hit: str = "",
                mfe_r: float = 0, mae_r: float = 0,
                bars_held: int = 0) -> None:
    """Log an outcome check to outcomes.jsonl."""
    record = {
        "event": "outcome",
        "signal_id": signal_id,
        "symbol": symbol,
        "status": status,
        "hit": hit,
        "mfe_r": round(mfe_r, 2),
        "mae_r": round(mae_r, 2),
        "bars_held": bars_held,
    }
    _append(data_dir / "outcomes.jsonl", record)


def log_scan(data_dir: Path, symbols_scanned: int,
             signals_emitted: int, duration_s: float,
             scores: dict[str, float] | None = None) -> None:
    """Log a scan cycle summary to scans.jsonl."""
    record = {
        "event": "scan",
        "symbols_scanned": symbols_scanned,
        "signals_emitted": signals_emitted,
        "duration_s": round(duration_s, 2),
    }
    if scores:
        record["scores"] = {k: round(v, 1) for k, v in scores.items()}
    _append(data_dir / "scans.jsonl", record)
