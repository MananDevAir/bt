"""Phase 6 smoke-test: Signal store + outcome tracking + performance + JSONL.

    python test_phase6.py              # full test
    python test_phase6.py --dry        # skip Telegram
"""
import logging
import sys
import os
import time

sys.path.insert(0, ".")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="  %(name)-30s %(levelname)-7s %(message)s",
)

from datetime import datetime, timezone
from pathlib import Path
from src import config as config_mod
from src.data.budget import Budget
from src.data.router import Router
from src.store import db
from src.store import signals as sig_store
from src.analysis.confluence import score_symbol
from src.analysis.levels import generate_plan
from src.llm.template import build_fact_sheet, narrate as template_narrate
from src.tracking.outcome_checker import check_outcomes
from src.tracking.performance import compute_stats
from src.tracking.report import daily_report, format_report
from src.logging_util import log_signal, log_scan

dry_run = "--dry" in sys.argv


def main():
    cfg = config_mod.load()
    if not dry_run:
        cfg.raw["dry_run"] = False

    conn = db.connect(cfg.db_path)
    budget = Budget(conn, 750, 7)
    router = Router(cfg, conn, budget)
    now = datetime.now(timezone.utc)
    data_dir = cfg.db_path.parent  # data/

    print(f"\n  PHASE 6 TEST  {now:%Y-%m-%d %H:%M} UTC")
    print("  " + "=" * 80)

    # ── Test 1: Signal store — save + cooldown ──────────────────────────────
    print("\n  TEST 1: Signal store + cooldown")
    print("  " + "-" * 50)

    scan_start = time.time()
    signals_emitted = 0
    scores: dict[str, float] = {}

    for sym in cfg.symbols:
        res = router.fetch_symbol(sym, now)
        if not res.ok:
            print(f"  {sym.name:10s}  data incomplete")
            continue

        signal = score_symbol(res.frames, sym.name, cfg)
        plan = generate_plan(signal, cfg)
        scores[sym.name] = signal.score

        # Only save actionable signals
        if signal.label == "NEUTRAL" or signal.direction == 0:
            trade_type = "—"
            print(f"  {sym.name:10s}  {signal.score:+7.1f}  {signal.label:14s}  skip")
            continue

        # Check cooldown
        if sig_store.is_on_cooldown(conn, sym.name, signal.direction):
            print(f"  {sym.name:10s}  {signal.score:+7.1f}  {signal.label:14s}  COOLDOWN")
            continue

        # Generate narration (template for speed)
        facts = build_fact_sheet(signal, plan)
        narration = template_narrate(facts)

        trade_type = plan.trade_type if plan else "?"
        brief = plan.brief_reason if plan else ""

        # Save to DB
        signal_id = sig_store.save_signal(
            conn, signal, plan, narration, "template",
            sent_ok=True, data_source=res.source,
        )

        # Log to JSONL
        plan_dict = {
            "entry_low": plan.entry_low, "entry_high": plan.entry_high,
            "sl": plan.sl, "tp1": plan.tp1, "tp2": plan.tp2, "tp3": plan.tp3,
            "rr": plan.rr, "trade_type": plan.trade_type,
        } if plan else None

        log_signal(data_dir, signal_id, sym.name,
                   "long" if signal.direction > 0 else "short",
                   signal.label, signal.score,
                   plan=plan_dict, narration_source="template", sent_ok=True)

        signals_emitted += 1
        print(f"  {sym.name:10s}  {signal.score:+7.1f}  {signal.label:14s}  "
              f"saved #{signal_id}  {trade_type:12s}  {brief}")

    scan_duration = time.time() - scan_start
    log_scan(data_dir, len(cfg.symbols), signals_emitted, scan_duration, scores)
    print(f"\n  Scan: {len(cfg.symbols)} symbols, {signals_emitted} signals, "
          f"{scan_duration:.1f}s")

    # ── Test 2: Cooldown check ──────────────────────────────────────────────
    print("\n  TEST 2: Cooldown verification")
    print("  " + "-" * 50)
    for sym in cfg.symbols:
        for dir_val, dir_name in [(1, "long"), (-1, "short")]:
            on_cd = sig_store.is_on_cooldown(conn, sym.name, dir_val)
            if on_cd:
                print(f"  {sym.name:10s} {dir_name:6s}  COOLDOWN active")
    print("  [OK]")

    # ── Test 3: Outcome checker ─────────────────────────────────────────────
    print("\n  TEST 3: Outcome checker")
    print("  " + "-" * 50)
    summary = check_outcomes(conn, cfg, data_dir=data_dir)
    print(f"  Checked: {summary['checked']}")
    print(f"  Won: {summary['won']}, Lost: {summary['lost']}, "
          f"Expired: {summary['expired']}, Open: {summary['still_open']}")
    print("  [OK]")

    # ── Test 4: Open signals query ──────────────────────────────────────────
    print("\n  TEST 4: Open signals")
    print("  " + "-" * 50)
    open_sigs = sig_store.get_open_signals(conn)
    for s in open_sigs:
        age_h = (time.time() * 1000 - s["ts"]) / 3600000
        print(f"  #{s['id']:4d}  {s['symbol']:10s}  {s['direction']:5s}  "
              f"{s['label']:14s}  {s['score']:+6.1f}  {age_h:.1f}h old")
    print(f"  Total open: {len(open_sigs)}")

    # ── Test 5: Performance stats ───────────────────────────────────────────
    print("\n  TEST 5: Performance stats")
    print("  " + "-" * 50)
    stats = compute_stats(conn, hours=24)
    print(f"  Total: {stats['total']}")
    print(f"  Won: {stats['wins']}, Lost: {stats['losses']}")
    print(f"  Win rate: {stats['win_rate']:.1f}%")
    print(f"  Avg MFE: {stats['avg_mfe_r']:.2f}R, Avg MAE: {stats['avg_mae_r']:.2f}R")
    print(f"  By symbol: {stats['by_symbol']}")
    print("  [OK]")

    # ── Test 6: Report generation ───────────────────────────────────────────
    print("\n  TEST 6: Daily report")
    print("  " + "-" * 50)
    report_msg = format_report(stats, "Daily Report", now.strftime("%Y-%m-%d"))
    # Print clean version
    import re
    clean = re.sub(r"<[^>]+>", "", report_msg)
    for line in clean.split("\n"):
        print(f"  {line}")
    print("  [OK]")

    # ── Test 7: JSONL logs ──────────────────────────────────────────────────
    print("\n  TEST 7: JSONL logs")
    print("  " + "-" * 50)
    for fname in ("signals.jsonl", "outcomes.jsonl", "scans.jsonl"):
        fpath = data_dir / fname
        if fpath.exists():
            lines = fpath.read_text(encoding="utf-8").strip().split("\n")
            print(f"  {fname}: {len(lines)} records")
        else:
            print(f"  {fname}: not created yet")
    print("  [OK]")

    # ── Test 8: Last signal query ───────────────────────────────────────────
    print("\n  TEST 8: Last signal")
    print("  " + "-" * 50)
    last = sig_store.get_last_signal(conn)
    if last:
        print(f"  #{last['id']} {last['symbol']} {last['direction']} "
              f"{last['label']} {last['score']:+.1f}")
    else:
        print("  No signals in DB")
    print("  [OK]")

    print(f"\n  Phase 6 test complete.")
    print()

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
