"""Signal Bot — main entry point for the production bot.

Usage:
    python run_bot.py               # start the scan loop (dry_run from config)
    python run_bot.py --live        # override dry_run to False (send real alerts)
    python run_bot.py --scan-once   # run one scan and exit
    python run_bot.py --report      # generate and send daily report
    python run_bot.py --status      # print current scores (no Telegram)
"""
from __future__ import annotations

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from src import config as config_mod
from src.store import db
from src.data.budget import Budget
from src.data.router import Router


def _setup_logging(level: str = "INFO") -> None:
    """Configure logging with a clean format."""
    fmt = "%(asctime)s  %(name)-28s %(levelname)-7s %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt="%H:%M:%S",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def cmd_loop(cfg, conn, live: bool = False):
    """Start the continuous scan loop."""
    if live:
        cfg.raw["dry_run"] = False
    from src.scheduler import run_loop
    run_loop(cfg, conn)


def cmd_scan_once(cfg, conn, live: bool = False, update_outcomes: bool = False):
    """Run a single scan cycle and exit."""
    if live:
        cfg.raw["dry_run"] = False
    budget = Budget(conn, 750, 7)
    router = Router(cfg, conn, budget)
    data_dir = cfg.db_path.parent

    from src.scanner import run_scan
    summary = run_scan(cfg, conn, budget, router, data_dir)

    print(f"\nScan complete:")
    print(f"  Symbols scanned: {summary['symbols_scanned']}")
    print(f"  Signals emitted: {summary['signals_emitted']}")
    print(f"  Signals sent:    {summary.get('signals_sent', 0)}")
    print(f"  Duration:        {summary['duration_s']:.1f}s")
    if summary.get("scores"):
        print(f"\n  Scores:")
        for sym, score in sorted(summary["scores"].items()):
            print(f"    {sym:10s}  {score:+.1f}")
    if summary.get("errors"):
        print(f"\n  Errors:")
        for err in summary["errors"]:
            print(f"    {err}")

    if update_outcomes:
        from src.tracking.outcome_checker import check_outcomes
        outcomes = check_outcomes(conn, cfg, data_dir=data_dir)
        print(f"\nOutcomes checked: {outcomes['checked']} (Won: {outcomes['won']}, Lost: {outcomes['lost']})")


def cmd_report(cfg, conn, live: bool = False):
    """Generate and send a daily report."""
    if live:
        cfg.raw["dry_run"] = False
    from src.tracking.report import daily_report
    stats = daily_report(conn, cfg, send=not cfg.get("dry_run", default=True))
    print(f"\nReport: {stats['total']} signals, "
          f"{stats['wins']}W/{stats['losses']}L, "
          f"{stats['win_rate']:.0f}% win rate")


def cmd_status(cfg, conn):
    """Print current scores for all symbols (no Telegram)."""
    from datetime import datetime, timezone
    budget = Budget(conn, 750, 7)
    router = Router(cfg, conn, budget)
    now = datetime.now(timezone.utc)

    from src.analysis.confluence import score_symbol
    from src.analysis.levels import generate_plan

    print(f"\n  {'Symbol':10s}  {'Score':>7s}  {'Label':14s}  {'Type':12s}  Reason")
    print("  " + "-" * 80)

    for sym in cfg.symbols:
        res = router.fetch_symbol(sym, now)
        if not res.ok:
            print(f"  {sym.name:10s}  {'---':>7s}  data incomplete")
            continue

        signal = score_symbol(res.frames, sym.name, cfg)
        plan = generate_plan(signal, cfg)

        trade_type = plan.trade_type if plan else "\u2014"
        brief = plan.brief_reason if plan else ""

        print(f"  {sym.name:10s}  {signal.score:+7.1f}  {signal.label:14s}  "
              f"{trade_type:12s}  {brief}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Asset Signal Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--live", action="store_true",
                        help="Override dry_run to False (send real Telegram alerts)")
    parser.add_argument("--scan-once", action="store_true",
                        help="Run one scan cycle and exit")
    parser.add_argument("--github-actions", action="store_true",
                        help="Run for GitHub Actions: one scan + outcome check in live mode")
    parser.add_argument("--report", action="store_true",
                        help="Generate and send daily report")
    parser.add_argument("--status", action="store_true",
                        help="Print current scores (no Telegram)")
    parser.add_argument("--log-level", default="INFO",
                        help="Logging level (DEBUG, INFO, WARNING)")
    args = parser.parse_args()

    _setup_logging(args.log_level)
    cfg = config_mod.load()
    conn = db.connect(cfg.db_path)

    try:
        if args.status:
            cmd_status(cfg, conn)
        elif args.report:
            cmd_report(cfg, conn, live=args.live)
        elif args.scan_once:
            cmd_scan_once(cfg, conn, live=args.live)
        elif args.github_actions:
            # Process any pending Telegram interactive commands
            try:
                from src.alerts.telegram import poll_commands
                poll_commands(cfg, conn)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).debug("Telegram polling skipped: %s", exc)

            sleep_cfg = cfg.get("sleep_window", default={}) or {}
            is_sleeping = False
            from datetime import datetime, timezone, timedelta
            IST = timezone(timedelta(hours=5, minutes=30))
            ist_now = datetime.now(IST)
            if sleep_cfg.get("enabled", False):
                start_h = int(sleep_cfg.get("start_hour_ist", 0))
                end_h = int(sleep_cfg.get("end_hour_ist", 5))
                if start_h <= ist_now.hour < end_h:
                    is_sleeping = True
                    
            if is_sleeping:
                import logging
                logging.getLogger(__name__).info("Night mode active (%02d:00-%02d:00 IST) - skipping scan", start_h, end_h)
            else:
                cmd_scan_once(cfg, conn, live=True, update_outcomes=True)

            # Check if any new commands arrived during the scan
            try:
                from src.alerts.telegram import poll_commands
                poll_commands(cfg, conn)
            except Exception:
                pass

            # Daily summary at 9 PM IST (21:00)
            if ist_now.hour == 21 and ist_now.minute < 15:
                try:
                    from src.store import signals as sig_store
                    from src.alerts.telegram import send_text
                    today_start_ms = int(datetime(ist_now.year, ist_now.month, ist_now.day,
                                                  tzinfo=IST).timestamp() * 1000)
                    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                    # Count today's scans and signals
                    sig_count = conn.execute(
                        "SELECT COUNT(*) FROM signals WHERE ts > ?", (today_start_ms,)
                    ).fetchone()[0]
                    open_count = conn.execute(
                        "SELECT COUNT(*) FROM signals WHERE status = 'open'"
                    ).fetchone()[0]
                    won_count = conn.execute(
                        "SELECT COUNT(*) FROM signals WHERE ts > ? AND status = 'won'",
                        (today_start_ms,)
                    ).fetchone()[0]
                    lost_count = conn.execute(
                        "SELECT COUNT(*) FROM signals WHERE ts > ? AND status = 'lost'",
                        (today_start_ms,)
                    ).fetchone()[0]
                    summary_msg = (
                        f"\U0001f4ca <b>Daily Summary</b>\n"
                        f"\U0001f4cb Signals today: {sig_count}  |  Open: {open_count}\n"
                        f"\U0001f7e2 Won: {won_count}  |  \U0001f534 Lost: {lost_count}\n"
                        f"\u2705 Bot healthy\n"
                        f"\U0001f552 {ist_now.strftime('%d %b %Y, %I:%M %p IST')}"
                    )
                    send_text(summary_msg)
                except Exception as exc:
                    import logging
                    logging.getLogger(__name__).warning("Daily summary failed: %s", exc)

            # Daily report at midnight UTC
            utc_now = datetime.now(timezone.utc)
            if utc_now.hour == 0 and utc_now.minute < 15:
                try:
                    from src.tracking.report import daily_report
                    daily_report(conn, cfg, send=True)
                except Exception as exc:
                    import logging
                    logging.getLogger(__name__).warning("Daily report failed: %s", exc)

            # Lightweight DB maintenance (checkpoint WAL, expire old mutes)
            try:
                from src.maintenance import cleanup
                data_dir = cfg.db_path.parent
                cleanup(conn, data_dir, execute=True)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).debug("Maintenance cleanup skipped: %s", exc)
        else:
            cmd_loop(cfg, conn, live=args.live)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
