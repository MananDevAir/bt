"""Backtest CLI — run walk-forward replay from the command line.

Usage:
    python -m src.backtest.replay                    # all symbols, last 60 days
    python -m src.backtest.replay --symbol BTC       # one symbol
    python -m src.backtest.replay --days 30          # last 30 days
    python -m src.backtest.replay --step 1h          # step every 1h
    python -m src.backtest.replay --save             # save markdown report
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src import config as config_mod
from src.store import db
from src.backtest.replay import replay_symbol, replay_all
from src.backtest.report import print_report, save_report


def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--symbol", "-s", type=str, default=None,
                        help="Symbol to backtest (default: all)")
    parser.add_argument("--days", "-d", type=int, default=60,
                        help="Number of days to replay (default: 60)")
    parser.add_argument("--step", type=str, default="4h",
                        choices=["15m", "1h", "4h", "1d"],
                        help="Step interval (default: 4h)")
    parser.add_argument("--save", action="store_true",
                        help="Save markdown report to data/backtest_report.md")
    parser.add_argument("--log-level", default="WARNING",
                        help="Logging level (default: WARNING)")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="  %(name)-28s %(levelname)-7s %(message)s",
    )

    cfg = config_mod.load()
    conn = db.connect(cfg.db_path)

    if args.symbol:
        result = replay_symbol(conn, cfg, args.symbol,
                               days=args.days, step=args.step)
    else:
        result = replay_all(conn, cfg, days=args.days, step=args.step)

    print_report(result)

    if args.save:
        report_path = cfg.db_path.parent / "backtest_report.md"
        save_report(result, report_path)
        print(f"  Report saved to: {report_path}")

    conn.close()


if __name__ == "__main__":
    main()
