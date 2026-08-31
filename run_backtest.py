"""Run full backtest — all symbols, 60 days."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["PYTHONIOENCODING"] = "utf-8"

from src import config as config_mod
from src.store import db
from src.backtest.replay import replay_all
from src.backtest.report import print_report, save_report

cfg = config_mod.load()
conn = db.connect(cfg.db_path)

print("Running full backtest...")
result = replay_all(conn, cfg, days=60, step="4h")
print_report(result)

report_path = cfg.db_path.parent / "backtest_report.md"
save_report(result, report_path)
print(f"\nReport saved to: {report_path}")

conn.close()
