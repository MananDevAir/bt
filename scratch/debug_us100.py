import sys, os
sys.path.insert(0, '.')
os.environ['PYTHONIOENCODING'] = 'utf-8'

from src import config as config_mod
from src.store import db
from src.backtest.replay import replay_symbol

cfg = config_mod.load()
conn = db.connect(cfg.db_path)

print("Debugging US100...")
r = replay_symbol(conn, cfg, 'US100', days=60, step='4h')

# We want to see what scores it's getting and if it's hitting gates.
# Wait, replay_symbol doesn't log all scores, only the ones that pass watch.
# Let's temporarily mock the watch threshold in config down to 0 to see what it scores.
cfg.raw.setdefault("thresholds", {})["watch"] = 0
cfg.raw.setdefault("symbol_overrides", {})["US100"] = {"watch": 0}
r_all = replay_symbol(conn, cfg, 'US100', days=60, step='4h')

scores = [s.score for s in r_all.signals]
if scores:
    print(f"Total scored points: {len(scores)}")
    print(f"Max score: {max(scores):.1f}")
    print(f"Min score: {min(scores):.1f}")
    print(f"Average absolute score: {sum(abs(x) for x in scores)/len(scores):.1f}")
    
    # How many above 18?
    above_18 = sum(1 for s in r_all.signals if abs(s.score) >= 18)
    print(f"Scores >= 18: {above_18}")
else:
    print("Zero signals even at watch=0. It must be getting blocked by a hard gate (like HTF conflict or Volatility).")

conn.close()
