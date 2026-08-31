"""Smoke-test Phase 4: confluence engine + trade plan on all 9 symbols."""
import sys
sys.path.insert(0, ".")

from src import config as config_mod
from src.data.budget import Budget
from src.data.router import Router
from src.store import db
from src.analysis.confluence import score_symbol
from src.analysis.levels import generate_plan
from datetime import datetime, timezone

cfg = config_mod.load()
conn = db.connect(cfg.db_path)
budget = Budget(conn, 750, 7)
router = Router(cfg, conn, budget)
now = datetime.now(timezone.utc)

print(f"\n  CONFLUENCE SCAN  {now:%Y-%m-%d %H:%M} UTC")
print(f"  profile={cfg.get('profile')}  timeframes={', '.join(cfg.timeframes)}")
print("  " + "=" * 90)
print(f"  {'Symbol':<10}{'Score':>7}  {'Label':<14}{'Conf%':>6}  {'Gates':<20}  {'Plan':>6}")
print("  " + "-" * 90)

for sym in cfg.symbols:
    res = router.fetch_symbol(sym, now)
    if not res.ok:
        print(f"  {sym.name:<10}  -- data incomplete --")
        continue

    signal = score_symbol(res.frames, sym.name, cfg)
    plan = generate_plan(signal, cfg)

    gate_str = ", ".join(signal.gates.keys()) if signal.gates else "all pass"
    plan_str = f"RR={plan.rr}" if plan else "no plan"

    print(f"  {sym.name:<10}{signal.score:>+7.1f}  {signal.label:<14}{signal.confidence:>5.0f}%  "
          f"{gate_str:<20}  {plan_str:>6}")

    # Print trade plan details for actionable signals
    if plan and signal.label not in ("NEUTRAL",):
        d = "LONG" if plan.direction > 0 else "SHORT"
        print(f"            {d}  Entry: {plan.entry_low:>12,.2f} - {plan.entry_high:,.2f}  "
              f"(via {plan.source})")
        print(f"            SL: {plan.sl:>12,.2f}  "
              f"TP1: {plan.tp1:>12,.2f} / TP2: {plan.tp2:>12,.2f} / TP3: {plan.tp3:>12,.2f}")
        print(f"            R:R={plan.rr:.2f}  Risk={plan.risk_pct:.2f}%  "
              f"({plan.risk_atr:.1f} ATR)  Hold={plan.holding_horizon}")

print()
conn.close()
