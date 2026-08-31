"""Smoke-test Phase 3: price action + SMC on live BTC 1d data."""
import sys
sys.path.insert(0, ".")

from src import config as config_mod
from src.data.budget import Budget
from src.data.router import Router
from src.store import db
from src.analysis.price_action import price_action_scan
from src.analysis.smc import smc_scan
from datetime import datetime, timezone

cfg = config_mod.load()
conn = db.connect(cfg.db_path)
budget = Budget(conn, 750, 7)
router = Router(cfg, conn, budget)
now = datetime.now(timezone.utc)

btc = cfg.symbols[0]  # BTC
res = router.fetch_symbol(btc, now)
df = res.frames.get("1d")

if df is None or df.empty:
    print("No data for BTC 1d!")
    sys.exit(1)

print(f"\n  BTC 1d: {len(df)} bars, last close = ${df['close'].iloc[-1]:,.2f}")
print("  " + "=" * 70)

# ---- Price Action ----
print("\n  PRICE ACTION")
print("  " + "-" * 70)
pa = price_action_scan(df)

print(f"  Candlestick patterns (last 5 bars):")
if pa["patterns"]:
    for p in pa["patterns"]:
        bias = {1: "BULL", -1: "BEAR", 0: "NEUT"}[p.bias]
        print(f"    {p.name:<18} {bias}  strength={p.strength:.2f}  @ {p.ts}")
else:
    print("    (none)")

print(f"\n  S/R Zones ({len(pa['sr_zones'])} found):")
for z in pa["sr_zones"][:6]:
    print(f"    {z.kind:<12} {z.mid:>12,.2f}  (touches={z.touches}, strength={z.strength:.2f})")

print(f"\n  Fibonacci levels:")
for f in pa["fibonacci"][:8]:
    print(f"    {f.kind:<14} {f.ratio:>6.3f}  @ {f.price:>12,.2f}")

print(f"\n  Key levels:")
for kl in pa["key_levels"][:5]:
    print(f"    {kl.label:<20} @ {kl.price:>12,.2f}")

# ---- SMC ----
print("\n  SMART MONEY CONCEPTS")
print("  " + "-" * 70)
smc = smc_scan(df)

print(f"  Structure bias: {smc['structure_bias']}")
print(f"  Last 5 labels:")
for sp in smc["structure"][-5:]:
    print(f"    {sp.label.value:<4}  {sp.pivot.kind:<5} @ {sp.pivot.price:>12,.2f}  bias={'+' if sp.bias > 0 else '-'}")

print(f"\n  BOS/CHoCH ({len(smc['bos'])} found):")
for b in smc["bos"][-5:]:
    d = "BULL" if b.direction > 0 else "BEAR"
    print(f"    {b.kind:<6} {d}  broke {b.broken_level:>12,.2f}  @ {b.ts}")

pd_z = smc["premium_discount"]
print(f"\n  Premium/Discount: {pd_z.zone} ({pd_z.pct:.1f}%)")
print(f"    Range: {pd_z.range_low:,.2f} - {pd_z.range_high:,.2f}  EQ={pd_z.equilibrium:,.2f}")

print(f"\n  Order blocks ({len(smc['order_blocks'])} total, "
      f"{len(smc['order_blocks_unmitigated'])} unmitigated):")
for ob in smc["order_blocks_unmitigated"][-3:]:
    d = "BULL" if ob.direction > 0 else "BEAR"
    print(f"    {d} OB  {ob.lo:>12,.2f} - {ob.hi:>12,.2f}  disp={ob.displacement:.1f} ATR")

print(f"\n  FVGs ({len(smc['fvg'])} total, {len(smc['fvg_open'])} open):")
for f in smc["fvg_open"][-3:]:
    d = "BULL" if f.direction > 0 else "BEAR"
    print(f"    {d} FVG  {f.lo:>12,.2f} - {f.hi:>12,.2f}  filled={f.filled_pct:.0%}")

print(f"\n  Liquidity sweeps ({len(smc['liquidity_sweeps'])}):")
for s in smc["liquidity_sweeps"][-3:]:
    d = "BULL" if s.direction > 0 else "BEAR"
    print(f"    {d} sweep  level={s.swept_level:>12,.2f}  wick_beyond={s.wick_beyond:>8,.2f}")

print(f"\n  Equal levels ({len(smc['equal_levels'])}):")
for e in smc["equal_levels"][-3:]:
    print(f"    {e.kind:<14}  @ {e.price:>12,.2f}  count={e.count}")

print()
conn.close()
