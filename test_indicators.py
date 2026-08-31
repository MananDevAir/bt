"""Quick smoke-test: fetch BTC 1d, run all indicators, print last values."""
import sys
sys.path.insert(0, ".")

from src import config as config_mod
from src.data.budget import Budget
from src.data.router import Router
from src.store import db
from src.analysis.indicators import classic_frame
from src.analysis.modern import modern_frame, volume_profile, divergences, volatility_regime
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

print(f"\n  BTC 1d: {len(df)} bars, last close = {df['close'].iloc[-1]:,.2f}")
print("  " + "-" * 60)

# Classic indicators
ci = classic_frame(df)
last = ci.iloc[-1]
print(f"  RSI(14)     = {last['rsi']:.1f}")
print(f"  MACD line   = {last['macd_macd']:.2f}")
print(f"  MACD signal = {last['macd_signal']:.2f}")
print(f"  MACD hist   = {last['macd_hist']:.2f}")
print(f"  Stoch K/D   = {last['stoch_k']:.1f} / {last['stoch_d']:.1f}")
print(f"  BB %B       = {last['bb_percent_b']:.3f}")
print(f"  ATR(14)     = {last['atr']:.2f}  ({last['atr_pct']:.2f}%)")
print(f"  ADX         = {last['adx']:.1f}  (+DI={last['plus_di']:.1f}, -DI={last['minus_di']:.1f})")
print(f"  EMA20/50/200= {last['ema20']:.0f} / {last['ema50']:.0f} / {last['ema200']:.0f}")
print(f"  OBV slope   = {'up' if last['obv'] > last['obv_ema'] else 'down'}")

# Modern indicators
mi = modern_frame(df)
mlast = mi.iloc[-1]
print(f"\n  SuperTrend  = {mlast['st']:.0f}  dir={'UP' if mlast['st_dir'] > 0 else 'DOWN'}")
print(f"  VWAP(sess)  = {mlast['vwap']:.0f}")
print(f"  Ichimoku    tenkan={mlast['tenkan']:.0f}  kijun={mlast['kijun']:.0f}")
print(f"  MFI(14)     = {mlast['mfi']:.1f}")

# Point-in-time
vp = volume_profile(df)
if vp:
    print(f"  Vol Profile POC={vp.poc:,.0f}  VAL={vp.val:,.0f}  VAH={vp.vah:,.0f}")

regime = volatility_regime(df)
print(f"  Vol Regime  = {regime['regime']}  (pctl={regime['percentile']:.0%})  squeeze={regime['squeeze']}")

divs = divergences(df)
if divs:
    for d in divs:
        print(f"  Divergence  {d.kind} on {d.osc}, {d.bars_ago} bars ago")
else:
    print("  Divergence  none active")

print()
conn.close()
