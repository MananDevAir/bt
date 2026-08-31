"""Phase 1 entrypoint: prove the data foundation works.

Fetches every configured symbol across every configured timeframe and prints a
coverage table (rows, first/last closed candle, last close, source, notes) plus
the Twelve Data credit spend for the day.

    python -m src.main            # full scan
    python -m src.main --crypto   # skip TradFi (no key needed)
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone

from . import config as config_mod
from .data import sessions
from .data.budget import Budget
from .data.router import Router
from .store import db


def _fmt_ts(ts) -> str:
    return ts.strftime("%Y-%m-%d %H:%M") if ts is not None else "-"


def run(crypto_only: bool = False) -> int:
    cfg = config_mod.load()
    conn = db.connect(cfg.db_path)
    budget = Budget(
        conn,
        daily_cap=int(cfg.get("budget", "twelvedata_daily_cap", default=750)),
        per_min=int(cfg.get("budget", "twelvedata_per_min", default=7)),
    )
    router = Router(cfg, conn, budget)
    now = datetime.now(timezone.utc)

    symbols = [s for s in cfg.symbols if not (crypto_only and s.source != "binance")]

    print(f"\n  scan at {now:%Y-%m-%d %H:%M} UTC   profile={cfg.get('profile')}   "
          f"timeframes={', '.join(cfg.timeframes)}")
    print("  " + "-" * 108)
    header = (f"  {'symbol':<10}{'session':<9}{'tf':<6}{'bars':>6}  "
              f"{'last closed candle':<18}{'close':>12}  {'source':<12}")
    print(header)
    print("  " + "-" * 108)

    ok_count = 0
    all_notes: list[str] = []

    for sym in symbols:
        res = router.fetch_symbol(sym, now)
        state = sessions.describe(sym.session, now)
        if res.ok:
            ok_count += 1
        first_line = True
        for tf in cfg.timeframes:
            df = res.frames.get(tf)
            if df is None or df.empty:
                bars, last, close = 0, None, float("nan")
            else:
                bars, last, close = len(df), df.index[-1], float(df["close"].iloc[-1])
            name = sym.name if first_line else ""
            sess = state if first_line else ""
            src = res.source if first_line else ""
            close_txt = "-" if bars == 0 else f"{close:,.2f}"
            print(f"  {name:<10}{sess:<9}{tf:<6}{bars:>6}  "
                  f"{_fmt_ts(last):<18}{close_txt:>12}  {src:<12}")
            first_line = False
        for note in res.notes:
            all_notes.append(f"{sym.name}: {note}")
        print("  " + "." * 108)

    print(f"\n  {ok_count}/{len(symbols)} symbols fully covered")
    print(f"  twelvedata: {budget.report()}")
    if all_notes:
        print("\n  notes")
        for note in all_notes:
            print(f"    - {note}")
    print()

    conn.close()
    return 0 if ok_count else 1


if __name__ == "__main__":
    sys.exit(run(crypto_only="--crypto" in sys.argv))
