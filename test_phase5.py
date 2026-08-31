"""Phase 5 smoke-test: LLM narration + Telegram delivery on live signals.

Runs a full scan, generates narration for actionable signals via HF (with
fallback), and sends a real formatted alert to Telegram.

    python test_phase5.py              # full test (sends to Telegram)
    python test_phase5.py --dry        # dry run (prints, no Telegram send)
    python test_phase5.py --template   # force template narration (no HF)
"""
import logging
import sys
import os

sys.path.insert(0, ".")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="  %(name)-25s %(levelname)-7s %(message)s",
)

from datetime import datetime, timezone
from src import config as config_mod
from src.data.budget import Budget
from src.data.router import Router
from src.store import db
from src.analysis.confluence import score_symbol
from src.analysis.levels import generate_plan
from src.llm.explain import explain
from src.llm.template import build_fact_sheet, narrate as template_narrate
from src.alerts.telegram import send_signal, send_text
from src.alerts.formatter import format_signal

# Parse args
dry_run = "--dry" in sys.argv
force_template = "--template" in sys.argv


def main():
    cfg = config_mod.load()

    # Override dry_run if --dry flag passed
    if not dry_run:
        cfg.raw["dry_run"] = False  # send for real

    conn = db.connect(cfg.db_path)
    budget = Budget(conn, 750, 7)
    router = Router(cfg, conn, budget)
    now = datetime.now(timezone.utc)

    print(f"\n  PHASE 5 TEST  {now:%Y-%m-%d %H:%M} UTC")
    print(f"  mode={'DRY RUN' if dry_run else 'LIVE'}  "
          f"narration={'TEMPLATE' if force_template else 'HF+FALLBACK'}")
    print("  " + "=" * 80)

    # ── Test 1: Template narration ──────────────────────────────────────────
    print("\n  TEST 1: Template narration")
    print("  " + "-" * 40)
    test_facts = {
        "symbol": "BTC", "direction": 1, "score": 42.5,
        "label": "BUY", "confidence": 72,
        "htf_bias": "bullish", "mtf_bias": "bullish", "ltf_state": "pullback",
        "triggers": ["EMA 20>50>200 aligned bull", "SuperTrend UP",
                     "bullish BOS on 1h", "RSI 55 turning up"],
        "entry_low": 84500.0, "entry_high": 84700.0,
        "sl": 83800.0, "tp1": 85400.0, "tp2": 86100.0, "tp3": 87200.0,
        "rr": 2.3, "risk_pct": 0.83, "risk_atr": 1.4,
        "source": "order_block", "invalidation": "4h close below 83,800",
        "holding_horizon": "hours",
    }
    tmpl = template_narrate(test_facts)
    print(f"  Template: {tmpl}")
    print("  [OK]")

    # ── Test 2: HF narration with fallback chain ───────────────────────────
    if not force_template:
        print("\n  TEST 2: HF narration (fallback chain)")
        print("  " + "-" * 40)

        # Test with a real-ish fact sheet
        from src.llm.explain import _get_tokens, _get_models, _call_hf, _build_user_prompt
        tokens = _get_tokens()
        models = _get_models(cfg)
        print(f"  Tokens available: {len(tokens)}")
        print(f"  Models: {', '.join(models)}")

        prompt = _build_user_prompt(test_facts)
        success = False
        for i, token in enumerate(tokens):
            masked = token[:8] + "..." + token[-4:]
            for model in models:
                print(f"  Trying token={masked} model={model}...", end=" ")
                reply = _call_hf(token, model, prompt, timeout=15)
                if reply:
                    print(f"OK ({len(reply)} chars)")
                    print(f"  Reply: {reply[:200]}")
                    success = True
                    break
                else:
                    print("FAILED")
            if success:
                break

        if not success:
            print("  All HF attempts failed — template will be used in production")
        print("  [OK]")

    # ── Test 3: Live scan + narration + formatted message ──────────────────
    print("\n  TEST 3: Live scan + narration")
    print("  " + "-" * 40)

    signals_sent = 0
    best_signal = None
    best_plan = None
    best_score = -999

    for sym in cfg.symbols:
        res = router.fetch_symbol(sym, now)
        if not res.ok:
            print(f"  {sym.name:10s}  data incomplete, skipping")
            continue

        signal = score_symbol(res.frames, sym.name, cfg)
        plan = generate_plan(signal, cfg)

        gate_str = ", ".join(signal.gates.keys()) if signal.gates else "all pass"
        plan_str = f"RR={plan.rr:.1f}" if plan else "no plan"

        print(f"  {sym.name:10s}  {signal.score:+7.1f}  {signal.label:14s}  "
              f"{gate_str:20s}  {plan_str}")

        # Track the best actionable signal for sending
        if signal.label != "NEUTRAL" and abs(signal.score) > best_score:
            best_score = abs(signal.score)
            best_signal = signal
            best_plan = plan

    # ── Test 4: Send the best signal to Telegram ───────────────────────────
    if best_signal:
        print(f"\n  TEST 4: Sending best signal to Telegram")
        print(f"  " + "-" * 40)
        print(f"  Best: {best_signal.symbol} {best_signal.label} "
              f"(score={best_signal.score:+.1f})")

        # Get narration
        if force_template:
            facts = build_fact_sheet(best_signal, best_plan)
            narration = template_narrate(facts)
            narr_source = "template"
        else:
            narration, narr_source = explain(best_signal, best_plan, cfg)

        print(f"  Narration ({narr_source}): {narration[:150]}...")

        # Format message
        msg = format_signal(best_signal, best_plan, narration, narr_source)
        print(f"  Message length: {len(msg)} chars")

        # Send
        ok = send_signal(best_signal, best_plan, narration, narr_source, cfg)
        if ok:
            signals_sent += 1
            print("  [SENT]" if not dry_run else "  [DRY RUN - logged]")
        else:
            print("  [SEND FAILED]")
    else:
        print("\n  No actionable signals right now (all NEUTRAL)")
        # Send a status message instead
        print("\n  TEST 4: Sending status overview to Telegram")
        print("  " + "-" * 40)

        status_data = []
        for sym in cfg.symbols:
            res = router.fetch_symbol(sym, now)
            if not res.ok:
                continue
            signal = score_symbol(res.frames, sym.name, cfg)
            status_data.append({
                "symbol": sym.name,
                "score": signal.score,
                "label": signal.label,
            })

        from src.alerts.formatter import format_status
        status_msg = format_status(status_data)

        if not dry_run:
            result = send_text(status_msg)
            print("  [SENT]" if result else "  [SEND FAILED]")
        else:
            import re
            clean = re.sub(r"<[^>]+>", "", status_msg)
            for line in clean.split("\n"):
                print(f"  {line}")
            print("  [DRY RUN]")

    print(f"\n  Phase 5 test complete. Signals sent: {signals_sent}")
    print()

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
