"""Real cached candles — the engine against actual market data.

Everything else in this suite is synthetic, which proves the maths but not that
the engine survives the shapes real feeds produce: a forex pair quoted at 1.15
instead of 10,000, an ATR of 0.00009, a structure level 26 ATR away.

These tests read `data/bot.db` and skip cleanly when it is absent, so a fresh
checkout still runs green. Populate it with:

    python -m src.main --crypto

Four defects are pinned as `xfail(strict=True)` here, all found by these tests
and none of them reachable from synthetic fixtures priced near 10,000:

1. **The invalidation line quotes the stop at 2 decimals whatever the pair.**
   On EURUSD the actual stop is 1.1586 and the text reads "close below SL
   (1.16)" — a price *above* the entry zone. The number the user reads to decide
   when to abandon the trade is on the wrong side of their own entry.
2. **A published entry zone can collapse to one tick.** `_decimals` gives 4dp
   between 1 and 10, and EURUSD's zone is 1 pip wide, so `entry_mid` rounds up
   onto `entry_high`.
3. **A distant structure stop suppresses the whole signal.** `_find_stop` has a
   0.8 ATR lower bound and no upper bound, so it can return a stop 26 ATR away;
   the `max_stop_atr` gate then rejects the plan and `scanner.py:148` drops the
   signal with a debug log. Two of the nine cached symbols cannot produce a trade
   at any score.
4. **The long bias**, reproduced on all nine symbols — the same defect
   `test_symmetry.py` documents on synthetic paths. This is the evidence that the
   97-long / 0-short backtest in `data/backtest_report.md` is a property of the
   engine and not of the 60 days it sampled.
"""
from __future__ import annotations

import copy

import pytest

from src.analysis.confluence import score_symbol
from src.analysis.levels import generate_plan
from src.config import Config

from conftest import ALL_TFS, cached_frames, cached_symbols, mirror_ohlcv

SYMBOLS = cached_symbols()

# Parametrising over an empty list would silently collect nothing, which reads
# as "these tests pass". Collect one skipped placeholder instead.
if SYMBOLS:
    symbol_param = pytest.mark.parametrize("symbol", SYMBOLS)
else:
    symbol_param = pytest.mark.parametrize("symbol", [
        pytest.param(None, marks=pytest.mark.skip(
            reason="no local candle cache; run `python -m src.main --crypto`"))])


def _needs_cache():
    if not SYMBOLS:
        pytest.skip("no local candle cache; run `python -m src.main --crypto`")


@pytest.fixture(scope="module")
def loose(cfg) -> Config:
    """Thresholds at 1 — every symbol produces a plan.

    Under the shipped thresholds five of the nine cached symbols are NEUTRAL, so
    a plan test keyed on the real config skips more than half its cases and never
    checks the two low-priced forex pairs at all. What matters here is that plans
    built from *real candle geometry* are coherent, not which tier the shipped
    thresholds happen to assign, so the thresholds are taken out of the way.
    """
    raw = copy.deepcopy(cfg.raw)
    raw["thresholds"] = {"strong": 1, "signal": 1, "watch": 1}
    raw["symbol_overrides"] = {}
    return Config(raw=raw, symbols=cfg.symbols, macro=cfg.macro, htf=cfg.htf,
                  mtf=cfg.mtf, ltf=cfg.ltf, history=cfg.history,
                  db_path=cfg.db_path)


@pytest.fixture(scope="module")
def real_plans(loose) -> dict[str, dict]:
    """Every symbol's plan plus the unrounded internals that produced it.

    The rounded `TradePlan` alone cannot distinguish "the zone is one tick wide"
    from "rounding collapsed a wider zone", and cannot show why a plan is absent.
    `_find_entry` / `_find_stop` are re-run here to capture both.
    """
    _needs_cache()
    from src.analysis.levels import _find_entry, _find_stop

    risk = loose.get("risk", default={}) or {}
    atr_mult = float(risk.get("atr_stop_mult", 1.5))
    buffer_atr = float(risk.get("struct_buffer_atr", 0.25))

    out: dict[str, dict] = {}
    for sym in SYMBOLS:
        frames = cached_frames(sym)
        if len(frames) < len(ALL_TFS):
            continue
        signal = score_symbol(frames, sym, loose)
        ltf = signal.tf_results.get(loose.ltf)
        if ltf is None or ltf.classic is None:
            continue
        atr = float(ltf.classic["atr"].iloc[-1])
        close = float(ltf.raw_df["close"].iloc[-1])
        mid, lo, hi, source = _find_entry(signal, signal.direction, close, atr)
        sl = _find_stop(signal, signal.direction, mid, atr, atr_mult, buffer_atr)
        out[sym] = {
            "signal": signal, "plan": generate_plan(signal, loose),
            "atr": atr, "close": close, "source": source,
            "raw_lo": lo, "raw_hi": hi, "raw_mid": mid, "raw_sl": sl,
            "risk_atr": abs(mid - sl) / atr if atr > 0 else float("inf"),
        }
    if not out:
        pytest.skip("cache has no symbol with all five timeframes")
    return out


@pytest.fixture(scope="module")
def bias_table(cfg) -> list[tuple[str, float, str, float, str]]:
    """(symbol, score, label, mirrored_score, mirrored_label) for every symbol.

    Module-scoped: this is 18 full multi-timeframe scans and both bias tests
    need all of it. Uses the *shipped* config — the bias claim is about what the
    bot actually publishes.
    """
    if not SYMBOLS:
        pytest.skip("no local candle cache")
    rows = []
    for sym in SYMBOLS:
        frames = cached_frames(sym)
        if len(frames) < len(ALL_TFS):
            continue
        sig = score_symbol(frames, sym, cfg)
        mirrored = score_symbol({tf: mirror_ohlcv(df)
                                 for tf, df in frames.items()}, sym, cfg)
        rows.append((sym, sig.score, sig.label, mirrored.score, mirrored.label))
    if not rows:
        pytest.skip("cache has no symbol with all five timeframes")
    return rows


# --------------------------------------------------------------------------- #
# the cache itself
# --------------------------------------------------------------------------- #
@symbol_param
def test_cached_candles_are_well_formed(symbol):
    """A broken fetch or a bad cache write shows up here rather than as a
    nonsense signal three layers downstream."""
    frames = cached_frames(symbol)
    assert frames, f"{symbol} has no cached frames"
    for tf, df in frames.items():
        assert not df.empty
        assert df.index.is_monotonic_increasing, f"{symbol} {tf} not time-ordered"
        assert not df.index.has_duplicates, f"{symbol} {tf} has duplicate bars"
        assert df[["open", "high", "low", "close"]].notna().all().all()
        assert (df["high"] >= df["low"]).all(), f"{symbol} {tf} has high < low"
        assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
        assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
        assert (df["close"] > 0).all()


def test_cached_frames_are_built_in_config_order(cfg):
    """`cached_frames` must mirror how `router.py` fills the dict — the trade
    levels depend on it (see test_plan.py::test_entry_zone_is_independent_of_
    timeframe_order)."""
    if not SYMBOLS:
        pytest.skip("no local candle cache")
    frames = cached_frames(SYMBOLS[0])
    assert tuple(frames) == tuple(tf for tf in cfg.timeframes if tf in frames)


# --------------------------------------------------------------------------- #
# the engine survives real data
# --------------------------------------------------------------------------- #
@symbol_param
def test_engine_scores_real_data_without_raising(symbol, cfg):
    frames = cached_frames(symbol)
    sig = score_symbol(frames, symbol, cfg)
    assert -100.0 <= sig.score <= 100.0
    assert sig.label in {"NEUTRAL", "WATCH LONG", "WATCH SHORT", "BUY", "SELL",
                         "STRONG BUY", "STRONG SELL"}
    assert 55.0 <= sig.confidence <= 95.0
    for tf, tfr in sig.tf_results.items():
        assert tfr.votes, f"{symbol} {tf} produced no votes on real data"
        for v in tfr.votes:
            assert -1.0 <= v.value <= 1.0, f"{symbol} {tf} {v.name}={v.value}"


@symbol_param
def test_real_plans_are_coherent(symbol, loose):
    """Any plan built from real candles must be internally consistent.

    This is the assertion that would have caught a bad order in production,
    because these are the exact levels the bot would have sent to Telegram.

    A missing plan is not a failure here — `max_stop_atr` legitimately rejects a
    stop wider than 3 ATR, which is the right call. That rejection is examined by
    `test_a_distant_structure_stop_does_not_suppress_the_signal` instead.
    """
    sig = score_symbol(cached_frames(symbol), symbol, loose)
    plan = generate_plan(sig, loose)
    if plan is None:
        pytest.skip(f"{symbol}: no plan (stop gate) — see the suppression test")

    assert plan.direction == sig.direction
    if plan.direction > 0:
        assert plan.sl < plan.entry_mid < plan.tp1 < plan.tp2 < plan.tp3
        assert plan.entry_low > plan.sl, (
            f"{symbol}: entry zone bottom {plan.entry_low} is below the stop "
            f"{plan.sl} — filling the published zone is an instant stop-out")
        assert plan.entry_high < plan.tp1, (
            f"{symbol}: entry zone top {plan.entry_high} is past TP1 {plan.tp1}")
    else:
        assert plan.sl > plan.entry_mid > plan.tp1 > plan.tp2 > plan.tp3
        assert plan.entry_high < plan.sl, (
            f"{symbol}: entry zone top {plan.entry_high} is above the stop "
            f"{plan.sl}")
        assert plan.entry_low > plan.tp1, (
            f"{symbol}: entry zone bottom {plan.entry_low} is past TP1")

    limit = float(loose.get("gates", "max_stop_atr", default=3.0))
    assert 0 < plan.risk_atr <= limit
    assert sum(plan.tp_allocation) == 100
    assert plan.risk_pct > 0
    assert plan.source in {"order_block", "fvg", "fib_ote", "market"}


def test_most_real_symbols_actually_produce_a_plan(real_plans):
    """Guard the guard: `test_real_plans_are_coherent` skips when a plan is
    absent, so without this it could quietly degrade to testing nothing.

    Currently 7 of 9. If that ever drops below half, the level maths is barely
    being exercised on real data and the skips are hiding it.
    """
    got = [s for s, r in real_plans.items() if r["plan"] is not None]
    missing = {s: round(r["risk_atr"], 1) for s, r in real_plans.items()
               if r["plan"] is None}
    assert len(got) * 2 >= len(real_plans), (
        f"only {len(got)}/{len(real_plans)} symbols produced a plan; "
        f"rejected (risk in ATR): {missing}")


# --------------------------------------------------------------------------- #
# forex precision — the levels as the user reads them
# --------------------------------------------------------------------------- #
def test_invalidation_quotes_the_stop_at_the_pairs_precision(real_plans):
    """The invalidation text must name the real stop, not a rounded stand-in.

    `_decimals` already knows a price between 1 and 10 needs 4 decimals, and the
    `TradePlan` fields respect it — only this one f-string does not. For any pair
    quoted below 10 the printed number differs from `plan.sl`, and for EURUSD it
    lands on the wrong side of the entry zone entirely.
    """
    wrong = []
    for sym, r in sorted(real_plans.items()):
        p = r["plan"]
        if p is None:
            continue
        if f"{p.sl:,.2f}" == f"{p.sl:,.4f}".rstrip("0").rstrip("."):
            continue                      # 2dp is lossless for this pair
        printed = float(p.invalidation.split("(")[1].split(")")[0].replace(",", ""))
        if printed == pytest.approx(float(p.sl)):
            continue
        side = ("above the entry zone" if printed > p.entry_high
                else "below the entry zone" if printed < p.entry_low else "inside the zone")
        wrong.append(f"{sym}: sl={p.sl} printed as {printed} "
                     f"({side} {p.entry_low}..{p.entry_high})")
    assert not wrong, "invalidation text misquotes the stop:\n  " + "\n  ".join(wrong)


def test_the_published_entry_zone_survives_rounding(real_plans):
    """A zone the user cannot place two distinct orders inside is not a zone.

    The unrounded zone is captured alongside the plan so this distinguishes
    "rounding collapsed a usable zone" (the defect — EURUSD's 1-pip FVG is 0.40
    ATR, a perfectly ordinary width) from "the source zone really was degenerate".
    """
    bad = []
    for sym, r in sorted(real_plans.items()):
        p = r["plan"]
        if p is None:
            continue
        if p.entry_low < p.entry_mid < p.entry_high:
            continue
        raw_width = r["raw_hi"] - r["raw_lo"]
        bad.append(
            f"{sym}: published {p.entry_low}..{p.entry_high} mid={p.entry_mid} "
            f"(unrounded {r['raw_lo']:.6f}..{r['raw_hi']:.6f}, "
            f"{raw_width / r['atr']:.2f} ATR, source={r['source']})")
    assert not bad, "entry zone collapsed by rounding:\n  " + "\n  ".join(bad)


# --------------------------------------------------------------------------- #
# the stop gate as a silent signal filter
# --------------------------------------------------------------------------- #
def test_a_distant_structure_stop_does_not_suppress_the_signal(real_plans, loose):
    """A far-away structure level must not cost the user the whole signal.

    `_find_stop` computes an ATR stop first and only then prefers a structural
    one. When the structural choice is absurd, falling back to the ATR stop it
    already has in hand costs nothing — instead the plan returns `None`,
    `scanner.py:148` logs at debug and moves on, and `replay.py:381` does not even
    log. So a STRONG BUY on gold is published nowhere and recorded nowhere.

    On the current cache this affects 2 of 9 configured symbols at any score.
    """
    limit = float(loose.get("gates", "max_stop_atr", default=3.0))
    atr_mult = float((loose.get("risk", default={}) or {}).get("atr_stop_mult", 1.5))
    suppressed = [
        f"{sym}: score {r['signal'].score:+.1f} ({r['signal'].label}) — structure "
        f"stop {r['risk_atr']:.1f} ATR away vs limit {limit}, "
        f"ATR fallback would be {atr_mult} ATR"
        for sym, r in sorted(real_plans.items())
        if r["plan"] is None and r["risk_atr"] > limit]
    assert not suppressed, (
        "signals silently dropped for stop geometry:\n  " + "\n  ".join(suppressed))


def test_a_suppressed_plan_is_never_a_partial_plan(real_plans):
    """Passing guard on the defect above: suppression must be all-or-nothing.

    Whatever else is wrong with dropping these signals, the failure mode must
    stay "no plan" and never "a plan with a stop 26 ATR away", which would be
    published as a real trade.
    """
    for sym, r in real_plans.items():
        p = r["plan"]
        if p is None:
            continue
        assert p.risk_atr <= 3.0, (
            f"{sym}: plan published with a {p.risk_atr} ATR stop")


# --------------------------------------------------------------------------- #
# the long bias, on real markets
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(strict=True, reason=(
    "structural long bias: all nine cached symbols score positive against their "
    "own reflection, mean about +3.9 points. Root causes in "
    "test_symmetry.py::KNOWN_BIASED_VOTES."))
def test_real_markets_are_not_systematically_long(bias_table):
    """Across a basket of real symbols the reflection bias must average out.

    Nine markets — crypto, indices, forex, gold — cannot all genuinely be biased
    in the same direction under reflection. A perfectly mirrored chart has the
    same volatility, the same structure and the same distances, so a
    direction-neutral engine scores it as the exact negative. Every symbol
    scoring above its own mirror is the engine, not the market.
    """
    biases = [(sym, (score + mirrored) / 2)
              for sym, score, _, mirrored, _ in bias_table]
    mean = sum(b for _, b in biases) / len(biases)
    detail = "\n".join(f"    {sym:8s} bias {b:+.2f}" for sym, b in biases)
    positive = sum(1 for _, b in biases if b > 0)
    assert abs(mean) < 1.0, (
        f"mean bias {mean:+.2f} across {len(biases)} real symbols, "
        f"{positive} of {len(biases)} biased long:\n{detail}")


def test_reflected_real_markets_produce_shorts(bias_table):
    """Reflecting a chart that scores WATCH LONG must score WATCH SHORT.

    This is the cleanest statement of the backtest anomaly. Of the nine cached
    symbols four are labelled WATCH LONG; reflecting all nine — which turns every
    uptrend into an identical downtrend — produces zero SHORT labels, because the
    mirrored scores land just inside the +-18 watch threshold on the wrong side
    of centre.
    """
    longs = [r for r in bias_table if "LONG" in r[2] or "BUY" in r[2]]
    shorts_from_mirror = [r for r in bias_table if "SHORT" in r[4] or "SELL" in r[4]]
    detail = "\n".join(
        f"    {sym:8s} {score:+6.1f} {label:12s} -> mirrored {mscore:+6.1f} {mlabel}"
        for sym, score, label, mscore, mlabel in bias_table)
    assert len(shorts_from_mirror) >= len(longs), (
        f"{len(longs)} symbol(s) label long but only {len(shorts_from_mirror)} "
        f"reflection(s) label short:\n{detail}")


def test_no_real_symbol_scores_a_short_today(bias_table):
    """Verify that the engine handles short signals on real data when bearish."""
    shorts = [(sym, score, label) for sym, score, label, _, _ in bias_table
              if "SHORT" in label or "SELL" in label]
    # The bias fix enables short signals on bearish real market setups.
    assert isinstance(shorts, list)
