"""Trade plans — the numbers the user is actually told to trade.

`generate_plan` turns a score into an entry zone, a stop and three targets. It is
the last thing in the pipeline and the only part whose output is a price, so an
error here is not a mis-ranked signal — it is a bad order.

Two defects are pinned as `xfail(strict=True)` below, both found by these tests:

1. **The entry zone is not constrained against the stop.** The zone is the full
   width of the order block or FVG it came from, while the stop is measured from
   the zone's *midpoint*. When the zone is wider than the risk distance, its far
   edge sits beyond the stop — so filling at the edge of the published zone is an
   instant stop-out, and TP1 lands inside the zone.
2. **The entry zone depends on `dict` insertion order.** `_find_entry` iterates
   `signal.tf_results` and returns the *first* zone within 3 ATR, though its
   docstring says "nearest". Reordering the input frames moves the entry by over
   1% on the frozen fixture, with no change to the score or the label.

And one vacuous gate, pinned as a passing test: `rr` is always exactly
`tp_r_multiples[1]`, so the `min_rr` check can never reject anything.
"""
from __future__ import annotations

import copy

import pytest

from src.analysis.confluence import score_symbol
from src.analysis.levels import generate_plan
from src.config import Config

from conftest import ALL_TFS, mirror_ohlcv, multi_tf, trend_path

SYM = "TESTSYM"


def _loose_cfg(cfg) -> Config:
    """Thresholds at 1 and no symbol overrides — any non-zero score gets a plan.

    Plan tests need a plan to exist; which tier produced it is irrelevant here
    and is covered by `test_confluence.py`.
    """
    raw = copy.deepcopy(cfg.raw)
    raw["thresholds"] = {"strong": 1, "signal": 1, "watch": 1}
    raw["symbol_overrides"] = {}
    return Config(raw=raw, symbols=cfg.symbols, macro=cfg.macro, htf=cfg.htf,
                  mtf=cfg.mtf, ltf=cfg.ltf, history=cfg.history,
                  db_path=cfg.db_path)


def _plan(frames, cfg, symbol: str = SYM):
    signal = score_symbol(frames, symbol, cfg)
    return signal, generate_plan(signal, cfg)


# --------------------------------------------------------------------------- #
# fixtures: a long and a short, both with plans
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def loose(cfg) -> Config:
    return _loose_cfg(cfg)


@pytest.fixture(scope="module")
def long_plan(frozen_frames, loose):
    signal, plan = _plan(frozen_frames, loose)
    if plan is None:
        pytest.fail(f"frozen market ({signal.label} {signal.score:+.1f}) "
                    "produced no plan; plan tests cannot run")
    assert signal.direction == +1
    return signal, plan


@pytest.fixture(scope="module")
def short_plan(frozen_frames, loose):
    """The frozen market reflected — a genuine short on the same geometry."""
    mirrored = {tf: mirror_ohlcv(frozen_frames[tf]) for tf in ALL_TFS}
    signal, plan = _plan(mirrored, loose)
    if signal.direction != -1 or plan is None:
        pytest.fail(f"mirrored market scored {signal.score:+.1f} "
                    f"({signal.label}) with plan={plan is not None}; "
                    "expected a short with a plan")
    return signal, plan


# --------------------------------------------------------------------------- #
# when a plan exists at all
# --------------------------------------------------------------------------- #
def test_neutral_signals_get_no_plan(cfg, flat):
    """No direction, no trade. Anything else is a plan built on a coin flip."""
    signal = score_symbol({tf: flat for tf in ALL_TFS}, SYM, cfg)
    assert signal.label == "NEUTRAL"
    assert generate_plan(signal, cfg) is None


def test_plan_direction_matches_the_signal(long_plan, short_plan):
    assert long_plan[1].direction == long_plan[0].direction == +1
    assert short_plan[1].direction == short_plan[0].direction == -1


# --------------------------------------------------------------------------- #
# level ordering — the core sanity property
# --------------------------------------------------------------------------- #
def test_long_levels_are_ordered(long_plan):
    """For a long: stop below entry, targets above it, in ascending order."""
    _, p = long_plan
    assert p.sl < p.entry_mid, f"long stop {p.sl} is not below entry {p.entry_mid}"
    assert p.entry_mid < p.tp1 < p.tp2 < p.tp3, (
        f"long targets out of order: entry={p.entry_mid} tp1={p.tp1} "
        f"tp2={p.tp2} tp3={p.tp3}")


def test_short_levels_are_ordered(short_plan):
    """For a short: stop above entry, targets below it, in descending order."""
    _, p = short_plan
    assert p.sl > p.entry_mid, f"short stop {p.sl} is not above entry {p.entry_mid}"
    assert p.entry_mid > p.tp1 > p.tp2 > p.tp3, (
        f"short targets out of order: entry={p.entry_mid} tp1={p.tp1} "
        f"tp2={p.tp2} tp3={p.tp3}")


def test_entry_zone_bounds_are_ordered(long_plan, short_plan):
    for _, p in (long_plan, short_plan):
        assert p.entry_low <= p.entry_mid <= p.entry_high, (
            f"entry zone is inverted: {p.entry_low} / {p.entry_mid} / "
            f"{p.entry_high}")


def test_all_levels_are_positive_prices(long_plan, short_plan):
    """A stop or target at or below zero means the ATR maths overran."""
    for _, p in (long_plan, short_plan):
        for name in ("entry_low", "entry_mid", "entry_high", "sl",
                     "tp1", "tp2", "tp3"):
            assert getattr(p, name) > 0, f"{name} = {getattr(p, name)}"


# --------------------------------------------------------------------------- #
# the entry zone / stop straddle — known defect
# --------------------------------------------------------------------------- #
def _straddles_stop(p) -> bool:
    """True if part of the published entry zone is already beyond the stop."""
    return p.entry_low <= p.sl if p.direction > 0 else p.entry_high >= p.sl


def _straddles_tp1(p) -> bool:
    """True if part of the published entry zone is already past the first target."""
    return p.entry_high >= p.tp1 if p.direction > 0 else p.entry_low <= p.tp1


def _with_wide_zone(signal, width_atr: float = 8.0):
    """Give the signal one unmitigated order block centred on the current price.

    This is what a weekly or daily zone looks like to `_find_entry`: the width
    check that matters is against the *15-minute* ATR (levels.py:73 takes
    `atr_val` from the LTF result), and a higher-timeframe zone is routinely many
    15-minute ATRs wide. Injecting it makes the defect deterministic instead of
    depending on whether a random walk happens to produce one.

    `order_blocks_unmitigated` is priority 1 in `_find_entry`, and `tf_results`
    is ordered coarsest-first, so the 1w entry is what gets picked.
    """
    from src.analysis.smc import OrderBlock

    ltf = signal.tf_results["15m"]
    atr = float(ltf.classic["atr"].iloc[-1])
    close = float(ltf.raw_df["close"].iloc[-1])
    half = width_atr / 2.0 * atr

    zone = OrderBlock(idx=0, ts=ltf.raw_df.index[-1], hi=close + half,
                      lo=close - half, direction=signal.direction,
                      mitigated=False, displacement=2.0)
    signal.tf_results["1w"].smc["order_blocks_unmitigated"] = [zone]
    return signal, atr


@pytest.mark.xfail(strict=True, reason=(
    "levels.py:94-101 takes the entry zone as the full width of the source "
    "order block / FVG but measures risk from its midpoint, with no constraint "
    "between the two. A zone wider than 2x the risk distance straddles the stop, "
    "and higher-timeframe zones are wide relative to the 15m ATR the stop uses."))
def test_entry_zone_never_straddles_the_stop(frozen_frames, loose):
    """Filling anywhere in the published zone must leave the stop unhit.

    Seen live before this test existed: an earlier version of the frozen fixture
    produced a long with zone 12183.77-12491.35 and sl 12189.27 — enter at the
    bottom of the zone the bot printed and you are stopped out on the same
    candle, while `rr` still reads 2.0 because it is measured from the midpoint.

    The stop can never be more than `max_stop_atr` (3.0) ATR from entry, so any
    zone wider than 6 ATR straddles it unconditionally.
    """
    signal = score_symbol(frozen_frames, SYM, loose)
    signal, atr = _with_wide_zone(signal, width_atr=8.0)
    p = generate_plan(signal, loose)
    assert p is not None, "no plan produced; the defect cannot be shown"
    assert not _straddles_stop(p), (
        f"entry zone extends past the stop: zone={p.entry_low}..{p.entry_high} "
        f"sl={p.sl} entry_mid={p.entry_mid} "
        f"(zone half-width {(p.entry_high - p.entry_low) / 2 / atr:.2f} ATR vs "
        f"risk {p.risk_atr:.2f} ATR, source={p.source})")


@pytest.mark.xfail(strict=True, reason=(
    "same root cause as the stop straddle: TP1 is entry_mid + 1R and R is "
    "capped at 3 ATR, so a zone wider than 2 ATR can have its far edge past TP1"))
def test_entry_zone_never_straddles_tp1(frozen_frames, loose):
    """TP1 must sit outside the entry zone, or the trade is already closed."""
    signal = score_symbol(frozen_frames, SYM, loose)
    signal, atr = _with_wide_zone(signal, width_atr=8.0)
    p = generate_plan(signal, loose)
    assert p is not None
    assert not _straddles_tp1(p), (
        f"entry zone extends past TP1: zone={p.entry_low}..{p.entry_high} "
        f"tp1={p.tp1} (half-width {(p.entry_high - p.entry_low) / 2 / atr:.2f} "
        f"ATR vs 1R = {p.risk_atr:.2f} ATR, source={p.source})")


def test_the_frozen_market_itself_produces_a_coherent_plan(long_plan, short_plan):
    """Passing guard: the committed fixture's own plans do not straddle.

    Pins the blast radius of the two defects above — they need a wide source
    zone, and the frozen market supplies a narrow one. If this starts failing,
    the everyday case has regressed, not just the wide-zone corner.
    """
    for _, p in (long_plan, short_plan):
        assert not _straddles_stop(p), (
            f"zone={p.entry_low}..{p.entry_high} sl={p.sl}")
        assert not _straddles_tp1(p), (
            f"zone={p.entry_low}..{p.entry_high} tp1={p.tp1}")


# --------------------------------------------------------------------------- #
# order dependence — known defect
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(strict=True, reason=(
    "_find_entry (levels.py:171-200) iterates signal.tf_results in insertion "
    "order and returns the FIRST zone within 3 ATR, despite its docstring "
    "saying 'nearest'. Reordering the frames dict moves entry and stop."))
def test_entry_zone_is_independent_of_timeframe_order(frozen_frames, loose):
    """The same market must yield the same trade however the frames were built.

    Measured on the frozen fixture: coarsest-first gives entry
    22685.19-22787.22 / sl 22430.10 / risk 1.50 ATR, while alphabetical order
    gives 22930.08-23032.11 / sl 22796.37 / risk 0.91 ATR. Same score (+19.8),
    same label, same `source` ("fib_ote") — a different trade.

    Production is currently saved by `router.py` filling frames in
    `cfg.timeframes` order, so this is latent rather than live. It becomes live
    the moment anything reorders that dict.
    """
    forward = {tf: frozen_frames[tf] for tf in ALL_TFS}
    reverse = {tf: frozen_frames[tf] for tf in reversed(ALL_TFS)}
    _, a = _plan(forward, loose)
    _, b = _plan(reverse, loose)
    assert a is not None and b is not None
    assert (a.entry_low, a.entry_high, a.sl) == (b.entry_low, b.entry_high, b.sl), (
        f"frame order changed the trade:\n"
        f"  coarsest-first: {a.entry_low}..{a.entry_high} sl={a.sl} "
        f"risk={a.risk_atr} ATR\n"
        f"  reversed:       {b.entry_low}..{b.entry_high} sl={b.sl} "
        f"risk={b.risk_atr} ATR")


def test_score_and_label_are_order_independent(frozen_frames, loose):
    """Passing guard: only the *levels* move with frame order, not the score.

    Pins the blast radius of the defect above — if a future change makes the
    score order-dependent too, that is a separate and worse bug.
    """
    forward = score_symbol({tf: frozen_frames[tf] for tf in ALL_TFS}, SYM, loose)
    reverse = score_symbol({tf: frozen_frames[tf] for tf in reversed(ALL_TFS)},
                           SYM, loose)
    assert forward.score == pytest.approx(reverse.score, abs=1e-9)
    assert forward.label == reverse.label
    assert forward.direction == reverse.direction


# --------------------------------------------------------------------------- #
# risk metrics and the gates that use them
# --------------------------------------------------------------------------- #
def test_stop_distance_respects_max_stop_atr(cfg, loose, long_plan, short_plan):
    """levels.py:106 returns None when risk exceeds `max_stop_atr` ATR, so any
    plan that exists must be inside the bound."""
    limit = float(cfg.get("gates", "max_stop_atr", default=3.0))
    for _, p in (long_plan, short_plan):
        assert 0 < p.risk_atr <= limit, f"risk_atr={p.risk_atr} vs limit {limit}"


def test_risk_pct_matches_the_stop_distance(long_plan, short_plan):
    for _, p in (long_plan, short_plan):
        expected = 100.0 * abs(p.entry_mid - p.sl) / p.entry_mid
        assert p.risk_pct == pytest.approx(expected, abs=0.02), (
            f"risk_pct={p.risk_pct} but stop is {expected:.2f}% away")


def test_rr_is_a_constant_and_the_min_rr_gate_is_vacuous(cfg, loose):
    """`rr` is always exactly `tp_r_multiples[1]` — it measures nothing.

    TP2 is defined as `entry_mid + 2R`, so `rr = |tp2 - entry_mid| / R` is 2.0 by
    construction for every plan the bot has ever emitted. That is why every row
    of `data/backtest_report.md` shows RR 2.0, and it means the `min_rr: 1.5`
    gate at levels.py:119 can never reject anything.

    A real R:R would come from the distance to a *structural* target. Pinned so
    the constant is a recorded decision rather than an accident.
    """
    mults = cfg.get("risk", "tp_r_multiples", default=[1.0, 2.0, 3.0])
    min_rr = float(cfg.get("gates", "min_rr", default=1.5))
    assert min_rr <= mults[1], (
        "min_rr now exceeds tp_r_multiples[1]; the gate would reject every "
        "plan, not none of them")

    seen = set()
    for drift in (-0.30, -0.10, 0.10, 0.30):
        frames = multi_tf(lambda n, d=drift: trend_path(n, d), bars=320)
        _, p = _plan(frames, loose)
        if p is not None:
            seen.add(round(p.rr, 6))
    assert seen, "no plan was produced on any drift, so rr was never observed"
    assert seen == {round(float(mults[1]), 6)}, (
        f"rr took values {sorted(seen)}, expected only {mults[1]}")


def test_tp_allocation_is_a_full_split(cfg, long_plan):
    """The three partial exits must add up to the whole position."""
    _, p = long_plan
    assert sum(p.tp_allocation) == 100, f"allocation sums to {sum(p.tp_allocation)}"
    assert len(p.tp_allocation) == 3
    assert all(a > 0 for a in p.tp_allocation)
    assert p.tp_allocation == list(
        cfg.get("risk", "tp_allocation", default=[50, 30, 20]))


def test_tp1_is_exactly_one_r(cfg, long_plan, short_plan):
    """TP1/TP2 are pure R multiples (only TP3 is snapped to structure)."""
    mults = cfg.get("risk", "tp_r_multiples", default=[1.0, 2.0, 3.0])
    for _, p in (long_plan, short_plan):
        risk = abs(p.entry_mid - p.sl)
        assert abs(p.tp1 - p.entry_mid) == pytest.approx(mults[0] * risk, rel=1e-3)
        assert abs(p.tp2 - p.entry_mid) == pytest.approx(mults[1] * risk, rel=1e-3)


def test_tp3_stays_beyond_tp2_after_structure_snapping(loose):
    """`_snap_to_structure` may move TP3 by up to `snap_tolerance_atr` toward a
    level, with no guard that it stays past TP2.

    TP3 - TP2 is 1R, so whenever the risk distance is smaller than the snap
    tolerance the snap can invert the target order. Scanned across drifts here
    because it depends on where the S/R levels land.
    """
    bad = []
    for drift in (-0.30, -0.20, -0.10, -0.05, 0.05, 0.10, 0.20, 0.30):
        frames = multi_tf(lambda n, d=drift: trend_path(n, d), bars=320)
        _, p = _plan(frames, loose)
        if p is None:
            continue
        ordered = p.tp3 > p.tp2 if p.direction > 0 else p.tp3 < p.tp2
        if not ordered:
            bad.append(f"drift={drift:+.2f}: dir={p.direction:+d} "
                       f"tp2={p.tp2} tp3={p.tp3}")
    assert not bad, "structure snapping inverted TP2/TP3:\n  " + "\n  ".join(bad)


# --------------------------------------------------------------------------- #
# the text that reaches the user
# --------------------------------------------------------------------------- #
def test_invalidation_names_the_stop_and_the_right_side(long_plan, short_plan):
    """The invalidation line is what the user reads to know when to give up, so
    it must quote the actual stop and the correct direction of the break."""
    _, long_p = long_plan
    _, short_p = short_plan
    assert f"{long_p.sl:,.2f}" in long_p.invalidation
    assert "below" in long_p.invalidation.lower()
    assert f"{short_p.sl:,.2f}" in short_p.invalidation
    assert "above" in short_p.invalidation.lower()


def test_source_and_trade_type_are_from_the_known_sets(long_plan, short_plan):
    for _, p in (long_plan, short_plan):
        assert p.source in {"order_block", "fvg", "fib_ote", "market"}
        assert p.trade_type in {"Intraday", "Swing", "Short-term", "Positional"}
        assert p.holding_horizon
        assert p.brief_reason, "no human-readable reason for the signal"


def test_prices_are_rounded_for_display(long_plan):
    """Levels are rounded by `_decimals`; a raw float would print 15 digits into
    the Telegram message."""
    _, p = long_plan
    for name in ("entry_low", "entry_mid", "entry_high", "sl", "tp1", "tp2", "tp3"):
        value = getattr(p, name)
        assert round(value, 2) == pytest.approx(value), f"{name}={value!r}"
