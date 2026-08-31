"""Confluence scoring — score bounds, label mapping, gates, config plumbing.

`score_symbol` is the one function whose output the user actually sees, and it
has a lot of arithmetic between the votes and the label: per-category averaging,
per-timeframe multipliers, normalisation to -100..+100, threshold mapping,
per-symbol threshold overrides, then four gates that can rewrite the label. Each
of those steps is tested here against a property rather than a recorded number.

Direction correctness is *not* tested here — that is `test_symmetry.py`, which
documents the known long bias. This file tests that the machinery around the
score behaves as configured, whatever the score happens to be.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from src.analysis.confluence import DEFAULT_WEIGHTS, score_symbol
from src.config import Config

from conftest import ALL_TFS, BASE, make_ohlcv, multi_tf, trend_path

SYM = "TESTSYM"


def _cfg_with(cfg, **overrides) -> Config:
    """A copy of the real config with top-level keys replaced."""
    raw = copy.deepcopy(cfg.raw)
    raw.update(overrides)
    return Config(raw=raw, symbols=cfg.symbols, macro=cfg.macro, htf=cfg.htf,
                  mtf=cfg.mtf, ltf=cfg.ltf, history=cfg.history,
                  db_path=cfg.db_path)


def _thresholds(cfg, symbol: str = SYM) -> tuple[int, int, int]:
    """The (watch, signal, strong) actually in force for a symbol."""
    t = cfg.get("thresholds", default={}) or {}
    watch, signal = int(t.get("watch", 18)), int(t.get("signal", 40))
    strong = int(t.get("strong", 65))
    ov = (cfg.get("symbol_overrides", default={}) or {}).get(symbol)
    if isinstance(ov, dict):
        watch = int(ov.get("watch", watch))
        signal = int(ov.get("signal", signal))
        strong = int(ov.get("strong", strong))
    return watch, signal, strong


DOWNGRADED = {"WATCH LONG", "WATCH SHORT"}


def _expected_label(score: float, watch: int, signal: int, strong: int) -> str:
    a = abs(score)
    if a >= strong:
        return "STRONG BUY" if score > 0 else "STRONG SELL"
    if a >= signal:
        return "BUY" if score > 0 else "SELL"
    if a >= watch:
        return "WATCH LONG" if score > 0 else "WATCH SHORT"
    return "NEUTRAL"


# --------------------------------------------------------------------------- #
# the fixture order must match production
# --------------------------------------------------------------------------- #
def test_test_timeframe_order_matches_config(cfg):
    """`ALL_TFS` exists so fixtures are built in the order the bot builds them.

    `router.py` fills `res.frames` by iterating `cfg.timeframes`, and several
    things downstream depend on that insertion order (see
    `test_plan.py::test_entry_zone_is_independent_of_timeframe_order`). If the
    two ever drift, the whole suite silently stops testing production's path.
    """
    assert tuple(cfg.timeframes) == ALL_TFS


# --------------------------------------------------------------------------- #
# score bounds and internal consistency
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("drift", [-0.30, -0.10, 0.0, +0.10, +0.30])
def test_score_stays_in_range(neutral_cfg, drift):
    sig = score_symbol(multi_tf(lambda n: trend_path(n, drift)), SYM, neutral_cfg)
    assert -100.0 <= sig.score <= 100.0


@pytest.mark.parametrize("drift", [-0.30, -0.10, 0.0, +0.10, +0.30])
def test_direction_agrees_with_the_sign_of_the_score(neutral_cfg, drift):
    sig = score_symbol(multi_tf(lambda n: trend_path(n, drift)), SYM, neutral_cfg)
    assert sig.direction == (+1 if sig.score > 0 else (-1 if sig.score < 0 else 0))


def test_score_is_raw_over_max_possible(neutral_cfg, frozen_frames):
    """The reported score must be the normalisation of the reported internals."""
    sig = score_symbol(frozen_frames, SYM, neutral_cfg)
    assert sig.max_possible > 0
    expected = max(-100.0, min(100.0, 100.0 * sig.raw_score / sig.max_possible))
    assert sig.score == pytest.approx(round(expected, 1))


def test_confidence_is_clamped_to_its_documented_band(neutral_cfg, frozen_frames):
    """`confidence` is floored at 55 and capped at 95 — never 0-100.

    Worth pinning because a reader of the Telegram message sees "55%" and may
    read it as a real probability; it is the floor, not a measurement.
    """
    sig = score_symbol(frozen_frames, SYM, neutral_cfg)
    assert 55.0 <= sig.confidence <= 95.0


# --------------------------------------------------------------------------- #
# vote contract
# --------------------------------------------------------------------------- #
def test_every_vote_is_in_range_and_in_a_weighted_category(cfg, frozen_frames):
    """A vote outside -1..+1 silently breaks normalisation; a vote in a category
    with no configured weight is dropped on the floor and never scored."""
    weights = cfg.get("weights", default=DEFAULT_WEIGHTS) or DEFAULT_WEIGHTS
    sig = score_symbol(frozen_frames, SYM, cfg)
    assert sig.tf_results, "no timeframe produced a result"
    for tf, tfr in sig.tf_results.items():
        assert tfr.votes, f"{tf} produced no votes at all"
        for v in tfr.votes:
            assert -1.0 <= v.value <= 1.0, f"{tf} {v.name}={v.value}"
            assert v.category in weights, (
                f"{tf} {v.name} is in category {v.category!r}, which has no "
                f"weight — it cannot affect the score")


def test_every_timeframe_contributes_votes(cfg, frozen_frames):
    """All five timeframes must score. A silently-skipped one costs its full
    `tf_multiplier` share of the score and nothing upstream logs it."""
    sig = score_symbol(frozen_frames, SYM, cfg)
    assert set(sig.tf_results) == set(ALL_TFS)


# --------------------------------------------------------------------------- #
# label mapping
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("drift", [-0.45, -0.25, -0.10, 0.0, +0.10, +0.25, +0.45])
def test_label_follows_the_configured_thresholds(neutral_cfg, drift):
    """The label must be the threshold mapping of the score.

    The only permitted deviation is a gate downgrade, which caps BUY/STRONG at
    WATCH — so a mismatch is only acceptable when a downgrade gate fired and the
    label moved *down* a tier.
    """
    sig = score_symbol(multi_tf(lambda n: trend_path(n, drift)), SYM, neutral_cfg)
    watch, signal, strong = _thresholds(neutral_cfg)
    expected = _expected_label(sig.score, watch, signal, strong)
    if sig.label == expected:
        return
    downgrades = [k for k, v in sig.gates.items() if v.get("action") == "downgrade"]
    assert downgrades and sig.label in DOWNGRADED, (
        f"score {sig.score:+.1f} mapped to {sig.label!r}, expected {expected!r} "
        f"(thresholds watch={watch} signal={signal} strong={strong}, "
        f"gates={list(sig.gates)})")


def test_a_score_below_watch_is_neutral(neutral_cfg, flat):
    watch, _, _ = _thresholds(neutral_cfg)
    sig = score_symbol({tf: flat for tf in ALL_TFS}, SYM, neutral_cfg)
    assert abs(sig.score) < watch
    assert sig.label == "NEUTRAL"


def test_raising_the_watch_threshold_silences_a_marginal_signal(cfg, frozen_frames):
    """The frozen market scores just under +20 — a WATCH LONG at watch=18.

    Push the threshold above it and the same market must go quiet. This is the
    mechanism `symbol_overrides` uses, so it is worth proving it bites.
    """
    loud = score_symbol(frozen_frames, SYM, _cfg_with(
        cfg, thresholds={"strong": 65, "signal": 40, "watch": 18},
        symbol_overrides={}))
    quiet = score_symbol(frozen_frames, SYM, _cfg_with(
        cfg, thresholds={"strong": 65, "signal": 40, "watch": 80},
        symbol_overrides={}))
    assert loud.score == pytest.approx(quiet.score), "threshold moved the score"
    assert loud.label != "NEUTRAL"
    assert quiet.label == "NEUTRAL", (
        f"score {quiet.score:+.1f} still labelled {quiet.label!r} at watch=80")


def test_symbol_overrides_take_precedence_over_global_thresholds(cfg, frozen_frames):
    """`symbol_overrides` is how the backtest-tuned per-symbol watch levels are
    applied. Same data, same score, different symbol name, different label."""
    base = {"strong": 65, "signal": 40, "watch": 18}
    tuned = _cfg_with(cfg, thresholds=base,
                      symbol_overrides={"PICKY": {"watch": 90}})
    normal = score_symbol(frozen_frames, "TESTSYM", tuned)
    picky = score_symbol(frozen_frames, "PICKY", tuned)
    assert normal.score == pytest.approx(picky.score)
    assert normal.label != "NEUTRAL"
    assert picky.label == "NEUTRAL", (
        f"override ignored: {picky.score:+.1f} -> {picky.label!r}")


# --------------------------------------------------------------------------- #
# weights and multipliers
# --------------------------------------------------------------------------- #
def test_zero_weights_produce_no_signal(cfg, frozen_frames):
    """With every weight at zero `max_possible` is 0 and the engine must bail to
    a clean NEUTRAL rather than dividing by zero."""
    zeroed = _cfg_with(cfg, weights={k: 0 for k in DEFAULT_WEIGHTS})
    sig = score_symbol(frozen_frames, SYM, zeroed)
    assert sig.score == 0.0
    assert sig.label == "NEUTRAL"
    assert sig.direction == 0


def test_a_single_weighted_category_isolates_that_category(cfg, frozen_frames):
    """Weighting only `trend` must give the average trend vote, scaled to 100.

    This is the clearest statement of what the score means, and it catches a
    normalisation change that a whole-config test would average away.
    """
    only_trend = _cfg_with(cfg, weights={"trend": 10}, symbol_overrides={})
    sig = score_symbol(frozen_frames, SYM, only_trend)
    tf_mult = cfg.get("tf_multiplier", default={}) or {}

    num = den = 0.0
    for tf, tfr in sig.tf_results.items():
        mult = float(tf_mult.get(tf, 1.0))
        trend = [v.value for v in tfr.votes if v.category == "trend"]
        if trend:
            num += mult * 10 * (sum(trend) / len(trend))
        den += mult * 10
    assert sig.score == pytest.approx(round(100.0 * num / den, 1))


def test_timeframe_multipliers_shift_the_score(cfg, frozen_frames):
    """Weighting the weekly 4x vs 1x must change the answer — otherwise
    `tf_multiplier` in config.yaml is decoration."""
    flat_mult = _cfg_with(cfg, tf_multiplier={tf: 1.0 for tf in ALL_TFS},
                          symbol_overrides={})
    htf_heavy = _cfg_with(cfg, tf_multiplier={"1w": 20.0, "1d": 1.0, "4h": 1.0,
                                              "1h": 1.0, "15m": 1.0},
                          symbol_overrides={})
    a = score_symbol(frozen_frames, SYM, flat_mult).score
    b = score_symbol(frozen_frames, SYM, htf_heavy).score
    assert a != pytest.approx(b), (
        f"tf_multiplier had no effect: both scored {a:+.1f}")


def test_absent_categories_still_dilute_the_score(cfg, frozen_frames):
    """Documented quirk, not a bug: `max_possible` adds every configured
    category's weight for every timeframe, whether or not that category emitted
    any votes (confluence.py:415). So adding a weighted category that never
    fires pulls every score toward zero.

    Pinned because it is surprising, and because a future "fix" that only
    counts categories with votes would move every score in the project —
    including the thresholds tuned against them.
    """
    with_ghost = _cfg_with(cfg, weights={**DEFAULT_WEIGHTS, "ghost": 100},
                           symbol_overrides={})
    baseline = score_symbol(frozen_frames, SYM,
                            _cfg_with(cfg, symbol_overrides={}))
    diluted = score_symbol(frozen_frames, SYM, with_ghost)
    assert abs(diluted.score) < abs(baseline.score), (
        f"a never-firing category did not dilute: {baseline.score:+.1f} -> "
        f"{diluted.score:+.1f}")


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #
def test_gate_actions_are_from_the_known_set(cfg, frozen_frames):
    sig = score_symbol(frozen_frames, SYM, cfg)
    for name, g in sig.gates.items():
        assert g.get("action") in {"note", "downgrade", "drop"}, (
            f"gate {name} has action {g.get('action')!r}")
        assert g.get("detail"), f"gate {name} has no detail for the user"


def test_a_downgrade_gate_caps_the_label_at_watch(cfg, frozen_frames):
    """A conflicting higher timeframe must not leave a BUY/SELL standing.

    The frozen market is a long whose 1d structure is bearish, so `htf_conflict`
    fires on it. Thresholds are dropped to 1 so the *ungated* label would be
    STRONG BUY — otherwise the score sits in the WATCH band anyway and the cap
    is unobservable.
    """
    tuned = _cfg_with(cfg, thresholds={"strong": 1, "signal": 1, "watch": 1},
                      symbol_overrides={})
    sig = score_symbol(frozen_frames, SYM, tuned)
    assert sig.score > 0
    assert "htf_conflict" in sig.gates, f"gates: {list(sig.gates)}"
    assert sig.label == "WATCH LONG", (
        f"htf_conflict fired but the label is {sig.label!r} at "
        f"{sig.score:+.1f} (would be STRONG BUY ungated)")


def test_a_downgrade_gate_never_silences_a_watch(cfg, frozen_frames):
    """The cap floors at WATCH — it cannot demote a signal to NEUTRAL.

    So a conflicted signal is still published and still generates a trade plan;
    the gate costs it 10 confidence points and a tier, nothing more. Worth
    pinning because "downgrade" reads as if it might suppress the alert.
    """
    sig = score_symbol(frozen_frames, SYM, _cfg_with(cfg, symbol_overrides={}))
    assert any(g.get("action") == "downgrade" for g in sig.gates.values())
    assert sig.label in DOWNGRADED
    assert sig.direction != 0


def test_adx_gate_fires_on_the_frozen_market(cfg, frozen_frames):
    """`min_adx` flags range conditions. The frozen market sits at ADX 19 on 4h,
    just under the configured 20, so the note must appear — and must say which
    timeframe and value triggered it, since that text reaches the user."""
    sig = score_symbol(frozen_frames, SYM, cfg)
    assert "adx_weak" in sig.gates, f"gates: {list(sig.gates)}"
    detail = sig.gates["adx_weak"]["detail"]
    assert "ADX=" in detail and "range mode" in detail
    assert any(tf in detail for tf in cfg.mtf), (
        f"gate does not name the timeframe it read: {detail!r}")


def test_adx_gate_cannot_fire_on_a_dead_flat_tape(cfg, flat):
    """Documented blind spot: on a zero-range series ADX is NaN on every bar
    (+DM and -DM are both 0, so DX is 0/0), and the gate skips NaN. So the
    flattest possible market — the one most deserving a "range mode" note —
    gets none.

    Not obviously wrong (NaN means "unknown", not "low"), but it means
    `adx_weak` cannot be relied on as the range detector for dead symbols.
    """
    from src.analysis import indicators as ind

    assert ind.adx(flat["high"], flat["low"], flat["close"], 14)["adx"].isna().all()
    sig = score_symbol({tf: flat for tf in ALL_TFS}, SYM, cfg)
    assert "adx_weak" not in sig.gates


def test_volatility_gate_fires_on_a_dead_tape(cfg):
    """A series whose range collapses at the end must trip `vol_dead`."""
    n = 400
    closes = np.concatenate([
        BASE + np.random.default_rng(3).normal(0, 60, n - 120).cumsum(),
        np.full(120, BASE),          # volatility goes to zero
    ])
    frames = {tf: make_ohlcv(closes, freq="1h") for tf in ALL_TFS}
    sig = score_symbol(frames, SYM, cfg)
    assert "vol_dead" in sig.gates, (
        f"a flatlining tape produced no dead-volatility gate: {list(sig.gates)}")
    assert sig.gates["vol_dead"]["action"] == "downgrade"


def test_gate_passed_is_currently_unfalsifiable(cfg, frozen_frames):
    """`gate_passed` is False only when some gate emits `action: "drop"`
    (confluence.py:542) — and no gate ever does. Every gate is "note" or
    "downgrade", so the field is a constant True and the `if not gate_passed`
    branches downstream are dead code.

    Asserted rather than left implicit so that adding a real drop gate breaks
    this test and forces a look at the consumers of the flag.
    """
    from src.analysis import confluence

    src = confluence.__file__
    with open(src, encoding="utf-8") as fh:
        body = fh.read()
    assert '"action": "drop"' not in body, (
        "a gate now emits a drop action — `gate_passed` is finally meaningful; "
        "check every consumer and delete this test")
    assert score_symbol(frozen_frames, SYM, cfg).gate_passed is True


# --------------------------------------------------------------------------- #
# degenerate inputs
# --------------------------------------------------------------------------- #
def test_no_frames_is_a_clean_neutral(cfg):
    sig = score_symbol({}, SYM, cfg)
    assert (sig.score, sig.label, sig.direction) == (0.0, "NEUTRAL", 0)
    assert sig.tf_results == {}


def test_short_frames_are_skipped_not_scored(cfg):
    """Under 30 bars a timeframe is dropped (confluence.py:394). A fresh listing
    must not be scored on 20 candles."""
    short = make_ohlcv(BASE + np.arange(20, dtype=float), freq="1h")
    sig = score_symbol({tf: short for tf in ALL_TFS}, SYM, cfg)
    assert sig.tf_results == {}
    assert sig.label == "NEUTRAL"


def test_a_partial_timeframe_set_still_scores(cfg, frozen_frames):
    """Live data is often missing a timeframe; the engine must score what it has."""
    sig = score_symbol({"1h": frozen_frames["1h"]}, SYM, cfg)
    assert set(sig.tf_results) == {"1h"}
    assert -100.0 <= sig.score <= 100.0


def test_none_and_empty_frames_are_tolerated(cfg, frozen_frames):
    import pandas as pd
    frames = {"1w": None, "1d": pd.DataFrame(), "1h": frozen_frames["1h"]}
    sig = score_symbol(frames, SYM, cfg)
    assert set(sig.tf_results) == {"1h"}


def test_scoring_an_unknown_symbol_name_uses_global_thresholds(cfg, frozen_frames):
    """A symbol with no override entry must not crash on the lookup."""
    sig = score_symbol(frozen_frames, "NOT_IN_CONFIG_AT_ALL", cfg)
    watch, signal, strong = _thresholds(cfg, "NOT_IN_CONFIG_AT_ALL")
    expected = _expected_label(sig.score, watch, signal, strong)
    downgraded = any(v.get("action") == "downgrade" for v in sig.gates.values())
    assert sig.label == expected or downgraded
