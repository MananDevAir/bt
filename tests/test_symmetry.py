"""Direction symmetry — the diagnostic behind the long-only backtest.

The 60-day backtest emitted 97 signals and every one was `WATCH LONG`. These
tests establish why, offline and deterministically.

The method: take a price path, reflect it vertically (`mirror_ohlcv`), and score
both. A perfect reflection maps every up-move to an equal down-move, every
support to an equal resistance, every bull order block to a bear one — distances
and proportions are preserved exactly. So a direction-symmetric engine must
score the mirror as the precise negative of the original. Anything else is a
directional bias.

What this suite found (measured, not inferred):

* A clear trend is scored symmetrically — the engine *can* short (see
  `test_downtrend_produces_a_short`, which passes). Strong drift washes the bias
  out entirely.
* A driftless random walk scores +25.6 → `WATCH LONG`, while the same chart
  reflected scores −16.0 → `NEUTRAL`. The bias is roughly +4.8 points and peaks
  in exactly the low-drift chop where most bars live. With `thresholds.watch: 18`
  that is enough to tip sideways markets into LONG and never into SHORT.
* Four votes fail to flip sign under reflection. `equal_levels` (+0.6) and
  `fib_ote` (+0.7) are unconditional; `supertrend` (±1.0, the heaviest vote in
  the engine) and `liq_sweep` are conditional. See `KNOWN_BIASED_VOTES` for the
  verified cause of each.

The bias tests are `xfail(strict=True)`: they document known bugs, and the moment
one is fixed the XPASS fails the suite and forces the marker to be removed.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis.confluence import _compute_tf, score_symbol

from conftest import BASE, make_ohlcv, mirror_ohlcv, multi_tf, trend_path

# Votes are discrete constants (±1.0, ±0.8, ±0.4 ...), so an honest mirror
# reproduces them exactly. Allow only float noise.
VOTE_TOL = 1e-9

# Known-biased votes, with the verified source of each asymmetry.
KNOWN_BIASED_VOTES = {
    # Structural — always bullish, on every shape tested.
    "equal_levels":
        "smc.py:441 builds the list grouped by kind (`for kind in (\"high\", "
        "\"low\")`), so equal_highs all precede equal_lows. confluence.py:185 "
        "then reads `eq[-2:]` as if it were chronological, which always lands "
        "in the equal_lows group: a constant +0.6 whenever two equal lows "
        "exist. Measured at exactly +0.600 on all four shapes and all four "
        "reflections.",
    "fib_ote":
        "confluence.py:281-287 votes +0.7 for any retracement in the 0.618-"
        "0.705 zone and has no bearish branch, so a bearish retracement into "
        "OTE also votes long.",
    # Conditional — only bite when price stays inside the bands / on some data.
    "supertrend":
        "modern.py:52 seeds st_dir = +1.0 unconditionally. Direction only "
        "flips on a 3xATR band break, so a market that never breaks the lower "
        "band keeps the seeded UP — and the seed does not mirror. Bites in "
        "chop only; a decisive trend does break a band.",
    "liq_sweep":
        "the detector caps its output at `unique[-10:]` (smc.py:412) and the "
        "vote reads `sweeps[-2:]` (confluence.py:290). Which two sweeps "
        "survive both truncations is not preserved under reflection, so the "
        "pair that votes can be same-signed both ways. Data-dependent.",
}


def _votes_by_name(df, tf: str = "1h") -> dict[str, float]:
    """Vote name -> summed value for one frame.

    Summed because one name can emit several votes (e.g. two order blocks in
    range), and it is the total contribution that has to mirror.
    """
    result = _compute_tf(df, tf)
    out: dict[str, float] = {}
    for v in result.votes:
        out[v.name] = out.get(v.name, 0.0) + v.value
    return out


def _asymmetric_votes(df) -> list[tuple[str, float, float]]:
    """Return (name, original, mirrored) for every vote that fails to flip."""
    orig = _votes_by_name(df)
    mirr = _votes_by_name(mirror_ohlcv(df))
    offenders: list[tuple[str, float, float]] = []
    for name in sorted(set(orig) | set(mirr)):
        o, m = orig.get(name, 0.0), mirr.get(name, 0.0)
        if abs(m + o) > VOTE_TOL:
            offenders.append((name, o, m))
    return offenders


def _shape(kind: str, n: int = 400):
    """Named price paths, all deterministic."""
    if kind == "trend_up":
        return make_ohlcv(trend_path(n, +0.18), freq="1h")
    if kind == "trend_down":
        return make_ohlcv(trend_path(n, -0.18), freq="1h")
    if kind == "chop":
        return make_ohlcv(trend_path(n, 0.0), freq="1h")
    if kind == "walk":
        rng = np.random.default_rng(42)
        return make_ohlcv(BASE + rng.normal(0.0, 1.0, n).cumsum() * 8.0, freq="1h")
    raise ValueError(kind)


def _bias(cfg, drift_pct: float, bars: int = 320) -> tuple[float, float]:
    """Score a path and its reflection. Returns (score, mirrored_score).

    A symmetric engine gives (x, -x); the midpoint is the directional bias.
    """
    frames = multi_tf(lambda n: trend_path(n, drift_pct), bars=bars)
    mirrored = {tf: mirror_ohlcv(df) for tf, df in frames.items()}
    return (score_symbol(frames, "TESTSYM", cfg).score,
            score_symbol(mirrored, "TESTSYM", cfg).score)


# --------------------------------------------------------------------------- #
# the engine can read direction in a clear trend — these pass
# --------------------------------------------------------------------------- #
def test_downtrend_produces_a_short(neutral_cfg, downtrend_frames):
    """A market falling on every timeframe must score negative and label short.

    This passes: the engine is not blind to downtrends. The long-only backtest
    is caused by the chop bias below, not by an inability to short.
    """
    sig = score_symbol(downtrend_frames, "TESTSYM", neutral_cfg)
    assert sig.score < 0, (
        f"falling market scored {sig.score:+.1f} (label={sig.label})")
    assert sig.direction == -1
    assert "SELL" in sig.label or "SHORT" in sig.label, (
        f"falling market labelled {sig.label!r} at {sig.score:+.1f}")


def test_uptrend_produces_a_long(neutral_cfg, uptrend_frames):
    """Control case for the test above — a rising market must score positive."""
    sig = score_symbol(uptrend_frames, "TESTSYM", neutral_cfg)
    assert sig.score > 0, f"rising market scored {sig.score:+.1f}"
    assert sig.direction == +1
    assert "BUY" in sig.label or "LONG" in sig.label, (
        f"rising market labelled {sig.label!r} at {sig.score:+.1f}")


def test_strong_trends_are_scored_symmetrically(neutral_cfg):
    """At high drift the trend dominates and the bias washes out.

    Kept as a passing guard: it localises the bug to the low-drift regime, so a
    future regression in strong-trend handling is distinguishable from the
    known chop bias.
    """
    for drift in (-0.40, -0.25, +0.25, +0.40):
        score, mirrored = _bias(neutral_cfg, drift)
        bias = (score + mirrored) / 2
        assert abs(bias) < 1.0, (
            f"drift={drift:+.2f}%/bar: {score:+.1f} vs mirrored {mirrored:+.1f} "
            f"— bias {bias:+.2f}")


def test_flat_market_scores_neutral(neutral_cfg, flat):
    """A dead-flat market has no direction, so the score must sit at neutral."""
    frames = {tf: flat for tf in ("1w", "1d", "4h", "1h", "15m")}
    sig = score_symbol(frames, "TESTSYM", neutral_cfg)
    assert sig.label == "NEUTRAL", (
        f"flat market labelled {sig.label!r} at {sig.score:+.1f}")


# --------------------------------------------------------------------------- #
# the long bias — known-failing, documents the bug
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(strict=True, reason=(
    "structural long bias: a zero-drift walk scores +25.6 (WATCH LONG) while "
    "its reflection scores -16.0 (NEUTRAL). Root causes in KNOWN_BIASED_VOTES."))
def test_zero_drift_market_is_not_a_signal(neutral_cfg):
    """A market with no drift must not produce a directional signal.

    This is the test that explains the 97-long / 0-short backtest: at
    `thresholds.watch: 18`, a +4.8 point bias pushes driftless chop over the
    line into WATCH LONG, while the mirror-image chart stays NEUTRAL.
    """
    score, mirrored = _bias(neutral_cfg, 0.0)
    frames = multi_tf(lambda n: trend_path(n, 0.0), bars=320)
    sig = score_symbol(frames, "TESTSYM", neutral_cfg)
    assert sig.label == "NEUTRAL", (
        f"driftless market labelled {sig.label!r} at {score:+.1f}; the same "
        f"chart reflected scores {mirrored:+.1f} — bias {(score + mirrored) / 2:+.2f}")


@pytest.mark.xfail(strict=True, reason=(
    "structural long bias peaks at roughly +4.8 points in the low-drift regime"))
@pytest.mark.parametrize("drift", [-0.10, -0.05, 0.0, +0.05, +0.10])
def test_low_drift_is_scored_symmetrically(neutral_cfg, drift):
    """The reflection of a weakly-trending market must score its exact negative.

    Low drift is where most bars live, and where the bias is largest.
    """
    score, mirrored = _bias(neutral_cfg, drift)
    bias = (score + mirrored) / 2
    assert abs(bias) < 1.0, (
        f"drift={drift:+.2f}%/bar: {score:+.1f} vs mirrored {mirrored:+.1f} "
        f"— bias {bias:+.2f} toward {'long' if bias > 0 else 'short'}")


@pytest.mark.xfail(strict=True, reason=(
    "4 votes never flip sign under reflection — see KNOWN_BIASED_VOTES"))
@pytest.mark.parametrize("shape", ["chop", "walk"])
def test_every_vote_flips_sign_under_reflection(shape):
    """Each individual vote must negate when the price path is reflected.

    This is the test that names the culprit: the failure message lists every
    vote whose mirrored value is not the negative of its original.
    """
    offenders = _asymmetric_votes(_shape(shape))
    detail = "\n".join(
        f"    {name:20s} original={o:+.3f}  mirrored={m:+.3f}  "
        f"(expected {-o:+.3f})\n      cause: {KNOWN_BIASED_VOTES.get(name, 'unknown')}"
        for name, o, m in offenders)
    assert not offenders, (
        f"{len(offenders)} vote(s) do not flip sign when the chart is "
        f"reflected — each is a directional bias:\n{detail}")


@pytest.mark.parametrize("shape", ["trend_up", "trend_down"])
def test_votes_flip_in_clear_trends(shape):
    """In a clear trend the conditional biases disappear.

    Passing guard that pins the blast radius: only `equal_levels` survives here,
    so a regression that breaks trending markets too will show up as a new name.
    """
    offenders = _asymmetric_votes(_shape(shape))
    unexpected = [o for o in offenders if o[0] not in KNOWN_BIASED_VOTES]
    assert not unexpected, f"new asymmetric votes: {unexpected}"


@pytest.mark.xfail(strict=True, reason=(
    "smc.py:441 groups equal_highs before equal_lows, so confluence.py:185's "
    "`eq[-2:]` always reads the equal_lows group"))
@pytest.mark.parametrize("shape", ["chop", "walk", "trend_up"])
def test_equal_levels_list_is_chronological(shape):
    """`confluence.py:185` reads `eq[-2:]` as "the two most recent" levels.

    That is only meaningful if the list is ordered by time. It is not — the
    detector emits all highs then all lows — so the slice deterministically
    picks equal_lows and votes +0.3 for each. This test asserts the ordering
    contract the caller assumes.
    """
    from src.analysis import smc

    levels = smc.detect_equal_levels(_shape(shape), 0.1, 3, 3)
    if len(levels) < 2:
        pytest.skip("not enough equal levels on this shape")
    stamps = [e.last_ts for e in levels]
    assert stamps == sorted(stamps), (
        "equal_levels is not time-ordered: "
        f"{[(e.kind, str(e.last_ts)[:16]) for e in levels]}")


def test_equal_levels_vote_is_not_a_constant():
    """The vote must depend on the chart, not just on list ordering.

    Measured at exactly +0.600 for every shape and every reflection — a vote
    that never varies carries no information and only adds a long tilt.
    """
    values = set()
    for shape in ("chop", "walk", "trend_up", "trend_down"):
        df = _shape(shape)
        for frame in (df, mirror_ohlcv(df)):
            v = _votes_by_name(frame).get("equal_levels")
            if v is not None:
                values.add(round(v, 6))
    assert len(values) > 1, (
        f"equal_levels voted the same value {values} on every shape and every "
        "reflection")


def test_supertrend_seed_does_not_decide_direction():
    """SuperTrend must not report UP on a chart that is falling from bar one.

    Passing guard: in a decisive downtrend the band break does happen, so the
    seed gets overridden. The chop case below is where it does not.
    """
    from src.analysis.modern import supertrend

    df = _shape("trend_down", 400)
    st_dir = supertrend(df)["st_dir"].dropna()
    assert not st_dir.empty, "supertrend produced no direction at all"
    frac_up = float((st_dir > 0).mean())
    assert frac_up < 0.5, (
        f"SuperTrend called UP on {frac_up:.0%} of bars of a falling market")


@pytest.mark.xfail(strict=True, reason=(
    "modern.py:52 seeds st_dir = +1.0, so a low-volatility walk that never "
    "breaks a 3xATR band reports UP on 100% of bars — and so does its mirror"))
def test_supertrend_direction_mirrors_in_a_quiet_walk():
    """Reflecting a chart must invert SuperTrend's up/down split.

    The cleanest isolation of the seed bug. Measured on `walk`: 100% UP on the
    original and 100% UP on the reflection, where the reflection must be 0%.
    Neither run ever breaks a band, so both just report the hardcoded seed for
    all 400 bars — SuperTrend contributes its full +1.0 (the heaviest single
    vote in the engine, category `trend`, weight 30) on the strength of a
    constant.

    `chop` and the two clean trends do mirror correctly, so this is specific to
    markets quiet enough to stay inside a 3xATR envelope.
    """
    from src.analysis.modern import supertrend

    df = _shape("walk", 400)
    up_orig = float((supertrend(df)["st_dir"].dropna() > 0).mean())
    up_mirr = float((supertrend(mirror_ohlcv(df))["st_dir"].dropna() > 0).mean())
    assert abs(up_mirr - (1.0 - up_orig)) < 0.05, (
        f"SuperTrend called UP on {up_orig:.0%} of bars, and on {up_mirr:.0%} "
        f"of the reflected bars (expected {1 - up_orig:.0%})")


def test_supertrend_mirrors_in_trends_and_chop():
    """Passing guard: the seed only decides direction in a quiet walk.

    Pins the blast radius so a future regression in trending markets is
    distinguishable from the known seed bug.
    """
    from src.analysis.modern import supertrend

    for shape in ("chop", "trend_up", "trend_down"):
        df = _shape(shape, 400)
        up_orig = float((supertrend(df)["st_dir"].dropna() > 0).mean())
        up_mirr = float((supertrend(mirror_ohlcv(df))["st_dir"].dropna() > 0).mean())
        assert abs(up_mirr - (1.0 - up_orig)) < 0.10, (
            f"{shape}: UP on {up_orig:.0%} of bars vs {up_mirr:.0%} reflected "
            f"(expected {1 - up_orig:.0%})")
