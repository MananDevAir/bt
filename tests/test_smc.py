"""Smart Money Concepts — structure detection against hand-built shapes.

Every test here constructs a price path whose features are known by
construction, so the assertion is "the detector found the thing I put there",
not "the number matches whatever it printed last time".

`smc_scan` output feeds the `structure` category, which carries weight 30 —
equal-heaviest with `trend`. A detector that silently returns nothing costs the
engine a third of its opinion, and nothing upstream would notice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import smc
from src.analysis.pivots import pivots

from conftest import BASE, make_ohlcv


# --------------------------------------------------------------------------- #
# builders for known shapes
# --------------------------------------------------------------------------- #
def zigzag(legs: list[float], bars_per_leg: int = 12) -> pd.DataFrame:
    """Piecewise-linear path through the given price levels.

    `zigzag([100, 120, 110, 140])` rises, pulls back, then makes a higher high —
    a textbook bullish HH/HL sequence.
    """
    closes: list[float] = [legs[0]]
    for target in legs[1:]:
        start = closes[-1]
        closes.extend(np.linspace(start, target, bars_per_leg + 1)[1:])
    return make_ohlcv(np.asarray(closes, dtype=float), freq="1h")


def staircase(n_steps: int, step: float, bars_per_leg: int = 12,
              pullback: float = 0.4) -> pd.DataFrame:
    """Repeated impulse + partial pullback. `step > 0` is bullish HH/HL."""
    legs = [BASE]
    for _ in range(n_steps):
        legs.append(legs[-1] + step)
        legs.append(legs[-1] - step * pullback)
    return zigzag(legs, bars_per_leg)


# --------------------------------------------------------------------------- #
# pivots — everything downstream is built on these
# --------------------------------------------------------------------------- #
def test_pivots_finds_the_turns_of_a_zigzag():
    df = zigzag([BASE, BASE + 400, BASE + 150, BASE + 600, BASE + 300])
    pv = pivots(df, 3, 3)
    assert any(p.kind == "high" for p in pv), "no swing highs in a zigzag"
    assert any(p.kind == "low" for p in pv), "no swing lows in a zigzag"


def test_pivots_needs_right_bars_confirmed():
    """A pivot cannot be reported until `right` bars have closed after it —
    otherwise the detector is reading the future."""
    df = zigzag([BASE, BASE + 400, BASE + 150, BASE + 600])
    pv = pivots(df, 3, 3)
    assert pv, "expected at least one pivot"
    assert max(p.idx for p in pv) <= len(df) - 1 - 3


def test_pivots_finds_nothing_in_a_flat_market(flat):
    """A dead-flat series has no turns. Anything found is an artifact."""
    assert not pivots(flat, 3, 3)


# --------------------------------------------------------------------------- #
# market structure
# --------------------------------------------------------------------------- #
def test_structure_reads_a_rising_staircase_as_bullish():
    result = smc.smc_scan(staircase(6, +400.0))
    assert result["structure_bias"] == "bullish", (
        f"HH/HL staircase read as {result['structure_bias']!r}")


def test_structure_reads_a_falling_staircase_as_bearish():
    result = smc.smc_scan(staircase(6, -400.0))
    assert result["structure_bias"] == "bearish", (
        f"LH/LL staircase read as {result['structure_bias']!r}")


def test_structure_labels_are_valid_enum_members():
    struct = smc.market_structure(staircase(6, +400.0))
    assert struct, "no structure points found"
    valid = set(smc.StructureLabel)
    assert all(p.label in valid for p in struct)
    assert all(p.bias in (-1, 0, +1) for p in struct)


def test_structure_points_are_chronological():
    struct = smc.market_structure(staircase(6, +400.0))
    idxs = [p.pivot.idx for p in struct]
    assert idxs == sorted(idxs)


# --------------------------------------------------------------------------- #
# break of structure / change of character
# --------------------------------------------------------------------------- #
def test_bos_direction_matches_the_break():
    """A staircase up only breaks highs; a staircase down only breaks lows."""
    up = smc.smc_scan(staircase(6, +400.0))["bos"]
    down = smc.smc_scan(staircase(6, -400.0))["bos"]
    assert up, "rising staircase produced no BOS"
    assert down, "falling staircase produced no BOS"
    assert all(b.direction == +1 for b in up), (
        f"bearish BOS in a pure uptrend: {[b.kind for b in up if b.direction < 0]}")
    assert all(b.direction == -1 for b in down)


def test_bos_close_price_actually_passed_the_broken_level():
    """The definition of a break: the close is beyond the swing level."""
    for step in (+400.0, -400.0):
        for b in smc.smc_scan(staircase(6, step))["bos"]:
            if b.direction > 0:
                assert b.close_price > b.broken_level
            else:
                assert b.close_price < b.broken_level


def test_bos_kinds_are_bos_or_choch():
    events = smc.smc_scan(zigzag(
        [BASE, BASE + 500, BASE + 200, BASE + 800, BASE - 300, BASE - 600]))["bos"]
    assert events
    assert {b.kind for b in events} <= {"bos", "choch"}


def test_choch_appears_when_trend_reverses():
    """Up-staircase then down-staircase: the turn must register a CHoCH."""
    legs = [BASE]
    for _ in range(4):
        legs += [legs[-1] + 400, legs[-1] + 400 - 160]
    for _ in range(4):
        legs += [legs[-1] - 400, legs[-1] - 400 + 160]
    events = smc.smc_scan(zigzag(legs))["bos"]
    assert any(b.kind == "choch" for b in events), (
        f"reversal produced no CHoCH, only {[b.kind for b in events]}")


# --------------------------------------------------------------------------- #
# fair value gaps
# --------------------------------------------------------------------------- #
def _with_gap(direction: int) -> pd.DataFrame:
    """A clean 3-candle displacement that must register as an FVG."""
    closes = list(np.full(60, BASE) + np.linspace(0, 5, 60))
    jump = 600.0 * direction
    closes += [BASE + jump, BASE + jump * 1.4, BASE + jump * 1.5]
    closes += list(np.full(40, BASE + jump * 1.45))
    return make_ohlcv(np.asarray(closes, dtype=float), freq="1h")


@pytest.mark.parametrize("direction", [+1, -1])
def test_fvg_detected_on_a_displacement_candle(direction):
    fvgs = smc.detect_fvg(_with_gap(direction))
    assert fvgs, f"no FVG found on a {'bullish' if direction > 0 else 'bearish'} gap"
    assert any(f.direction == direction for f in fvgs), (
        f"gap direction misread: got {[f.direction for f in fvgs]}")


def test_fvg_bounds_are_ordered_and_fill_is_a_fraction():
    for f in smc.detect_fvg(_with_gap(+1)):
        assert f.hi > f.lo, "FVG top must exceed its bottom"
        assert 0.0 <= f.filled_pct <= 1.0
        assert f.still_open == (f.filled_pct < 0.5)


def test_no_fvg_in_a_flat_market(flat):
    assert not smc.detect_fvg(flat)


# --------------------------------------------------------------------------- #
# order blocks
# --------------------------------------------------------------------------- #
def test_order_blocks_have_sane_bounds_and_direction():
    obs = smc.smc_scan(staircase(6, +400.0))["order_blocks"]
    for ob in obs:
        assert ob.hi >= ob.lo
        assert ob.direction in (-1, +1)
        assert ob.displacement >= 0


def test_unmitigated_order_blocks_are_a_subset():
    result = smc.smc_scan(staircase(6, +400.0))
    assert set(id(o) for o in result["order_blocks_unmitigated"]) <= \
        set(id(o) for o in result["order_blocks"])
    assert all(not o.mitigated for o in result["order_blocks_unmitigated"])


# --------------------------------------------------------------------------- #
# liquidity sweeps
# --------------------------------------------------------------------------- #
def test_liquidity_sweep_direction_is_signed():
    df = zigzag([BASE, BASE + 500, BASE + 100, BASE + 480, BASE - 200,
                 BASE + 460, BASE + 900])
    for s in smc.detect_liquidity_sweeps(df):
        assert s.direction in (-1, +1)
        assert s.wick_beyond >= 0


def test_no_liquidity_sweep_in_a_flat_market(flat):
    assert not smc.detect_liquidity_sweeps(flat)


# --------------------------------------------------------------------------- #
# equal levels
# --------------------------------------------------------------------------- #
def test_equal_highs_detected_on_a_double_top():
    """Two swings to the same price is the definition of equal highs."""
    df = zigzag([BASE, BASE + 500, BASE + 150, BASE + 500, BASE + 150])
    eq = smc.detect_equal_levels(df, 0.5, 3, 3)
    assert any(e.kind == "equal_highs" for e in eq), (
        f"double top produced {[e.kind for e in eq]}")


def test_equal_lows_detected_on_a_double_bottom():
    df = zigzag([BASE + 500, BASE, BASE + 350, BASE, BASE + 350])
    eq = smc.detect_equal_levels(df, 0.5, 3, 3)
    assert any(e.kind == "equal_lows" for e in eq), (
        f"double bottom produced {[e.kind for e in eq]}")


def test_equal_levels_cluster_at_least_two_swings():
    df = zigzag([BASE, BASE + 500, BASE + 150, BASE + 500, BASE + 150])
    for e in smc.detect_equal_levels(df, 0.5, 3, 3):
        assert e.count >= 2
        assert e.last_ts >= e.first_ts


# --------------------------------------------------------------------------- #
# premium / discount
# --------------------------------------------------------------------------- #
def test_premium_when_price_sits_at_the_top_of_range(ramp):
    pd_zone = smc.premium_discount(ramp)
    assert pd_zone.zone == "premium", f"top of range read as {pd_zone.zone!r}"
    assert pd_zone.pct > 50


def test_discount_when_price_sits_at_the_bottom_of_range(dump):
    pd_zone = smc.premium_discount(dump)
    assert pd_zone.zone == "discount", f"bottom of range read as {pd_zone.zone!r}"
    assert pd_zone.pct < 50


def test_premium_discount_range_is_consistent(walk):
    z = smc.premium_discount(walk)
    assert z.range_low <= z.equilibrium <= z.range_high
    assert 0.0 <= z.pct <= 100.0
    assert z.zone in {"premium", "discount", "equilibrium"}


# --------------------------------------------------------------------------- #
# the scan contract + no lookahead
# --------------------------------------------------------------------------- #
SCAN_KEYS = {
    "structure", "structure_bias", "bos", "order_blocks",
    "order_blocks_unmitigated", "fvg", "fvg_open", "liquidity_sweeps",
    "equal_levels", "premium_discount",
}


def test_smc_scan_key_contract(walk):
    """The confluence layer reads these keys by hand."""
    assert set(smc.smc_scan(walk)) == SCAN_KEYS


def test_smc_scan_survives_a_short_frame():
    """Short frames happen on fresh symbols — must return empties, not raise."""
    result = smc.smc_scan(make_ohlcv(np.full(35, BASE) + np.arange(35), freq="1h"))
    assert set(result) == SCAN_KEYS


def test_smc_scan_never_looks_ahead(walk):
    """Events detected on a truncated frame must match the full-frame events.

    Only events strictly before the cut are compared: `right`-bar confirmation
    means the last few bars legitimately have no verdict yet.
    """
    cut = 420
    full = smc.smc_scan(walk)
    partial = smc.smc_scan(walk.iloc[:cut])
    horizon = cut - 10  # ignore the unconfirmed tail

    def keys(bos_list):
        return {(b.idx, b.direction, b.kind, round(b.broken_level, 6))
                for b in bos_list if b.idx < horizon}

    assert keys(partial["bos"]) == keys(full["bos"])
