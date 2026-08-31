"""Golden-file determinism — PLAN.md §17.

> "a fixed 500-candle series must always produce a byte-identical signal JSON"

The confluence engine's contract is *same candles → same score, always*: no
randomness, no wall-clock, no LLM input. This test freezes both ends of that
promise. The input is a committed fixture (`fixtures/candles_500.json`) so the
test does not depend on numpy's RNG staying byte-stable; the output is a
committed snapshot (`fixtures/golden_signal.json`) covering every vote, not just
the headline score, so a drift of 0.001 in one indicator is caught rather than
averaged away.

Refresh the snapshot after an *intentional* engine change:

    UPDATE_GOLDEN=1 python -m pytest tests/test_golden.py

Then read the diff before committing it. A golden file updated without reading
the diff is worse than no golden file — it converts a caught regression into a
recorded one.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.confluence import score_symbol
from src.analysis.levels import generate_plan

from conftest import CANDLES_JSON as CANDLES

FIXTURES = Path(__file__).resolve().parent / "fixtures"
GOLDEN = FIXTURES / "golden_signal.json"

SYMBOL = "GOLDEN"
UPDATING = os.environ.get("UPDATE_GOLDEN") == "1"

# Round to 6 decimals when snapshotting. Tighter than any decision the engine
# makes, loose enough to survive platform float formatting.
PRECISION = 6


# --------------------------------------------------------------------------- #
# load / serialise
# --------------------------------------------------------------------------- #
def _r(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if value != value else round(value, PRECISION)  # NaN -> None


def snapshot(frames: dict[str, pd.DataFrame], cfg) -> dict:
    """Serialise a full scoring run into a stable, diffable dict.

    Votes are sorted by (category, name, detail) rather than left in emission
    order, so a pure reordering inside a vote function is not reported as a
    behaviour change — only values are.
    """
    signal = score_symbol(frames, SYMBOL, cfg)
    plan = generate_plan(signal, cfg)

    out: dict = {
        "symbol": signal.symbol,
        "direction": signal.direction,
        "score": _r(signal.score),
        "label": signal.label,
        "confidence": _r(signal.confidence),
        "raw_score": _r(signal.raw_score),
        "max_possible": _r(signal.max_possible),
        "gate_passed": signal.gate_passed,
        "gates": {k: {gk: _r(gv) if isinstance(gv, (int, float))
                      and not isinstance(gv, bool) else gv
                      for gk, gv in v.items()} if isinstance(v, dict) else v
                  for k, v in sorted(signal.gates.items())},
        "timeframes": {},
    }

    for tf in sorted(signal.tf_results):
        tfr = signal.tf_results[tf]
        votes = sorted(
            ({"name": v.name, "category": v.category,
              "value": _r(v.value), "detail": v.detail} for v in tfr.votes),
            key=lambda d: (d["category"], d["name"], d["detail"] or ""))
        out["timeframes"][tf] = {
            "vote_count": len(votes),
            "vote_sum": _r(sum(v.value for v in tfr.votes)),
            "structure_bias": (tfr.smc or {}).get("structure_bias"),
            "regime": (tfr.regime or {}).get("regime"),
            "votes": votes,
        }

    out["plan"] = None if plan is None else {
        "direction": plan.direction,
        "entry_low": _r(plan.entry_low),
        "entry_high": _r(plan.entry_high),
        "entry_mid": _r(plan.entry_mid),
        "sl": _r(plan.sl),
        "tp1": _r(plan.tp1),
        "tp2": _r(plan.tp2),
        "tp3": _r(plan.tp3),
        "rr": _r(plan.rr),
        "risk_pct": _r(plan.risk_pct),
        "risk_atr": _r(plan.risk_atr),
        "tp_allocation": list(plan.tp_allocation),
        "invalidation": plan.invalidation,
        "holding_horizon": plan.holding_horizon,
        "source": plan.source,
        "trade_type": plan.trade_type,
        "brief_reason": plan.brief_reason,
    }
    return out


def dumps(payload: dict) -> str:
    """The canonical on-disk form — this is the "byte-identical" in the spec."""
    return json.dumps(payload, indent=2, sort_keys=True,
                      ensure_ascii=True) + "\n"


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def frames(frozen_frames) -> dict[str, pd.DataFrame]:
    """A private deep copy — `test_scoring_does_not_mutate_its_input` would
    otherwise be able to corrupt the session-scoped frames for every other file."""
    return {tf: df.copy(deep=True) for tf, df in frozen_frames.items()}


# --------------------------------------------------------------------------- #
# the frozen input itself
# --------------------------------------------------------------------------- #
def test_candle_fixture_is_intact(frames):
    """Guard the guard: if the input drifts, a golden mismatch means nothing."""
    assert set(frames) == {"1w", "1d", "4h", "1h", "15m"}
    for tf, df in frames.items():
        assert len(df) == 500, f"{tf} has {len(df)} bars, expected 500"
        assert df.index.is_monotonic_increasing, f"{tf} is not time-ordered"
        assert not df.index.has_duplicates, f"{tf} has duplicate timestamps"
        assert df.notna().all().all(), f"{tf} contains NaN"
        assert (df["high"] >= df["low"]).all(), f"{tf} has high < low"
        assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
        assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
        assert (df["volume"] > 0).all(), f"{tf} has non-positive volume"


# --------------------------------------------------------------------------- #
# determinism — same input, same output
# --------------------------------------------------------------------------- #
def test_scoring_is_repeatable_within_a_process(frames, cfg):
    """Two runs on identical input must agree byte for byte.

    Catches hidden state: a cached indicator, a mutated input frame, a set whose
    iteration order leaks into a detail string.
    """
    assert dumps(snapshot(frames, cfg)) == dumps(snapshot(frames, cfg))


def test_scoring_does_not_mutate_its_input(frames, cfg):
    """The engine must treat candles as read-only.

    If it mutates them, the second scan of a live cycle sees different data
    from the first and `test_scoring_is_repeatable_within_a_process` would be
    the only thing standing between that and silent corruption.
    """
    before = {tf: df.copy(deep=True) for tf, df in frames.items()}
    snapshot(frames, cfg)
    for tf, original in before.items():
        pd.testing.assert_frame_equal(frames[tf], original, check_freq=False,
                                      obj=f"{tf} mutated by score_symbol")


def test_score_is_independent_of_timeframe_dict_order(frames, cfg):
    """Reversing the insertion order of the frames must not move the score.

    `score_symbol` iterates `frames.items()` and accumulates into a float; the
    total is order-independent only if nothing downstream depends on iteration
    order. Float addition is not associative, so this also bounds the drift.
    """
    forward = score_symbol(frames, SYMBOL, cfg).score
    reverse = score_symbol({tf: frames[tf] for tf in reversed(list(frames))},
                           SYMBOL, cfg).score
    assert forward == pytest.approx(reverse, abs=1e-9)


# --------------------------------------------------------------------------- #
# the golden file
# --------------------------------------------------------------------------- #
def test_matches_golden_file(frames, cfg):
    """The 500-candle fixture must reproduce the committed snapshot exactly."""
    current = dumps(snapshot(frames, cfg))

    if UPDATING or not GOLDEN.exists():
        GOLDEN.write_text(current, encoding="utf-8")
        if not UPDATING:
            pytest.fail(
                f"golden file was missing and has been written to {GOLDEN.name}. "
                "Review it, commit it, and re-run.")
        pytest.skip(f"UPDATE_GOLDEN=1 — rewrote {GOLDEN.name}")

    expected = GOLDEN.read_text(encoding="utf-8")
    if current == expected:
        return

    # Point at the first divergence rather than dumping two 2000-line blobs.
    got, want = json.loads(current), json.loads(expected)
    diffs: list[str] = []

    for key in ("score", "label", "direction", "confidence", "raw_score"):
        if got.get(key) != want.get(key):
            diffs.append(f"  {key}: {want.get(key)!r} -> {got.get(key)!r}")

    for tf in sorted(set(got["timeframes"]) | set(want["timeframes"])):
        g = got["timeframes"].get(tf, {})
        w = want["timeframes"].get(tf, {})
        gv = {(v["category"], v["name"], v["detail"]): v["value"]
              for v in g.get("votes", [])}
        wv = {(v["category"], v["name"], v["detail"]): v["value"]
              for v in w.get("votes", [])}
        for k in sorted(set(gv) | set(wv)):
            if gv.get(k) != wv.get(k):
                diffs.append(f"  {tf} {k[0]}/{k[1]}: {wv.get(k)!r} -> "
                             f"{gv.get(k)!r}  [{k[2]}]")

    if got.get("plan") != want.get("plan"):
        diffs.append(f"  plan: {want.get('plan')!r}\n     -> {got.get('plan')!r}")

    detail = "\n".join(diffs[:40]) or "  (structural difference — diff the files)"
    more = f"\n  ... and {len(diffs) - 40} more" if len(diffs) > 40 else ""
    pytest.fail(
        "engine output drifted from the golden file "
        f"({GOLDEN.name}). Expected -> got:\n{detail}{more}\n\n"
        "If the change is intentional: UPDATE_GOLDEN=1 python -m pytest "
        "tests/test_golden.py — then read the diff before committing.")


def test_golden_file_is_canonically_formatted():
    """The snapshot on disk must be in the exact form `dumps` produces.

    Otherwise a hand-edit or an editor's trailing-newline habit shows up as an
    engine regression on the next run.
    """
    if not GOLDEN.exists():
        pytest.skip("no golden file yet")
    text = GOLDEN.read_text(encoding="utf-8")
    assert text == dumps(json.loads(text)), (
        f"{GOLDEN.name} is not canonically formatted; regenerate with "
        "UPDATE_GOLDEN=1")
