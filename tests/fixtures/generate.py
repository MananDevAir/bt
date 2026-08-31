"""Regenerate the frozen test fixtures.

Run from the project root:

    python tests/fixtures/generate.py

The candle fixture is committed so the golden-file test does not depend on
numpy's RNG implementation staying byte-stable across versions. Regenerate it
only if you deliberately want a different input series — doing so invalidates
`golden_signal.json`, which must then be refreshed with:

    UPDATE_GOLDEN=1 python -m pytest tests/test_golden.py

Construction: one long 15-minute random walk, resampled up to each timeframe,
last 500 bars of each kept. This matters. Generating five *independent* walks
gives frames that disagree about the current price (the first version of this
file had 15m ending at 12,220 while 1d ended at 17,329), so the trade plan gets
its entry from one timeframe and its bias from another and no assertion about
the result means anything. Resampling one path guarantees what real data
guarantees: every timeframe shares the same last close, and a higher timeframe
is exactly the aggregate of the lower ones.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

BARS = 500          # bars kept per timeframe
BASE = 10_000.0
SEED = 20260101
START = "2026-01-01"

# Coarsest timeframe needs BARS buckets, so the base path must span
# 500 weeks + slack. 1 week = 672 fifteen-minute bars.
WEEKS = BARS + 6
BASE_BARS = WEEKS * 672

# Per-15m-bar log-return parameters. Both are tiny because they compound over
# ~340k bars: DRIFT is set so the whole span rises by roughly e^0.5 (~1.6x),
# which reads as a moderate weekly uptrend and as near-driftless chop on 15m —
# the same way a real multi-year series does. Deliberately moderate: a steep
# trend saturates most votes at +-1.0 and the snapshot then exercises fewer
# branches of the vote functions.
DRIFT = 0.5 / BASE_BARS
VOL = 0.0011

TF_FREQ = {"1w": "7D", "1d": "1D", "4h": "4h", "1h": "1h", "15m": "15min"}
OHLC = {"open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum"}


def base_path() -> pd.DataFrame:
    """The single 15-minute series every timeframe is derived from."""
    from conftest import make_ohlcv

    rng = np.random.default_rng(SEED)
    steps = rng.normal(DRIFT, VOL, BASE_BARS)
    closes = BASE * np.exp(np.cumsum(steps))
    vols = rng.uniform(700.0, 1500.0, BASE_BARS)
    return make_ohlcv(closes, start=START, freq="15min", vol=vols)


def resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if freq == "15min":
        return df
    # No `origin` kwarg: pandas rejects it for non-tick freqs like 7D, and the
    # base path starts exactly at midnight UTC so day/week buckets align to bar
    # boundaries on their own. Deterministic either way — the input is frozen.
    return df.resample(freq).agg(OHLC).dropna()


def main() -> None:
    df = base_path()
    payload: dict[str, list[list[float]]] = {}
    tails: dict[str, pd.DataFrame] = {}

    for tf, freq in TF_FREQ.items():
        tail = resample(df, freq).tail(BARS)
        if len(tail) < BARS:
            raise SystemExit(
                f"{tf}: only {len(tail)} buckets from {BASE_BARS} base bars; "
                "raise WEEKS")
        tails[tf] = tail
        payload[tf] = [
            [int(ts.timestamp() * 1000), round(float(r.open), 6),
             round(float(r.high), 6), round(float(r.low), 6),
             round(float(r.close), 6), round(float(r.volume), 6)]
            for ts, r in tail.iterrows()
        ]

    # The invariant that makes the fixture coherent — assert it here rather than
    # discovering it as a confusing test failure later.
    last = {tf: round(float(t["close"].iloc[-1]), 6) for tf, t in tails.items()}
    if len(set(last.values())) != 1:
        raise SystemExit(f"timeframes disagree about the last close: {last}")

    out = HERE / "candles_500.json"
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")

    total = sum(len(v) for v in payload.values())
    print(f"wrote {out.relative_to(ROOT)} — {len(payload)} timeframes, "
          f"{total} bars, last close {next(iter(last.values())):,.2f}")
    for tf in TF_FREQ:
        t = tails[tf]
        print(f"  {tf:>4}  {t.index[0].date()} -> {t.index[-1].date()}  "
              f"close {t['close'].iloc[0]:>10,.2f} -> {t['close'].iloc[-1]:>10,.2f}")


if __name__ == "__main__":
    main()
