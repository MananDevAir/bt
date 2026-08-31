"""Timeframe resampling: build 1h / 4h / 1d bars from a deeper 15m series.

This is the trick that keeps Twelve Data inside its 800-credit/day free tier -
one 15m pull becomes three timeframes locally instead of three paid calls.

The final bucket is always dropped unless it is provably complete, because a
partially-filled higher-timeframe bar is exactly the repainting bug this bot is
designed never to have.
"""
from __future__ import annotations

import pandas as pd

# pandas offset aliases ('H' was removed in pandas 3.x - use lowercase 'h')
RULES = {"15m": "15min", "30m": "30min", "1h": "1h", "2h": "2h", "4h": "4h", "1d": "1D"}

AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}

# how many minutes each timeframe spans
MINUTES = {"15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240, "1d": 1440}


def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate an OHLCV frame up to `timeframe`, closed bars only."""
    rule = RULES.get(timeframe)
    if rule is None:
        raise ValueError(f"no resample rule for {timeframe!r}")
    if df is None or df.empty:
        return df

    out = (
        df.resample(rule, label="left", closed="left", origin="epoch")
        .agg(AGG)
        .dropna(subset=["open", "high", "low", "close"])
    )
    return _drop_incomplete(df, out, timeframe)


def _drop_incomplete(src: pd.DataFrame, out: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Remove a trailing bucket that the source data cannot fully cover."""
    if out.empty:
        return out
    span = pd.Timedelta(minutes=MINUTES[timeframe])
    last_bucket_start = out.index[-1]
    # the source must extend to the very end of the bucket for it to be closed
    if src.index[-1] < last_bucket_start + span - pd.Timedelta(minutes=1):
        return out.iloc[:-1]
    return out


def build_all(base: pd.DataFrame, base_tf: str, wanted: list[str]) -> dict[str, pd.DataFrame]:
    """Return {timeframe: frame} for every wanted timeframe >= base_tf."""
    result: dict[str, pd.DataFrame] = {}
    base_minutes = MINUTES[base_tf]
    for tf in wanted:
        if MINUTES[tf] < base_minutes:
            continue  # cannot invent finer granularity than we fetched
        result[tf] = base.copy() if tf == base_tf else resample(base, tf)
    return result
