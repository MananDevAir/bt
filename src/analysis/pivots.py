"""Swing pivot detection - shared by the divergence engine and the structure map.

A pivot high at bar `i` means `high[i]` is the strict maximum of the window
`[i-left, i+right]`. Because it needs `right` bars *after* it, a pivot is only
confirmed `right` bars late. That lag is deliberate: confirming earlier would
mean re-drawing structure as new candles arrive, which is exactly the
repainting behaviour the whole design rejects.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class Pivot:
    idx: int              # positional index into the frame
    ts: pd.Timestamp
    price: float
    kind: str             # "high" | "low"


def _find(values: np.ndarray, left: int, right: int, want_max: bool) -> list[int]:
    n = len(values)
    out: list[int] = []
    for i in range(left, n - right):
        window = values[i - left: i + right + 1]
        centre = values[i]
        if np.isnan(centre) or np.isnan(window).any():
            continue
        if want_max:
            if centre == window.max() and (window[:left] < centre).all() \
                    and (window[left + 1:] < centre).all():
                out.append(i)
        else:
            if centre == window.min() and (window[:left] > centre).all() \
                    and (window[left + 1:] > centre).all():
                out.append(i)
    return out


def pivots(df: pd.DataFrame, left: int = 3, right: int = 3) -> list[Pivot]:
    """All confirmed pivots, oldest first, highs and lows interleaved by time."""
    highs = _find(df["high"].to_numpy(dtype=float), left, right, True)
    lows = _find(df["low"].to_numpy(dtype=float), left, right, False)
    found = [Pivot(i, df.index[i], float(df["high"].iloc[i]), "high") for i in highs]
    found += [Pivot(i, df.index[i], float(df["low"].iloc[i]), "low") for i in lows]
    found.sort(key=lambda p: p.idx)
    return found


def last(pv: list[Pivot], kind: str, count: int = 1) -> list[Pivot]:
    """The most recent `count` pivots of one kind, oldest first."""
    matching = [p for p in pv if p.kind == kind]
    return matching[-count:] if count <= len(matching) else matching
