"""Modern indicators: SuperTrend, VWAP, Ichimoku, MFI, volume profile,
divergence, volatility regime.

These carry the weight the classics miss. SuperTrend and Ichimoku give a
trend read that survives chop; VWAP is where institutional fills actually
average out; volume profile says which prices the market agreed on; divergence
is the earliest honest warning that momentum is leaving a move.

Same rules as `indicators.py`: vectorised where possible, never forward-looking,
aligned to the input index.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import atr, bollinger, ema, rsi
from .pivots import Pivot, pivots

__all__ = [
    "supertrend", "vwap_session", "vwap_rolling", "anchored_vwap", "ichimoku",
    "mfi", "volume_profile", "VolumeProfile", "divergences", "Divergence",
    "volatility_regime", "modern_frame",
]


# --------------------------------------------------------------------------- #
# SuperTrend
# --------------------------------------------------------------------------- #
def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> pd.DataFrame:
    """ATR trailing-stop trend. `dir` is +1 up / -1 down, `flip` marks changes.

    The band recursion is genuinely sequential, so this is a Python loop over
    (at most) a few hundred bars - measured at well under a millisecond.
    """
    h, l, c = (df[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    a = atr(df["high"], df["low"], df["close"], n).to_numpy(dtype=float)
    hl2 = (h + l) / 2.0
    up_basic, dn_basic = hl2 + mult * a, hl2 - mult * a

    size = len(df)
    upper = np.full(size, np.nan)
    lower = np.full(size, np.nan)
    direction = np.zeros(size, dtype=float)
    line = np.full(size, np.nan)

    started = False
    for i in range(size):
        if np.isnan(a[i]):
            continue
        if not started:
            upper[i], lower[i], direction[i] = up_basic[i], dn_basic[i], 1.0
            line[i] = lower[i]
            started = True
            continue
        pu, pl = upper[i - 1], lower[i - 1]
        upper[i] = up_basic[i] if (up_basic[i] < pu or c[i - 1] > pu) else pu
        lower[i] = dn_basic[i] if (dn_basic[i] > pl or c[i - 1] < pl) else pl
        if c[i] > pu:
            direction[i] = 1.0
        elif c[i] < pl:
            direction[i] = -1.0
        else:
            direction[i] = direction[i - 1]
        line[i] = lower[i] if direction[i] > 0 else upper[i]

    out = pd.DataFrame({"st": line, "st_dir": direction,
                        "st_upper": upper, "st_lower": lower}, index=df.index)
    out["st_dir"] = out["st_dir"].replace(0.0, np.nan)
    out["st_flip"] = out["st_dir"].diff().fillna(0.0) != 0.0
    out.loc[out["st_dir"].isna(), "st_flip"] = False
    return out


# --------------------------------------------------------------------------- #
# VWAP family
# --------------------------------------------------------------------------- #
def _typical(df: pd.DataFrame) -> pd.Series:
    return (df["high"] + df["low"] + df["close"]) / 3.0


def vwap_session(df: pd.DataFrame) -> pd.Series:
    """VWAP re-anchored every UTC day.

    A single universal anchor beats per-exchange session opens here: crypto has
    no session, and for TradFi the UTC day still resets before the cash open.
    """
    tp, vol = _typical(df), df["volume"]
    day = pd.Series(df.index.normalize(), index=df.index)
    pv = (tp * vol).groupby(day).cumsum()
    vv = vol.groupby(day).cumsum()
    return pv / vv.where(vv > 0)


def vwap_rolling(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp, vol = _typical(df), df["volume"]
    pv = (tp * vol).rolling(n, min_periods=n).sum()
    vv = vol.rolling(n, min_periods=n).sum()
    return pv / vv.where(vv > 0)


def anchored_vwap(df: pd.DataFrame, anchor: int | pd.Timestamp) -> pd.Series:
    """VWAP measured from one bar forward - the level a swing's participants own.

    `anchor` is a positional index or a timestamp; bars before it are NaN.
    """
    pos = anchor if isinstance(anchor, int) else int(df.index.searchsorted(anchor))
    pos = max(0, min(pos, len(df) - 1))
    tp, vol = _typical(df).iloc[pos:], df["volume"].iloc[pos:]
    pv, vv = (tp * vol).cumsum(), vol.cumsum()
    return (pv / vv.where(vv > 0)).reindex(df.index)


# --------------------------------------------------------------------------- #
# Ichimoku
# --------------------------------------------------------------------------- #
def _mid(df: pd.DataFrame, n: int) -> pd.Series:
    return (df["high"].rolling(n, min_periods=n).max()
            + df["low"].rolling(n, min_periods=n).min()) / 2.0


def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26,
             senkou: int = 52) -> pd.DataFrame:
    """Tenkan/Kijun plus the cloud *as it sits under the current bar*.

    The spans are shifted forward by `kijun`, so the cloud we read at bar i was
    computed at bar i-26 - projection, not lookahead. Chikou is exposed as
    `chikou_delta` (close now minus close 26 bars ago) rather than a
    back-shifted series, which is the same information without the temptation
    to read a future value.
    """
    conv, base = _mid(df, tenkan), _mid(df, kijun)
    span_a = ((conv + base) / 2.0).shift(kijun)
    span_b = _mid(df, senkou).shift(kijun)
    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([span_a, span_b], axis=1).min(axis=1)
    return pd.DataFrame({
        "tenkan": conv, "kijun": base,
        "span_a": span_a, "span_b": span_b,
        "cloud_top": cloud_top, "cloud_bot": cloud_bot,
        "chikou_delta": df["close"] - df["close"].shift(kijun),
    })


# --------------------------------------------------------------------------- #
# Money Flow Index
# --------------------------------------------------------------------------- #
def mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """RSI's volume-weighted cousin: momentum that only counts if size showed up."""
    tp = _typical(df)
    flow = tp * df["volume"]
    delta = tp.diff()
    pos = flow.where(delta > 0, 0.0).rolling(n, min_periods=n).sum()
    neg = flow.where(delta < 0, 0.0).rolling(n, min_periods=n).sum()
    ratio = pos / neg.where(neg > 0)
    out = 100.0 - 100.0 / (1.0 + ratio)
    return out.mask((neg == 0) & pos.notna(), 100.0)


# --------------------------------------------------------------------------- #
# Volume profile
# --------------------------------------------------------------------------- #
class VolumeProfile:
    """Where volume actually traded, not just how much of it there was.

    poc  - price of the heaviest bin (the market's fair-value magnet)
    vah/val - edges of the 70% value area
    hvns/lvns - heavy and thin bins; price tends to stall at HVNs and slice LVNs
    """

    __slots__ = ("poc", "vah", "val", "centres", "volumes", "hvns", "lvns")

    def __init__(self, poc: float, vah: float, val: float,
                 centres: np.ndarray, volumes: np.ndarray) -> None:
        self.poc, self.vah, self.val = poc, vah, val
        self.centres, self.volumes = centres, volumes
        peak = volumes.max() if volumes.size else 0.0
        self.hvns = [float(c) for c, v in zip(centres, volumes) if peak and v >= 0.70 * peak]
        self.lvns = [float(c) for c, v in zip(centres, volumes) if peak and v <= 0.20 * peak]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"VolumeProfile(poc={self.poc:.2f}, val={self.val:.2f}, vah={self.vah:.2f})"


def volume_profile(df: pd.DataFrame, bars: int = 200, bins: int = 48,
                   value_area: float = 0.70) -> VolumeProfile | None:
    """Volume spread across each bar's true high-low range, not dumped on one price."""
    win = df.tail(bars)
    if len(win) < 10:
        return None
    lo = float(win["low"].min())
    hi = float(win["high"].max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None

    edges = np.linspace(lo, hi, bins + 1)
    bin_lo, bin_hi = edges[:-1][None, :], edges[1:][None, :]
    bar_lo = win["low"].to_numpy(dtype=float)[:, None]
    bar_hi = win["high"].to_numpy(dtype=float)[:, None]
    vol = win["volume"].to_numpy(dtype=float)[:, None]

    overlap = np.clip(np.minimum(bar_hi, bin_hi) - np.maximum(bar_lo, bin_lo), 0.0, None)
    span = overlap.sum(axis=1, keepdims=True)
    # a doji has zero range: put its whole volume in the bin holding its price
    flat = span[:, 0] <= 0
    if flat.any():
        idx = np.clip(np.searchsorted(edges, bar_hi[flat, 0], side="right") - 1, 0, bins - 1)
        overlap[np.where(flat)[0], idx] = 1.0
        span[flat, 0] = 1.0
    weights = np.nan_to_num(overlap / span) * np.nan_to_num(vol)
    per_bin = weights.sum(axis=0)

    centres = (edges[:-1] + edges[1:]) / 2.0
    total = per_bin.sum()
    if total <= 0:
        return None
    order = np.argsort(per_bin)[::-1]
    taken, acc = [], 0.0
    for i in order:
        taken.append(i)
        acc += per_bin[i]
        if acc >= value_area * total:
            break
    return VolumeProfile(float(centres[order[0]]), float(centres[max(taken)]),
                        float(centres[min(taken)]), centres, per_bin)


# --------------------------------------------------------------------------- #
# Divergence
# --------------------------------------------------------------------------- #
class Divergence:
    """One divergence between price and an oscillator, measured pivot to pivot."""

    __slots__ = ("kind", "osc", "bias", "from_ts", "to_ts", "bars_ago",
                 "price_delta", "osc_delta")

    def __init__(self, kind: str, osc: str, bias: int, from_ts: pd.Timestamp,
                 to_ts: pd.Timestamp, bars_ago: int,
                 price_delta: float, osc_delta: float) -> None:
        self.kind, self.osc, self.bias = kind, osc, bias
        self.from_ts, self.to_ts, self.bars_ago = from_ts, to_ts, bars_ago
        self.price_delta, self.osc_delta = price_delta, osc_delta

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Divergence({self.kind} {self.osc} {self.bars_ago} bars ago)"


def _pair_divergence(a: Pivot, b: Pivot, osc: pd.Series, name: str,
                     last_idx: int) -> Divergence | None:
    o_a, o_b = float(osc.iloc[a.idx]), float(osc.iloc[b.idx])
    if not (np.isfinite(o_a) and np.isfinite(o_b)):
        return None
    p_delta, o_delta = b.price - a.price, o_b - o_a
    bars_ago = last_idx - b.idx
    if a.kind == "low":
        if p_delta < 0 and o_delta > 0:
            kind = "bullish"                     # lower low, higher oscillator
        elif p_delta > 0 and o_delta < 0:
            kind = "hidden-bullish"              # higher low, weaker oscillator
        else:
            return None
        bias = +1
    else:
        if p_delta > 0 and o_delta < 0:
            kind = "bearish"                     # higher high, lower oscillator
        elif p_delta < 0 and o_delta > 0:
            kind = "hidden-bearish"
        else:
            return None
        bias = -1
    return Divergence(kind, name, bias, a.ts, b.ts, bars_ago, p_delta, o_delta)


def divergences(df: pd.DataFrame, oscillators: dict[str, pd.Series] | None = None,
                left: int = 3, right: int = 3, max_bars_ago: int = 30,
                max_gap: int = 60) -> list[Divergence]:
    """Regular and hidden divergences on the two most recent same-kind pivots.

    Only pivots that are close enough to matter are compared: the newer leg must
    be within `max_bars_ago` of the last bar and the two legs within `max_gap`
    of each other, otherwise we would be quoting ancient history as a signal.
    """
    if oscillators is None:
        oscillators = {"rsi": rsi(df["close"], 14)}
    pv = pivots(df, left, right)
    last_idx = len(df) - 1
    found: list[Divergence] = []
    for kind in ("low", "high"):
        legs = [p for p in pv if p.kind == kind][-2:]
        if len(legs) < 2:
            continue
        a, b = legs
        if last_idx - b.idx > max_bars_ago or b.idx - a.idx > max_gap:
            continue
        for name, series in oscillators.items():
            d = _pair_divergence(a, b, series, name, last_idx)
            if d is not None:
                found.append(d)
    return found


# --------------------------------------------------------------------------- #
# Volatility regime
# --------------------------------------------------------------------------- #
_REGIMES = ((0.15, "dead"), (0.40, "quiet"), (0.75, "normal"),
            (0.95, "expanding"), (1.01, "extreme"))


def volatility_regime(df: pd.DataFrame, lookback: int = 200) -> dict:
    """Where current volatility sits in its own recent history.

    Absolute ATR is meaningless across BTC and EURUSD; a percentile is not. The
    gate rejects "dead" (nothing moves, stops get ground down) and "extreme"
    (stops are unaffordable and structure is unreliable).
    """
    a = atr(df["high"], df["low"], df["close"], 14)
    atr_pct = 100.0 * a / df["close"]
    if len(df) < 20:
        return {"atr": float("nan"), "atr_pct": float("nan"),
                "percentile": float("nan"), "regime": "unknown",
                "squeeze": False, "bandwidth_percentile": float("nan")}
    rank = atr_pct.rolling(min(lookback, len(df)), min_periods=20).rank(pct=True)
    pct = float(rank.iloc[-1]) if len(rank) and np.isfinite(rank.iloc[-1]) else float("nan")
    label = "unknown"
    if np.isfinite(pct):
        label = next(name for edge, name in _REGIMES if pct < edge)

    bandwidth = bollinger(df["close"], 20, 2.0)["bandwidth"]
    bw_rank = bandwidth.rolling(min(lookback, len(df)), min_periods=20).rank(pct=True)
    bw_pct = float(bw_rank.iloc[-1]) if len(bw_rank) and np.isfinite(bw_rank.iloc[-1]) else float("nan")

    return {
        "atr": float(a.iloc[-1]) if np.isfinite(a.iloc[-1]) else float("nan"),
        "atr_pct": float(atr_pct.iloc[-1]) if np.isfinite(atr_pct.iloc[-1]) else float("nan"),
        "percentile": pct,
        "regime": label,
        "squeeze": bool(np.isfinite(bw_pct) and bw_pct < 0.20),
        "bandwidth_percentile": bw_pct,
    }


# --------------------------------------------------------------------------- #
# bundle
# --------------------------------------------------------------------------- #
def modern_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Every modern indicator that is a per-bar series, as aligned columns.

    Point-in-time objects (volume profile, divergences, regime) are not series
    and are fetched separately by the confluence layer.
    """
    out = supertrend(df)
    out["vwap"] = vwap_session(df)
    out["vwap_roll"] = vwap_rolling(df, 20)
    out = out.join(ichimoku(df))
    out["mfi"] = mfi(df, 14)
    out["ema21"] = ema(df["close"], 21)
    return out

