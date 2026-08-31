"""Classic indicators, hand-rolled on pandas/numpy.

No pandas_ta / TA-Lib here: pandas_ta is broken on pandas 3 + numpy 2 (it still
does `from numpy import NaN`) and TA-Lib needs a C toolchain we cannot assume on
a free host. Everything below is vectorised, never looks forward, and returns
series aligned to the input index.

Conventions
-----------
* `n` is the lookback in bars. Windowed series are NaN until `n` bars exist, so
  a caller can always trust "not NaN" to mean "fully formed".
* Wilder-smoothed series (RSI, ATR, ADX) use the recursive `ewm(alpha=1/n)`
  form. TradingView seeds the recursion with an SMA of the first `n` values
  instead of the first value; the gap decays like (1-1/n)^k and is ~1e-9 by the
  time we reach the tail of a 300+ bar frame, which is where we read from.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "sma", "ema", "rma", "rsi", "macd", "stoch", "bollinger",
    "true_range", "atr", "adx", "obv", "roc", "classic_frame",
]


# --------------------------------------------------------------------------- #
# moving averages
# --------------------------------------------------------------------------- #
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing (RMA / SMMA): alpha = 1/n."""
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


# --------------------------------------------------------------------------- #
# momentum
# --------------------------------------------------------------------------- #
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI. 0 when only losses, 100 when only gains."""
    delta = close.diff()
    avg_gain = rma(delta.clip(lower=0.0), n)
    avg_loss = rma((-delta).clip(lower=0.0), n)
    rs = avg_gain / avg_loss.where(avg_loss > 0)
    out = 100.0 - 100.0 / (1.0 + rs)
    # avg_loss == 0 -> rs is inf/NaN; a flat-or-up window is RSI 100.
    out = out.mask((avg_loss == 0) & avg_gain.notna(), 100.0)
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def stoch(high: pd.Series, low: pd.Series, close: pd.Series,
          n: int = 14, smooth_k: int = 3, d: int = 3) -> pd.DataFrame:
    hh = high.rolling(n, min_periods=n).max()
    ll = low.rolling(n, min_periods=n).min()
    span = (hh - ll).where(lambda s: s > 0)
    raw_k = 100.0 * (close - ll) / span
    # a dead-flat window is neither overbought nor oversold
    raw_k = raw_k.mask((hh - ll) == 0, 50.0)
    k = raw_k.rolling(smooth_k, min_periods=smooth_k).mean()
    return pd.DataFrame({"k": k, "d": k.rolling(d, min_periods=d).mean()})


def roc(close: pd.Series, n: int = 10) -> pd.Series:
    """Rate of change in percent."""
    return 100.0 * close.pct_change(n)


# --------------------------------------------------------------------------- #
# volatility
# --------------------------------------------------------------------------- #
def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    """Bands + bandwidth + %B. ddof=0 to match TradingView's stdev."""
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    upper, lower = mid + k * sd, mid - k * sd
    width = (upper - lower).where(lambda s: s > 0)
    return pd.DataFrame({
        "mid": mid,
        "upper": upper,
        "lower": lower,
        "bandwidth": 100.0 * (upper - lower) / mid,
        "percent_b": (close - lower) / width,
    })


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    return pd.concat([high - low, (high - prev).abs(), (low - prev).abs()],
                     axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        n: int = 14) -> pd.Series:
    return rma(true_range(high, low, close), n)


def adx(high: pd.Series, low: pd.Series, close: pd.Series,
        n: int = 14) -> pd.DataFrame:
    """Wilder ADX with +DI/-DI. ADX >= 20 is our "market is trending" gate."""
    up, down = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    atr_n = rma(true_range(high, low, close), n).where(lambda s: s > 0)
    plus_di = 100.0 * rma(plus_dm, n) / atr_n
    minus_di = 100.0 * rma(minus_dm, n) / atr_n
    total = (plus_di + minus_di).where(lambda s: s > 0)
    dx = 100.0 * (plus_di - minus_di).abs() / total
    # rma() skips the leading NaNs of dx, so adx only appears once it is real
    # (index 2n-2), matching Wilder's double-smoothing warm-up.
    return pd.DataFrame({"adx": rma(dx, n),
                         "plus_di": plus_di, "minus_di": minus_di})


# --------------------------------------------------------------------------- #
# volume
# --------------------------------------------------------------------------- #
def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-balance volume. Flat closes contribute nothing."""
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume).cumsum()


# --------------------------------------------------------------------------- #
# bundle
# --------------------------------------------------------------------------- #
def classic_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Every classic indicator for one OHLCV frame, as aligned columns.

    Column names are the contract the confluence layer reads, so they are stable
    even if the internals change.
    """
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
    out = pd.DataFrame(index=df.index)

    for n in (20, 50, 200):
        out[f"ema{n}"] = ema(c, n)
    out["sma50"] = sma(c, 50)
    out["sma200"] = sma(c, 200)

    out["rsi"] = rsi(c, 14)
    out["roc"] = roc(c, 10)
    out = out.join(macd(c).add_prefix("macd_"))
    out = out.join(stoch(h, l, c).add_prefix("stoch_"))
    out = out.join(bollinger(c).add_prefix("bb_"))

    out["atr"] = atr(h, l, c, 14)
    out["atr_pct"] = 100.0 * out["atr"] / c
    out = out.join(adx(h, l, c, 14))

    out["obv"] = obv(c, v)
    out["obv_ema"] = ema(out["obv"], 21)
    out["vol_sma20"] = sma(v, 20)
    out["vol_ratio"] = v / out["vol_sma20"].where(lambda s: s > 0)
    return out

