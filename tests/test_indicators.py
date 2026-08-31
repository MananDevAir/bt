"""Classic indicators — properties that must hold for any input.

These are hand-rolled in `src/analysis/indicators.py` (pandas_ta is broken on
pandas 3 / numpy 2), so nothing upstream guarantees they are right. Each test
asserts a mathematical property rather than a magic number, so the suite
survives a legitimate refactor but not a wrong one.

Two properties matter most and are checked for every windowed series:

* **No lookahead.** `f(series[:k])` must equal `f(series)[:k]`. A signal bot
  that peeks at future bars backtests beautifully and loses money live.
* **Warm-up honesty.** A windowed series is NaN until `n` bars exist, so
  "not NaN" can be trusted to mean "fully formed".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import indicators as ind

from conftest import BASE, make_ohlcv


# --------------------------------------------------------------------------- #
# moving averages
# --------------------------------------------------------------------------- #
def test_sma_matches_manual_mean(walk):
    c = walk["close"]
    got = ind.sma(c, 20)
    assert np.isnan(got.iloc[18])                      # not enough bars yet
    assert got.iloc[19] == pytest.approx(c.iloc[:20].mean())
    assert got.iloc[-1] == pytest.approx(c.iloc[-20:].mean())


def test_sma_of_a_constant_is_that_constant(flat):
    assert ind.sma(flat["close"], 50).iloc[-1] == pytest.approx(BASE)


def test_ema_is_nan_until_span_bars_exist(walk):
    got = ind.ema(walk["close"], 50)
    assert got.iloc[:49].isna().all()
    assert got.iloc[49:].notna().all()


def test_ema_tracks_but_lags_a_rising_series(ramp):
    """On a monotonic rise the EMA must sit below price and keep climbing."""
    c = ramp["close"]
    e = ind.ema(c, 20).dropna()
    assert (e < c.loc[e.index]).all()
    assert e.diff().dropna().gt(0).all()


def test_rma_is_a_slower_ema(walk):
    """Wilder smoothing with alpha=1/n reacts less than the equivalent EMA."""
    c = walk["close"]
    fast = ind.ema(c, 14).dropna()
    slow = ind.rma(c, 14).dropna()
    common = fast.index.intersection(slow.index)
    assert slow.loc[common].diff().abs().mean() < fast.loc[common].diff().abs().mean()


# --------------------------------------------------------------------------- #
# momentum
# --------------------------------------------------------------------------- #
def test_rsi_stays_in_bounds(walk):
    r = ind.rsi(walk["close"], 14).dropna()
    assert r.between(0.0, 100.0).all()


def test_rsi_is_100_on_an_unbroken_rise(ramp):
    """Documented edge case: a window with no losses is RSI 100, not NaN."""
    assert ind.rsi(ramp["close"], 14).iloc[-1] == pytest.approx(100.0)


def test_rsi_is_0_on_an_unbroken_fall(dump):
    assert ind.rsi(dump["close"], 14).iloc[-1] == pytest.approx(0.0)


def test_rsi_of_a_flat_series_is_not_a_signal(flat):
    """A flat window has zero average loss. The code maps that to 100 rather
    than NaN, so assert the documented behaviour explicitly."""
    r = ind.rsi(flat["close"], 14).dropna()
    assert not r.empty
    assert (r == 100.0).all()


def test_macd_histogram_is_line_minus_signal(walk):
    m = ind.macd(walk["close"])
    assert m["hist"].dropna().equals((m["macd"] - m["signal"]).dropna())


def test_macd_is_positive_in_an_uptrend_and_negative_in_a_downtrend(ramp, dump):
    assert ind.macd(ramp["close"])["macd"].iloc[-1] > 0
    assert ind.macd(dump["close"])["macd"].iloc[-1] < 0


def test_stoch_stays_in_bounds(walk):
    s = ind.stoch(walk["high"], walk["low"], walk["close"])
    assert s["k"].dropna().between(0.0, 100.0).all()
    assert s["d"].dropna().between(0.0, 100.0).all()


def test_stoch_of_a_dead_flat_window_is_50(flat):
    """A zero-range window is neither overbought nor oversold."""
    assert ind.stoch(flat["high"], flat["low"], flat["close"])["k"].iloc[-1] == \
        pytest.approx(50.0)


def test_roc_is_percent_change(walk):
    c = walk["close"]
    expected = 100.0 * (c.iloc[-1] / c.iloc[-11] - 1.0)
    assert ind.roc(c, 10).iloc[-1] == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# volatility
# --------------------------------------------------------------------------- #
def test_bollinger_bands_are_ordered(walk):
    b = ind.bollinger(walk["close"]).dropna()
    assert (b["lower"] <= b["mid"]).all()
    assert (b["mid"] <= b["upper"]).all()


def test_bollinger_percent_b_locates_price_in_the_band(walk):
    """%B = 0 at the lower band, 1 at the upper."""
    c, b = walk["close"], ind.bollinger(walk["close"])
    row = b.dropna().iloc[-1]
    expected = (c.iloc[-1] - row["lower"]) / (row["upper"] - row["lower"])
    assert row["percent_b"] == pytest.approx(expected)


def test_bollinger_collapses_on_a_flat_series(flat):
    """Zero-variance input gives zero bandwidth. `percent_b` is deliberately NaN
    there (the width is masked, not divided by), so only bandwidth is asserted."""
    b = ind.bollinger(flat["close"])
    assert b["bandwidth"].iloc[-1] == pytest.approx(0.0)
    assert np.isnan(b["percent_b"].iloc[-1])


def test_true_range_is_never_negative(walk):
    tr = ind.true_range(walk["high"], walk["low"], walk["close"]).dropna()
    assert (tr >= 0).all()


def test_true_range_covers_the_bar_range(walk):
    """TR is at least high-low, by definition the max of three spans."""
    tr = ind.true_range(walk["high"], walk["low"], walk["close"])
    assert (tr >= (walk["high"] - walk["low"]) - 1e-9).all()


def test_atr_is_positive_and_warms_up(walk):
    a = ind.atr(walk["high"], walk["low"], walk["close"], 14)
    assert a.iloc[:13].isna().all()
    assert (a.dropna() > 0).all()


def test_atr_scales_with_volatility():
    """Double every bar's range and ATR must double."""
    quiet = make_ohlcv(BASE + np.arange(200, dtype=float))
    loud = quiet.copy()
    mid = (loud["high"] + loud["low"]) / 2
    loud["high"] = mid + 2 * (quiet["high"] - mid)
    loud["low"] = mid - 2 * (mid - quiet["low"])
    ratio = ind.atr(loud["high"], loud["low"], loud["close"], 14).iloc[-1] / \
        ind.atr(quiet["high"], quiet["low"], quiet["close"], 14).iloc[-1]
    assert 1.5 < ratio < 2.5


# --------------------------------------------------------------------------- #
# ADX — the min_adx gate depends on this
# --------------------------------------------------------------------------- #
def test_adx_stays_in_bounds(walk):
    a = ind.adx(walk["high"], walk["low"], walk["close"], 14).dropna()
    assert a["adx"].between(0.0, 100.0).all()
    assert a["plus_di"].between(0.0, 100.0).all()
    assert a["minus_di"].between(0.0, 100.0).all()


def test_adx_di_leads_correctly_by_direction(ramp, dump):
    up = ind.adx(ramp["high"], ramp["low"], ramp["close"], 14).iloc[-1]
    down = ind.adx(dump["high"], dump["low"], dump["close"], 14).iloc[-1]
    assert up["plus_di"] > up["minus_di"], "rising market: +DI must lead"
    assert down["minus_di"] > down["plus_di"], "falling market: -DI must lead"


def test_adx_is_high_in_a_trend_and_low_in_chop(ramp, walk):
    trend = ind.adx(ramp["high"], ramp["low"], ramp["close"], 14)["adx"].iloc[-1]
    chop = ind.adx(walk["high"], walk["low"], walk["close"], 14)["adx"].iloc[-1]
    assert trend > 25, f"clean trend gave ADX={trend:.1f}, gate needs >= 20"
    assert trend > chop


def test_adx_warmup_is_double_smoothed(walk):
    """ADX needs 2n-2 bars before it is real (Wilder double smoothing)."""
    a = ind.adx(walk["high"], walk["low"], walk["close"], 14)["adx"]
    assert a.iloc[:25].isna().all()
    assert a.iloc[26:].notna().all()


# --------------------------------------------------------------------------- #
# volume
# --------------------------------------------------------------------------- #
def test_obv_accumulates_on_up_closes(ramp, dump):
    assert ind.obv(ramp["close"], ramp["volume"]).iloc[-1] > 0
    assert ind.obv(dump["close"], dump["volume"]).iloc[-1] < 0


def test_obv_ignores_flat_closes(flat):
    assert ind.obv(flat["close"], flat["volume"]).iloc[-1] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# the bundle + the no-lookahead contract
# --------------------------------------------------------------------------- #
EXPECTED_COLUMNS = {
    "ema20", "ema50", "ema200", "sma50", "sma200", "rsi", "roc",
    "macd_macd", "macd_signal", "macd_hist", "stoch_k", "stoch_d",
    "bb_mid", "bb_upper", "bb_lower", "bb_bandwidth", "bb_percent_b",
    "atr", "atr_pct", "adx", "plus_di", "minus_di",
    "obv", "obv_ema", "vol_sma20", "vol_ratio",
}


def test_classic_frame_column_contract(walk):
    """The confluence layer reads these names by hand; renaming one breaks it."""
    got = set(ind.classic_frame(walk).columns)
    assert EXPECTED_COLUMNS <= got, f"missing: {EXPECTED_COLUMNS - got}"


def test_classic_frame_is_index_aligned(walk):
    assert ind.classic_frame(walk).index.equals(walk.index)


def test_classic_frame_never_looks_ahead(walk):
    """Truncating the input must not change any earlier value.

    The single most important property in the file: if it fails, every backtest
    number the project has produced is fiction.
    """
    cut = 420
    full = ind.classic_frame(walk)
    partial = ind.classic_frame(walk.iloc[:cut])
    pd.testing.assert_frame_equal(
        full.iloc[:cut], partial, check_freq=False,
        obj="classic_frame truncated vs full")


@pytest.mark.parametrize("fn,args", [
    (ind.sma, (20,)), (ind.ema, (20,)), (ind.rma, (14,)), (ind.rsi, (14,)),
    (ind.roc, (10,)),
])
def test_series_indicators_never_look_ahead(walk, fn, args):
    c = walk["close"]
    cut = 300
    full = fn(c, *args)
    partial = fn(c.iloc[:cut], *args)
    pd.testing.assert_series_equal(full.iloc[:cut], partial, check_freq=False)


@pytest.mark.parametrize("fn", [ind.atr, ind.adx])
def test_hlc_indicators_never_look_ahead(walk, fn):
    cut = 300
    full = fn(walk["high"], walk["low"], walk["close"], 14)
    partial = fn(walk["high"].iloc[:cut], walk["low"].iloc[:cut],
                 walk["close"].iloc[:cut], 14)
    if isinstance(full, pd.DataFrame):
        pd.testing.assert_frame_equal(full.iloc[:cut], partial, check_freq=False)
    else:
        pd.testing.assert_series_equal(full.iloc[:cut], partial, check_freq=False)
