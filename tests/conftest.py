"""Shared fixtures: synthetic frames with known shapes, plus real cached candles.

Everything here is offline and deterministic — no network, no wall-clock. A test
that passes on a Sunday must pass on a Tuesday.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

# Price level for synthetic series. Far enough from zero that mirroring a path
# around its own mean never approaches 0 (atr_pct = atr/close would explode).
BASE = 10_000.0

# All timeframes the confluence engine knows about, coarsest first.
ALL_TFS = ("1w", "1d", "4h", "1h", "15m")

_TF_FREQ = {"1w": "7D", "1d": "1D", "4h": "4h", "1h": "1h", "15m": "15min"}


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def make_ohlcv(closes: np.ndarray, start: str = "2026-01-01",
               freq: str = "15min", vol: np.ndarray | None = None) -> pd.DataFrame:
    """Build a valid OHLCV frame around a close path (high/low bracket the body)."""
    n = len(closes)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    wiggle = 0.004 * np.abs(closes) + 1e-9
    high = np.maximum(opens, closes) + wiggle
    low = np.minimum(opens, closes) - wiggle
    if vol is None:
        rng = np.random.default_rng(7)
        vol = rng.uniform(800, 1200, n)
    return pd.DataFrame({"open": opens, "high": high, "low": low,
                         "close": closes, "volume": np.asarray(vol, dtype=float)},
                        index=idx)


def mirror_ohlcv(df: pd.DataFrame, pivot: float | None = None) -> pd.DataFrame:
    """Reflect an OHLCV frame vertically: every up-move becomes an equal down-move.

    high/low swap roles, volume is untouched, the index is unchanged. A
    direction-symmetric analysis engine must score the mirror as the exact
    negative of the original — that is what `test_symmetry.py` asserts.
    """
    if pivot is None:
        pivot = float(df["close"].mean())
    two_p = 2.0 * pivot
    out = pd.DataFrame(index=df.index)
    out["open"] = two_p - df["open"]
    out["high"] = two_p - df["low"]      # reflected low becomes the new high
    out["low"] = two_p - df["high"]
    out["close"] = two_p - df["close"]
    out["volume"] = df["volume"].to_numpy()
    return out


def trend_path(n: int, drift_pct: float, seed: int = 11,
               vol_pct: float = 0.6) -> np.ndarray:
    """Geometric random walk with drift — a realistic trend.

    `drift_pct` is the mean per-bar return in percent, `vol_pct` its stdev. A
    straight arithmetic line would be cleaner but is degenerate: constant ATR
    makes `volatility_regime`'s percentile rank meaningless and trips the
    extreme-vol gate on noise alone.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift_pct, vol_pct, n) / 100.0
    return BASE * np.exp(np.cumsum(steps))


def multi_tf(closes_fn, bars: int = 320) -> dict[str, pd.DataFrame]:
    """Build one frame per timeframe from the same shape function.

    Represents "the market looks the same on every timeframe" — the cleanest
    input for asking which direction the engine reports.
    """
    frames: dict[str, pd.DataFrame] = {}
    for tf in ALL_TFS:
        frames[tf] = make_ohlcv(closes_fn(bars), freq=_TF_FREQ[tf])
    return frames


# --------------------------------------------------------------------------- #
# price-shape fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def ramp() -> pd.DataFrame:
    """Strictly rising close: every trend indicator must agree it is bullish."""
    return make_ohlcv(BASE + np.arange(300, dtype=float) * 8.0)


@pytest.fixture
def dump() -> pd.DataFrame:
    """Strictly falling close — the exact mirror of `ramp`.

    The suite had no bearish fixture before, which is part of why the long-only
    backtest went unquestioned for as long as it did.
    """
    return make_ohlcv(BASE + np.arange(299, -1, -1, dtype=float) * 8.0)


@pytest.fixture
def walk() -> pd.DataFrame:
    """Deterministic random walk - the realistic case with pivots and chop."""
    rng = np.random.default_rng(42)
    steps = rng.normal(0.0, 1.0, 600).cumsum()
    return make_ohlcv(BASE + steps * 8.0)


@pytest.fixture
def flat() -> pd.DataFrame:
    """Dead-flat close: no indicator may claim a direction."""
    return make_ohlcv(np.full(300, BASE))


@pytest.fixture
def uptrend_frames() -> dict[str, pd.DataFrame]:
    """Bullish on every timeframe."""
    return multi_tf(lambda n: trend_path(n, +0.18))


@pytest.fixture
def downtrend_frames() -> dict[str, pd.DataFrame]:
    """Bearish on every timeframe. Must produce a SHORT."""
    return multi_tf(lambda n: trend_path(n, -0.18))


# --------------------------------------------------------------------------- #
# the frozen 500-candle fixture (see fixtures/generate.py)
# --------------------------------------------------------------------------- #
CANDLES_JSON = FIXTURE_DIR / "candles_500.json"


def load_frozen_frames() -> dict[str, pd.DataFrame]:
    """Rebuild the committed OHLCV frames from `fixtures/candles_500.json`.

    All five timeframes are resamplings of one 15-minute path, so they agree
    about the last close — the property that makes trade-plan assertions on this
    data meaningful.
    """
    raw = json.loads(CANDLES_JSON.read_text(encoding="utf-8"))
    frames: dict[str, pd.DataFrame] = {}
    for tf in ALL_TFS:                       # coarsest first, a stable order
        rows = raw[tf]
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low",
                                         "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        frames[tf] = df.set_index("ts").astype(float)
    return frames


@pytest.fixture(scope="session")
def frozen_frames() -> dict[str, pd.DataFrame]:
    """The frozen multi-timeframe market. Session-scoped: it is read-only.

    Scoring it must not mutate it — `test_golden.py` asserts exactly that, so a
    shared instance is safe and saves re-parsing 2500 bars per test.
    """
    if not CANDLES_JSON.exists():
        pytest.fail(f"missing {CANDLES_JSON}; run "
                    "`python tests/fixtures/generate.py`")
    return load_frozen_frames()


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def cfg():
    """The real config.yaml — tests should exercise shipped thresholds."""
    from src import config as config_mod
    return config_mod.load(ROOT / "config.yaml")


@pytest.fixture(scope="session")
def neutral_cfg(cfg):
    """config.yaml with symbol_overrides stripped.

    Direction tests use a symbol name the overrides do not touch, but stripping
    them makes that independence explicit rather than incidental.
    """
    import copy
    from src.config import Config

    raw = copy.deepcopy(cfg.raw)
    raw.pop("symbol_overrides", None)
    return Config(raw=raw, symbols=cfg.symbols, macro=cfg.macro, htf=cfg.htf,
                  mtf=cfg.mtf, ltf=cfg.ltf, history=cfg.history,
                  db_path=cfg.db_path)


# --------------------------------------------------------------------------- #
# real cached candles (optional — skipped on a machine with no cache)
# --------------------------------------------------------------------------- #
DB_PATH = ROOT / "data" / "bot.db"


def _require_cache():
    if not DB_PATH.exists():
        pytest.skip("no local cache; run `python -m src.main --crypto` first")


def load_cached(symbol: str, tf: str, bars: int = 500) -> pd.DataFrame:
    """One timeframe of real candles from the Phase 1 cache."""
    from src.data import cache
    from src.store import db
    conn = db.connect(DB_PATH)
    try:
        return cache.load(conn, symbol, tf, bars)
    finally:
        conn.close()


def cached_symbols() -> list[str]:
    """Symbols with candles for every timeframe, in config order.

    Returns [] when there is no cache, so collection-time parametrisation
    degrades to a skip rather than an error.
    """
    if not DB_PATH.exists():
        return []
    from src.store import db
    conn = db.connect(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT symbol, COUNT(DISTINCT timeframe) n FROM candles "
            "GROUP BY symbol HAVING n >= ? ORDER BY symbol",
            (len(ALL_TFS),)).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def cached_frames(symbol: str, bars: int = 500) -> dict[str, pd.DataFrame]:
    """Real multi-timeframe frames, built in the order `router.py` builds them."""
    frames: dict[str, pd.DataFrame] = {}
    for tf in ALL_TFS:
        df = load_cached(symbol, tf, bars)
        if df is not None and not df.empty:
            frames[tf] = df
    return frames


@pytest.fixture
def cached_btc() -> pd.DataFrame:
    """Real BTC 1h candles from the Phase 1 cache, if this machine has any."""
    _require_cache()
    frame = load_cached("BTC", "1h", 500)
    if frame.empty:
        pytest.skip("cache has no BTC 1h rows")
    return frame
