"""Source router: symbol -> the right fetcher, with fallback + cache.

Callers ask for "all timeframes for this symbol" and get a dict back. The
router tries the primary source first, falls back to the secondary source on
failure, and finally serves from the SQLite cache if both fail.

Optimisations (v2):
  - Freshness-aware skipping: HTF data that hasn't changed since the last
    scan is served from cache instead of re-fetched.
  - Parallel fetching: timeframes for a single symbol are fetched concurrently
    using a thread pool, cutting wall-clock time ~5×.

Data source chain:
  Crypto:   Binance -> Hyperliquid -> cache
  Indices:  Hyperliquid -> yfinance -> Twelve Data -> cache
  Forex:    Hyperliquid -> yfinance -> Twelve Data -> cache
  DJI:      yfinance -> Twelve Data -> cache
"""
from __future__ import annotations

import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd

from ..config import Config, Symbol, SourceSpec
from . import cache, crypto, resample, sessions
from .budget import Budget, BudgetExceeded

log = logging.getLogger(__name__)

# How long each timeframe's data stays "fresh" — if the cache's latest bar
# is within this window of now, skip the HTTP fetch entirely.
# Set to slightly more than 1 bar duration so we re-fetch right after a new
# bar closes, but not 15 minutes too early.
TF_FRESHNESS_S: dict[str, int] = {
    "15m":  900,        # 15 min — always re-fetch (matches scan interval)
    "1h":   3_600,      # 1 hour
    "4h":   14_400,     # 4 hours
    "1d":   86_400,     # 1 day
    "1w":   604_800,    # 1 week
}

# Max parallel threads per symbol fetch (one thread per timeframe)
MAX_WORKERS = 5


@dataclass
class FetchResult:
    symbol: Symbol
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    source: str = ""
    notes: list[str] = field(default_factory=list)
    ok: bool = False

    def note(self, msg: str) -> None:
        self.notes.append(msg)


class Router:
    def __init__(self, cfg: Config, conn: sqlite3.Connection, budget: Budget) -> None:
        self.cfg = cfg
        self.conn = conn
        self.budget = budget
        self.td_key = Config.secret("TWELVEDATA_KEY")
        self.fallbacks = list(cfg.get("exchange_fallbacks", default=[]) or [])
        self._daily_pulled: set[str] = set()

    def fetch_symbol(self, sym: Symbol, now: datetime | None = None) -> FetchResult:
        res = FetchResult(symbol=sym)
        open_now = sessions.is_open(sym.session, now)
        if not open_now:
            res.note(f"{sym.session} closed - serving cache")

        # Try primary source
        success = self._try_source(sym.primary, sym, res, open_now, now)

        # If primary failed, try fallback
        if not success:
            res.note(f"primary {sym.primary.source} failed, trying fallback {sym.fallback.source}")
            self._try_source(sym.fallback, sym, res, open_now, now)

        # Fill any gaps from cache
        self._fill_from_cache(sym, res)
        res.ok = all(not f.empty for f in res.frames.values()) and bool(res.frames)
        return res

    def _try_source(self, spec: SourceSpec, sym: Symbol, res: FetchResult,
                    open_now: bool, now: datetime | None = None) -> bool:
        """Attempt to fetch all timeframes from a source. Returns True if it got data."""
        try:
            if spec.source == "binance":
                self._fetch_binance(spec.ticker, sym, res, now)
            elif spec.source == "hyperliquid":
                self._fetch_hyperliquid(spec.ticker, sym, res, now)
            elif spec.source == "yfinance":
                self._fetch_yfinance(spec.ticker, sym, res, open_now, now)
            elif spec.source == "twelvedata":
                self._fetch_tradfi(spec.ticker, sym, res, open_now)
            else:
                res.note(f"unknown source: {spec.source}")
                return False

            # Check if we actually got data
            got_data = any(
                tf in res.frames and not res.frames[tf].empty
                for tf in self.cfg.timeframes
            )
            if got_data:
                res.source = spec.source
            return got_data

        except BudgetExceeded as exc:
            res.note(f"budget stop ({spec.source}): {exc}")
            return False
        except Exception as exc:
            res.note(f"{spec.source} error: {exc}")
            return False

    # ------------------------------------------------------------------ #
    # Freshness check — skip re-fetch if cache is recent enough
    # ------------------------------------------------------------------ #
    def _is_cache_fresh(self, sym_name: str, tf: str,
                        now: datetime | None = None) -> bool:
        """True if the cache already has a bar within the current TF window."""
        freshness = TF_FRESHNESS_S.get(tf, 3600)
        last_ts = cache.last_timestamp(self.conn, sym_name, tf)
        if last_ts is None:
            return False
        utc_now = now or datetime.now(timezone.utc)
        age_s = (utc_now - last_ts).total_seconds()
        if age_s < freshness:
            log.debug("%s/%s cache fresh (age %.0fs < %ds), skipping fetch",
                      sym_name, tf, age_s, freshness)
            return True
        return False

    def _tfs_needing_fetch(self, sym: Symbol, res: FetchResult,
                           now: datetime | None = None) -> list[str]:
        """Return only timeframes that actually need a fresh HTTP fetch."""
        need: list[str] = []
        for tf in self.cfg.timeframes:
            if tf in res.frames and not res.frames[tf].empty:
                continue
            if self._is_cache_fresh(sym.name, tf, now):
                # Serve from cache instead of fetching
                cached = cache.load(self.conn, sym.name, tf,
                                    self.cfg.history.get(tf, 300))
                if not cached.empty:
                    res.frames[tf] = cached
                    continue
            need.append(tf)
        return need

    # ------------------------------------------------------------------ #
    # Parallel fetch helper
    # ------------------------------------------------------------------ #
    def _fetch_tf(self, fetch_fn, ticker: str, tf: str,
                  limit: int) -> tuple[str, pd.DataFrame]:
        """Fetch one timeframe — designed to run in a thread."""
        try:
            df = fetch_fn(ticker, tf, limit)
            return (tf, df)
        except Exception as exc:
            log.debug("Fetch %s/%s failed: %s", ticker, tf, exc)
            return (tf, pd.DataFrame())

    def _parallel_fetch(self, fetch_fn, ticker: str, sym: Symbol,
                        res: FetchResult, tfs: list[str]) -> None:
        """Fetch multiple timeframes in parallel threads."""
        if not tfs:
            return
        futures = {}
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(tfs))) as pool:
            for tf in tfs:
                limit = self.cfg.history.get(tf, 300)
                fut = pool.submit(self._fetch_tf, fetch_fn, ticker, tf, limit)
                futures[fut] = tf

            for fut in as_completed(futures):
                tf, df = fut.result()
                if not df.empty:
                    res.frames[tf] = df
                    cache.save(self.conn, sym.name, tf, df)

    # ------------------------------------------------------------------ #
    # Source-specific fetchers (now with freshness + parallel)
    # ------------------------------------------------------------------ #

    def _fetch_binance(self, ticker: str, sym: Symbol, res: FetchResult,
                       now: datetime | None = None) -> None:
        tfs = self._tfs_needing_fetch(sym, res, now)
        if not tfs:
            return

        def _binance_one(t: str, tf: str, limit: int) -> pd.DataFrame:
            try:
                return crypto.fetch(t, tf, limit)
            except crypto.FetchError:
                if self.fallbacks:
                    return crypto.fetch_via_ccxt(t, tf, limit, self.fallbacks)
                raise

        # Parallel fetch all needed timeframes
        futures = {}
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(tfs))) as pool:
            for tf in tfs:
                limit = self.cfg.history.get(tf, 300)
                fut = pool.submit(_binance_one, ticker, tf, limit)
                futures[fut] = tf

            for fut in as_completed(futures):
                tf = futures[fut]
                try:
                    df = fut.result()
                    if not df.empty:
                        res.frames[tf] = df
                        cache.save(self.conn, sym.name, tf, df)
                except Exception as exc:
                    res.note(f"binance {tf}: {exc}")

    def _fetch_hyperliquid(self, ticker: str, sym: Symbol, res: FetchResult,
                           now: datetime | None = None) -> None:
        from . import hyperliquid
        tfs = self._tfs_needing_fetch(sym, res, now)
        if not tfs:
            return
        self._parallel_fetch(hyperliquid.fetch, ticker, sym, res, tfs)

    def _fetch_yfinance(self, ticker: str, sym: Symbol, res: FetchResult,
                        open_now: bool, now: datetime | None = None) -> None:
        from . import yfinance_source
        tfs = self._tfs_needing_fetch(sym, res, now)
        if not tfs:
            return
        self._parallel_fetch(yfinance_source.fetch, ticker, sym, res, tfs)

    # ---- Twelve Data: one deep 15m pull + resample (credit-budgeted) ---------
    def _fetch_tradfi(self, ticker: str, sym: Symbol, res: FetchResult,
                      open_now: bool) -> None:
        from . import tradfi
        if not self.td_key:
            res.note("skipped: no TWELVEDATA_KEY")
            return
        if not open_now:
            return

        base_tf = self.cfg.ltf
        deep = max(self.cfg.history.get(base_tf, 500), 5000)
        base = tradfi.fetch(ticker, base_tf, deep, self.td_key, self.budget)
        if base.empty:
            res.note("twelvedata returned no rows")
            return

        intraday = [tf for tf in self.cfg.timeframes if tf != self.cfg.htf]
        for tf, df in resample.build_all(base, base_tf, intraday).items():
            if tf in res.frames and not res.frames[tf].empty:
                continue
            trimmed = df.tail(self.cfg.history.get(tf, 300))
            res.frames[tf] = trimmed
            cache.save(self.conn, sym.name, tf, trimmed)
        res.note(f"TD 1 credit: {base_tf} x{len(base)} -> {', '.join(intraday)}")

        self._fetch_daily_td(ticker, sym, res)

    def _fetch_daily_td(self, ticker: str, sym: Symbol, res: FetchResult) -> None:
        """Daily bars via Twelve Data — only once per UTC day."""
        from . import tradfi
        htf = self.cfg.htf
        if htf in res.frames and not res.frames[htf].empty:
            return
        stamp = f"{sym.name}:{datetime.now(timezone.utc):%Y-%m-%d}"
        if stamp in self._daily_pulled:
            return
        cached = cache.load(self.conn, sym.name, htf, self.cfg.history.get(htf, 300))
        if not cached.empty:
            fresh_enough = cached.index[-1].date() >= (
                datetime.now(timezone.utc).date() - timedelta(days=1)
            )
            if fresh_enough:
                res.frames[htf] = cached
                return
        try:
            df = tradfi.fetch(ticker, htf, self.cfg.history.get(htf, 300),
                              self.td_key, self.budget)
        except (tradfi.FetchError, BudgetExceeded) as exc:
            res.note(f"TD daily skipped: {exc}")
            return
        if not df.empty:
            res.frames[htf] = df
            cache.save(self.conn, sym.name, htf, df)
            self._daily_pulled.add(stamp)
            res.note("TD 1 credit: daily refresh")

    # ---- Cache backfill -----------------------------------------------------
    def _fill_from_cache(self, sym: Symbol, res: FetchResult) -> None:
        for tf in self.cfg.timeframes:
            if tf in res.frames and not res.frames[tf].empty:
                continue
            cached = cache.load(self.conn, sym.name, tf, self.cfg.history.get(tf, 300))
            if not cached.empty:
                res.frames[tf] = cached
                res.note(f"{tf} from cache ({len(cached)} bars)")
