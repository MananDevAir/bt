"""Typed configuration loader.

Reads config.yaml plus .env and validates the parts the rest of the bot
depends on, so a typo fails here instead of three modules later.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

VALID_SOURCES = {"binance", "hyperliquid", "yfinance", "twelvedata"}
VALID_SESSIONS = {"always", "us_cash", "fx_week"}


@dataclass(frozen=True)
class SourceSpec:
    source: str
    ticker: str


@dataclass(frozen=True)
class Symbol:
    name: str
    primary: SourceSpec
    fallback: SourceSpec
    session: str

    # Convenience aliases for code that just needs the default ticker
    @property
    def ticker(self) -> str:
        return self.primary.ticker

    @property
    def source(self) -> str:
        return self.primary.source

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.name}({self.primary.ticker})"


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    symbols: tuple[Symbol, ...]
    macro: str | None
    htf: str
    mtf: tuple[str, ...]
    ltf: str
    history: dict[str, int]
    db_path: Path

    @property
    def timeframes(self) -> tuple[str, ...]:
        """All timeframes, highest first: (1w, 1d, 4h, 1h, 15m)."""
        if self.macro:
            return (self.macro, self.htf, *self.mtf, self.ltf)
        return (self.htf, *self.mtf, self.ltf)

    def get(self, *keys: str, default: Any = None) -> Any:
        """Nested lookup: cfg.get('gates', 'min_rr')."""
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    @staticmethod
    def secret(name: str) -> str | None:
        value = os.environ.get(name, "").strip()
        return value or None


def _parse_source_spec(entry: dict, key: str, sym_name: str) -> SourceSpec:
    """Parse a {source: ..., ticker: ...} block."""
    spec = entry.get(key)
    if not isinstance(spec, dict):
        raise ValueError(f"{sym_name}: '{key}' must be a dict with 'source' and 'ticker'")
    source = str(spec.get("source", "")).lower()
    ticker = str(spec.get("ticker", ""))
    if source not in VALID_SOURCES:
        raise ValueError(f"{sym_name}: unknown {key} source {source!r}")
    if not ticker:
        raise ValueError(f"{sym_name}: {key} ticker is empty")
    return SourceSpec(source=source, ticker=ticker)


def load(path: Path | str | None = None) -> Config:
    load_dotenv(ROOT / ".env")
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")

    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    # Apply profile overlay (conservative / balanced / aggressive)
    from .profiles import apply_profile
    raw = apply_profile(raw)

    tf = raw.get("timeframes") or {}
    macro = tf.get("macro")  # optional — e.g. "1w"
    htf, ltf = tf.get("htf"), tf.get("ltf")
    mtf = tuple(tf.get("mtf") or ())
    if not htf or not ltf or not mtf:
        raise ValueError("config.timeframes needs htf, mtf and ltf")

    symbols: list[Symbol] = []
    seen: set[str] = set()
    for entry in raw.get("symbols") or []:
        name = str(entry["name"])
        primary = _parse_source_spec(entry, "primary", name)
        fallback = _parse_source_spec(entry, "fallback", name)
        session = str(entry.get("session", "always")).lower()
        if session not in VALID_SESSIONS:
            raise ValueError(f"{name}: unknown session {session!r}")
        if name in seen:
            raise ValueError(f"duplicate symbol name {name!r}")
        seen.add(name)
        symbols.append(Symbol(name=name, primary=primary, fallback=fallback, session=session))
    if not symbols:
        raise ValueError("config.symbols is empty")

    history = {str(k): int(v) for k, v in (raw.get("history") or {}).items()}
    all_tfs = (htf, *mtf, ltf) if not macro else (macro, htf, *mtf, ltf)
    for timeframe in all_tfs:
        history.setdefault(timeframe, 300)
    if macro:
        history.setdefault(macro, 104)  # ~2 years of weekly bars

    db_path = ROOT / str(raw.get("db_path", "data/bot.db"))
    db_path.parent.mkdir(parents=True, exist_ok=True)

    return Config(
        raw=raw,
        symbols=tuple(symbols),
        macro=str(macro) if macro else None,
        htf=str(htf),
        mtf=mtf,
        ltf=str(ltf),
        history=history,
        db_path=db_path,
    )
