# Multi-Asset Market Analysis & Signal Bot — Master Plan

> **Status:** design locked, ready to build
> **Date:** 2026-08-29
> **Principle:** the bot ANALYSES and REPORTS. It never places an order.

---

## 1. Objective

An always-running market analyst that watches crypto, tokenized gold, US indices
and forex across three timeframes, applies classic + modern indicators + price
action / Smart Money Concepts, and pushes a complete **signal plan**
(direction, entry, stop-loss, take-profits, R:R, confidence, rationale) to
Telegram.

**Golden rule of the architecture:**

> **The rules decide. The AI only describes.**

The confluence engine is 100% deterministic — same candles always produce the
same signal, so it is backtestable and cannot be hallucinated. The Hugging Face
model is used *only* to turn the numbers into a readable paragraph. If HF is
down or out of free credits, a templated explanation is used and the bot keeps
working.

---

## 2. Locked decisions

| Area | Decision | Reason |w
|---|---|---|
| Strictness | **Balanced** (R:R >= 1.5, both indicator families + structure must agree) | Signal quality without going silent for weeks |
| Language | Python 3.13 (installed: 3.13.14, pip 26.1.2) | Already on machine |
| Indicators | **Hand-rolled in pure pandas/numpy** | `pandas_ta` 0.3.14b is broken on numpy 2.x / py3.13 (`from numpy import NaN`). No fragile dependency. |
| Crypto data | **Binance** (keyless public API) | Free, no API key, unlimited, all timeframes |
| TradFi data (primary) | **Hyperliquid** (keyless `api.hyperliquid.xyz`) | 1200 req/min, no key, covers indices + forex via HIP-3 builder markets |
| TradFi data (DJI only) | **yfinance** (Yahoo Finance scraper) | Free, no key, `^DJI` for Dow Jones |
| TradFi data (fallback) | **Twelve Data free** (800 credits/day) | Emergency fallback if Hyperliquid/yfinance fail |
| LLM | **HF Serverless Inference API** + templated fallback | User's choice; credit-metered so fallback is mandatory |
| Alerts | **Telegram Bot API** | Free, instant push |
| Storage | **SQLite** (file) -> optional Neon Postgres later | Zero setup, zero cost |
| Compute | **GitHub Actions cron** (phase 1) -> **Oracle Cloud Always Free VM** (phase 7) | Free; VM needed for true 24/7 crypto |
| Monthly cost | **$0 / Rs.0** | Every component on a permanent free tier |

---

## 3. Watchlist & data routing

| # | Symbol in config | API ticker | Primary source | Fallback source | Session |
|---|---|---|---|---|---|
| 1 | BTC | `BTC/USDT` (Binance) / `BTC` (HL) | Binance Spot | Hyperliquid | 24/7 |
| 2 | ETH | `ETH/USDT` (Binance) / `ETH` (HL) | Binance Spot | Hyperliquid | 24/7 |
| 3 | XAUUSDT (gold) | `XAUT/USDT` (Binance) / `PAXG` (HL) | Binance Spot | Hyperliquid (PAXG) | 24/7 |
| 4 | US100 / NAS100 | `xyz:XYZ100` (HL) | Hyperliquid | yfinance `^NDX` | 24/7 (HL perps) |
| 5 | US500 | `xyz:SP500` (HL) | Hyperliquid | yfinance `^GSPC` | 24/7 (HL perps) |
| 6 | US30 | `^DJI` (yfinance) | yfinance | Twelve Data `DJI` | US cash hours |
| 7 | EURUSD | `xyz:EUR` (HL) | Hyperliquid | yfinance `EURUSD=X` | 24/7 (HL perps) |
| 8 | GBPUSD | `xyz:GBP` (HL) | Hyperliquid | yfinance `GBPUSD=X` | 24/7 (HL perps) |
| 9 | USDJPY | `xyz:JPY` (HL) | Hyperliquid | yfinance `USDJPY=X` | 24/7 (HL perps) |

Notes:
- **Hyperliquid indices/forex are HIP-3 builder-deployed perpetual futures**
  (via trade[XYZ]). Officially licensed S&P 500 perp. They trade 24/7 with
  funding rates anchoring them to the real index/forex values.
- `XAUT/USDT` is **tokenized gold** — it tracks spot XAU/USD closely but can
  carry a small premium/discount. `PAXG` on Hyperliquid is the fallback.
- **Dow Jones** is the only symbol not on Hyperliquid. yfinance is primary;
  Twelve Data is the backup (uses ~2 credits/scan from the 800/day budget).
- Adding/removing symbols is a one-line edit in `config.yaml`. The router
  tries the primary source first, falls back automatically on failure.

---

## 4. Timeframe model (LTF / MTF / HTF / Macro)

| Role | Timeframe | Job in the decision |
|---|---|---|
| **Macro** | `1w` | Sets the **macro trend context**. Weekly structure, 200-week MA. Weight x4 |
| **HTF** | `1d` | Sets the **bias**. Nothing trades against it. Weight x3 |
| **MTF** | `4h` + `1h` | **Confirms** the swing and structure. Weight x2 |
| **LTF** | `15m` | **Triggers** the entry, defines the precise zone. Weight x1 |

Everything is computed per timeframe, then merged by the confluence engine.

### How timeframes are built

- **Crypto (Binance, unlimited):** fetch each timeframe **directly** —
  5 calls per symbol (`1w`, `1d`, `4h`, `1h`, `15m`).
- **TradFi via Hyperliquid (unlimited):** fetch each timeframe **directly** —
  same as crypto. Hyperliquid supports `1w`, `1d`, `4h`, `1h`, `15m` natively
  with up to 5000 candles per request. 1200 req/min rate limit is more than enough.
- **DJI via yfinance:** fetch `1wk`, `1d`, `1h`, `15m` natively. Resample `1h`
  to `4h` in pandas. yfinance has no official rate limit but we throttle to
  be polite.
- **Twelve Data (fallback only):** same deep-pull + resample strategy as before
  if ever needed.

---

## 5. Data source reliability & fallback chain

```
Symbol request arrives at DataRouter
  │
  ├─ Crypto (BTC/ETH/XAUT) ──► Binance Spot (keyless)
  │                                 └─ fail ──► Hyperliquid perps
  │
  ├─ Indices/Forex (US100/US500/EUR/GBP/JPY) ──► Hyperliquid HIP-3 (keyless)
  │                                                  └─ fail ──► yfinance
  │                                                                └─ fail ──► Twelve Data (credit budget)
  │
  └─ Dow Jones (US30) ──► yfinance (keyless)
                              └─ fail ──► Twelve Data (credit budget)
```

### Hyperliquid API specs
| Feature | Value |
|---|---|
| Endpoint | `POST https://api.hyperliquid.xyz/info` |
| Auth | **NONE** for market data |
| Rate limit | **1200 requests/min** |
| Candle intervals | 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 8h, 12h, 1d, 3d, 1w, 1M |
| History depth | Full history since listing |
| Format | JSON POST body, e.g. `{"type": "candleSnapshot", "req": {"coin": "xyz:SP500", "interval": "1h", ...}}` |

### Twelve Data credit budget (fallback only — rarely used)

Free plan = **8 credits/min, 800 credits/day**, resets 00:00 UTC.
Now only consumed when both primary + secondary sources fail for a symbol.
Expected daily usage: **< 50 credits** under normal conditions.

Hard safety rails still in `src/data/budget.py`:
- Persistent credit ledger in SQLite, reset at 00:00 UTC.
- **Hard stop at 750 credits**.
- Rate limiter: max 7 requests/min.
- On HTTP 429: exponential backoff 2s/4s/8s, then skip the symbol this scan.
- Last good candles are cached in SQLite so a skipped fetch degrades gracefully.

---

## 6. Analysis engine — Layer 1: classic indicators

All hand-implemented in `src/analysis/indicators.py` (pure pandas/numpy).

| Indicator | Params | Used for |
|---|---|---|
| EMA | 20, 50, 200 | Trend direction, dynamic S/R |
| SMA | 50, 200 | Golden/death cross context |
| RSI | 14 | Overbought/oversold + **divergence** |
| MACD | 12/26/9 | Momentum shift, histogram slope |
| Stochastic | 14/3/3 | Reversal timing in ranges |
| Bollinger Bands | 20, 2sd | Volatility squeeze / mean reversion |
| ADX (+DI/-DI) | 14 | **Trend strength gate** (ADX<20 => range mode) |
| ATR | 14 | Stop-loss sizing, volatility normalisation |
| OBV | - | Volume accumulation / distribution |

---

## 7. Layer 2: modern indicators (2026 practice)

| Indicator | Params | Signal it produces |
|---|---|---|
| **SuperTrend** | ATR 10, mult 3.0 | Clean long/short trend flip |
| **VWAP** (session) | daily anchor | Intraday fair value; above = buyers in control |
| **Anchored VWAP** | from last major swing | Institutional cost basis of the current leg |
| **Ichimoku Cloud** | 9/26/52 | Cloud position, TK cross, future cloud bias |
| **MFI** | 14 | Volume-weighted momentum |
| **Volume Profile** (lite) | 50-bin histogram over lookback | HVN/LVN, **POC** as a magnet & S/R |
| **Divergence engine** | RSI & MACD vs pivots | Regular + hidden bullish/bearish divergence |
| **Volatility regime** | ATR percentile, BB width | Expansion vs contraction; blocks signals in dead tape |

## 8. Layer 3: price action + Smart Money Concepts

In `src/analysis/price_action.py` and `src/analysis/smc.py`.

| Concept | Detection rule (deterministic) |
|---|---|
| **Swing points** | Fractal pivot: high > N bars each side (N=2 LTF, 3 HTF) |
| **Market structure** | Label sequence HH / HL / LH / LL from swings |
| **BOS** (break of structure) | Candle **body close** beyond last swing high/low (body close, not wick — cuts fakeouts) |
| **CHoCH** (change of character) | BOS in the direction opposite to the prevailing structure = early reversal |
| **Order blocks** | Last opposite-colour candle before a displacement leg that caused a BOS; zone = its high-low |
| **Fair Value Gap (FVG)** | 3-candle imbalance: gap between candle1.high and candle3.low (bullish) with tracked % fill |
| **Liquidity sweep** | Wick pierces a prior swing high/low then closes back inside (stop hunt) |
| **Equal highs/lows** | Two swings within 0.1x ATR = liquidity pool |
| **Premium / Discount** | Position within the dealing range: >70% premium, <30% discount, 50% = equilibrium |
| **Support / Resistance** | Clustered swing levels + volume-profile HVN, merged within 0.3x ATR |
| **Supply / demand zones** | Base candles before an impulsive move, unmitigated |
| **Candlestick patterns** | Engulfing, pin bar / hammer, shooting star, doji, inside bar, marubozu |
| **Key levels** | Prev-day high/low/close, daily & weekly open, round numbers |
| **Fibonacci** | 0.382 / 0.5 / 0.618 / 0.705 (OTE) / -0.27 / -0.62 extensions from the live swing |

---

## 9. Confluence engine (the decision maker)

`src/analysis/confluence.py`. Every check returns a vote in **-1 .. +1**.
Votes are grouped into 5 categories, each category has a weight, and each
timeframe has a multiplier.

### Category weights (sum = 100)

| Category | Weight | Contents |
|---|---|---|
| Trend | 30 | EMA stack, SuperTrend, Ichimoku, ADX/DI, SMA cross |
| Structure (PA/SMC) | 30 | HH-HL labels, BOS, CHoCH, premium/discount |
| Momentum | 20 | RSI, MACD, Stochastic, divergences |
| Zones | 12 | Order block, FVG, S/R proximity, fib OTE, liquidity sweep |
| Volume | 8 | OBV slope, MFI, volume vs average, POC relation |

### Timeframe multipliers

`Macro(1w) = 4.0`, `HTF(1d) = 3.0`, `MTF(4h) = 2.0`, `MTF(1h) = 2.0`, `LTF(15m) = 1.0`

```
raw   = SUM over timeframes( tf_mult * SUM over categories( cat_weight * vote ) )
score = 100 * raw / max_possible_raw          # normalised to -100 .. +100
```

### Score -> label (Balanced profile)

| Score | Label |
|---|---|
| >= +65 | **STRONG BUY** |
| +40 .. +65 | **BUY** |
| +18 .. +40 | **WATCH LONG** |
| -18 .. +18 | **NEUTRAL** (not sent) |
| -18 .. -40 | **WATCH SHORT** |
| -40 .. -65 | **SELL** |
| <= -65 | **STRONG SELL** |

### Hard gates (a signal is downgraded or dropped if any fail)

1. **HTF conflict gate** — if HTF bias opposes the signal direction, the best
   possible label is `WATCH` (never BUY/SELL).
2. **ADX gate** — trend-following signals need `ADX >= 20` on the MTF.
   Below that the engine switches to range logic (S/R bounce only).
3. **R:R gate** — computed plan must have `R:R >= 1.5`, else dropped.
4. **Volatility gate** — ATR percentile < 15 (dead tape) or > 95 (news chaos)
   => downgrade to `WATCH`.
5. **Stop-distance gate** — if structural stop is farther than `3 x ATR`, drop
   (risk too wide to be meaningful).
6. **Cooldown** — same symbol + same direction cannot re-alert within 4 hours
   unless the score improves by >= 15 points. Kills spam.
7. **Candle-close gate** — signals are only evaluated on **closed** candles.
   No repainting on a live bar.

**Confidence %** = `50 + |score|/2`, adjusted -10 for each failed soft check,
capped 55-95. Never shown as 100% — the model is not certain, ever.

---

## 10. Trade plan generator (entry / SL / TP / R:R)

`src/analysis/levels.py`. All values are **suggestions for the user** — nothing
is executed.

### Entry zone
- Prefer the **nearest unmitigated order block or FVG** in the signal direction
  on the LTF/MTF. Zone = that block's high-low.
- Fallback: fib 0.618-0.705 (OTE) band of the live swing.
- Fallback 2: current close +- 0.25 x ATR (market-ish entry).
- Reported as a **range**, plus a single mid price used for the maths.

### Stop loss (structure first, ATR as sanity bound)
```
long:
  sl_structure = last_swing_low  - 0.20 * ATR      # beyond the level, not on it
  sl_atr       = entry_mid       - 1.50 * ATR
  SL           = sl_structure
  if (entry_mid - sl_structure) < 0.8 * ATR:  SL = sl_atr    # too tight -> widen
  if (entry_mid - SL)           > 3.0 * ATR:  DROP SIGNAL     # too wide
short: mirrored (swing high + 0.20*ATR, etc.)
```

### Take profits (R multiples snapped to real structure)
```
R = |entry_mid - SL|

TP1 = entry_mid + 1.0R    -> close 50%
TP2 = entry_mid + 2.0R    -> close 30%
TP3 = next HTF S/R level in direction, else entry_mid + 3.0R  -> runner 20%

Snapping: if a real level (S/R cluster, POC, prev-day H/L, equal highs) lies
within 0.30 x ATR of a raw TP, move the TP to just BEFORE that level
(0.05 x ATR inside), because price reacts there.
```

### Reported metrics
`R:R` (to TP2), `risk %` suggestion (default 1% of account), distance to SL in
% and in ATR, invalidation condition (HTF/MTF close beyond SL), expected
holding horizon from the trigger timeframe, and the nearest opposing level.

---

## 11. LLM narration layer (Hugging Face serverless)

`src/llm/explain.py`

- Endpoint: HF Inference Providers router, OpenAI-compatible
  `https://router.huggingface.co/v1/chat/completions`, auth `HF_TOKEN`.
- Model (configurable): `Qwen/Qwen3-8B-Instruct`, fallback
  `meta-llama/Llama-3.1-8B-Instruct`.
- Input: a compact **JSON fact sheet** of the signal (never raw candles) —
  score, per-timeframe states, triggered checks, levels, R:R.
- Output: 2-4 sentences, max ~90 words, plain English.
- **The prompt forbids inventing direction or levels.** It may only rephrase the
  facts it is given. Temperature 0.3.
- Post-check: if the reply mentions a direction that contradicts the rule
  engine, or contains any number not present in the fact sheet, it is **rejected**
  and the template is used instead.

**Fallback chain (never blocks an alert):**
```
HF call (timeout 12s, 1 retry)
  -> on 402/429/5xx/timeout/validation-fail ->
     deterministic template built from the same fact sheet
```
HF free serverless is credit-metered (roughly $0.10/month of included credits,
$2/month on PRO), so the template will carry a real share of the traffic. That
is by design — the bot must not depend on the LLM.

Rate control: max 1 HF call per emitted signal, global cap `hf_daily_cap: 100`
in config, cached by `(symbol, direction, score bucket)` for 4 h.

---

## 12. Telegram alert format

`src/alerts/telegram.py` — HTML parse mode, one message per signal.

```
ETH/USDT  -  BUY   (Confidence 78%)
Score +52  |  1d up . 4h up . 1h up . 15m pullback

Entry   3,120 - 3,140      (1h bullish order block)
Stop    3,048              (-2.3% | 1.5 ATR, below swing low)
TP1     3,210   1.0R  (50%)
TP2     3,290   2.0R  (30%)
TP3     3,400   3.1R  (20%)  next daily resistance
R:R     1 : 2.3            Risk 1% of account

Triggers
  BOS on 1h . swept liquidity below 3,100 . reclaimed OB
  RSI 45 turning up . SuperTrend flipped long . above VWAP
  ADX 24 . discount zone (32%)

Invalidation: 4h close below 3,048

Why: <LLM or template paragraph>

Analysis only - not financial advice. No order was placed.
```

Also implemented: `/status`, `/watchlist`, `/last`, `/mute <symbol>` commands,
and a daily 08:00 UTC market-overview digest for all 9 symbols.

---

## 13. Repository layout

```
C:\Manan\bt\
├── PLAN.md                     this document
├── README.md                   setup + run instructions
├── requirements.txt
├── config.yaml                 watchlist, timeframes, weights, thresholds
├── .env.example                HF_TOKEN, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
│                               (TWELVEDATA_KEY optional — fallback only)
├── .gitignore                  .env, *.db, __pycache__, logs/
├── src/
│   ├── main.py                 entrypoint: one full scan cycle
│   ├── scheduler.py            session-aware loop, cooldowns, candle-close timing
│   ├── config.py               typed config loader + validation
│   ├── data/
│   │   ├── router.py           symbol -> source dispatch (primary/fallback chain)
│   │   ├── crypto.py           Binance Spot fetch (keyless)
│   │   ├── hyperliquid.py      Hyperliquid POST /info fetch (keyless, HIP-3)
│   │   ├── yfinance_source.py  yfinance fetch (keyless, DJI + fallback)
│   │   ├── tradfi.py           Twelve Data fetch (keyed, fallback only)
│   │   ├── budget.py           credit ledger + rate limiter + backoff (TD only)
│   │   ├── resample.py         15m -> 1h / 4h / 1d
│   │   ├── cache.py            SQLite candle cache
│   │   └── sessions.py         market-hours calendar (US cash, FX week, 24/7)
│   ├── analysis/
│   │   ├── indicators.py       Layer 1 - classic
│   │   ├── modern.py           Layer 2 - SuperTrend, VWAP, Ichimoku, VP, divergence
│   │   ├── price_action.py     swings, structure, S/R, patterns, fib, key levels
│   │   ├── smc.py              BOS, CHoCH, order blocks, FVG, liquidity, prem/disc
│   │   ├── confluence.py       voting + weights + gates -> label & score
│   │   └── levels.py           entry / SL / TP / R:R generator
│   ├── llm/
│   │   ├── explain.py          HF call + validation
│   │   └── template.py         deterministic fallback narration
│   ├── alerts/
│   │   ├── telegram.py         sender + formatter + bot commands
│   │   └── formatter.py        message building
│   ├── store/
│   │   ├── db.py               SQLite schema + migrations
│   │   └── signals.py          signal history, dedupe, cooldown state
│   ├── tracking/
│   │   ├── outcome_checker.py  follow-up job: did SL/TP1/TP2/TP3 hit?
│   │   ├── performance.py      hit-rate, win/loss, avg R, per-symbol stats
│   │   └── report.py           generate daily/weekly performance reports
│   └── backtest/
│       ├── replay.py           walk-forward candle replay of the engine
│       └── report.py           backtest hit-rate, avg R, MFE/MAE
├── logs/
│   ├── signals.jsonl           append-only log of every signal emitted
│   ├── outcomes.jsonl          append-only log of every outcome check
│   ├── scans.jsonl             per-scan metadata (timing, errors, symbols)
│   └── daily_report.md         auto-generated daily performance summary
├── tests/
│   ├── test_indicators.py      known-value checks vs reference series
│   ├── test_smc.py             synthetic candles -> expected BOS/CHoCH/OB/FVG
│   ├── test_confluence.py      crafted scenarios -> expected label
│   ├── test_levels.py          SL/TP maths incl. all gate branches
│   ├── test_tracking.py        outcome checker against known price paths
│   └── fixtures/               saved candle JSON so tests need no network
└── deploy/
    ├── github-actions.yml      cron every 15 min (phase 1)
    ├── systemd.service         Oracle VM always-on (phase 7)
    └── Dockerfile              optional
```

---

## 14. config.yaml (shape)

```yaml
profile: balanced            # conservative | balanced | aggressive

timeframes:
  htf: 1d
  mtf: [4h, 1h]
  ltf: 15m

symbols:
  - name: BTC
    primary:  {source: binance,      ticker: "BTC/USDT"}
    fallback: {source: hyperliquid,  ticker: "BTC"}
    session: always
  - name: ETH
    primary:  {source: binance,      ticker: "ETH/USDT"}
    fallback: {source: hyperliquid,  ticker: "ETH"}
    session: always
  - name: XAUUSDT
    primary:  {source: binance,      ticker: "XAUT/USDT"}
    fallback: {source: hyperliquid,  ticker: "PAXG"}
    session: always
  - name: US100
    primary:  {source: hyperliquid,  ticker: "xyz:XYZ100"}
    fallback: {source: yfinance,     ticker: "^NDX"}
    session: always     # HL perps trade 24/7
  - name: US500
    primary:  {source: hyperliquid,  ticker: "xyz:SP500"}
    fallback: {source: yfinance,     ticker: "^GSPC"}
    session: always
  - name: US30
    primary:  {source: yfinance,     ticker: "^DJI"}
    fallback: {source: twelvedata,   ticker: "DJI"}
    session: us_cash
  - name: EURUSD
    primary:  {source: hyperliquid,  ticker: "xyz:EUR"}
    fallback: {source: yfinance,     ticker: "EURUSD=X"}
    session: always
  - name: GBPUSD
    primary:  {source: hyperliquid,  ticker: "xyz:GBP"}
    fallback: {source: yfinance,     ticker: "GBPUSD=X"}
    session: always
  - name: USDJPY
    primary:  {source: hyperliquid,  ticker: "xyz:JPY"}
    fallback: {source: yfinance,     ticker: "USDJPY=X"}
    session: always

weights:      {trend: 30, structure: 30, momentum: 20, zones: 12, volume: 8}
tf_multiplier: {"1d": 3.0, "4h": 2.0, "1h": 2.0, "15m": 1.0}

thresholds:
  strong: 65
  signal: 40
  watch: 18

gates:
  min_rr: 1.5
  min_adx: 20
  max_stop_atr: 3.0
  atr_pct_min: 15
  atr_pct_max: 95
  cooldown_hours: 4
  cooldown_score_override: 15

risk:
  default_risk_pct: 1.0
  atr_stop_mult: 1.5
  structure_stop_buffer_atr: 0.20
  tp_r_multiples: [1.0, 2.0, 3.0]
  tp_allocation: [50, 30, 20]
  snap_tolerance_atr: 0.30

llm:
  provider: huggingface
  model: "Qwen/Qwen3-8B-Instruct"
  enabled: true
  daily_cap: 100
  timeout_s: 12

budget:
  twelvedata_daily_cap: 750
  twelvedata_per_min: 7

tracking:
  outcome_check_interval_hours: 1    # how often to re-check open signals
  max_signal_age_days: 7             # stop tracking after 7 days
  report_hour_utc: 0                 # daily report at 00:00 UTC
  weekly_report_day: sunday          # weekly summary day

logging:
  log_dir: "logs"
  signals_file: "signals.jsonl"      # every signal emitted
  outcomes_file: "outcomes.jsonl"     # every outcome check
  scans_file: "scans.jsonl"          # per-scan metadata
  daily_report_file: "daily_report.md"
  rotate_days: 90                    # keep logs for 90 days

scan_interval_minutes: 15
```

---

## 15. Storage schema (SQLite -> optional Neon Postgres)

```sql
candles(symbol, timeframe, ts, open, high, low, close, volume)  PK(symbol,timeframe,ts)

signals(id, ts, symbol, direction, label, score, confidence,
        entry_low, entry_high, sl, tp1, tp2, tp3, rr, atr,
        triggers_json, htf_bias, mtf_bias, ltf_state,
        narration, narration_source, sent_ok,
        data_source,                -- which source provided the candles
        status)                     -- 'open' | 'won' | 'lost' | 'expired' | 'invalidated'

outcomes(signal_id, checked_ts, hit, mfe_r, mae_r, bars_held, note,
         tp1_hit_ts, tp2_hit_ts, tp3_hit_ts, sl_hit_ts,   -- exact timestamps
         price_at_check,                                    -- price when checked
         entry_filled)                                      -- did price reach entry zone?

performance_daily(
    day_utc       TEXT PRIMARY KEY,
    total_signals INTEGER,
    wins          INTEGER,        -- TP1+ hit before SL
    losses        INTEGER,        -- SL hit before any TP
    open_signals  INTEGER,
    win_rate      REAL,
    avg_r         REAL,           -- average R achieved
    best_r        REAL,
    worst_r       REAL,
    by_symbol     TEXT,           -- JSON: {"BTC": {wins: 3, losses: 1}, ...}
    by_label      TEXT            -- JSON: {"BUY": {wins: 5, losses: 2}, ...}
)

api_usage(day_utc, provider, credits_used)
mutes(symbol, until_ts)
```

`outcomes` is filled by the **outcome checker** (`src/tracking/outcome_checker.py`)
which runs every hour (configurable). For every signal in `status = 'open'`:

1. Fetch candles from signal time to now.
2. Walk forward bar by bar and check: did price reach the entry zone? If yes,
   did it hit SL or any TP first?
3. Record MFE (max favourable excursion in R) and MAE (max adverse excursion).
4. Update `signals.status` to `won`, `lost`, `expired` (past max age), or
   `invalidated` (HTF close beyond SL).
5. Append a line to `logs/outcomes.jsonl` for permanent record.

This makes the bot **fully self-auditing** — after a few weeks you have real
hit rates per symbol, per label, per timeframe, and per profile.

---

## 16. Signal tracking & performance logging system

This is the feedback loop that makes the bot improvable. Every signal is
tracked from emission to resolution, and the data feeds back into tuning.

### What gets logged (append-only JSONL files)

| Log file | What it records | When |
|---|---|---|
| `logs/signals.jsonl` | Every signal the bot emits: symbol, direction, label, score, entry/SL/TP, data source, all indicator states, raw confluence votes | On signal emission |
| `logs/outcomes.jsonl` | Outcome check: signal_id, current price, SL/TP hit status, MFE/MAE in R, bars elapsed, status change | Every outcome check cycle (hourly) |
| `logs/scans.jsonl` | Per-scan metadata: start/end time, symbols scanned, data source used, errors, latency per source, signals emitted count | Every scan cycle (15 min) |
| `logs/daily_report.md` | Auto-generated daily performance summary in markdown | Daily at `report_hour_utc` |

### Outcome checker logic (`src/tracking/outcome_checker.py`)

Runs every `outcome_check_interval_hours` (default: 1h):

```
For each signal WHERE status = 'open' AND age < max_signal_age_days:
  1. Fetch candles from signal_ts to now
  2. Check: did price reach the entry zone?
     - NO  → status stays 'open', record price_at_check
     - YES → walk forward from entry:
       a. If SL hit first       → status = 'lost',  record mae_r, bars_held
       b. If TP1 hit first      → record tp1_hit_ts, continue checking TP2/TP3
       c. If TP2 hit            → status = 'won',   record mfe_r
       d. If TP3 hit            → status = 'won',   record mfe_r (best case)
       e. If expired (age > 7d) → status = 'expired'
  3. If HTF closes beyond SL   → status = 'invalidated' (structure broken)
  4. Append result to outcomes.jsonl
  5. Update signals table in SQLite
```

### Performance reporter (`src/tracking/performance.py`)

Generates stats on demand and on schedule:

```
📊 Daily Performance Report — 2026-08-29
─────────────────────────────────────────
Signals emitted:  12
Entry filled:      9  (75%)
Wins (TP1+):       6  (67% of filled)
Losses (SL):       2  (22% of filled)
Still open:        1

Avg R achieved:   1.4R
Best trade:       ETH BUY +2.8R
Worst trade:      US100 SELL -1.0R

By Symbol:
  BTC     3 signals, 2W 1L  (67%)
  ETH     2 signals, 2W 0L  (100%)
  US500   2 signals, 1W 1L  (50%)
  ...

By Label:
  STRONG BUY   2 signals, 2W 0L  (100%)
  BUY          5 signals, 3W 1L  (75%)
  SELL         3 signals, 1W 1L  (50%)
  ...

Notes:
  ⚠ US30 signals underperforming (25% win rate, 4 signals)
  ✅ BTC STRONG BUY has 100% hit rate (3/3 over 2 weeks)
```

This report is:
- **Sent to Telegram** daily (configurable hour)
- **Saved to** `logs/daily_report.md` (overwritten daily)
- **Weekly summary** generated on `weekly_report_day` with aggregated stats
- **Queryable via Telegram**: `/stats`, `/stats BTC`, `/stats 7d`

### What you can improve from this data

1. **Which symbols are profitable?** Drop or reduce weight on losers.
2. **Which labels work?** If `WATCH LONG` never fills, raise the threshold.
3. **Is the R:R gate too loose?** If avg R achieved < 1.0, tighten `min_rr`.
4. **Are certain indicator combos better?** The `triggers_json` field lets you
   correlate which confluence patterns produce winners.
5. **Data source quality** — if Hyperliquid-sourced signals outperform
   yfinance-sourced ones, that tells you something about data quality.

---

## 17. Validation & testing plan

1. **Unit tests** with saved candle fixtures (no network) for every indicator
   against reference values, and synthetic candle patterns for each SMC concept.
2. **Golden-file test** — a fixed 500-candle series must always produce a
   byte-identical signal JSON. This proves determinism.
3. **Walk-forward backtest** (`src/backtest/replay.py`) — replays 1-2 years of
   candles bar by bar, evaluating only closed bars, generating the same
   signals the live bot would, then scoring them against what price did next.
   Report: signal count, win rate at TP1/TP2, average R, MFE/MAE, worst
   drawdown of the signal series, per-symbol and per-label breakdown.
4. **Paper-forward run** — 2 weeks live in `dry_run: true` (logs, no Telegram
   spam) before trusting anything.
5. **Threshold tuning** happens only against backtest output AND live tracking
   data, never by eyeballing the last few alerts.
6. **Tracking tests** (`tests/test_tracking.py`) — synthetic price paths fed
   to the outcome checker to verify it correctly identifies wins/losses/expired.

**Backtest depth:** Binance + Hyperliquid both provide deep free history.
yfinance gives ~1 year of daily and ~60 days of intraday. Twelve Data free
gives ~52 days of 15m. Limitations are stated in the report, not hidden.

---

## 18. Build phases

| Phase | Deliverable | How it is proven |
|---|---|---|
| **1. Data foundation** | config loader, sessions, Binance fetch, Hyperliquid fetch, yfinance fetch, router with fallback chain, resampler, SQLite cache | Script prints a candle table for **all 9 symbols x 4 timeframes** with row counts and last timestamps |
| **2. Indicator engines** | `indicators.py`, `modern.py` | Unit tests pass; values spot-checked against a reference series |
| **3. Price action + SMC** | `price_action.py`, `smc.py` | Synthetic-candle tests for swings/BOS/CHoCH/OB/FVG/sweeps; printed structure map for BTC 1d |
| **4. Confluence + levels** | `confluence.py`, `levels.py` | Crafted scenario tests hit expected labels; every gate branch covered; full signal JSON printed for all 9 symbols |
| **5. Narration + Telegram** | `explain.py`, `template.py`, `telegram.py` | Real message delivered to your chat; HF-down path verified by forcing a failure |
| **6. Signal tracking + logging** | `outcome_checker.py`, `performance.py`, `report.py`, JSONL logging | Outcome checker correctly resolves test signals; daily report generated; `/stats` command works in Telegram |
| **7. Deploy + backtest** | GitHub Actions cron, systemd unit for Oracle VM, `replay.py` + report | Cron runs green; backtest report produced for BTC/ETH/XAUT; live tracking running |

Phases 1-4 need **no keys at all** (Binance and Hyperliquid are both keyless),
so we can build and verify the entire bot before you hand over the Telegram
token and HF token. Twelve Data key is optional (fallback only).

### Dependencies (`requirements.txt`)
```
ccxt            # crypto candles via Binance (keyless)
requests        # Hyperliquid, Twelve Data, HF, Telegram
yfinance        # Yahoo Finance for DJI + fallback
pandas          # dataframes / resampling
numpy           # maths
PyYAML          # config
python-dotenv   # .env
pytest          # tests
```
No `pandas_ta`, no `TA-Lib` (broken / needs C build on py3.13). Indicators are
ours, which also means no silent behaviour change on a dependency bump.

---

## 19. Deployment

**Phase 1 target — GitHub Actions cron (free)**
- `*/15 * * * *` on a **public** repo = unlimited free minutes; on a private repo
  the free quota is 2,000 min/month, so 15-min cadence is the right choice there.
- Secrets stored as repo secrets, never in the repo.
- SQLite state is committed back as an artifact / cache (or skipped — cooldowns
  fall back to in-run only).
- Caveat: GitHub may skip scheduled runs on inactive repos and cron firing is
  best-effort, sometimes minutes late. Fine for 15m signals, not for scalping.

**Phase 7 target — Oracle Cloud Always Free VM**
- Always Free ARM allocation was **halved to 2 OCPU / 12 GB on 2026-06-15** —
  still far more than this bot needs.
- Runs as a `systemd` service, restart on failure, true 24/7 for crypto,
  proper persistent SQLite, and websocket streaming possible later.
- Outcome checker and daily reports run as separate scheduled tasks on the VM.

---

## 20. Risks, limits and honest caveats

| Risk | Reality | Mitigation |
|---|---|---|
| Tokenized gold != spot gold | `XAUT/USDT` can trade at a premium/discount, thinner book | Documented; `PAXG` on Hyperliquid as cross-check |
| Hyperliquid HIP-3 perps != spot indices | Funding rates anchor price but small deviations possible | yfinance + Twelve Data fallback for cross-validation |
| yfinance is unofficial | Yahoo can change the backend anytime | Twelve Data fallback; bot logs source used per signal |
| Twelve Data 800/day | Free tier is tight (but now fallback only) | Expected < 50 credits/day usage; hard cap 750 |
| HF free credits are tiny | Metered, will run out under load | Template fallback + daily cap + 4h cache; LLM is never load-bearing |
| SMC is a heuristic framework | Order blocks / FVG are not proven institutional footprints | Implemented as explicit rules; validated by backtest + live tracking |
| Overfitting the weights | Easy to tune into fantasy | Tune only on walk-forward output + live tracking data |
| GitHub cron unreliability | Runs can be late or skipped | Move to Oracle VM in phase 7 |
| Repainting | A signal that changes after the fact is worthless | Closed-candle-only evaluation, enforced in code and tested |
| Free tiers change | Oracle already cut theirs in June 2026 | No lock-in: data layer is behind a router, host is swappable |
| Signal tracking bias | Only measuring signals we emitted, not ones we missed | Addressed in backtest; tracking measures what we DID say, not what we should have |

### Scope guardrails (non-negotiable)
- **No order placement, no exchange API keys with trade permission, ever.**
- Every message carries "analysis only, not financial advice".
- SL/TP are *suggestions* for your own decision, not instructions.
- Backtest before believing any number this bot prints.

---

## 21. Cost summary

| Component | Service | Cost |
|---|---|---|
| Crypto + gold candles | Binance Spot (keyless) | $0 |
| Indices + forex candles | Hyperliquid (keyless) | $0 |
| Dow Jones candles | yfinance (keyless) | $0 |
| Fallback data | Twelve Data free (800/day, rarely used) | $0 |
| Compute | GitHub Actions -> Oracle Always Free VM | $0 |
| LLM narration | HF serverless free credits + template fallback | $0 |
| Alerts + tracking reports | Telegram Bot API | $0 |
| Storage + logs | SQLite (file) + JSONL files | $0 |
| **Total** | | **$0 / month** |

---

## 22. What I need from you (when convenient)

| Item | Needed for | Blocking? |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` (from @BotFather) | Phase 5 | No |
| `HF_TOKEN` (huggingface.co settings -> Access Tokens) | Phase 5 | No |
| `TWELVEDATA_KEY` (free signup) | Fallback only | No — bot runs without it |

Phases 1-4 start immediately with **zero keys**. Binance and Hyperliquid are
both keyless. Twelve Data key is only needed if primary + secondary sources
fail simultaneously, which should be rare.

