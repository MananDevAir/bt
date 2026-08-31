# Signal Bot

Personal multi-asset analysis bot. Scans crypto, indices, and forex every 15 minutes, scores them with a deterministic confluence engine, and sends trade ideas to Telegram. **Analyses and reports only — never places orders.**

---

## Table of Contents

- [Quick Start](#quick-start)
- [How It Works — The Complete Pipeline](#how-it-works--the-complete-pipeline)
- [Commands](#commands)
- [Profiles](#profiles)
- [Configuration](#configuration)
- [Backtest](#backtest)
- [Tests](#tests)
- [Project Structure](#project-structure)

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set up keys (copy and fill in)
cp .env.example .env

# 3. Dry run — scan once, print scores, no Telegram
python run_bot.py --scan-once

# 4. Start the loop (scans every 15 min)
python run_bot.py

# 5. Go live (sends real Telegram alerts)
python run_bot.py --live
```

### Environment Variables (`.env`)

| Variable | Required? | What it does |
|----------|-----------|-------------|
| `TELEGRAM_BOT_TOKEN` | For alerts | Create via @BotFather |
| `TELEGRAM_CHAT_ID` | For alerts | Your chat ID |
| `HF_TOKEN` | Optional | AI narration (falls back to templates without it) |
| `TWELVEDATA_KEY` | Optional | Only for US30 fallback. Everything else works without it |

---

## How It Works — The Complete Pipeline

### The Loop

When you run `python run_bot.py`, the scheduler starts an infinite loop:

```
┌──────────────────────────────────────────────────────────────┐
│  Every 15 minutes:                                           │
│    1. SCAN — fetch data → score → plan → narrate → alert     │
│                                                              │
│  Every 1 hour:                                               │
│    2. TRACK — check if SL/TP hit on open signals             │
│                                                              │
│  Every day at 00:00 UTC:                                     │
│    3. REPORT — daily win/loss stats to Telegram              │
│    4. CLEANUP — rotate old logs, trim DB                     │
│                                                              │
│  Every Sunday at 00:00 UTC:                                  │
│    5. WEEKLY — performance summary for the week              │
│                                                              │
│  Between scans (every 5 seconds):                            │
│    6. POLL — check for Telegram commands (/status, /last)    │
└──────────────────────────────────────────────────────────────┘
```

The bot catches `Ctrl+C` / `SIGTERM` gracefully — it finishes the current cycle, sends a shutdown message to Telegram, then exits.

---

### Step 1: Data Fetch

For each symbol in the watchlist, the **router** fetches candles across **5 timeframes**: `1w`, `1d`, `4h`, `1h`, `15m`.

```
BTC → try Binance first → if down, try Hyperliquid → if down, use SQLite cache
US100 → try Hyperliquid perp → if down, try yfinance → if down, use cache
US30 → try yfinance → if down, try Twelve Data (budget-controlled) → cache
```

**Key rules:**
- **Only closed candles** — the live (incomplete) candle is always dropped. Signals evaluate only on finished bars. This prevents repainting.
- **Freshness check** — data older than 2× the timeframe duration is rejected (e.g., 15m data older than 30 min = stale → skip symbol).
- **Session filter** — US30 is only scanned during US cash hours (9:30 AM – 4 PM ET). Crypto scans 24/7.
- **Twelve Data budget** — capped at 750 credits/day, 7/minute. Only used as a last-resort fallback for US indices.
- **SQLite cache** — every candle fetched is cached. If both primary and fallback sources fail, the cache serves the last known data.

The `4h` timeframe is built by resampling `1h` candles (since most sources don't serve 4h natively). Incomplete buckets are dropped.

---

### Step 2: Confluence Scoring

The heart of the bot. The **confluence engine** is fully deterministic: same candles → same score, always. No randomness, no LLM input, no wall-clock dependency.

#### Indicator Computation

For each timeframe, two indicator frames are computed:

**Classic indicators** (`indicators.py`):
- RSI (14) with Wilder smoothing
- MACD (12, 26, 9) with signal line and histogram
- Bollinger Bands (20, 2σ)
- ATR (14) — true range based
- ADX (14) with DI+/DI-
- EMA crossovers (9/21 and 50/200)
- Stochastic RSI (14, 14, 3, 3)
- OBV (cumulative volume delta)
- Keltner Channels (20, 1.5 ATR)

**Modern indicators** (`modern.py`):
- SuperTrend (10, 3.0) — direction + flip detection
- VWAP with standard deviation bands
- Ichimoku Cloud (9, 26, 52)
- MFI (14) — Money Flow Index
- Volume Profile (VPOC, VAH, VAL from 48 bins)
- RSI and MACD divergences (regular and hidden)
- Volatility regime (ATR percentile rank over 100 bars)

**SMC analysis** (`smc.py`):
- Market structure — BOS (Break of Structure) and CHoCH (Change of Character)
- Order blocks — bullish/bearish institutional zones
- Fair value gaps (FVGs) — imbalance areas price tends to fill
- Supply/demand zones
- Liquidity sweeps — stop hunts above/below recent highs/lows
- Structure bias (bullish/bearish/neutral based on recent BOS/CHoCH sequence)

**Price action** (`price_action.py`):
- Candle patterns — engulfing, pin bar, morning/evening star, doji, inside bar
- Key levels — swing highs/lows, recent S/R
- Fibonacci retracement (0.618, 0.5, 0.382 OTE zone)
- Support/resistance strength scoring

#### Voting System

Each indicator check produces a **vote**: a value from -1 (strongly bearish) to +1 (strongly bullish) with a category tag and detail string.

Example votes for BTC on the 4h timeframe:
```
trend      | EMA crossover    | +0.8 | "9 EMA > 21 EMA, widening"
trend      | SuperTrend       | +1.0 | "price above SuperTrend, flipped 3 bars ago"
structure  | BOS              | +0.7 | "bullish BOS at 84,200"
structure  | Order block      | +0.6 | "price in bullish OB zone [83,800-84,100]"
momentum   | RSI              | +0.4 | "RSI 58 — mild bullish"
momentum   | MACD             | +0.6 | "histogram expanding"
zones      | FVG              | +0.5 | "unfilled bullish FVG [83,500-83,900]"
volume     | OBV              | +0.3 | "OBV rising"
```

#### Scoring Math

```
For each timeframe (1w, 1d, 4h, 1h, 15m):
  1. Group votes by category (trend, structure, momentum, zones, volume)
  2. Average each category's votes → category score (-1 to +1)
  3. Multiply by category weight (trend=30, structure=30, momentum=20, zones=12, volume=8)
  4. Multiply by timeframe multiplier (1w=4×, 1d=3×, 4h=2×, 1h=2×, 15m=1×)
  5. Sum all → raw score

Final score = (raw_score / max_possible_score) × 100 → range -100 to +100
```

The score maps to a label:

| Score Range | Label |
|-------------|-------|
| +65 to +100 | **STRONG BUY** |
| +40 to +65 | **BUY** |
| +18 to +40 | **WATCH LONG** |
| -18 to +18 | **NEUTRAL** |
| -18 to -40 | **WATCH SHORT** |
| -40 to -65 | **SELL** |
| -65 to -100 | **STRONG SELL** |

#### Hard Gates

Even if the score is high, the signal is killed if any of these fail:

| Gate | Rule | Why |
|------|------|-----|
| **HTF conflict** | Daily bias must not oppose the signal direction | Don't go long if the daily chart is bearish |
| **ADX gate** | ADX(14) ≥ 20 on the lowest timeframe | No signal in a trendless, choppy market |
| **Volatility gate** | ATR percentile between 15th and 95th | No signal in dead markets or crash-tier vol |
| **Macro conflict** | Weekly bias must not oppose the direction | Respect the larger trend |
| **R:R gate** | Reward-to-risk ≥ 1.5 (balanced profile) | Never take a bad risk/reward trade |
| **Stop-distance gate** | Stop distance ≤ 3.0 ATR | Reject trades with unreasonably wide stops |

If a gate kills the signal, the reason is logged in `signal.gates` (e.g., `"rr_too_low": {"detail": "R:R 1.23 < minimum 1.5"}`).

---

### Step 3: Trade Plan Generation

If the score passes thresholds and gates, the **levels engine** generates a trade plan:

#### Entry Zone Selection (priority order)

1. **Order block** — most recent untested OB in the signal direction
2. **Fair value gap** — nearest unfilled FVG
3. **Fibonacci OTE** — 0.618–0.5 retracement zone of the last swing
4. **Market entry** — current close ± 0.3 ATR (fallback)

#### Stop Loss

Placed at the nearest **structure level** that invalidates the trade:
- For longs: below the last swing low or bearish order block
- For shorts: above the last swing high or bullish order block
- Buffer: 0.20 ATR beyond the structure level
- Fallback: 1.5 ATR from entry

#### Take Profits

Three targets at fixed R multiples from entry:

| Target | R Multiple | Allocation |
|--------|-----------|------------|
| TP1 | 1.0R | 50% |
| TP2 | 2.0R | 30% |
| TP3 | 3.0R | 20% |

TP3 is snapped to the nearest structure level (within 0.3 ATR tolerance) to make it more likely to fill.

#### Trade Type Classification

Based on the timeframe that triggered the strongest votes:

| Type | Holding Period | When |
|------|---------------|------|
| Intraday | Close within hours | 15m/1h dominant |
| Swing | Hours to 1-2 days | 4h dominant |
| Short-term | 1-5 days | 1d dominant |
| Positional | Days to weeks | 1w dominant |

---

### Step 4: LLM Narration

The bot generates a 2-3 sentence AI explanation of *why* the signal was generated. This is purely descriptive — the AI never influences the score or plan.

**5-level fallback chain:**

```
1. Try Qwen3-8B-Instruct via HF API (primary token)
   ↓ fails
2. Try Llama-3.1-8B-Instruct via HF API (same token)
   ↓ fails
3. Try Qwen3 with backup HF token
   ↓ fails
4. Try Llama with backup token
   ↓ fails
5. Use deterministic template (no API needed) ← always works
```

**Validation**: The LLM output is checked for direction agreement. If it says "bearish" but the signal is LONG, the output is rejected and the next fallback is tried.

**Daily cap**: 100 LLM calls/day. After that, all narrations use the template.

The **template fallback** builds a fact sheet from the signal data and formats it as a readable paragraph. Example:

> *"BTC shows STRONG BUY at +72.3. Trend on 1d: bullish (EMA stack aligned, SuperTrend supporting). Structure: bullish BOS at 84,200. Key trigger: RSI divergence on 4h. Entry zone 83,800–84,100 offers 2.1:1 reward-to-risk."*

---

### Step 5: Anti-Spam Filters

Before sending the alert, two filters run:

#### Time-Based Cooldown
No duplicate alerts for the same symbol + direction within **4 hours** (configurable). Exception: if the new score is **15+ points higher** than the previous signal's score, the cooldown is overridden.

#### Price-Level Deduplication
If there's already an **open** signal for the same symbol + direction whose entry zone overlaps the new one (within ±1 ATR tolerance), the signal is suppressed. This prevents alert spam when price oscillates around the same level.

---

### Step 6: Telegram Alert

The formatted alert looks like this:

```
🟢 BTC  •  STRONG BUY
📊 Score: +72.3  |  Confidence: 88%
🔄 Type: Swing  —  hold hours to 1-2 days

🔍 Bullish BOS + OB retest at 84,200 with RSI divergence support

📊 Timeframes: 1w ↑ | 1d ↑ | 4h ↑ | 1h → | 15m ↑

🎯 Trade Plan:
  Entry:  83,800 – 84,100
  SL:     83,200  (−1.2 ATR)
  TP1:    84,800  (+1.0R)  50%
  TP2:    85,500  (+2.0R)  30%
  TP3:    86,200  (+3.0R)  20%
  R:R:    2.1:1

🤖 AI: BTC is printing a higher low at a daily order block with
bullish MACD divergence on 4h. Structure favors long above 83,200.

🕒 30 Aug, 8:15 PM IST
```

The message uses HTML formatting and is sent via the Telegram Bot API with retry logic (3 attempts, respects 429 rate limits with retry-after backoff).

---

### Step 7: Signal Storage

Every signal is saved to **two places**:

1. **SQLite database** (`data/bot.db`) — `signals` table with full metadata (entry, SL, TP, score, narration, trade type, data source)
2. **JSONL log** (`data/signals.jsonl`) — append-only audit trail

Signals get a status: `open` → `won` | `lost` | `expired` | `invalidated`.

---

### Step 8: Outcome Tracking (Hourly)

Every hour, the **outcome checker** runs:

```
For each open signal:
  1. Get the current price from cached candles
  2. Check if entry was filled (price reached entry zone)
  3. If filled:
     - Check SL hit → mark "lost", send 🔴 alert
     - Check TP1/TP2/TP3 hit → mark "won", send 🟢 alert
     - Update MFE (max favorable excursion in R)
     - Update MAE (max adverse excursion in R)
  4. If signal is older than 48 hours → mark "expired"
```

Follow-up alerts are sent to Telegram:

```
🟢 BTC TP1 HIT — WIN
  Price: 84,800  |  +1.0R
  Signal #42 closed as WIN

🔴 ETH SL HIT
  Price: 3,180  |  MFE: 0.6R
  Signal #43 closed as LOSS
```

---

### Step 9: Performance Reports (Daily/Weekly)

At midnight UTC, the bot computes and sends stats:

- Total signals, wins, losses, expired
- Win rate (wins / closed signals)
- Average MFE/MAE in R multiples
- Best and worst R
- Breakdown by symbol and by label (BUY vs SELL performance)
- Daily snapshot saved to `performance_daily` table

Weekly report adds 7-day trends.

---

### Step 10: Maintenance (Daily)

At midnight UTC:
- Delete candles older than 90 days from the cache
- Archive JSONL log files over 10 MB
- Clean up expired signals from the signals table

---

## Commands

### CLI

```bash
python run_bot.py                # start 15-min scan loop (dry run)
python run_bot.py --live         # scan loop with real Telegram alerts
python run_bot.py --scan-once    # one scan, print results, exit
python run_bot.py --status       # current scores for all symbols
python run_bot.py --report       # generate daily performance report
python run_bot.py --log-level DEBUG  # verbose logging
```

### Telegram Bot Commands

| Command | What it does |
|---------|-------------|
| `/status` | Current scores for all symbols |
| `/last` | Details of the last emitted signal |
| `/report` | Today's win/loss stats |
| `/watchlist` | Active symbols |
| `/help` | List commands |

---

## Profiles

Change `profile:` in `config.yaml`:

```yaml
profile: balanced        # default — shipped settings
profile: conservative    # fewer trades, wider stops, R:R ≥ 2.0
profile: aggressive      # more trades, tighter stops, R:R ≥ 1.0
```

| Setting | Conservative | Balanced | Aggressive |
|---------|-------------|----------|------------|
| Watch threshold | 25 | 18 | 12 |
| Signal threshold | 50 | 40 | 30 |
| Strong threshold | 75 | 65 | 55 |
| Min R:R | 2.0 | 1.5 | 1.0 |
| Min ADX | 25 | 20 | 15 |
| Cooldown | 6h | 4h | 2h |
| Stop width | 2.0 ATR | 1.5 ATR | 1.2 ATR |
| TP targets | 1.5/3.0/4.5R | 1.0/2.0/3.0R | 0.8/1.5/2.5R |
| Best for | Swing/positional | General | Intraday |

Profiles are applied at config load time via deep-merge. The `balanced` profile changes nothing — it uses `config.yaml` values as-is.

---

## Configuration

### Watchlist

Each symbol needs a primary and fallback data source:

```yaml
symbols:
  - name: BTC
    primary:  {source: binance,     ticker: "BTC/USDT"}
    fallback: {source: hyperliquid, ticker: "BTC"}
    session: always          # always | us_cash | fx_week
```

Sources: `binance`, `hyperliquid`, `yfinance`, `twelvedata`

### Symbol Overrides

Per-symbol threshold tuning (from backtest results):

```yaml
symbol_overrides:
  US30:    {watch: 25}    # noisier symbol, needs stronger confluence
  US100:   {watch: 15}    # quiet on perps, lower threshold to catch setups
  EURUSD:  {watch: 15}    # thinner books on perps
```

---

## Backtest

```bash
python run_backtest.py            # backtest all symbols
python run_backtest.py BTC ETH    # specific symbols only
```

The backtest uses **walk-forward replay**: it slides a window across historical candles, running the full confluence + plan pipeline at each step, then checks if price hit SL or TP in subsequent bars. Results go to `data/backtest_report.md`.

---

## Tests

```bash
py -m pytest tests/ -v                              # full suite (189 tests)
py -m pytest tests/test_golden.py -v                # golden-file regression
UPDATE_GOLDEN=1 py -m pytest tests/test_golden.py   # refresh after engine changes
```

### Golden-File Test

A frozen 500-candle fixture is scored, and the output is compared byte-for-byte against a committed snapshot. If any indicator, vote, or plan value drifts even by 0.001, the test fails. This catches accidental regressions.

To update after intentional changes:
```bash
UPDATE_GOLDEN=1 py -m pytest tests/test_golden.py
# then READ THE DIFF before committing
```

---

## Project Structure

```
run_bot.py              ← entry point (4 CLI modes)
run_backtest.py         ← backtest runner
config.yaml             ← all settings (profiles, watchlist, thresholds, gates)
.env                    ← secrets (never commit)

src/
  config.py             ← config loader + validation + profile merge
  profiles.py           ← conservative/balanced/aggressive presets
  scanner.py            ← one scan cycle (fetch → score → plan → alert)
  scheduler.py          ← timed loop + Telegram command polling

  data/
    router.py           ← source dispatcher (primary → fallback → cache)
    crypto.py           ← Binance via ccxt (3 mirrors, 3 retries)
    hyperliquid.py      ← Hyperliquid HIP-3 API
    yfinance_source.py  ← Yahoo Finance (4h resampled from 1h)
    tradfi.py           ← Twelve Data (credit-budgeted, backoff)
    budget.py           ← API credit tracker (750/day, 7/min for Twelve Data)
    cache.py            ← SQLite candle cache (upsert on conflict)
    sessions.py         ← market hours (DST-aware via zoneinfo)
    resample.py         ← build HTF bars from LTF (incomplete buckets dropped)

  analysis/
    indicators.py       ← 9 classic indicators (RSI, MACD, BB, ATR, ADX, etc.)
    modern.py           ← 8 modern indicators (SuperTrend, VWAP, Ichimoku, etc.)
    smc.py              ← Smart Money Concepts (BOS, CHoCH, OBs, FVGs, sweeps)
    price_action.py     ← candle patterns, S/R, Fibonacci retracement
    confluence.py       ← voting engine (5 categories × 5 timeframes → score)
    levels.py           ← entry/SL/TP generator (OB → FVG → fib → market)
    pivots.py           ← fractal pivot detection for swing highs/lows

  llm/
    explain.py          ← HF API with 5-level model fallback + validation
    template.py         ← deterministic narration (no API, always works)

  alerts/
    telegram.py         ← send + retry + bot command polling
    formatter.py        ← HTML message formatting with IST timestamps

  store/
    db.py               ← SQLite schema (WAL mode, 6 tables)
    signals.py          ← signal CRUD + cooldown + price-level dedup

  tracking/
    outcome_checker.py  ← SL/TP hit tracking + MFE/MAE + Telegram follow-ups
    performance.py      ← win rate, R stats, by-symbol/by-label breakdown
    report.py           ← daily/weekly Telegram reports

  backtest/
    replay.py           ← walk-forward replay engine
    report.py           ← markdown report generator

  logging_util.py       ← JSONL append-only audit logging
  maintenance.py        ← DB cleanup, log rotation, archiving

tests/
  conftest.py           ← shared fixtures (synthetic frames, real cache)
  test_golden.py        ← golden-file determinism test
  test_confluence.py    ← confluence scoring + direction tests
  test_indicators.py    ← indicator correctness (RSI, MACD, ATR, etc.)
  test_plan.py          ← trade plan generation + gates
  test_smc.py           ← SMC detection accuracy
  test_symmetry.py      ← long/short mirror symmetry
  test_real_data.py     ← tests on cached real candles
  fixtures/
    candles_500.json    ← frozen 500-bar fixture (5 timeframes)
    golden_signal.json  ← committed engine output snapshot
    generate.py         ← fixture regeneration script

data/
  bot.db                ← SQLite database (auto-created)
  signals.jsonl         ← signal audit log (append-only)
  outcomes.jsonl        ← outcome audit log (append-only)
  scans.jsonl           ← scan cycle log (append-only)
```

---

## Design Principles

1. **The bot analyses and reports. It never places an order.**
2. **The rules decide. The AI only describes.** — The confluence score is purely deterministic. The LLM narration is cosmetic.
3. **Same candles → same score, always.** — No randomness, no time-of-day effects, no hidden state between scans.
4. **Only closed candles.** — Live bars are dropped to prevent repainting. A signal generated at 8:15 PM is based on the 8:00 PM candle, not the incomplete 8:15 one.
5. **Fail gracefully.** — Every data source, API, and network call has retry logic and fallback chains. The bot should never crash from a transient error.
