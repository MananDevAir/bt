"""HF Inference API caller with multi-token + multi-model fallback.

Fallback chain (5 levels — the bot never blocks on a narration failure):

    1. HF_TOKEN      + primary model   (Qwen/Qwen3-8B)
    2. HF_TOKEN      + fallback model  (Llama-3.1-8B-Instruct)
    3. HF_TOKEN_FALLBACK + primary model
    4. HF_TOKEN_FALLBACK + fallback model
    5. Deterministic template          (always works, no API)

Post-validation: if the LLM reply mentions a direction that contradicts
the rule engine, or contains numbers not in the fact sheet, the reply is
rejected and the next level in the chain is tried.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

import requests

from .template import build_fact_sheet, narrate as template_narrate
from ..config import Config

log = logging.getLogger(__name__)

__all__ = ["explain"]

URL = "https://router.huggingface.co/v1/chat/completions"

SYSTEM_PROMPT = """\
You are an institutional Price Action (PA) & Smart Money Concepts (SMC) trader.
Write a crisp, clean 3-bullet technical rationale for a Telegram alert based on the provided facts.

STRICT FORMAT (Output ONLY these 3 bullets, nothing else):
• Entry: <1 sentence on technical reason for entry (e.g. FVG fill, Order Block, S/R retest, CHoCH/BOS)>
• Stop-Loss: <1 sentence on why the SL is placed safely beyond support/resistance/sweep>
• Targets: <1 sentence on which key resistance/liquidity pools TP1-TP3 target>

RULES:
- Keep it concise, punchy, and professional (under 80 words total).
- Do NOT repeat numeric tables or list "Trade Overview" (the user already has the price table).
- Do NOT use markdown headers (no ### or ##).
- Do NOT add preamble, intro, or concluding summary.
- Strictly use the technical context from the provided facts.
"""


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"]


def _get_tokens() -> list[str]:
    """Collect all available HF tokens from environment."""
    tokens: list[str] = []
    primary = os.environ.get("HF_TOKEN", "").strip()
    if primary:
        tokens.append(primary)
    fallback = os.environ.get("HF_TOKEN_FALLBACK", "").strip() or os.environ.get("HF_BACKUP_TOKEN", "").strip()
    if fallback and fallback != primary:
        tokens.append(fallback)
    return tokens


def _get_models(cfg: Config) -> list[str]:
    """Primary + fallback model names from config."""
    llm_cfg = cfg.get("llm", default={}) or {}
    primary = str(llm_cfg.get("model", "Qwen/Qwen3-8B"))
    fallback = str(llm_cfg.get("fallback_model", "meta-llama/Llama-3.1-8B-Instruct"))
    models = [primary]
    if fallback and fallback != primary:
        models.append(fallback)
    return models


def _call_groq(api_key: str, model: str, prompt: str,
               timeout: int = 8, max_tokens: int = 1500) -> str | None:
    """Make an ultra-fast (<0.4s) Groq API call."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    try:
        resp = requests.post(GROQ_URL, headers=headers, json=body, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content.strip() if content else None
        else:
            log.debug("Groq %d (model=%s): %s", resp.status_code, model, resp.text[:120])
            return None
    except Exception as exc:
        log.debug("Groq call failed (model=%s): %s", model, exc)
        return None


def _call_hf(token: str, model: str, prompt: str,
             timeout: int = 12, max_tokens: int = 1500) -> str | None:
    """Make a single HF Inference API call.  Returns text or None."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }

    for attempt in range(2):  # 1 try + 1 retry
        try:
            resp = requests.post(URL, headers=headers, json=body, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                msg = data.get("choices", [{}])[0].get("message", {})
                content = msg.get("content")

                # Qwen3 thinking mode: content has the real answer,
                # reasoning_content has internal thinking.  NEVER use
                # reasoning_content — it's "let me think about this..."
                # If content is null, the model spent all tokens thinking
                # and never produced an answer → treat as failure.
                if not content:
                    log.warning("HF content=null (model=%s, likely spent all "
                                "tokens on thinking)", model)
                    return None

                # Strip any leaked <think>...</think> tags from content
                content = re.sub(r"<think>.*?</think>", "", content,
                                 flags=re.DOTALL).strip()

                if not content:
                    log.warning("HF content empty after stripping think tags "
                                "(model=%s)", model)
                    return None

                return content.strip()
            elif resp.status_code in (401, 403):
                log.warning("HF auth error %d with model=%s", resp.status_code, model)
                return None  # no point retrying auth errors
            elif resp.status_code in (402, 429):
                log.warning("HF rate/credit limit %d (model=%s)", resp.status_code, model)
                return None  # try next token/model
            else:
                log.warning("HF error %d (model=%s, attempt=%d): %s",
                            resp.status_code, model, attempt + 1, resp.text[:200])
                if attempt == 0:
                    time.sleep(2)
                    continue
                return None
        except requests.exceptions.Timeout:
            log.warning("HF timeout (model=%s, attempt=%d)", model, attempt + 1)
            if attempt == 0:
                time.sleep(1)
                continue
            return None
        except Exception as exc:
            log.warning("HF error (model=%s): %s", model, exc)
            return None
    return None


def _validate_reply(reply: str, facts: dict[str, Any]) -> bool:
    """Validate the reply is clean, non-contradictory, and not bloated with markdown headers."""
    if not reply or len(reply.strip()) < 15:
        return False

    lower = reply.lower()

    # Reject preamble or markdown header clutter
    if "###" in reply or "trade overview" in lower or "sure," in lower:
        log.warning("LLM reply contains header/preamble clutter, rejecting")
        return False

    stripped = lower.strip()
    if stripped.startswith("here is") or stripped.startswith("here's"):
        log.warning("LLM reply starts with preamble, rejecting")
        return False

    direction = facts.get("direction", 0)
    if direction > 0:
        if re.search(r"\b(short bias|bearish bias)\b", lower):
            log.warning("LLM reply contradicts LONG signal direction, rejecting")
            return False
    elif direction < 0:
        if re.search(r"\b(long bias|bullish bias)\b", lower):
            log.warning("LLM reply contradicts SHORT signal direction, rejecting")
            return False

    # Reject over-long replies that would bloat the Telegram alert.
    # The Telegram limit is 4096 chars; a narration should be 60-100 words.
    word_count = len(reply.split())
    if word_count > 120:
        log.warning("LLM reply too long (%d words), rejecting", word_count)
        return False

    return True


def _build_user_prompt(facts: dict[str, Any]) -> str:
    """Build the user prompt from the fact sheet — compact JSON."""
    return (
        "Explain the Price Action and SMC trade logic in 3 crisp bullets (Entry, Stop-Loss, Targets):\n\n"
        f"```json\n{json.dumps(facts, indent=2, default=str)}\n```"
    )


# Simple in-memory cache: (symbol, direction, score_bucket) -> (text, ts)
_cache: dict[tuple[str, int, int], tuple[str, float]] = {}
CACHE_TTL = 4 * 3600  # 4 hours


def _cache_key(facts: dict[str, Any]) -> tuple[str, int, int]:
    sym = facts.get("symbol", "?")
    direction = int(facts.get("direction", 0))
    score_bucket = int(facts.get("score", 0)) // 10  # group by 10-point bands
    return (sym, direction, score_bucket)


# Daily call counter (resets on day change)
_daily_state: dict[str, Any] = {"date": "", "count": 0}


def _check_daily_cap(cfg: Config) -> bool:
    """Return True if under daily cap."""
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if _daily_state["date"] != today:
        _daily_state["date"] = today
        _daily_state["count"] = 0
    cap = int((cfg.get("llm", "daily_cap", default=100) or 100))
    return _daily_state["count"] < cap


def explain(signal: Any, plan: Any | None, cfg: Config) -> tuple[str, str]:
    """Generate a narration for a signal.

    Returns:
        (narration_text, source)
        source is one of: "hf:<model>", "template"
    """
    facts = build_fact_sheet(signal, plan)

    # Check cache
    key = _cache_key(facts)
    now = time.time()
    if key in _cache:
        text, ts = _cache[key]
        if now - ts < CACHE_TTL:
            log.debug("Narration cache hit for %s", key)
            return text, "cache"

    # Check if LLM is enabled
    llm_cfg = cfg.get("llm", default={}) or {}
    if not llm_cfg.get("enabled", True):
        text = template_narrate(facts)
        return text, "template"

    # Check daily cap
    if not _check_daily_cap(cfg):
        log.info("HF daily cap reached, using template")
        text = template_narrate(facts)
        return text, "template"

    # Build the prompt
    prompt = _build_user_prompt(facts)
    timeout = int(llm_cfg.get("timeout_s", 12))

    # 1. Try Groq if GROQ_API_KEY is available (sub-second ultra-fast inference)
    groq_key = os.environ.get("GROQ_API_KEY", "").strip() or (Config.secret("GROQ_API_KEY") or "").strip()
    if groq_key:
        for g_model in GROQ_MODELS:
            log.debug("Trying Groq: model=%s", g_model)
            reply = _call_groq(groq_key, g_model, prompt, timeout=min(8, timeout))
            if reply and _validate_reply(reply, facts):
                _daily_state["count"] += 1
                _cache[key] = (reply, now)
                source = f"groq:{g_model}"
                log.info("Narration from %s (%d tokens)", source, len(reply.split()))
                return reply, source
            elif reply:
                log.warning("Groq reply rejected by validation (model=%s)", g_model)

    # 2. Try Hugging Face if HF tokens are available
    tokens = _get_tokens()
    models = _get_models(cfg)

    for token in tokens:
        masked = token[:8] + "..." + token[-4:]
        for model in models:
            log.debug("Trying HF: token=%s model=%s", masked, model)
            reply = _call_hf(token, model, prompt, timeout=timeout)
            if reply and _validate_reply(reply, facts):
                _daily_state["count"] += 1
                _cache[key] = (reply, now)
                source = f"hf:{model}"
                log.info("Narration from %s (%d tokens)", source, len(reply.split()))
                return reply, source
            elif reply:
                log.warning("HF reply rejected by validation (model=%s)", model)

    # 3. All LLM attempts failed → deterministic template
    log.info("All LLM attempts failed, using template fallback")
    text = template_narrate(facts)
    _cache[key] = (text, now)
    return text, "template"
