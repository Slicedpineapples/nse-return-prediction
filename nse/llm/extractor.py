"""
LLM event extraction using Claude.

The LLM is used ONLY for:
  1. Classifying and extracting structured fields from announcements.
  2. Generating trade explanation summaries.

It does NOT output trade decisions.  The ranker uses LLM outputs as numeric
features, not as trade signals.

Output schema (per document)
----------------------------
{
  "event_type": str,             # earnings | dividend | agm | management | regulatory | financing | other
  "sentiment_score": float,      # -2.0 to +2.0
  "surprise_score": float,       # -2.0 to +2.0
  "materiality_score": float,    # 0.0 to 1.0
  "time_horizon": str,           # "1d" | "3d" | "1m" | "structural"
  "persistence": str,            # "temporary" | "persistent"
  "key_numbers": dict,           # {"eps_growth": ..., "dividend_kes": ..., etc.}
  "risks_mentioned": list[str],
  "confidence": float,           # 0.0 to 1.0
  "summary_50_words": str
}
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import requests

from nse.config import LLM
from nse.db.schema import IssuerEvent, session_scope

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a financial analyst specialising in East African equities, particularly
the Nairobi Securities Exchange (NSE), Kenya.  Your job is to read company
announcements and extract structured information for use in a quantitative
trading model.

Be precise and conservative.  If you are uncertain, lower the confidence score
and the materiality score.  Do not hallucinate numbers not present in the text.
"""

_USER_TEMPLATE = """\
Company: {issuer_name}
Ticker: {ticker}
Published: {timestamp}

Announcement text:
---
{raw_text}
---

Please extract the following fields and return valid JSON only (no markdown, no explanation):

{{
  "event_type": "<earnings|dividend|agm|management|regulatory|litigation|financing|investor_briefing|other>",
  "sentiment_score": <float -2.0 to +2.0, where -2=very negative, 0=neutral, +2=very positive>,
  "surprise_score": <float -2.0 to +2.0, where -2=very negative surprise, 0=in-line, +2=very positive surprise>,
  "materiality_score": <float 0.0 to 1.0, where 0=immaterial, 1=highly material to stock price>,
  "time_horizon": "<1d|3d|1m|structural>",
  "persistence": "<temporary|persistent>",
  "key_numbers": {{
    "eps_growth_pct": null_or_float,
    "revenue_growth_pct": null_or_float,
    "dividend_kes": null_or_float,
    "payout_change_pct": null_or_float,
    "guidance_change": null_or_string
  }},
  "risks_mentioned": ["<risk 1>", "<risk 2>"],
  "confidence": <float 0.0 to 1.0>,
  "summary_50_words": "<concise summary in ≤50 words>"
}}
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_batch(batch_size: int = 50) -> int:
    """
    Pull unprocessed events from DB, run LLM extraction, store results.

    Returns number of events processed.
    """
    from nse.ingest.announcements_live import get_unprocessed_events

    events = get_unprocessed_events(batch_size=batch_size)
    if not events:
        log.info("No unprocessed events to extract")
        return 0

    processed = 0
    for event in events:
        try:
            result = extract_event(event)
            _store_extraction(event["event_id"], result)
            processed += 1
        except Exception as exc:
            log.warning("LLM extraction failed for %s: %s", event["event_id"], exc)
            continue

    log.info("LLM processed %d events", processed)
    return processed


def extract_event(event: dict) -> dict[str, Any]:
    """
    Run LLM extraction on a single event dict.

    Routes to Ollama or Anthropic based on LLM_PROVIDER env var.

    Parameters
    ----------
    event : dict with keys: ticker, timestamp_published, raw_text

    Returns
    -------
    Structured extraction dict.
    """
    raw_text = (event.get("raw_text") or "")[:4000]

    user_prompt = _USER_TEMPLATE.format(
        issuer_name=event.get("ticker", "Unknown"),
        ticker=event.get("ticker", ""),
        timestamp=str(event.get("timestamp_published", "")),
        raw_text=raw_text,
    )

    content = _call_llm(system=_SYSTEM_PROMPT, user=user_prompt)

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            result = json.loads(match.group(0))
        else:
            raise ValueError(f"LLM did not return valid JSON: {content[:200]}")

    return result


# ---------------------------------------------------------------------------
# Provider-agnostic LLM call
# ---------------------------------------------------------------------------

def _call_llm(system: str, user: str, max_tokens: int = 1024) -> str:
    """Call the configured LLM provider and return the response text."""
    provider = LLM["provider"]
    if provider == "ollama":
        return _call_ollama(system, user, max_tokens)
    return _call_anthropic(system, user, max_tokens)


def _call_ollama(system: str, user: str, max_tokens: int) -> str:
    url = f"{LLM['ollama_base_url']}/api/chat"
    payload = {
        "model": LLM["ollama_model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": LLM["temperature"],
            "num_predict": max_tokens,
        },
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def _call_anthropic(system: str, user: str, max_tokens: int) -> str:
    import anthropic as _anthropic
    from nse.config import get_anthropic_key
    client = _anthropic.Anthropic(api_key=get_anthropic_key())
    msg = client.messages.create(
        model=LLM["model"],
        max_tokens=max_tokens,
        temperature=LLM["temperature"],
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text.strip()


def generate_trade_explanation(
    ticker: str,
    score: float,
    features: dict,
    event_summaries: list[str],
) -> str:
    """Generate a human-readable explanation for a trade recommendation."""
    _ret_keys = {"ret_3d", "ret_5d", "vol_20d"}
    key_features = {}
    for k, v in features.items():
        if v is None or (isinstance(v, float) and v != v):
            continue
        if k not in ("ret_3d", "ret_5d", "vol_20d", "volume_zscore_20d",
                     "amihud_20d", "llm_max_materiality_today",
                     "llm_weighted_sentiment_3d", "beta_nse20_60d"):
            continue
        # Convert return/vol fractions to percentages for readability
        key_features[k] = round(v * 100, 2) if k in _ret_keys else round(v, 4)
    # Label units in the dict keys so the LLM knows the scale
    key_features = {
        (k + "_pct" if k in _ret_keys else k): v
        for k, v in key_features.items()
    }

    prompt = f"""You are a trading desk analyst at a Nairobi-based fund.

Ticker: {ticker}
Model score (0-1, higher = more attractive): {score:.3f}
Key quantitative features: {json.dumps(key_features, indent=2)}
Recent announcements: {chr(10).join(event_summaries) if event_summaries else 'None'}

Write a concise (3-5 sentence) investment rationale for a long trade on {ticker}.
Focus on what the features and announcements suggest about near-term (1-3 day) price direction.
Be specific about the evidence. Do not claim certainty. End with the key risk."""

    try:
        return _call_llm(system="You are a concise financial analyst.", user=prompt, max_tokens=300)
    except Exception as exc:
        log.warning("Explanation generation failed: %s", exc)
        return f"Score: {score:.3f}. Explanation unavailable."


# ---------------------------------------------------------------------------
# DB persistence
# ---------------------------------------------------------------------------

def _store_extraction(event_id: str, result: dict) -> None:
    """Write LLM extraction results back to issuer_events."""
    key_numbers = result.get("key_numbers", {})
    risks = result.get("risks_mentioned", [])

    updates = {
        "llm_sentiment": _safe_float(result.get("sentiment_score")),
        "llm_surprise": _safe_float(result.get("surprise_score")),
        "llm_materiality": _safe_float(result.get("materiality_score")),
        "llm_confidence": _safe_float(result.get("confidence")),
        "llm_horizon": result.get("time_horizon"),
        "llm_persistence": result.get("persistence"),
        "llm_summary": result.get("summary_50_words"),
        "llm_key_numbers": json.dumps(key_numbers) if key_numbers else None,
        "llm_risks_mentioned": json.dumps(risks) if risks else None,
        "llm_processed_at": datetime.utcnow(),
        "materiality_score": _safe_float(result.get("materiality_score")),
        "event_type": result.get("event_type") or "other",
    }

    with session_scope() as s:
        s.query(IssuerEvent).filter(IssuerEvent.event_id == event_id).update(updates)


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
