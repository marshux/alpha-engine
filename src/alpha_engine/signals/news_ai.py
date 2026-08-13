"""AI-driven news research signal: Claude + the web_search server tool
researches recent news for a symbol and returns a structured sentiment
score.

LIVE-ONLY -- do not call this from the backtester. `web_search` queries the
*current* web; a backtest evaluating a rebalance from, say, 2021 would get
2026 search results back, which is lookahead bias. This module is for
paper/live trading only, run at each real-world rebalance. See CLAUDE.md
for the full rationale. `backtest/engine.py` never imports this module.

Results are structured JSON (not free text) via `output_config.format`, so
the LLM's non-determinism is confined to signal generation -- the output is
just another numeric column feeding the deterministic decision engine in
decision/scorer.py. Results are cached by (symbol, date) so repeat runs on
the same trading day don't re-query the API.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import anthropic
import pandas as pd

from alpha_engine.signals.base import zscore

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache" / "news_ai"

MODEL = "claude-opus-5"

SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "number",
            "description": "Near-term sentiment for the stock, from -1 (very negative) to 1 (very positive).",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence in this assessment, from 0 (no relevant news found) to 1 (strong, clear signal).",
        },
        "event_tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Short tags for notable events found, e.g. 'earnings_beat', 'guidance_cut', 'lawsuit', 'product_launch'.",
        },
        "summary": {
            "type": "string",
            "description": "One-sentence summary of the key news driving the score.",
        },
    },
    "required": ["sentiment", "confidence", "event_tags", "summary"],
    "additionalProperties": False,
}


def _cache_path(symbol: str, date_str: str) -> Path:
    return CACHE_DIR / date_str / f"{symbol}.json"


def _research_symbol(client: anthropic.Anthropic, symbol: str, date_str: str) -> dict:
    cache_path = _cache_path(symbol, date_str)
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        tools=[{"type": "web_search_20260209", "name": "web_search"}],
        output_config={"format": {"type": "json_schema", "schema": SENTIMENT_SCHEMA}},
        messages=[
            {
                "role": "user",
                "content": (
                    f"Research recent news (last ~7 days) for the stock {symbol}. "
                    "Look for earnings, guidance changes, major announcements, "
                    "lawsuits, management changes, or other market-moving events. "
                    "Score the near-term sentiment based only on what you find; "
                    "if nothing relevant turns up, say so with low confidence "
                    "rather than guessing."
                ),
            }
        ],
    )

    if response.stop_reason == "refusal":
        result = {"sentiment": 0.0, "confidence": 0.0, "event_tags": [], "summary": "refused"}
    else:
        text = next(b.text for b in response.content if b.type == "text")
        result = json.loads(text)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result))
    return result


def compute(symbols: list[str], as_of: pd.Timestamp, api_key: str | None = None) -> pd.Series:
    """Researches each symbol's recent news and returns a z-scored sentiment
    signal, weighted by the model's own stated confidence so a "nothing
    found" result pulls toward neutral rather than injecting noise.
    """
    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    date_str = as_of.strftime("%Y-%m-%d")

    values: dict[str, float] = {}
    for symbol in symbols:
        result = _research_symbol(client, symbol, date_str)
        values[symbol] = result["sentiment"] * result["confidence"]

    return zscore(pd.Series(values, dtype=float))
