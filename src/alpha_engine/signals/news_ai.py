"""AI-driven sentiment research: Claude + the web_search server tool (and,
optionally, X's MCP server for direct X/Twitter search) researches current
sentiment around the AI sector as a whole and returns a structured score.

Deliberately macro, not per-ticker. An earlier version of this module made
one research call per symbol -- at full-universe scale (63 symbols) that
cost ~$27/rebalance (~$1,400/year at weekly cadence) for a component that
only ever fed a 0.1-weighted signal. A single sector-level read costs
roughly 1/63rd as much and is arguably a *better* fit for what this is
actually used for: modulating the AI-theme concentration hedge
(decision/portfolio.py::modulate_theme_cap), which is a portfolio-level
risk parameter, not a per-stock ranking input -- see live/runner.py for how
the resulting scalar is applied.

LIVE-ONLY -- do not call this from the backtester. `web_search` (and X
search) query the *current* web; a backtest evaluating a rebalance from,
say, 2021 would get 2026 results back, which is lookahead bias. This module
is for paper/live trading only, run at each real-world rebalance. See
CLAUDE.md for the full rationale. `backtest/engine.py` never imports this
module.

Results are structured JSON (not free text) via `output_config.format`, so
the LLM's non-determinism is confined to signal generation -- the output is
just a scalar feeding a deterministic formula in
decision/portfolio.py::modulate_theme_cap. Results are cached by date so
repeat runs on the same trading day don't re-query the API.

X search is opt-in, gated on X_MCP_AUTHORIZATION_TOKEN being set -- it's a
paid, separate-billing service (X's own pay-per-use API, not part of the
Anthropic bill), so it must not silently start costing money just because
ANTHROPIC_API_KEY is configured. Without the token, this falls back to
web_search alone. The token is a plain OAuth bearer token passed
per-request (`mcp_servers[].authorization_token`) -- unlike Managed Agents'
vault credentials, the Messages API does not auto-refresh it, so whoever
owns the X developer app needs to rotate it manually before it expires.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import anthropic
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache" / "news_ai"

# Sentiment scoring from search results doesn't need frontier-tier
# reasoning -- Sonnet handles this well at a fraction of Opus's token cost.
MODEL = "claude-sonnet-5"
MCP_CLIENT_BETA = "mcp-client-2025-11-20"
X_MCP_URL = "https://api.x.com/mcp"
X_MCP_SERVER_NAME = "x_search"

SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "number",
            "description": (
                "Near-term sentiment for the AI sector as a whole, from -1 "
                "(very negative) to 1 (very positive)."
            ),
        },
        "confidence": {
            "type": "number",
            "description": "Confidence in this assessment, from 0 (no relevant news found) to 1 (strong, clear signal).",
        },
        "event_tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Short tags for notable events found, e.g. 'earnings_beat', 'capex_cut', 'regulatory_action', 'model_release'.",
        },
        "summary": {
            "type": "string",
            "description": "One-sentence summary of the key news driving the score.",
        },
    },
    "required": ["sentiment", "confidence", "event_tags", "summary"],
    "additionalProperties": False,
}


def _cache_path(date_str: str) -> Path:
    return CACHE_DIR / date_str / "ai_sector_sentiment.json"


def _research_ai_sector_sentiment(
    client: anthropic.Anthropic, date_str: str, x_authorization_token: str | None
) -> dict:
    cache_path = _cache_path(date_str)
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    tools: list[dict] = [{"type": "web_search_20260209", "name": "web_search"}]
    request_kwargs: dict = {}
    if x_authorization_token:
        tools.append({"type": "mcp_toolset", "mcp_server_name": X_MCP_SERVER_NAME})
        request_kwargs["betas"] = [MCP_CLIENT_BETA]
        request_kwargs["mcp_servers"] = [
            {
                "type": "url",
                "url": X_MCP_URL,
                "name": X_MCP_SERVER_NAME,
                "authorization_token": x_authorization_token,
            }
        ]
        messages_create = client.beta.messages.create
        source_note = (
            "You also have direct X (Twitter) search -- use it explicitly for "
            "real-time analyst and market reaction, which often moves before "
            "mainstream coverage catches up. Weight credible, widely-shared "
            "reactions more heavily than one-off posts."
        )
    else:
        messages_create = client.messages.create
        source_note = ""

    response = messages_create(
        model=MODEL,
        max_tokens=4096,
        tools=tools,
        output_config={"format": {"type": "json_schema", "schema": SENTIMENT_SCHEMA}},
        messages=[
            {
                "role": "user",
                "content": (
                    "Research recent news and sentiment (last ~7 days) for the AI "
                    "sector as a whole -- major AI companies (e.g. Nvidia, "
                    "hyperscaler capex plans, foundation-model labs), earnings "
                    "and guidance touching AI spend, regulatory developments, "
                    "and notable model or product releases. "
                    f"{source_note} "
                    "Score the near-term sentiment for the sector based only on "
                    "what you find; if nothing relevant turns up, say so with "
                    "low confidence rather than guessing."
                ),
            }
        ],
        **request_kwargs,
    )

    if response.stop_reason == "refusal":
        result = {"sentiment": 0.0, "confidence": 0.0, "event_tags": [], "summary": "refused"}
    else:
        text = next(b.text for b in response.content if b.type == "text")
        result = json.loads(text)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result))
    return result


def compute_ai_sector_sentiment(
    as_of: pd.Timestamp,
    api_key: str | None = None,
    x_authorization_token: str | None = None,
) -> dict:
    """Single research call per rebalance (not per-symbol) -- returns the
    raw `{sentiment, confidence, event_tags, summary}` dict, consumed by
    decision/portfolio.py::modulate_theme_cap as a portfolio-level input,
    not a per-stock ranking signal.

    `x_authorization_token`, if provided (defaults to the
    X_MCP_AUTHORIZATION_TOKEN env var): enables X/Twitter search via X's MCP
    server alongside web_search. Left unset, this is web_search-only -- see
    the module docstring for why that's the safe default (X search is
    billed separately, per-action, by X).
    """
    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    x_authorization_token = x_authorization_token or os.environ.get("X_MCP_AUTHORIZATION_TOKEN")
    date_str = as_of.strftime("%Y-%m-%d")
    return _research_ai_sector_sentiment(client, date_str, x_authorization_token)
