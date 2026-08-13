"""Tests for signals/news_ai.py using a fake Anthropic client -- no real API
calls. Covers the caching logic and the opt-in X MCP search path. This
signal researches AI-sector sentiment as a whole (one call per rebalance),
not per-symbol -- see the module docstring for why.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import MagicMock

import pandas as pd
import pytest

from alpha_engine.signals import news_ai


@dataclass
class _FakeTextBlock:
    type: str
    text: str


@dataclass
class _FakeResponse:
    stop_reason: str
    content: list


def _fake_client(result: dict, stop_reason: str = "end_turn") -> MagicMock:
    client = MagicMock()
    if stop_reason == "refusal":
        response = _FakeResponse(stop_reason="refusal", content=[])
    else:
        response = _FakeResponse(
            stop_reason="end_turn",
            content=[_FakeTextBlock(type="text", text=json.dumps(result))],
        )
    client.messages.create.return_value = response
    client.beta.messages.create.return_value = response
    return client


def test_research_calls_api_and_caches_by_date(tmp_path, monkeypatch):
    monkeypatch.setattr(news_ai, "CACHE_DIR", tmp_path)
    result = {"sentiment": 0.5, "confidence": 0.8, "event_tags": ["capex_beat"], "summary": "s"}
    client = _fake_client(result)

    first = news_ai._research_ai_sector_sentiment(client, "2026-01-05", None)
    assert first == result
    assert client.messages.create.call_count == 1

    second = news_ai._research_ai_sector_sentiment(client, "2026-01-05", None)
    assert second == result
    assert client.messages.create.call_count == 1  # cache hit, no second call


def test_research_handles_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(news_ai, "CACHE_DIR", tmp_path)
    client = _fake_client({}, stop_reason="refusal")

    result = news_ai._research_ai_sector_sentiment(client, "2026-01-05", None)

    assert result["sentiment"] == 0.0
    assert result["confidence"] == 0.0


def test_research_omits_x_mcp_without_token(tmp_path, monkeypatch):
    monkeypatch.setattr(news_ai, "CACHE_DIR", tmp_path)
    result = {"sentiment": 0.0, "confidence": 0.0, "event_tags": [], "summary": ""}
    client = _fake_client(result)

    news_ai._research_ai_sector_sentiment(client, "2026-01-05", None)

    assert client.messages.create.call_count == 1
    assert client.beta.messages.create.call_count == 0
    tools = client.messages.create.call_args.kwargs["tools"]
    assert all(t["type"] != "mcp_toolset" for t in tools)


def test_research_uses_x_mcp_when_token_provided(tmp_path, monkeypatch):
    monkeypatch.setattr(news_ai, "CACHE_DIR", tmp_path)
    result = {"sentiment": 0.0, "confidence": 0.0, "event_tags": [], "summary": ""}
    client = _fake_client(result)

    news_ai._research_ai_sector_sentiment(client, "2026-01-05", "x-token-123")

    assert client.messages.create.call_count == 0
    assert client.beta.messages.create.call_count == 1
    call = client.beta.messages.create.call_args.kwargs
    assert call["betas"] == [news_ai.MCP_CLIENT_BETA]
    assert call["mcp_servers"] == [
        {
            "type": "url",
            "url": news_ai.X_MCP_URL,
            "name": news_ai.X_MCP_SERVER_NAME,
            "authorization_token": "x-token-123",
        }
    ]
    tool_types = [t["type"] for t in call["tools"]]
    assert "web_search_20260209" in tool_types
    assert "mcp_toolset" in tool_types


def test_compute_ai_sector_sentiment_returns_raw_result(tmp_path, monkeypatch):
    monkeypatch.setattr(news_ai, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(news_ai.anthropic, "Anthropic", lambda api_key=None: MagicMock())

    expected = {"sentiment": 0.3, "confidence": 0.6, "event_tags": [], "summary": "x"}
    monkeypatch.setattr(
        news_ai, "_research_ai_sector_sentiment", lambda client, date_str, token: expected
    )

    result = news_ai.compute_ai_sector_sentiment(pd.Timestamp("2026-01-05"), api_key="test")

    assert result == expected


def test_compute_ai_sector_sentiment_reads_x_token_from_env(tmp_path, monkeypatch):
    monkeypatch.setattr(news_ai, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("X_MCP_AUTHORIZATION_TOKEN", "from-env")
    monkeypatch.setattr(news_ai.anthropic, "Anthropic", lambda api_key=None: MagicMock())

    seen_tokens = []

    def fake_research(client, date_str, x_authorization_token):
        seen_tokens.append(x_authorization_token)
        return {"sentiment": 0.0, "confidence": 0.0, "event_tags": [], "summary": ""}

    monkeypatch.setattr(news_ai, "_research_ai_sector_sentiment", fake_research)

    news_ai.compute_ai_sector_sentiment(pd.Timestamp("2026-01-05"), api_key="test")

    assert seen_tokens == ["from-env"]
