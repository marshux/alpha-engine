"""Tests for signals/news_ai.py using a fake Anthropic client -- no real API
calls. Covers the caching and confidence-weighting logic, which is all pure
Python around the one non-deterministic call (the LLM request itself).
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
    return client


def test_research_symbol_calls_api_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(news_ai, "CACHE_DIR", tmp_path)
    result = {"sentiment": 0.5, "confidence": 0.8, "event_tags": ["earnings_beat"], "summary": "s"}
    client = _fake_client(result)

    first = news_ai._research_symbol(client, "AAPL", "2026-01-05")
    assert first == result
    assert client.messages.create.call_count == 1

    second = news_ai._research_symbol(client, "AAPL", "2026-01-05")
    assert second == result
    assert client.messages.create.call_count == 1  # cache hit, no second call


def test_research_symbol_handles_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(news_ai, "CACHE_DIR", tmp_path)
    client = _fake_client({}, stop_reason="refusal")

    result = news_ai._research_symbol(client, "XYZ", "2026-01-05")

    assert result["sentiment"] == 0.0
    assert result["confidence"] == 0.0


def test_compute_weights_by_confidence(tmp_path, monkeypatch):
    monkeypatch.setattr(news_ai, "CACHE_DIR", tmp_path)

    def fake_research(client, symbol, date_str):
        return {
            "AAPL": {"sentiment": 1.0, "confidence": 1.0, "event_tags": [], "summary": ""},
            "LOWCONF": {"sentiment": 1.0, "confidence": 0.1, "event_tags": [], "summary": ""},
        }[symbol]

    monkeypatch.setattr(news_ai, "_research_symbol", fake_research)
    monkeypatch.setattr(news_ai.anthropic, "Anthropic", lambda api_key=None: MagicMock())

    scores = news_ai.compute(["AAPL", "LOWCONF"], pd.Timestamp("2026-01-05"), api_key="test")

    # same raw sentiment, but AAPL's higher confidence should score higher
    assert scores["AAPL"] > scores["LOWCONF"]
