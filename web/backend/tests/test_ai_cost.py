"""Tests for app/ai_cost.py's pure cost-estimation math."""
from __future__ import annotations

import pytest

from app.ai_cost import estimate_cost_usd


def test_estimate_cost_usd_tokens_only():
    cost = estimate_cost_usd(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == pytest.approx(2.00 + 10.00)


def test_estimate_cost_usd_includes_web_search():
    cost = estimate_cost_usd(input_tokens=0, output_tokens=0, web_search_requests=1_000)
    assert cost == pytest.approx(10.00)


def test_estimate_cost_usd_includes_cache_read_and_write():
    cost = estimate_cost_usd(
        input_tokens=0,
        output_tokens=0,
        cache_creation_input_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
    )
    assert cost == pytest.approx(2.50 + 0.20)


def test_estimate_cost_usd_zero_usage_is_free():
    assert estimate_cost_usd(input_tokens=0, output_tokens=0) == 0.0
