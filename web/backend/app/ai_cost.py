"""Estimated USD cost of signals/news_ai.py's Anthropic API usage.

Computed from raw token/search counts (persisted as AiUsageEvent rows,
see models.py) rather than a dollar figure baked in at record time -- a
future pricing change only means updating the rate constants here, not
re-deriving every historical estimate from a number that was already
wrong.

Rates as of 2026-08-13 for claude-sonnet-5 (introductory pricing through
2026-08-31 -- verify at https://platform.claude.com/docs before trusting
this after that date):
  input:        $2.00 / MTok
  output:       $10.00 / MTok
  cache read:   $0.20 / MTok  (0.1x input, standard Anthropic discount)
  cache write:  $2.50 / MTok  (1.25x input, 5-minute TTL)
  web_search:   $10.00 / 1,000 searches, plus the token cost of results
                 returned (already covered by input_tokens above)
"""
from __future__ import annotations

INPUT_PRICE_PER_MTOK = 2.00
OUTPUT_PRICE_PER_MTOK = 10.00
CACHE_READ_PRICE_PER_MTOK = 0.20
CACHE_WRITE_PRICE_PER_MTOK = 2.50
WEB_SEARCH_PRICE_PER_1K = 10.00


def estimate_cost_usd(
    input_tokens: int,
    output_tokens: int,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    web_search_requests: int = 0,
) -> float:
    return (
        input_tokens / 1_000_000 * INPUT_PRICE_PER_MTOK
        + output_tokens / 1_000_000 * OUTPUT_PRICE_PER_MTOK
        + cache_creation_input_tokens / 1_000_000 * CACHE_WRITE_PRICE_PER_MTOK
        + cache_read_input_tokens / 1_000_000 * CACHE_READ_PRICE_PER_MTOK
        + web_search_requests / 1_000 * WEB_SEARCH_PRICE_PER_1K
    )
