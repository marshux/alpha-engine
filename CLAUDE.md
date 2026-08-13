# alpha-engine

A systematic swing-trading research engine: quant signals plus an AI news
signal feed a deterministic decision engine, which is backtested against
index benchmarks (SPY/VOO/VTI). Goal is to beat cap-weighted index funds
over days-to-weeks holding periods. Backtest-first — no live/paper order
execution yet.

## Core design constraint

**AI is allowed to research and score; it is never allowed to decide.**
`signals/news_ai.py` calls Claude with the `web_search` tool to research
news and returns a *structured* sentiment score (forced via
`output_config.format` — not free text). That score becomes just another
numeric column. Everything downstream of the signal table
(`decision/scorer.py`, `decision/portfolio.py`, `decision/risk.py`) is a
plain weighted-sum/rules pipeline with no LLM call anywhere in it. This is
what makes backtests trustworthy: given the same signal inputs, the
decision layer always produces the same trades. Preserve this separation in
any future work — don't let an LLM call creep into the decision path.

## Architecture

```
data/equities.py  →  signals/*.py  →  decision/scorer.py  →  decision/portfolio.py  →  backtest/engine.py
  (yfinance,           (momentum,        (weighted sum         (top-N, sizing,           (walk-forward loop,
   parquet cache)       mean_reversion,   of z-scored           caps)                      turnover costs,
                        volatility,       signals)                                         benchmark compare)
                        news_ai)
```

- **`data/equities.py`**: `yfinance` OHLCV fetch, read-through parquet cache
  at `data/cache/<symbol>.parquet` (gitignored). Cache is overwritten
  wholesale on a miss rather than incrementally merged — simpler, and
  yfinance full-history downloads are cheap.
- **`signals/base.py`**: every signal module exposes `compute(prices, as_of)
  -> pd.Series`, cross-sectionally z-scored so different signals are
  comparable and combinable by simple weighted sum.
- **`signals/momentum.py`, `mean_reversion.py`, `volatility.py`**: pure
  pandas/numpy, fully deterministic. `volatility.py` also exposes
  `raw_annualized_vol()` (the un-inverted, un-z-scored number), reused
  directly by `decision/portfolio.py` for volatility-scaled position sizing
  — avoids re-deriving raw vol from the inverted/z-scored signal.
- **`signals/news_ai.py`**: **live-only, never called by the backtester.**
  `web_search` queries the *current* web — using it inside a historical
  backtest would be lookahead bias (a 2021 rebalance would see 2026 search
  results). `backtest/engine.py` has no import of this module. Results are
  cached by `(symbol, date)` under `data/cache/news_ai/`.
- **`decision/scorer.py`**: weighted sum of the signal table. Missing
  signal values (e.g. `news_ai` before it's wired into a live run) are
  treated as neutral (0), not dropped — a symbol isn't excluded from
  ranking just because one signal has no data for it yet.
- **`decision/portfolio.py`**: top-N selection, then position sizing
  (equal-weight or volatility-scaled), then a per-position cap. Capping
  never redistributes freed-up capital to other positions — it's
  deliberately left as cash rather than adding redistribution complexity,
  so a fully-capped equal-weight portfolio can end up under-invested. Then
  a sector-exposure cap (currently a no-op for individual stocks — see
  "Known gaps" below).
- **`decision/risk.py`**: per-position stop-loss and a portfolio-level
  max-drawdown circuit breaker, both evaluated on realized data only.
- **`backtest/engine.py`**: walk-forward loop over rebalance dates. Tracks
  the portfolio as **weights**, not share counts — no fractional-share or
  cash-drag bookkeeping, which isn't the point of a v1 research backtester.
  One consequence: risk checks (circuit breaker, stop-loss) are only
  evaluated at rebalance granularity (weekly by default), not intraday —
  a single week's move can overshoot the circuit-breaker threshold before
  it's caught.

## Known gaps (deliberate v1 scope, not oversights)

- **Circuit breaker never re-enters.** Once portfolio drawdown breaches
  `risk.max_drawdown_pct`, the engine flattens to cash — and since cash
  stays flat, the drawdown-from-peak never recovers below the threshold, so
  the strategy stays in cash for the rest of the backtest. Confirmed in the
  first end-to-end run (2019-2024 sector-ETF backtest, flattened after the
  2020 COVID crash and never re-entered — see equity curve in
  `output/equity_curve.csv`). A re-entry rule (fixed cooldown, drawdown
  recovery threshold, etc.) is a real strategy design decision, not
  something to guess at — raise it explicitly before changing.
- **No sector classification for individual stocks.** `universe.yaml`'s
  `large_caps` list has no GICS sector mapping, so
  `decision/portfolio.apply_sector_cap` only meaningfully caps sector ETFs
  (which are their own sector). Single-position caps still apply to
  individual stocks.
- **Universe is a hand-curated static list**, not live S&P 500 membership —
  deliberate, to avoid survivorship bias silently creeping into backtests
  from following an index's *current* constituents into the past.

## Config

- **`config/universe.yaml`**: explicit ticker lists (large-caps, 11 SPDR
  sector ETFs, benchmarks). Edit directly to change the tradable universe.
- **`config/strategy.yaml`**: signal weights, rebalance cadence, portfolio
  construction params, risk limits, cost model, backtest params. This is
  the primary lever for changing strategy behavior without touching code.

## Running a backtest

```
.venv/bin/python3 -m alpha_engine.cli backtest --start 2019-01-01 --end 2024-12-31
```

`--symbols` overrides the configured universe (useful for a fast sanity
check on just the sector ETFs); `--config` points at an alternate
`strategy.yaml`. Prints a strategy-vs-benchmark summary table and writes
`output/equity_curve.csv`.

## Environment setup

This machine has no system `pip`/`venv` bootstrap (`ensurepip` missing, no
sudo). The `.venv` here was built by downloading `python3-pip` /
`python3-venv` / `python3-distutils` `.deb`s with `apt-get download`
(no root needed), extracting with `dpkg-deb -x`, and using the extracted
pip (via `PYTHONPATH`) to bootstrap a real pip into a `python3 -m venv
--without-pip` environment. Once `.venv/bin/pip` exists, everything after
that is normal (`pip install -e ".[dev]"`). If `.venv` is ever deleted, redo
that bootstrap rather than assuming `python3 -m venv` alone will work.

## Testing

```
.venv/bin/python3 -m pytest tests/ -v
```

All signal/decision/backtest tests run against synthetic price data with
hand-reasoned expected outputs (e.g. a known uptrend should score higher
than a known downtrend) — no network calls, no API key needed.
`test_news_ai.py` mocks the Anthropic client entirely; it never hits the
real API.

## AI news signal — setup for live use

Requires `ANTHROPIC_API_KEY` (see `.env.example`). Not wired into any CLI
command yet — it's a live-trading-phase feature, added once paper trading
starts. When integrating it, pass a `news_signal` DataFrame into
`run_backtest()`/a live runner (indexed by date, one column per symbol) —
never call `signals/news_ai.compute()` from inside a historical backtest
loop.
