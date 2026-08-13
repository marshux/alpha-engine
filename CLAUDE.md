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
data/equities.py       →  signals/*.py           →  decision/scorer.py  →  decision/portfolio.py  →  backtest/engine.py  →  backtest/report.py
data/fundamentals.py      (momentum,                 (weighted sum          (top-N, sizing,           (walk-forward loop,     backtest/walkforward.py
  (yfinance,                mean_reversion,            of z-scored            caps)                     turnover costs,
   parquet cache)           volatility, value,         signals)                                          circuit breaker)
                            quality, news_ai)
```

- **`data/equities.py`**: `yfinance` OHLCV fetch, read-through parquet cache
  at `data/cache/<symbol>.parquet` (gitignored). Cache is overwritten
  wholesale on a miss rather than incrementally merged — simpler, and
  yfinance full-history downloads are cheap.
- **`data/fundamentals.py`**: quarterly balance-sheet + income-statement
  fetch, same cache pattern at `data/cache/fundamentals/<symbol>.parquet`.
  yfinance's fundamentals history is shallow (roughly the last two years of
  quarters, not a deep point-in-time archive) — `latest_report_as_of()`
  never selects a report dated after the query date, but coverage is thin
  or empty for older backtest dates. Opt-in via `--fundamentals` (see
  below) since it's a slower, separate fetch from price data.
- **`signals/base.py`**: every signal module exposes `compute(prices, as_of)
  -> pd.Series`, cross-sectionally z-scored so different signals are
  comparable and combinable by simple weighted sum. `value.py`/`quality.py`
  diverge from this shape (they need fundamentals data, and value.py needs
  both prices and fundamentals) — `backtest/engine.py` calls them
  explicitly rather than through the uniform `SIGNAL_MODULES` dict, the
  same way it already special-cases `news_ai.py`.
- **`signals/momentum.py`, `mean_reversion.py`, `volatility.py`**: pure
  pandas/numpy, fully deterministic. `volatility.py` also exposes
  `raw_annualized_vol()` (for volatility-scaled sizing) and
  `trailing_returns_matrix()` (for min-variance sizing) — both un-inverted,
  un-z-scored, reused directly by `decision/portfolio.py` rather than
  re-deriving raw numbers from a z-scored signal.
- **`signals/value.py`**: cheapness via trailing price-to-book (needs price
  + fundamentals). **`signals/quality.py`**: ROE + low-leverage composite
  (fundamentals only). Both silently score neutral for symbols/dates
  without fundamentals coverage — same missing-signal handling as
  everywhere else, not a special case.
- **`signals/news_ai.py`**: **live-only, never called by the backtester.**
  `web_search` queries the *current* web — using it inside a historical
  backtest would be lookahead bias (a 2021 rebalance would see 2026 search
  results). `backtest/engine.py` has no import of this module. Results are
  cached by `(symbol, date)` under `data/cache/news_ai/`.
- **`decision/scorer.py`**: weighted sum of the signal table. Missing
  signal values are treated as neutral (0), not dropped — a symbol isn't
  excluded from ranking just because one signal has no data for it yet.
- **`decision/portfolio.py`**: top-N selection, then position sizing
  (`equal_weight`, `volatility_scaled`, or `min_variance` — the closed-form
  global minimum-variance portfolio via `numpy.linalg.pinv`, long-only
  approximated by clipping negative weights and renormalizing rather than a
  full QP, so no `scipy` dependency), then a per-position cap, then a
  sector-exposure cap (`config/sectors.yaml` — hand-curated GICS sectors,
  same rationale as `universe.yaml` being hand-curated: deterministic, no
  flaky data dependency). Capping never redistributes freed-up capital to
  other positions — deliberately left as cash, so a fully-capped
  equal-weight portfolio can end up under-invested.
- **`decision/risk.py`**: per-position stop-loss and a portfolio-level
  max-drawdown circuit breaker, both evaluated on realized data only.
- **`backtest/engine.py`**: walk-forward loop over rebalance dates. Tracks
  the portfolio as **weights**, not share counts — no fractional-share or
  cash-drag bookkeeping, which isn't the point of a v1 research backtester.
  One consequence: risk checks (circuit breaker, stop-loss) are only
  evaluated at rebalance granularity (weekly by default), not intraday —
  a single week's move can overshoot the circuit-breaker threshold before
  it's caught. The circuit breaker re-enters after a configurable cooldown
  (`risk.circuit_breaker_cooldown_periods`), resetting its drawdown
  reference point to the post-cooldown equity level — see "Known gaps"
  below for why this matters. `run_backtest()` takes optional `start`/`end`
  params that restrict *rebalancing* to a sub-range without slicing
  `prices` — signals still see full lookback history from before `start`;
  this is what `backtest/walkforward.py` uses per-window.
- **`backtest/metrics.py`**: CAGR, Sharpe, Sortino, Calmar, downside
  deviation, win rate, max drawdown, multi-benchmark alpha.
- **`backtest/report.py`**: matplotlib tear sheet (equity curve vs
  benchmarks, drawdown, rolling Sharpe, turnover) written to
  `output/tearsheet.png` on every `backtest` run.
- **`backtest/walkforward.py`**: rolling train/test window validation —
  optionally grid-searches `strategy.yaml` overrides (dotted-path -> list
  of candidate values) on each train window by Sharpe, always evaluates
  out-of-sample on the following test window. This is what actually
  answers "does this hold up out-of-sample or is it curve-fit to one
  period" — a single full-period backtest can't tell you that. See
  `alpha-engine walkforward --help`.

## Known gaps (deliberate v1 scope, not oversights)

- **Fundamentals point-in-time depth is limited by yfinance**, not by
  design — see `data/fundamentals.py`. A multi-year backtest with
  `--fundamentals` will have thin-to-empty value/quality coverage for
  older dates; this degrades gracefully (neutral score) rather than
  failing, but don't expect deep historical coverage.
- **Circuit-breaker re-entry timing is a fixed cooldown**, not a
  market-conditions-aware re-entry rule (e.g. waiting for a broader
  recovery signal) — chosen for simplicity. `risk.circuit_breaker_cooldown_periods`
  in `config/strategy.yaml` controls it.
- **Universe is a hand-curated static list**, not live S&P 500 membership —
  deliberate, to avoid survivorship bias silently creeping into backtests
  from following an index's *current* constituents into the past. Same
  rationale applies to `config/sectors.yaml`.
- **Min-variance sizing's long-only constraint is an approximation**
  (clip negative weights + renormalize), not the exact constrained
  optimum a full QP solver would give — a deliberate trade-off to avoid
  a `scipy` dependency.

## Config

- **`config/universe.yaml`**: explicit ticker lists (large-caps, 11 SPDR
  sector ETFs, benchmarks). Edit directly to change the tradable universe.
- **`config/sectors.yaml`**: symbol -> GICS sector for every symbol in
  `universe.yaml` (enforced by `tests/test_config.py`) — add an entry
  whenever a new ticker is added to the universe.
- **`config/strategy.yaml`**: signal weights, rebalance cadence, portfolio
  construction params, risk limits, cost model, backtest params. This is
  the primary lever for changing strategy behavior without touching code.

## Running a backtest

```
.venv/bin/python3 -m alpha_engine.cli backtest --start 2019-01-01 --end 2024-12-31
```

`--symbols` overrides the configured universe (useful for a fast sanity
check on just the sector ETFs); `--config` points at an alternate
`strategy.yaml`; `--fundamentals` fetches fundamentals and enables the
value/quality signals (slower, thin coverage on older dates — see "Known
gaps"). Prints a strategy summary plus a per-benchmark (SPY/VOO/VTI)
comparison table, and writes `output/equity_curve.csv` +
`output/tearsheet.png`.

## Running walk-forward validation

```
.venv/bin/python3 -m alpha_engine.cli walkforward --start 2019-01-01 --end 2024-12-31 \
    --train-years 3 --test-years 1 --step-years 1
```

Add `--param-grid path/to/grid.yaml` (dotted-config-path -> list of
candidate values, e.g. `portfolio.top_n: [10, 15, 20]`) to grid-search on
each train window instead of just checking out-of-sample consistency with
the configured `strategy.yaml`. Writes `output/walkforward.csv`.

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
real API. Where hand-predicting an outcome on noisy synthetic data would be
fragile (e.g. walk-forward grid search), tests instead recompute the
expected result independently using the same building blocks and assert
the two agree — see `test_walkforward.py`'s
`test_run_walk_forward_grid_search_picks_higher_train_sharpe_config` for
the pattern.

## AI news signal — setup for live use

Requires `ANTHROPIC_API_KEY` (see `.env.example`). Not wired into any CLI
command yet — it's a live-trading-phase feature, added once paper trading
starts. When integrating it, pass a `news_signal` DataFrame into
`run_backtest()`/a live runner (indexed by date, one column per symbol) —
never call `signals/news_ai.compute()` from inside a historical backtest
loop.
