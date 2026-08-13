# alpha-engine

A systematic swing-trading research engine: quant signals plus an AI news
signal feed a deterministic decision engine, which is backtested against
index benchmarks (SPY/VOO/VTI). Goal is to beat cap-weighted index funds
over days-to-weeks holding periods. Backtest-first — no live/paper order
execution yet.

## Core design constraint

**AI is allowed to research and score; it is never allowed to decide.**
`signals/news_ai.py` calls Claude with the `web_search` tool (and,
optionally, X search) to research current AI-sector sentiment and returns a
*structured* score (forced via `output_config.format` — not free text).
That score never enters the per-stock signal table at all — it feeds one
plain deterministic formula, `decision/portfolio.py::modulate_theme_cap`,
which scales the AI-theme concentration cap up or down. Everything
downstream of signal generation (`decision/scorer.py`,
`decision/portfolio.py`, `decision/risk.py`) is a plain weighted-sum/rules
pipeline with no LLM call anywhere in it. This is what makes backtests
trustworthy: given the same signal inputs, the decision layer always
produces the same trades. Preserve this separation in any future work —
don't let an LLM call creep into the decision path.

## Architecture

```
data/equities.py       →  signals/*.py           →  decision/scorer.py  →  decision/portfolio.py  →  backtest/engine.py  →  backtest/report.py
data/fundamentals.py      (momentum,                 (weighted sum          (top-N, sizing,           (walk-forward loop,     backtest/walkforward.py
  (yfinance,                mean_reversion,            of z-scored            caps -- incl. AI theme    turnover costs,
   parquet cache)           volatility, value,         signals)               cap, live-only modulated  circuit breaker)
                            quality, ai_theme)                                by signals/news_ai.py)
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
  explicitly rather than through the uniform `SIGNAL_MODULES` dict.
  `signals/news_ai.py` diverges further still: it isn't a per-symbol signal
  at all (see its own entry below).
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
- **`signals/news_ai.py`**: **live-only, never called by the backtester,
  and deliberately macro rather than per-symbol.** `web_search` (and,
  optionally, X search — see below) queries the *current* web — using
  either inside a historical backtest would be lookahead bias (a 2021
  rebalance would see 2026 search results). `backtest/engine.py` has no
  import of this module. An earlier per-ticker version of this signal made
  one research call per symbol (63 calls/rebalance at full universe) at
  real, meaningful cost (~$27/rebalance, ~$1,400/year at weekly cadence)
  for a component that only ever fed a 0.1-weighted per-stock signal;
  `compute_ai_sector_sentiment()` instead makes a single call researching
  AI-sector sentiment as a whole, ~60x cheaper, and its output
  (`{sentiment, confidence, ...}`) is consumed by
  `decision/portfolio.py::modulate_theme_cap` as a portfolio-level risk
  input, not joined into the per-stock signal table at all. Results are
  cached by date under `data/cache/news_ai/`. X/Twitter search via X's own
  MCP server (`api.x.com/mcp`) is opt-in, gated on
  `X_MCP_AUTHORIZATION_TOKEN` being set — it's billed separately and
  per-action by X itself, not bundled into the Anthropic API bill, so it
  must not turn on just because `ANTHROPIC_API_KEY` is configured. Without
  the token, this signal is `web_search`-only. See `.env.example`. (As of
  this writing, Anthropic's MCP connector fails to reach X's MCP server
  server-side — `Error while communicating with MCP server` — even though
  the token and URL are independently verified correct via a raw MCP
  `initialize` call; left wired up and dormant pending a fix on Anthropic's
  side, no code changes needed once it resolves.)
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

## Paper trading (live)

`alpha-engine live-rebalance [--dry-run]` runs one real rebalance against
an Alpaca **paper** account — the only place `signals/news_ai.py`'s live
AI-sector-sentiment research is ever actually called (see that module's
docstring for why it must never run inside `backtest/engine.py`). Requires
the `live` extra (`pip install -e ".[live]"`) and `ALPACA_API_KEY`/
`ALPACA_SECRET_KEY` (paper-trading keys — see `.env.example`).

- **`broker/alpaca.py`**: thin `alpaca-py` wrapper. `paper=True` is
  hardcoded in `get_client()`, not a runtime toggle — going live with real
  money is meant to be a deliberate code change later, not a config flip.
  `compute_orders()` diffs target weights against currently-held Alpaca
  positions (unioned, so a position dropped from `target_weights` gets an
  implicit sell) and returns notional (dollar-denominated) market orders,
  sidestepping fractional-share rounding. `submit_orders()` submits
  sells before buys so a same-batch sell's freed cash is available before
  the corresponding buy is attempted.
- **`live/state.py`**: the circuit breaker's `breaker_peak`/
  `cooldown_remaining` and the last rebalance date, persisted as JSON at
  `data/live_state.json` (not the web dashboard's SQLite DB) — this state
  has no equivalent on Alpaca's side, and living inside `alpha_engine`
  itself is what lets the CLI and a future scheduler job share it without
  either depending on the other.
- **`live/runner.py`**: `run_live_rebalance()` mirrors
  `backtest/engine.py`'s per-date risk-check control flow exactly
  (cooldown → circuit breaker → regime filter → stop-loss) evaluated once
  per call instead of once per historical date. Position entry prices for
  the stop-loss come directly from Alpaca's own `avg_entry_price` rather
  than being tracked separately. Idempotent per calendar day (guards on
  `live/state.py`'s `last_rebalance_date`) so an accidental double-run
  doesn't double-submit orders. `decision/scorer.py`,
  `decision/portfolio.py`, and `decision/risk.py` are reused completely
  unchanged — this is what keeps live trading bound by the same
  deterministic decision layer a backtest already validated.

Known v1 scope limits, not oversights: paper trading only (see above);
weekly cadence only, no intraday execution; order submission is treated
as the audit record, not reconciled against actual fills.
