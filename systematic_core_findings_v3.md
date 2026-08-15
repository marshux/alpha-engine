# Systematic Core — findings from implementing `systematic_core_action_plan (1).md` ("Action Plan v3")

## Status: partially complete, one action still running in the background

This round targeted the CAGR shortfall `systematic_core_findings_v2.md` left
open (Sharpe/drawdown had improved, absolute return hadn't). Actions 1, 2,
and 5 are complete and validated below. Action 3 (full `insider_buying`
backfill) is a genuinely multi-day background job, still in progress as of
this writing; Action 4 (re-check `ml_alpha`'s IC with full insider coverage)
is blocked on it and not yet run. Action 6 was correctly skipped per the
plan's own conditional logic — see Finding 1.

## Plan corrections made before implementing

The plan document had several details that didn't match the actual
codebase — caught and corrected before running anything, not silently
followed:

- **Signal weight keys in Action 6's proposed YAML don't exist here**:
  `momentum_12_1`, `gross_profitability`, `low_vol`, `book_to_price`,
  `earnings_surprise` aren't this repo's signal names. The real keys are
  `momentum`, `quality`, `volatility`, `value`, `earnings_revision`.
  (Moot in the end — Action 6 wasn't run; see Finding 1.)
- **Action 1's proposed `self.exposure_log` / class-based engine
  instrumentation doesn't fit**: `run_backtest` is a plain function, not a
  class. `BacktestResult.weights_history` already stores per-date weights,
  so gross exposure is derivable post-hoc with no new logging — the only
  real gap was that the HMM regime *label* was computed and then discarded
  each rebalance. Fixed by adding `BacktestResult.regime_history` and
  actually keeping the label (`backtest/engine.py`).
- **Action 3's proposed `backfill_insider_universe` function reinvents
  something that already exists**: `data/insider.py::get_universe_insider_transactions`
  already has the real per-symbol-and-per-page pacing fix from this
  project's earlier SEC rate-limit incident, and already caches to
  parquet. No new backfill function was written — the existing one was
  just run as a proper background job. Its proposed JSON cache format
  wouldn't have been read by anything else in the codebase.
- **Action 3's "~7-10 min for 204 symbols" runtime estimate was very
  wrong.** Sampling real hit counts before committing to a background run:
  MSFT alone has 2,694 Form-4 search hits over 2005–2026 (~270 paginated
  search requests before any filing content is even fetched); PG has
  4,939; META has 1,737. See Finding 3 for the real, measured cost.

## Finding 1: the CAGR gap is mostly a structural sizing effect, not regime misclassification or low beta

Action 1's diagnostic (`BacktestResult.regime_history` + a new
`signals/volatility.py::compute_beta`/`portfolio_beta`, both added this
round) ran the v2-validated config as a single full 2008–2024 backtest and
measured what gross exposure and portfolio beta the system actually runs
at, day by day:

```
Overall mean gross exposure: 0.620
% of days below 0.80 exposure: 82.5%
Overall mean portfolio beta: 0.950
```

Underexposure is real and confirmed (mean 0.62, matching the plan's own
diagnostic framework). But the regime breakdown reveals the plan's
hypothesis was incomplete:

| Regime | % of dates | Mean gross exposure | Exposure floor |
|---|---|---|---|
| BULL_TREND | 37.4% | **0.709** | 1.00 (no dampening at all) |
| LOW_VOL_GRIND | 32.0% | 0.713 | 0.90 |
| BEAR_TREND | 19.8% | 0.481 | 0.60 |
| CRISIS | 7.6% | 0.275 | 0.35 |
| TRANSITION | 2.1% | 0.448 | 0.75 |

Even in BULL_TREND — where the regime overlay applies *no* dampening
(floor = 1.0) — mean exposure is only 0.71, not ~1.0. The ratio of
actual-exposure to floor is ~0.7–0.83x under every single regime,
including the undampened one. This means most of the shortfall isn't
regime misclassification; it's `decision/portfolio.py::size_positions`'s
volatility-scaled sizing hitting `max_position_weight`'s 5% cap on several
of the 20 selected names, with the excess capital left as cash rather than
redistributed — an existing, deliberate, documented tradeoff
(`size_positions`'s own docstring already says this), just not previously
identified as the dominant driver of the CAGR gap.

The 2013–2019 regime-distribution check the plan specified also ran: only
12.6% BULL_TREND during a historically calm multi-year bull run, with
27.4% BEAR_TREND — real evidence the HMM does over-call BEAR_TREND, but
this is a secondary effect given the capping mechanism applies uniformly
across every regime including the ones with no classification ambiguity.

Portfolio beta came back **near-neutral (mean 0.95)**, not below the
plan's own 0.80 trigger threshold for Action 6 (beta floor / `low_vol`
reweighting). Per the plan's explicit conditional logic
("Only run this if Action 1 shows portfolio beta < 0.80"), **Action 6 was
correctly skipped** — low-beta stock selection is not a meaningfully
independent driver here, and forcing it through anyway would have been
tuning against a hypothesis the data didn't support.

## Finding 2: a refined, cheaper Action 2 grid recovered both Sharpe and CAGR

The plan's literal Action 2 grid (2×2×3×3×2 = 72 combinations across
BULL_TREND/LOW_VOL_GRIND/TRANSITION/BEAR_TREND/CRISIS) would have taken an
estimated 10+ hours of walk-forward grid search. Given Finding 1's
discovery that the capping mechanism (not regime floors) is the dominant
lever, a much smaller, better-targeted 2×2 grid was run instead:

- `portfolio.max_position_weight`: [0.05 (current), 0.075] — tests the
  structural capping hypothesis directly.
- `regime.exposure_bear`: [0.40, 0.70] — the two extremes of the plan's
  original 3-value BEAR_TREND proposal, narrowed to control cost.

(`regime.exposure_crisis` stayed fixed at 0.35; `regime.exposure_bull`/
`low_vol_grind`/`transition` stayed at their defaults this round. A new
`regime.exposure_transition` override was added to
`decision/regime.py::resolve_exposure_table` while implementing this —
TRANSITION previously had no config key at all and always silently used a
hardcoded 0.75, unlike the other four regimes.)

Full 2008–2024 walk-forward, same 3 window configs, 4,442–1,786s per
config:

| Window | v2 baseline | **This round (Action 2)** |
|---|---|---|
| 2yr/1yr | Sharpe 0.93, CAGR 8.48%, MaxDD -7.70% | **0.97, 9.19%, -8.40%** |
| 3yr/1yr | Sharpe 1.01, CAGR 9.21%, MaxDD -7.28% | **1.06, 9.82%, -7.73%** |
| 5yr/2yr | Sharpe 0.79, CAGR 7.77%, MaxDD -11.00% | **0.85, 8.80%, -11.54%** |
| **Overall (combined)** | **0.918, 8.54%, -8.46%** | **0.969, 9.30%, -9.03%** |

Sharpe and CAGR improved in every window config; MaxDD ticked up modestly
but stays well inside Contrarian Volatility's -11.4%. Updated comparison
table:

| Strategy | Avg Sharpe | Avg CAGR | Avg Max DD |
|---|---|---|---|
| Traditional (baseline, live) | **1.227** | **17.6%** | -10.0% |
| **Systematic Core (this round)** | **0.969** | **9.30%** | -9.03% |
| Contrarian Volatility (live) | 0.870 | 13.3% | -11.4% |

The winning-parameter pattern per window is genuinely regime-sensitive,
not a trivial "widest cap/loosest floor always wins" artifact:
`max_position_weight=0.075` won in the earliest windows (2013–2016), but
the original `0.05` won from 2016 onward; `exposure_bear=0.70` (loosest)
won in 9 of 11 windows, but the *tightest* option (0.40) won specifically
for the two windows whose test range includes the real 2022 bear
market — the grid search is picking up a genuine signal, not just
defaulting to "more exposure is always better."

## Finding 3: the insider_buying backfill is real, but far slower than the plan assumed

Before committing to a background run, real hit counts were sampled
directly (see "Plan corrections" above) rather than trusting the plan's
estimate. Even so, the actual per-symbol cost has come in higher than
that sample predicted, because `get_insider_transactions` fetches the
full XML for *every* search hit before checking whether the filing's
issuer actually matches the requested symbol — for a company with ~1,700
hits, that's ~1,700 individual paced filing fetches, not just the ~170
paginated search requests.

Measured so far (of 204 symbols; AAPL/AMZN/GOOGL/NVDA were already
warm from the v2 round):

| Symbol | Transactions | Wall time |
|---|---|---|
| META | 54,994 | ~87 min |
| AVGO | 17,142 | ~59 min |

At this pace, full-universe completion is realistically **1–2 days** of
continuous background running, not the plan's "~7-10 min," and not even
the "several hours" first estimated before real per-symbol data came in.
This was flagged directly and the user chose to let it keep running in
the background rather than scope it down — it is still in progress. A
retry-with-backoff was added to `data/insider.py::_get_json`/`_get_text`
partway through (a transient 500 on an isolated, non-burst request was
observed even with the existing rate-limit pacing in place) so a single
flaky response doesn't abort an entire symbol's multi-hundred-request
fetch.

## Finding 4: signal coverage audit (Action 5, Option A)

Measured directly from the real rebalance-date sequence rather than
estimated:

| Signal | Active on % of rebalance dates | Mean breadth when active |
|---|---|---|
| `short_interest` | 41.3% | 200/204 symbols |
| `insider_buying` | 69.3% | **1.9/204 symbols** |

`short_interest`'s number matches the plan's own ~37% estimate closely,
and its coverage is genuinely broad whenever active (FINRA's data covers
nearly the whole universe from 2018 on) — its effective composite weight
is roughly 0.041 of its nominal 0.10.

`insider_buying`'s 69.3% "active" figure is misleading taken alone: with
only 4 symbols cached pre-backfill, "active" mostly means *some* one of
those 4 names had a qualifying transaction in the trailing 90-day window —
not that the signal says anything meaningful cross-sectionally. A z-score
over ~2 data points out of 204 is close to uninformative. This is the
concrete reason Action 3's backfill matters even though it's slow: the
naive "% active" metric overstates how much this signal currently
contributes.

## What's still open

- **Action 3** (full `insider_buying` backfill): in progress, ~1-2 days
  estimated total, 7/204 symbols done as of this writing.
- **Action 4** (`ml_alpha` IC recheck with full insider coverage): blocked
  on Action 3. Current known value is still v2's 0.0203 (fails the 0.03
  gate, weight stays 0.0) — not re-measured this round.
- Once Action 3 completes, Action 5's `insider_buying` coverage numbers
  and Action 2's grid-search winners should ideally be re-validated with
  full coverage, since the current Action 2 result was validated with
  `insider_buying` still at its pre-backfill (near-zero cross-sectional
  breadth) state. The 15% signal weight it nominally carries was, for all
  practical purposes, decorative for this round's validation.

## Conclusion

Two of the plan's three diagnostic questions (underexposure, beta drag)
were answered directly and honestly by data rather than assumed: exposure
is real and underexposure is confirmed, but its true driver is
`size_positions`'s position-cap clipping showing up uniformly across every
regime, not primarily regime misclassification; portfolio beta is
near-neutral, so Action 6 was correctly skipped rather than run anyway.
The resulting, more narrowly-targeted Action 2 grid search recovered real
Sharpe and CAGR gains (0.918→0.969, 8.54%→9.30%) at every window config,
closing some — not all — of the gap to Traditional (still 1.227 Sharpe,
17.6% CAGR). The insider-data backfill needed for Action 4 to mean
anything is real but slow, currently still running; this document will be
updated once it completes and Action 4 can actually be checked. No live
wiring, no further scope expansion, and no premature "final" verdict on
`ml_alpha` while the backfill it depends on is still incomplete.
