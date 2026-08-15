# Systematic Core postmortem: why it underperforms, and how faithfully it follows the guide

## Update — after implementing `systematic_core_improvements.md`

All six code fixes plus the guide corrections from `systematic_core_improvements.md`
were implemented and validated (see that file's own "Quick Reference" table
for the list; briefly: ratio-based `top_n`, an IC validation gate, the
`BAA`/`AAA` credit-spread fix, HMM degeneracy detection with a rule-based
fallback, tunable/walk-forward-grid-searched exposure floors, and an opt-in
REDUCE risk tier). Two things worth stating plainly:

1. **The IC gate found the `ml_alpha` signal genuinely fails validation.**
   Measured mean IC on a held-out 2021–2022 period: **-0.0120** (needs ≥0.03).
   Per the gate's own rule, its weight was set to 0.0 for the revalidation
   below — it was not a defensible signal to keep weighting.
2. **The net result of applying every fix is a worse Sharpe than before the
   fixes, not a better one.** This is the honest, load-bearing finding of
   this update:

| Version | Avg Sharpe | Avg CAGR | Avg Max DD |
|---|---|---|---|
| Traditional (baseline, live) | **1.227** | **17.6%** | -10.0% |
| Systematic Core — pre-fix (bugs present, `ml_alpha` weighted 0.35, `top_n=30`) | 1.130 | 9.8% | -9.2% |
| Systematic Core — all fixes applied (`ml_alpha`=0, ratio-based `top_n`≈6, tuned exposure floors, REDUCE tier) | **0.730** | 10.5% | -11.4% |
| Contrarian Volatility (live) | 0.870 | 13.3% | -11.4% |

Each individual fix was independently well-justified (two were real bugs;
the IC gate is the guide's own stated requirement; the data-source fix
closed a real, confirmed licensing gap). But the ratio-based `top_n` fix
— framed in this same document as the single highest-priority,
highest-confidence improvement — concentrated the portfolio from 30
positions down to roughly 6 (10% of the 63-symbol universe), and that
concentration cost more in idiosyncratic risk than it recovered in
selectivity. The walk-forward exposure-floor grid search is a second,
smaller data point in the same direction: given the now-more-concentrated,
`ml_alpha`-free portfolio, it consistently selected the *loosest* available
`BEAR_TREND` floor (0.60 of 3 candidates, in 9 of 11 windows) rather than a
tighter one — the train-Sharpe-optimal choice was to stay more invested
through weakness, not less, which cuts against the postmortem's original
"tighten the floors" framing.

Net: the fixes were correct engineering (real bugs are now fixed, a real
signal-validation gate now exists and is being honored, a real data
restriction is closed), but the improvement plan's central performance
hypothesis — that over-diversification and an untrusted ML signal were the
main drag — was not confirmed by re-running the numbers. Systematic Core
remains behind Traditional, and the specific combination of fixes applied
here trails even the pre-fix version on risk-adjusted return. This is
reported as-is rather than re-tuned further without checking in again,
since further changes at this point are strategy-design choices, not bug
fixes.

## Bottom line (original findings, before the update above)

Systematic Core (HMM regime classifier + LightGBM `ml_alpha` signal, built
against `systematic_trading_engineering_guide.md` per the user's explicit
"full literal implementation" choice) does not beat the existing
Traditional strategy. Averaged across the same three walk-forward window
configs used for every strategy this session (2008–2024, full universe):

| Strategy | Avg Sharpe | Avg CAGR | Avg Max DD |
|---|---|---|---|
| Traditional (baseline, live) | **1.227** | **17.6%** | -10.0% |
| Systematic Core | 1.130 | 9.8% | -9.2% |
| Contrarian Volatility (live) | 0.870 | 13.3% | -11.4% |

Per window config, Systematic Core's Sharpe never beats Traditional's,
ties it once, and falls clearly behind in the longest (5yr-train/2yr-test)
window — the one that fully contains the 2022 bear market:

| Window config | Systematic Core Sharpe | Traditional Sharpe |
|---|---|---|
| 2yr train / 1yr test | 1.146 | 1.194 |
| 3yr train / 1yr test | 1.238 | 1.238 (tied) |
| 5yr train / 2yr test | 0.971 | 1.258 |

This is the result **after** two real implementation bugs were found and
fixed (below) — earlier, buggy runs of the same strategy looked
artificially competitive with or even superior to Traditional, purely
because the regime classifier wasn't actually doing anything. The honest
number, with the classifier genuinely working, is a real but modest
underperformance, paired with a real but modest drawdown improvement.

This document explains why, and how faithfully the implementation
actually follows the guide it was built against.

---

## Two real bugs found during validation (engineering record)

Both were caught by running real walk-forward validation against real
market data, not by code review — exactly the kind of thing the guide's
own Part III ("Effective Backtesting is the most important safeguard in
the entire system") argues for.

### Bug 1 — unsafe default direction in regime interpretation

`decision/regime.py::RegimeClassifier._interpret_regimes` originally
defaulted any state it couldn't confidently classify to `"BULL_TREND"`
(1.0 gross exposure) — the single most aggressive possible outcome. On
real 2022 data, most states landed in that ambiguous branch, so the
"downside protection" mechanism was silently investing at full exposure
through most of a bear market. Fixed to default to a conservative
`"TRANSITION"` label (0.75 exposure) instead, and simplified the
per-state classification logic itself from the guide's illustrative
two-dimensional AND-thresholds (which rarely both held for a given
state's mean, given only 4 states summarizing years of data) to an
exhaustive return-sign-based split, so every state now gets a real label
instead of falling through to a default at all.

### Bug 2 — a single missing macro column silently zeroed the entire feature panel

`decision/regime.py::build_regime_features` used one blanket `.dropna()`
across all 8 feature columns. `BAMLH0A0HYM2` (the ICE BofA high-yield
credit spread, used for `credit_spread`) turned out to only have ~3 years
of history available through FRED's free/anonymous CSV access — the
other three macro series (`DGS10`, `DGS2`, `VIXCLS`) are unrestricted,
this one series specifically is not (confirmed directly against FRED's
endpoint, not a caching or fetch-code bug). Because of the blanket
`dropna()`, this single column's multi-year gap discarded the *entire*
feature panel for every date in that gap — meaning the HMM classifier
had **never actually fit on real data** for most of the validation
history; it was silently hitting the "insufficient history" fallback the
entire time, before Bug 1 was even found. Fixed to `fillna(0.0)` the
macro-derived columns specifically (matching the same fallback
`ml/features.py::build_ml_features` already used), while still dropping
rows only for genuine transient warmup gaps in the price-derived columns.

**This second bug is the more consequential one.** Every validation run
before it was fixed was measuring a strategy that never actually used its
own headline feature (regime timing) — the "looks competitive with
Traditional" numbers from those runs were not evidence the design works;
they were an artifact of a data pipeline that was quietly broken.

---

## Fidelity to the guide, section by section

| Guide section | What it describes | What was built | Fidelity |
|---|---|---|---|
| 2. Securities Universe | S&P 500 tier (~500 names), point-in-time index membership, delisted-stock inclusion | This repo's existing 63-symbol hand-curated universe (52 large caps + 11 sector ETFs), no delisting/survivorship data | **Low** — a deliberate, pre-existing repo choice (CLAUDE.md's documented rationale: avoids survivorship bias from *current* index membership drifting into historical backtests), not something this feature changed. But it means "n_long=50" from the guide's portfolio example (§3.7) isn't really achievable from a 63-symbol universe; `top_n=30` here is holding roughly half the entire universe, not the top 6% the guide's ratio implies. |
| 3.2–3.3 Technical & Fundamental signals | 12-1 momentum, low-vol, mean reversion, earnings yield, earnings revision/surprise, gross profitability, book-to-price, insider buying, short interest | Reused this repo's existing `momentum`/`mean_reversion`/`volatility`/`value`/`quality` signals — momentum and quality/value map closely to the guide's versions; earnings revision, earnings surprise, insider buying, and short interest were **not implemented at all** — no free data source for analyst estimates or SEC Form 4 filings in this repo | **Medium** — the signals that exist are faithful; roughly half the guide's Part I signal roster requires data this project doesn't have access to. |
| 3.4 Sentiment/Options flow | Put/call ratio, options skew | Not implemented | **None** — explicitly out of scope, no options data source. |
| 3.5 ML-based signals | Two-stage pipeline: rich feature set (10 base signals + interactions + macro/temporal context) → LightGBM → rank-transformed, IC-validated alpha score | Built the same architecture (`alpha_engine/ml/`), but with only **5** base signals (not 10) since the other 5 need unavailable data. **IC was never measured** — the guide's own backtesting checklist (§5.7) requires "each signal has positive IC on out-of-sample data (IC > 0.03)" before trusting it; this was never checked here. | **Medium-low** — the pipeline shape (feature builder → rank-transform → LightGBM → z-score) is faithful; the feature richness and the validation rigor around it are not. |
| 3.6 Regime detection | 4-state HMM on 8 macro/market features, regime-adaptive signal weights, continuous gross-exposure overlay | Built essentially as specified (`decision/regime.py`) — but discovered via two real bugs (above) that HMM fits on real data are genuinely fragile at this scale (small feature space, only a few thousand rows, a `GaussianHMM` with 4 components) — several fit attempts return degenerate `startprob_`/`transmat_` and have to fall back to a neutral default, a failure mode the guide's illustrative code doesn't address at all. | **Medium-high** — the architecture is faithful and, once the two bugs were fixed, genuinely functions (verified: correctly reads BULL_TREND through 2021, CRISIS at the Feb-Mar 2022 selloff, sustained BEAR_TREND through the rest of 2022). But it is measurably less protective than Traditional's binary trend filter during a *sustained* bear market — see root cause analysis below. |
| 3.7 Portfolio construction | Alpha-weighted, vol-targeted, capped by position/sector | Reused this repo's existing `build_portfolio` (rank → top-N → size → caps) with `sizing=volatility_scaled`, `top_n=30`, `max_position_weight=0.06`, `max_sector_weight=0.25` | **High** — faithful to the mechanism; the specific *parameters* (30 of 63 symbols, wide diversification) were set by analogy to the guide's ratios rather than independently tuned/validated for this universe size. |
| 3.8 Risk management | Position/sector/portfolio-level limits, drawdown-based REDUCE/HALT tiers | This repo's existing binary stop-loss + circuit-breaker (HALT at -15%, no intermediate REDUCE tier) | **Medium** — the guide's `RiskManager.check_portfolio_drawdown` has three tiers (NORMAL/REDUCE/HALT); this repo only ever had two (NORMAL/HALT), and that wasn't changed for Systematic Core. The circuit breaker's cooldown-then-reset design (pre-existing, shared with Traditional) also means cumulative drawdown across repeated reset cycles can exceed the configured single-cycle threshold — observed directly in the diagnosed window. |
| 4.4 Adaptive signal weighting | Online learning (Exponential Gradient) adjusting weights by rolling IC | **Not implemented.** Signal weights are static, hand-set config values (`SYSTEMATIC_CORE_OVERRIDES`), same as every other strategy in this repo. | **None** — a real, acknowledged gap. The regime-adaptive weight *table* is closer in spirit (§3.6's `REGIME_WEIGHTS`) but that's discrete/regime-keyed, not the continuous online IC-based reweighting §4.4 describes. |
| 5.4 Realistic cost model | sqrt-participation market impact model, tiered by liquidity | This repo's existing flat commission+slippage bps model, unchanged | **Low** — not upgraded for this strategy specifically; cost realism is identical across all three strategies, so it isn't the source of Systematic Core's relative underperformance, but it also means none of these validated Sharpe numbers account for the impact-cost realism the guide argues is "the #1 reason backtests look better than live performance." |
| 5.5 Statistical significance | t-stat > 2-3, Deflated Sharpe Ratio > 0.95, permutation test p < 0.05 | **Not implemented.** This session's validation reused the existing walk-forward methodology (mean OOS Sharpe/CAGR/MaxDD across 3 window configs) — the same bar Traditional and Contrarian Volatility were held to, but not the guide's more rigorous statistical machinery. | **None** — an explicit scope decision, not an oversight (the "full literal implementation" choice was scoped to the ML/HMM machinery specifically; the statistical-validation suite was part of the *other*, not-chosen option). This means the Sharpe gap reported above (1.13 vs 1.23) has never been tested for statistical significance — it may well be within noise. |
| 5.6 Regime robustness | Per-historical-regime breakdown (dotcom, GFC, COVID, 2022, etc.) | Not implemented as a separate report; the 5yr/2yr window's poor showing is *de facto* regime-robustness evidence (it's the window containing 2022), but no systematic regime-by-regime table was built. | **None** — informally covered, not formally implemented. |

---

## Root cause: why Systematic Core underperforms Traditional

Several factors compound, roughly in order of estimated impact:

1. **Diversification tax.** 30 positions vs. Traditional's 10, both drawn
   from the same 63-symbol universe. Even if the ranking signal is
   genuinely informative, spreading capital across positions ranked
   11–30 (which Traditional would never hold at all) dilutes whatever
   edge exists. This was a deliberate design choice (matching the
   guide's broad-diversification philosophy), but on a 63-symbol
   universe it means holding essentially half the available names —
   much less selective than the guide's own ~10%-of-universe ratio
   (50 of ~500).

2. **Unvalidated, feature-poor ML signal.** `ml_alpha` is trained on 5
   base factors (not the guide's 10) with no IC check ever performed. It
   is entirely possible — and was never ruled out — that this signal
   carries little genuine predictive power on this universe/data, in
   which case its 0.35 signal weight is partly just adding noise to the
   ranking rather than alpha.

3. **The continuous exposure floor is less protective than a binary
   flatten during a *sustained* bear market.** This is the clearest,
   most directly observed cause. In the diagnosed 2020-12-30..2022-12-30
   window, once the classifier was genuinely working, it correctly held
   `BEAR_TREND` for essentially all of April–December 2022 — but
   `BEAR_TREND`'s exposure floor is 60%, not 0%. Staying 60% invested
   through a ~9-month bear market still accumulates real losses that
   Traditional's all-cash binary trend filter avoids entirely. This is
   an inherent tradeoff of the design choice (continuous exposure was
   chosen specifically to avoid the binary filter's whipsaw/opportunity
   cost during ambiguous periods), not a bug — but it is a real, material
   cost during genuinely sustained downturns, and the `GROSS_EXPOSURE_BY_REGIME`
   table's specific values (0.35/0.60/0.90/1.0) were set by analogy to
   the guide's illustrative numbers, never independently tuned or
   validated for this strategy's actual universe and cost structure.

4. **No adaptive reweighting (guide §4.4).** Traditional's weights were
   walk-forward-selected for `top_n`/`sizing` specifically (documented in
   `config/strategy.yaml`'s own comments: "walk-forward picked equal_weight
   in 8/14 windows"). Systematic Core's weights were set once, by
   analogy to the guide's illustrative defaults, and never similarly
   validated/selected against this repository's own walk-forward
   harness before being used in the sweep.

5. **No statistical significance testing.** Without a deflated Sharpe
   ratio or permutation test (guide §5.5), it isn't actually established
   that the ~0.10 average Sharpe gap between Systematic Core and
   Traditional (1.13 vs 1.23) reflects a genuine difference in strategy
   quality rather than noise across a modest number of walk-forward
   windows (40 total across all three configs).

---

## Honest fidelity summary

The **mechanisms** described in the guide's more advanced sections (3.5
ML signals, 3.6 regime detection) were built close to as specified, and
after two real bug fixes, both genuinely function on real data. Where
this implementation is least faithful to the guide is:

- **Feature richness** (Part I signals) — roughly half the guide's
  signal roster needs data sources (analyst estimates, insider filings,
  short interest, options flow) this project doesn't have, a limitation
  inherited from the whole repo, not introduced by this feature.
- **Validation rigor** (Part III) — the guide's own backtesting checklist
  (§5.7: IC thresholds, deflated Sharpe, permutation tests, cost
  sensitivity analysis) was not applied to this strategy's signals or
  results, so several of the guide's own stated preconditions for
  "declaring any backtest result valid" were skipped.
- **Adaptive weighting** (§4.4) — not implemented; weights are static.
- **Tiered risk response** (§3.8's REDUCE tier, §5.4's realistic cost
  model) — this repo's simpler, pre-existing binary risk/flat-cost
  machinery was reused rather than upgraded to the guide's fuller design.

None of these gaps are hidden or silent — each was a scope decision made
explicit at the time (either by the user's own choice among options
presented, or flagged in code comments/docstrings as a deliberate
simplification) — but taken together, they explain both *why* this
implementation underperforms Traditional and *where* the highest-leverage
next steps would be if that gap is ever worth closing: validating the
`ml_alpha` signal's actual IC before trusting its weight, and tuning
`GROSS_EXPOSURE_BY_REGIME`'s floors against the walk-forward harness
rather than by analogy to the guide's illustrative numbers.
