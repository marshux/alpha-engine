# Systematic Core — findings from implementing `systematic_core_action_plan.md`

## Summary

The revised action plan's central hypothesis was that `systematic_core_findings.md`'s
diversification-tax problem was caused by too small a universe (63 symbols),
not by the ratio-based `top_n` fix itself — and that expanding the universe
while re-calibrating `top_n_pct`/`min`/`max` and adding three new
signals would recover performance. That hypothesis holds up. Universe
expansion (63 → 204 hand-curated large caps) plus re-calibrated position
sizing (20 positions instead of 6) produced a real, consistent Sharpe
improvement across all three walk-forward window configs versus the prior
round. Systematic Core now beats Contrarian Volatility on Sharpe and has
the lowest drawdown of the three strategies, but still trails Traditional
by a wide margin on both Sharpe and CAGR.

## What was implemented

- **Universe**: `config/universe.yaml` expanded from 63 to 204 symbols (52 → 193
  hand-picked large caps, unchanged sector ETFs/benchmarks), with matching
  GICS entries added to `config/sectors.yaml`. Kept static and hand-curated
  (not scraped from current index membership) — expanding this list by
  following today's index constituents into the historical backtest would
  reintroduce the exact survivorship bias `config/universe.yaml`'s existing
  comment block was written to avoid. This was flagged to the user before
  implementing and confirmed as the right call.
- **Fix 1, re-calibrated**: `portfolio.top_n_pct=0.10`, `top_n_min=10`,
  `top_n_max=30` — on 204 symbols this resolves to 20 positions (vs. 6 on
  the old 63-symbol universe), with the original wider position/sector caps
  reverted back to their tighter defaults (`max_position_weight=0.05`,
  `max_sector_weight=0.25`) now that 20 positions doesn't require artificial
  headroom to stay invested.
- **Fix 2, IC-tiered weighting**: `ml_alpha`'s weight is set from a freshly
  measured IC on the expanded universe with the new signal features
  included, not assumed. See Finding 1.
- **Three new signals**, all backed by real (not synthetic/assumed) external
  data:
  - `earnings_revision` (`signals/earnings_revision.py`): a PEAD (post-
    earnings-announcement drift) proxy — cumulative abnormal return
    (stock return minus benchmark return) over the 3 trading days *ending
    at* the earnings report date. This measures the *initial* reaction,
    which is what predicts future drift under PEAD — a window extending
    past the report date would leak the very drift the signal is supposed
    to predict.
  - `short_interest` (`data/short_interest.py`, `signals/short_interest.py`):
    real FINRA consolidated short-interest data (`api.finra.org`'s
    `consolidatedShortInterest` data group), negative z-score of
    days-to-cover. The action plan's proposed URL
    (`cdn.finra.org/.../CNMSshvol...`) turned out to be the wrong dataset —
    Reg SHO daily short *volume*, not bi-weekly short *interest* — caught
    by directly verifying the response schema before trusting it, not by
    assuming the plan's URL was correct. Full 204/204 universe coverage,
    but only back to ~2018 (FINRA's own API history depth), not the full
    2008–2024 backtest window.
  - `insider_buying` (`data/insider.py`, `signals/insider_buying.py`): real
    SEC EDGAR Form 4 open-market purchase/sale data (excludes grants,
    option exercises, gifts — only genuine open-market P/S transactions).
    **Coverage is partial: only 4 of 204 symbols (AAPL, AMZN, GOOGL,
    NVDA)** — see Finding 2 for why, and why a full-universe fetch wasn't
    forced anyway.
- All three signals wired into `signal_weights`, `backtest/engine.py`,
  `backtest/walkforward.py`, and `ml/features.py` (feeding `ml_alpha` too),
  gated the same way `value`/`quality` are gated on data availability —
  missing data degrades to neutral (0), never an error.
- 315 tests passing (226 root + 89 web-backend), no regressions.

## Finding 1: `ml_alpha`'s IC improved but still fails the gate

Re-measured on the expanded universe with the three new signals included as
features, same methodology as before (train 2015–2020, validate on
held-out 2021–2022):

```
mean_ic   =  0.0203   (was -0.0120 on the 63-symbol universe)
ic_std    =  0.2045
ic_ir     =  0.0991
n_periods =  104
passes (>= 0.03) = False
```

This is a real, positive improvement — it went from actively harmful
(negative IC) to weakly informative (positive IC, positive information
ratio) — directly supporting the action plan's premise that a larger,
more diverse universe gives the ML signal more cross-sectional dispersion
to work with. It's still below the 0.03 gate threshold, so per the tiering
rule (IC < 0.03 → weight 0.0), `ml_alpha`'s weight in the final validated
configuration remains 0.0. The gate is being honored as measured, not
adjusted to manufacture a pass.

## Finding 2: SEC EDGAR rate-limiting incident limited insider-data coverage

Pre-warming Form 4 data for a 60-symbol subset (only an inter-*page* pause,
not an inter-*symbol* pause) triggered cascading HTTP 500s from SEC's
full-text search endpoint starting at the 2nd symbol — including on the
very first page of that symbol's query, which ruled out a pagination-depth
explanation and pointed at burst/rate-limit detection instead. Confirmed
by: stopping the job, waiting briefly, and manually re-running the exact
same query moments later — it succeeded. Fixed by adding a pause between
symbols (not just between pages) in `get_universe_insider_transactions`.

Given SEC's fair-access pacing (keep it well under ~10 req/sec, and a
large-cap company can have hundreds of Form 4 filings across its history),
a full 204-symbol fetch is a genuine multi-hour cost. Rather than run that
unattended mid-session, a `get_cached_insider_transactions` helper was
added that reads only already-warmed cache entries and never touches the
network — used throughout the final IC check and validation sweep. The
practical result: **insider_buying only has real coverage for 4 symbols**
(AAPL, AMZN, GOOGL, NVDA — whatever was cached before the rate-limit
incident). For the other 200 symbols it degrades to neutral (0), same as
any other signal's missing-data handling. This is a real, known limitation
of this round's validation, not a bug — a full-coverage re-run would need
a dedicated multi-hour background fetch, not attempted here.

## Finding 3: the combined fixes recovered performance, confirming the diversification-tax diagnosis

Full walk-forward comparison, same 3 window configs (2/1/1, 3/1/1, 5/2/1),
full 2008–2024 history:

| Version | Avg Sharpe | Avg CAGR | Avg Max DD |
|---|---|---|---|
| Traditional (baseline, live) | **1.227** | **17.6%** | -10.0% |
| Systematic Core — 63-symbol universe, all 6 fixes (prior round) | 0.730 | 10.5% | -11.4% |
| **Systematic Core — 204-symbol universe, re-calibrated + 3 new signals** | **0.918** | **8.5%** | **-8.5%** |
| Contrarian Volatility (live) | 0.870 | 13.3% | -11.4% |

Per-window-config, the improvement over the prior round is consistent
across all three, not driven by one favorable window:

| Window config | Prior round (63 symbols) | This round (204 symbols) |
|---|---|---|
| 2yr train / 1yr test | 0.761 | **0.929** |
| 3yr train / 1yr test | 0.775 | **1.006** |
| 5yr train / 2yr test | 0.631 | **0.790** |

This directly supports the diagnosis in `systematic_core_findings.md`:
the ratio-based `top_n` fix itself was correct, but on a 63-symbol universe
it collapsed the portfolio to ~6 positions, and the resulting
concentration risk outweighed the benefit of fixing over-diversification.
On the properly-sized 204-symbol universe, the same ratio resolves to 20
positions — enough headroom that the position/sector caps could return to
their original, tighter defaults without leaving the portfolio
under-invested.

A second data point supports the same read: in the prior round, the
exposure-floor grid search (Fix 5) consistently chose the *loosest*
`BEAR_TREND` floor (0.60 of 0.30/0.45/0.60) in most windows — read at the
time as evidence the strategy needed to "stay invested to compensate" for
its own concentration risk. This round, the winning floor is genuinely
mixed across windows (roughly split between 0.30/0.45/0.60 depending on
the specific train window, no dominant choice), consistent with the
concentration-driven bias being gone now that the universe fix addressed
the root cause rather than a downstream symptom.

## Where it still falls short

Systematic Core (this round) now has the **lowest drawdown of the three
strategies** and **beats Contrarian Volatility on Sharpe** (0.918 vs.
0.870), but its CAGR (8.5%) is well below both Traditional (17.6%) and
Contrarian Volatility (13.3%) — its Sharpe improvement is coming partly
from lower volatility/drawdown, not from generating more return. The
5yr/2yr window (which includes the full 2022 bear market) remains the
weakest of the three configs (Sharpe 0.79), same pattern as every prior
round of this strategy. `ml_alpha` still contributes nothing (Finding 1),
and `insider_buying`'s near-total lack of coverage (Finding 2) means its
15%+10%+10% combined weight across the three new signals is realistically
carried mostly by `earnings_revision` and `short_interest` alone for all
but 4 symbols.

## Conclusion

The action plan's core bet — that the universe, not the `top_n` ratio
logic, was the real bottleneck — is validated by the numbers. Systematic
Core is meaningfully better than it was, and no longer the weakest of the
three strategies on a risk-adjusted basis. It remains behind Traditional
on both Sharpe and CAGR, and its two newest signals have real, documented
data-coverage gaps (short_interest's ~2018+ history, insider_buying's
4-symbol coverage) that a longer, dedicated data-backfill effort could
close but weren't closed here.

As with every strategy validated in this project, this is reported as-is
before any further action — no live wiring, no Key 2/Secret 2 use, no
further unrequested tuning. The next move (accept as a third live
strategy, keep iterating on data coverage first, or hold as research-only)
is a call for the user to make with these numbers in hand.
