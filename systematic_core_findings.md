# Systematic Core — findings from implementing `systematic_core_improvements.md`

## Summary

All six code fixes and the six guide corrections from
`systematic_core_improvements.md` were implemented, tested, and validated
against the same full 2008–2024 walk-forward harness used throughout this
project. Every fix was independently correct and well-justified. The net
effect on Systematic Core's performance was **negative** — average Sharpe
dropped from 1.13 to 0.73, moving it further behind Traditional, not
closer. This document records what was actually found, separate from the
prescriptive improvement plan and the original postmortem.

## What was implemented

| # | Fix | Status |
|---|---|---|
| 1 | Ratio-based `top_n` (`top_n_pct`/`top_n_min` instead of a hardcoded count) | Implemented in `decision/portfolio.py::resolve_top_n`, backward-compatible (Traditional/Contrarian Volatility unaffected) |
| 2 | IC validation gate for `ml_alpha` | Implemented in new `alpha_engine/ml/validation.py`; actually run against real data |
| 3 | Replace `BAMLH0A0HYM2` with `BAA`−`AAA` credit spread | Implemented in `data/macro.py`; confirmed full unrestricted history back to 1919 |
| 4 | HMM degeneracy detection + rule-based fallback | Implemented in `decision/regime.py` (`_is_degenerate`, `_simple_regime_fallback`); caught and fixed a real bug in the detection logic itself during testing (see below) |
| 5 | Walk-forward-tunable `GROSS_EXPOSURE_BY_REGIME` | Implemented (`resolve_exposure_table`, config-driven overrides) and actually grid-searched via the existing `run_walk_forward` `param_grid` mechanism |
| 6 | Opt-in REDUCE risk tier | Implemented in `backtest/engine.py`/`live/runner.py`, disabled by default, combines with regime exposure via `min()` |
| 7 | Guide corrections (gaps A–F) | Appended to `systematic_trading_engineering_guide.md` at the six specified sections |

All work is covered by new/updated tests — 276 tests passing across the
root engine suite and the web backend suite, no regressions.

## Finding 1: `ml_alpha`'s real IC is negative

Measured directly (`alpha_engine.ml.validation.compute_mean_ic`), training
on 2015–2020 and validating on the genuinely held-out 2021–2022 period:

```
mean_ic   = -0.0120
ic_std    =  0.2533
ic_ir     = -0.0474
n_periods =  104
passes (>= 0.03) = False
```

This is a real, negative result, not noise-adjacent-to-zero — `ic_std` is
21x larger than `mean_ic`, and the IC information ratio is effectively
zero. The signal was carrying a 0.35 composite weight for the entirety of
this session's earlier validation work before this was ever checked. Per
the IC gate, its weight was set to 0.0 for the final validated
configuration below (remaining signal weights renormalized to sum to 1.0).

## Finding 2: a real bug in the degeneracy-detection fix itself

While validating Fix 4, `_is_degenerate` initially flagged every
legitimately-fit diagonal-covariance HMM as degenerate. Root cause:
`GaussianHMM.covars_` always returns a full `(n_components, n_features,
n_features)` matrix regardless of `covariance_type` (hmmlearn's
`fill_covars` fills the off-diagonal entries with zeros for the "diag"
case) — the original check compared the *entire* matrix against a
near-zero threshold, so every structurally-zero off-diagonal entry looked
like a collapsed variance. Fixed to check only `np.diagonal(model.covars_,
axis1=-2, axis2=-1)`. Caught by the test suite before this ever reached a
real validation run — see `tests/test_regime.py::TestIsDegenerate` for the
regression tests (constructed fake model objects, not dependent on
hmmlearn actually producing a specific fit outcome).

## Finding 3: the fixes combined to hurt risk-adjusted performance

Full walk-forward comparison, same 3 window configs (2/1/1, 3/1/1, 5/2/1),
full 2008–2024 history, full 63-symbol universe:

| Version | Avg Sharpe | Avg CAGR | Avg Max DD |
|---|---|---|---|
| Traditional (baseline, live) | **1.227** | **17.6%** | -10.0% |
| Systematic Core — before this round's fixes | 1.130 | 9.8% | -9.2% |
| Systematic Core — all fixes applied | **0.730** | 10.5% | -11.4% |
| Contrarian Volatility (live) | 0.870 | 13.3% | -11.4% |

Per-window-config, the drop is consistent, not driven by one bad window:

| Window config | Before fixes | After fixes |
|---|---|---|
| 2yr train / 1yr test | 1.146 | 0.761 |
| 3yr train / 1yr test | 1.238 | 0.775 |
| 5yr train / 2yr test | 0.971 | 0.631 |

### Why

The most likely driver is Fix 1 — the ratio-based `top_n` change,
positioned in the improvement plan as the single highest-confidence,
highest-priority fix. On this repo's 63-symbol universe, `top_n_pct=0.10`
with `top_n_min=5` resolves to `max(5, int(63*0.10)) = 6` positions —
down from the previous fixed 30. Concentrating a portfolio from 30 names
to 6 substantially increases idiosyncratic (single-name) risk; the
improvement plan's framing ("the diversification tax") assumed that
fixing over-diversification would be a clean win, but it does not account
for the risk cost of the opposite failure mode (under-diversification) on
a universe this small. The wider position/sector caps that had to
accompany the smaller position count (`max_position_weight` 0.06→0.30,
`max_sector_weight` 0.25→0.50, needed simply to avoid leaving most of the
portfolio in cash at 6 positions) further concentrate single-position and
single-sector risk.

A second, smaller data point points the same direction: the exposure-floor
grid search (Fix 5) consistently chose the *loosest* available
`BEAR_TREND` floor (0.60 of the three candidates tested — 0.30/0.45/0.60)
in 9 of 11 windows in the longest window config, and a matching pattern in
the others. The train-Sharpe-optimal choice, given the now-more-
concentrated and `ml_alpha`-free portfolio, was to stay *more* invested
through weakness, not less — the opposite of the improvement plan's
"the 0.60 floor is too loose, tighten it" framing.

Removing `ml_alpha` (Finding 1) is very likely also a contributing factor,
though probably a smaller one than the concentration change — even a
signal with a slightly negative individual IC can occasionally still
serve a diversifying/decorrelating role inside a composite score; ensemble
effects are not always additive; this was not separately isolated (doing
so would mean re-running the sweep with `ml_alpha` zeroed but `top_n`
still fixed at 30, and separately with `top_n` ratio-based but `ml_alpha`
still weighted, to attribute the drop between the two changes — not done
here).

## Conclusion

Every individual fix in `systematic_core_improvements.md` was correct
engineering: two were real, confirmed bugs; the IC gate is a real
validation step now honored rather than skipped; the data-source fix
closed a real, confirmed FRED licensing restriction; the exposure floors
are now genuinely tunable instead of hardcoded illustrative numbers. None
of that is in question.

What the numbers do not support is the improvement plan's central
performance hypothesis — that fixing over-diversification and removing an
unvalidated ML signal would close the gap with Traditional. Measured
honestly, it widened the gap. Systematic Core, as currently configured
with all six fixes applied, has a worse risk-adjusted return than it did
before this round of changes, and remains behind both Traditional and
(on Sharpe specifically) even Contrarian Volatility is now closer to it
than before.

This is reported as-is rather than further re-tuned unilaterally, since
the next reasonable moves (e.g., a `top_n_pct` between the original 30-
position count and the new 6-position count, or isolating `ml_alpha`'s
removal from the concentration change to attribute the drop correctly)
are strategy-design judgment calls, not bug fixes — the same threshold
this document's predecessor (`systematic_core_postmortem.md`) used to stop
and check in rather than keep iterating alone.
