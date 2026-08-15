# Systematic Core — Action Plan v3
### Based on `systematic_core_findings_v2.md`

---

## Where Things Stand

| Strategy | Avg Sharpe | Avg CAGR | Avg Max DD | Implied Vol |
|---|---|---|---|---|
| Traditional (baseline) | **1.227** | **17.6%** | -10.0% | ~14.3% |
| Systematic Core (this round) | 0.918 | 8.5% | **-8.5%** | ~9.3% |
| Contrarian Volatility | 0.870 | 13.3% | -11.4% | ~15.3% |

The strategy now has the **best drawdown** and **best Sharpe among non-Traditional
strategies** — genuine, consistent progress. The remaining problem is not
risk-adjusted quality; it is **absolute return**. At 8.5% CAGR vs. SPY's
historical ~10%, the strategy barely covers the benchmark before costs.

The Sharpe/CAGR mismatch points to one root cause: **the strategy is running
at ~9.3% annualized volatility, half of Traditional's ~14.3%**. That gap is
not coming from the signal being wrong — it is coming from the system
systematically underweighting gross exposure. Fix that and CAGR lifts without
materially worsening the drawdown profile.

---

## Root Cause Analysis

### Why is annualized vol so low (~9.3%)?

Three compounding factors:

**1. Gross exposure floors are set by regime but the regime classifier may be
over-triggering TRANSITION/BEAR_TREND.**
The exposure table gives 0.75 for TRANSITION and 0.30–0.60 for BEAR_TREND.
If the HMM is spending a disproportionate amount of time in those two states
(common when the classifier is uncertain), average gross exposure could be
sitting materially below 1.0 even in benign markets. The prior round's grid
search returned a "mixed" result across windows — that is consistent with
regime misclassification at the margin, not a well-calibrated classifier.

**2. The low-volatility signal is selecting inherently low-beta stocks.**
The `low_vol` signal at 10% weight systematically tilts the portfolio toward
names with lower beta. Combined with `gross_profitability` (also tends to
select lower-beta quality names), the 20-stock portfolio likely carries a
market beta meaningfully below 1.0. Lower beta = lower CAGR in a bull market.

**3. `ml_alpha` and `insider_buying` contribute zero.**
Two signals totaling 25% of nominal composite weight (0.00 `ml_alpha` +
effectively-zero `insider_buying`) are dead weight. The remaining signals carry
100% of the load but their weights were renormalized to sum to 1.0 — meaning
the effective weighting is distorted from what was intended.

---

## Priority Actions

---

### Action 1 — Measure and Log Average Gross Exposure Per Regime

Before changing anything, instrument the backtest to record what gross exposure
the system actually runs at, day by day. This takes 30 minutes and tells you
definitively whether underexposure is the problem.

```python
# backtest/engine.py — add to the per-bar logging

def log_bar_state(self, date, weights, regime, dd_state):
    gross_exposure = weights.abs().sum()   # 0 to 1+
    self.exposure_log.append({
        'date':           date,
        'regime':         regime,
        'dd_state':       dd_state,
        'gross_exposure': gross_exposure,
        'n_positions':    (weights > 0.001).sum(),
    })

# After walk-forward: summarize
exposure_df = pd.DataFrame(engine.exposure_log)
print(exposure_df.groupby('regime')['gross_exposure'].describe())
print(f"\nOverall mean gross exposure: {exposure_df['gross_exposure'].mean():.3f}")
print(f"% of days below 0.80 exposure: {(exposure_df['gross_exposure'] < 0.80).mean():.1%}")
```

**Expected finding if underexposure is the cause**: mean gross exposure is
below 0.80, with TRANSITION/BEAR_TREND accounting for a large % of days even
in 2013–2019 (a sustained bull run where it should mostly be BULL_TREND).

**Expected finding if low-beta stock selection is the cause**: mean gross
exposure is above 0.90, but portfolio beta (weighted sum of individual betas)
is below 0.70.

Run both checks. They are not mutually exclusive.

```python
# Also compute portfolio beta vs SPY
def compute_portfolio_beta(weights, betas, date):
    """
    betas: pd.Series of 12-month rolling beta vs SPY, indexed by ticker.
    """
    aligned = weights.reindex(betas.index).fillna(0)
    portfolio_beta = (aligned * betas).sum() / aligned.sum()
    return portfolio_beta
```

---

### Action 2 — Recalibrate Gross Exposure Floors (Conditional on Action 1)

**Only run this if Action 1 confirms mean gross exposure < 0.85.**

The current floors were grid-searched on validation folds but the grid was
limited (0.30 / 0.45 / 0.60 for BEAR_TREND, 0.20 / 0.35 for CRISIS). Widen
the grid and add TRANSITION:

```python
param_grid = {
    'BULL_TREND':    [1.00, 1.10],   # test modest leverage in bull regime
    'LOW_VOL_GRIND': [0.90, 1.00],
    'TRANSITION':    [0.75, 0.85, 0.95],   # was only 0.65/0.75 before
    'BEAR_TREND':    [0.40, 0.55, 0.70],   # shift the whole range up
    'CRISIS':        [0.20, 0.35],          # keep crisis floor low
}
```

The working hypothesis: the prior grid's ceiling of 0.60 for BEAR_TREND and
0.75 for TRANSITION was too conservative for the 200-stock universe. A
strategy with Sharpe 0.918 and MaxDD -8.5% has room to take more risk.

**Guard rail**: any configuration that increases mean MaxDD beyond -12% on
walk-forward is rejected regardless of Sharpe improvement.

---

### Action 3 — Backfill `insider_buying` for Full Universe

Currently 4/204 symbols covered. This is not a signal — it is noise with a
label. Fix: run the full backfill as a dedicated background job with proper
per-symbol pacing.

```python
# data/insider.py — full backfill with rate limiting

import time
import logging

def backfill_insider_universe(tickers: list[str],
                               cache_dir: str,
                               lookback_years: int = 10,
                               pause_between_symbols: float = 2.0,
                               pause_between_pages: float = 0.5):
    """
    Fetches Form 4 data for all tickers with proper SEC rate limiting.
    2 seconds between symbols, 0.5 seconds between pages within a symbol.
    Estimated runtime for 204 symbols: ~7–10 minutes.
    Saves per-ticker JSON to cache_dir. Safe to interrupt and resume
    (already-cached tickers are skipped on re-run).
    """
    logger = logging.getLogger(__name__)
    done, skipped, failed = 0, 0, 0

    for i, ticker in enumerate(tickers):
        cache_path = Path(cache_dir) / f"{ticker}.json"
        if cache_path.exists():
            skipped += 1
            continue

        try:
            transactions = fetch_form4_transactions(
                ticker=ticker,
                lookback_years=lookback_years,
                page_pause=pause_between_pages
            )
            cache_path.write_text(json.dumps(transactions))
            done += 1
            logger.info(f"[{i+1}/{len(tickers)}] {ticker}: {len(transactions)} transactions cached")
        except Exception as e:
            failed += 1
            logger.warning(f"[{i+1}/{len(tickers)}] {ticker}: FAILED — {e}")
            # Write empty cache to avoid re-hammering failed tickers
            cache_path.write_text(json.dumps([]))

        time.sleep(pause_between_symbols)   # respect SEC fair-access guidance

    logger.info(f"Backfill complete: {done} cached, {skipped} skipped, {failed} failed")
```

Run this once offline. The `get_cached_insider_transactions` helper already
reads from this cache — no other changes needed once the cache is populated.

**Expected impact**: `insider_buying` signal goes from 4/204 coverage (~2%)
to 200+/204 coverage (~98%). Combined with `short_interest` and
`earnings_revision`, the three new signals will carry their intended composite
weight for the first time.

---

### Action 4 — Re-check `ml_alpha` IC After Data Backfill

`ml_alpha` IC moved from -0.012 to +0.020 when the universe expanded. It is
currently at 0.020 — just 0.010 below the gate. With full `insider_buying`
coverage added as a feature (currently contributing nothing for 200/204
tickers), the IC may cross 0.030.

Do not change the gate. Instead, re-run IC measurement after Action 3's
backfill is complete:

```python
# After insider_buying backfill — re-run IC check
ic_result = validate_ml_alpha_ic(
    signal_fn=compute_ml_alpha,        # now includes full insider_buying feature
    price_data=prices_204,
    validation_start='2018-01-01',     # limited by short_interest history depth
    validation_end='2020-12-31',
    min_ic=0.03
)

# IC tiering — unchanged from action plan v2
weight = get_weight_from_ic(ic_result['mean_ic'], base_weight=0.15)
print(f"ml_alpha IC: {ic_result['mean_ic']:.4f} → weight: {weight:.2f}")
```

If IC still does not pass 0.030 after full data coverage: accept that
`ml_alpha` is not a viable signal on this dataset and remove it permanently.
Redistribute its 15% nominal weight to the signals with confirmed positive IC.

---

### Action 5 — Address the `short_interest` History Gap

`short_interest` data is only available back to ~2018 via FINRA's API.
The full walk-forward runs 2008–2024, so this signal contributes nothing for
the first 10 years of the backtest.

Two options:

**Option A (recommended): Degrade gracefully, acknowledge the gap.**
The current implementation already degrades to neutral (0) when data is
missing — leave this in place. Measure what % of the total backtest days
the signal is actually active:

```python
def audit_signal_coverage(signal_log: pd.DataFrame) -> pd.DataFrame:
    """
    Reports what % of days each signal contributes a non-zero score.
    Signals with < 30% coverage are effectively decorative.
    """
    return signal_log.ne(0).mean().rename('pct_active').to_frame()
```

If `short_interest` is active only ~37% of days (2018–2024 / 2008–2024),
its effective weight in the composite is ~37% × 10% = ~3.7% — small but
not zero. Leave it; the coverage will grow as the strategy runs live.

**Option B: Source pre-2018 short interest from SEC Reg SHO.**
The SEC publishes Reg SHO short sale volume data going back to 2005 on
its website. This is daily short *volume* (not bi-weekly short *interest*)
but is correlated enough to serve as a proxy for the pre-2018 window:

```python
def fetch_sec_regsho_short_volume(ticker: str, date: str) -> float:
    """
    SEC Reg SHO daily short sale volume as % of total volume.
    Available at: https://www.sec.gov/cgi-bin/browse-edgar
    Alternative URL pattern: https://www.finra.org/sites/default/files/
    Proxy for short interest when FINRA API history is unavailable.
    """
    # Implementation: download daily Reg SHO file for the settlement date,
    # parse tab-separated format, extract ShortVolume / TotalVolume for ticker
    pass
```

Option A is sufficient for now. Pursue Option B only if the signal's full-period
IC (including the pre-2018 neutral years) is materially lower than its
post-2018 IC — run that comparison before investing time in Option B.

---

### Action 6 — Investigate and Reduce the Beta Drag (Conditional on Action 1)

**Only run this if Action 1 shows portfolio beta < 0.80.**

If the 20-stock portfolio is systematically selecting low-beta names (driven
by `low_vol` and `gross_profitability` tilts), CAGR will structurally lag
the benchmark in bull markets regardless of gross exposure.

Two targeted fixes:

**Fix A — Add a beta floor to the portfolio optimizer.**
Accept only stocks with beta > 0.60 in the eligible universe. Eliminates the
most defensive names while keeping the quality/momentum tilt.

```python
def apply_beta_floor(universe: list[str], betas: pd.Series,
                     min_beta: float = 0.60) -> list[str]:
    """Remove structurally low-beta names from the investable universe."""
    return [t for t in universe if betas.get(t, 1.0) >= min_beta]
```

**Fix B — Reduce `low_vol` signal weight.**
`low_vol` at 10% weight in a 200-stock universe is strong enough to
systematically select the bottom beta quintile. Cut to 5% and redistribute
to `momentum_12_1` or `earnings_revision`, both of which select higher-beta
names (momentum tends to pick names in trending, often higher-vol sectors).

```yaml
# config/strategy.yaml — if beta drag is confirmed
systematic_core:
  signal_weights:
    momentum_12_1:       0.25   # up from 0.20
    earnings_revision:   0.18   # up from 0.15
    short_interest:      0.12   # up from 0.10
    insider_buying:      0.12   # up from 0.10 — after backfill
    gross_profitability: 0.12   # unchanged
    low_vol:             0.06   # down from 0.10
    book_to_price:       0.08   # up from 0.08
    ml_alpha:            0.00   # unchanged until IC passes gate
    earnings_surprise:   0.07   # add (already available via PEAD proxy)
```

---

## Sequenced Execution

```
Day 1 (Diagnostic — no changes to strategy):
  [ ] Add gross exposure logging to backtest engine (Action 1)
  [ ] Re-run walk-forward with logging enabled (same parameters, no changes)
  [ ] Compute mean gross exposure by regime and overall
  [ ] Compute portfolio beta vs SPY by year
  [ ] Record findings: is CAGR gap from underexposure, low beta, or both?

Day 2 (Data backfill — offline job):
  [ ] Run insider_buying backfill for all 204 tickers (Action 3)
  [ ] Verify cache coverage: expect 195+ / 204 symbols cached
  [ ] Run signal coverage audit — confirm short_interest active ~37% of days

Day 3 (IC recheck + signal weight update):
  [ ] Re-run ml_alpha IC with full insider_buying feature (Action 4)
  [ ] If IC >= 0.03: enable ml_alpha at tiered weight (0.075 for IC 0.03–0.05)
  [ ] If IC < 0.03: redistribute its 15% nominal weight to confirmed signals
  [ ] Update signal_weights config with final validated values

Day 4 (Exposure or beta fix — based on Day 1 findings):
  [ ] If underexposure confirmed: run exposure floor grid search (Action 2)
  [ ] If beta drag confirmed: apply beta floor + reduce low_vol weight (Action 6)
  [ ] If both: fix underexposure first (bigger lever), then beta floor

Day 5 (Full walk-forward):
  [ ] Re-run 3 window configs with all changes applied
  [ ] Compare vs v2 baseline (0.918 Sharpe, 8.5% CAGR, -8.5% MaxDD)
  [ ] Report results — no further tuning without user confirmation
```

---

## Target After This Round

| Metric | v2 Baseline | Minimum Target | Stretch Target |
|---|---|---|---|
| Avg Sharpe | 0.918 | 0.95 | 1.10 |
| Avg CAGR | 8.5% | 12.0% | 15.0% |
| Avg Max DD | -8.5% | < -12% | < -10% |
| Implied vol | ~9.3% | ~12% | ~13.5% |

The primary goal is **lifting CAGR toward 12–15%** without meaningfully
worsening the drawdown. The strategy does not need to beat Traditional's
17.6% CAGR in this round — closing the gap to within 3–5% while maintaining
Sharpe above 1.0 would make it genuinely deployable alongside Traditional.

---

## What Would Indicate This Strategy Has a Ceiling

If after Actions 1–6, CAGR is still below 10% and Sharpe is below 1.0, the
strategy architecture itself may be the ceiling, not the parameters. Indicators
that would support stopping further iteration on Systematic Core and instead
building a complementary strategy:

- Portfolio beta persistently below 0.65 regardless of signal weight changes
- Gross exposure grid search consistently favoring floors above 0.85 in
  BEAR_TREND (means the regime signal adds no value)
- `ml_alpha` IC still below 0.02 after full data coverage (means features
  have no predictive power on this universe)
- 5yr/2yr Sharpe still below 0.70 (means the 2022 bear market problem is
  structural, not addressable by parameter tuning)

In that scenario: keep Systematic Core as a **low-drawdown defensive
satellite** (it has the best MaxDD of the three strategies) and build a
separate momentum-or-macro strategy with higher beta and higher CAGR target
to complement it.

---

*Supersedes `systematic_core_action_plan.md` (v2).*
*Read alongside `systematic_core_findings_v2.md`, `systematic_core_improvements.md`,*
*and `systematic_trading_engineering_guide.md`.*
