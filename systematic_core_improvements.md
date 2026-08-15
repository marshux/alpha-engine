# Systematic Core — Improvements & Action Plan
### Based on postmortem analysis of `systematic_core_postmortem.md`

---

## Quick Reference: Prioritized Action Items

| Priority | Change | Target | Expected Impact |
|---|---|---|---|
| 1 | `n_long` as ratio, not absolute | Implementation | High — directly fixes diversification tax |
| 2 | IC validation gate before `ml_alpha` use | Implementation | High — may reveal signal is adding noise |
| 3 | Replace `BAMLH0A0HYM2` with `BAA - AAA` | Implementation | High — restores full HMM feature panel |
| 4 | HMM degeneracy fallback to VIX+MA rule | Implementation | Medium — prevents silent failures on refit |
| 5 | Walk-forward tune `GROSS_EXPOSURE_BY_REGIME` | Implementation | Medium — directly targets 2022 bear drag |
| 6 | Add REDUCE tier to risk manager | Implementation | Low-Medium — smoother drawdown management |
| 7 | Update engineering guide (gaps A–F) | Guide | Prevents same failure modes in future builds |

---

## Part I — Implementation Fixes

---

### Fix 1 — Universe-to-`n_long` Ratio (Highest Priority)

**Problem**: `n_long=30` out of a 63-symbol universe means holding ~48% of all
available names. The top alpha signal in the world cannot overcome that level of
dilution. The engineering guide states `n_long=50` from ~500 stocks (10% of
universe) — but the code was written as a hardcoded integer, not a ratio.

**Fix**: Replace the hardcoded integer with a ratio-based calculation everywhere
`n_long` or `top_n` is set.

```python
# Before
n_long = 30

# After — always hold top 10% of the live universe, minimum 5 positions
n_long = max(5, int(len(universe) * 0.10))
# For a 63-symbol universe: max(5, int(63 * 0.10)) = max(5, 6) = 6
# For a 500-symbol universe: max(5, int(500 * 0.10)) = 50  ← matches the guide
```

Apply the same ratio logic wherever `top_n` is read from `config/strategy.yaml`.
Update the YAML to store the ratio, not the count:

```yaml
# config/strategy.yaml
systematic_core:
  top_n_pct: 0.10          # hold top 10% of universe by alpha rank
  top_n_min: 5             # floor: never fewer than 5 positions
  max_position_weight: 0.06
  max_sector_weight: 0.25
```

```python
# portfolio construction — read ratio, not count
top_n_pct = config.get('top_n_pct', 0.10)
top_n_min = config.get('top_n_min', 5)
n_long = max(top_n_min, int(len(universe) * top_n_pct))
```

---

### Fix 2 — IC Validation Gate for `ml_alpha`

**Problem**: `ml_alpha` is trained on 5 base factors (not the guide's 10) and
its Information Coefficient was never measured. Its 0.35 composite weight may be
assigning significant influence to a signal that carries no genuine predictive
power on this universe, effectively injecting noise into the ranking.

**Fix**: Add an IC validation step that runs on a held-out validation period
**before** any walk-forward sweep. If the signal's mean IC is below 0.03, its
weight is forced to zero until the signal is improved.

```python
# alpha_engine/ml/validate.py

from scipy.stats import spearmanr
import numpy as np
import pandas as pd

def compute_ic(signal_scores: pd.Series, forward_returns: pd.Series) -> float:
    """
    Information Coefficient: Spearman rank correlation between signal
    scores and realized forward returns.
    IC > 0.05 is strong. IC > 0.03 is the minimum acceptable threshold.
    """
    aligned = pd.concat([signal_scores, forward_returns], axis=1).dropna()
    if len(aligned) < 20:
        return 0.0
    ic, _ = spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return ic

def validate_ml_alpha_ic(
    signal_fn,
    price_data: pd.DataFrame,
    validation_start: str,
    validation_end: str,
    forward_horizon: int = 21,
    min_ic: float = 0.03,
) -> dict:
    """
    Runs IC sweep on validation period. Returns IC stats and a pass/fail flag.
    Call this BEFORE the walk-forward sweep, on a separate held-out period.
    """
    dates = pd.date_range(validation_start, validation_end, freq='B')
    ics = []

    for date in dates[::21]:  # sample monthly to avoid autocorrelation
        try:
            scores = signal_fn(date)
            fwd_returns = price_data.pct_change(forward_horizon).shift(-forward_horizon).loc[date]
            ic = compute_ic(scores, fwd_returns)
            ics.append(ic)
        except Exception:
            continue

    mean_ic = float(np.mean(ics)) if ics else 0.0
    ic_ir = float(np.mean(ics) / np.std(ics)) if len(ics) > 1 else 0.0  # IC Information Ratio

    result = {
        'mean_ic':     mean_ic,
        'ic_std':      float(np.std(ics)) if ics else 0.0,
        'ic_ir':       ic_ir,
        'n_periods':   len(ics),
        'passes':      mean_ic >= min_ic,
    }

    if not result['passes']:
        print(
            f"WARNING: ml_alpha mean IC={mean_ic:.4f} is below threshold {min_ic}.\n"
            f"Setting ml_alpha weight to 0.0 until signal is improved with more features.\n"
            f"Suggested fix: add earnings_revision, earnings_surprise, short_interest,\n"
            f"insider_buying, and options_skew signals to reach the guide's full feature set."
        )
    else:
        print(f"ml_alpha IC validated: mean_IC={mean_ic:.4f}, IC_IR={ic_ir:.2f} — OK to use.")

    return result


def get_effective_signal_weights(base_weights: dict, ic_results: dict) -> dict:
    """
    Returns weights with any failed signals zeroed out and remainder renormalized.
    """
    effective = dict(base_weights)
    for signal, result in ic_results.items():
        if not result.get('passes', True):
            effective[signal] = 0.0

    total = sum(effective.values())
    if total > 0:
        effective = {k: v / total for k, v in effective.items()}
    return effective
```

Usage at strategy initialization:

```python
# In systematic_core strategy setup
ic_result = validate_ml_alpha_ic(
    signal_fn=self.compute_ml_alpha,
    price_data=self.prices,
    validation_start='2018-01-01',
    validation_end='2020-12-31',   # held-out validation period, never the test set
    min_ic=0.03
)
weights = get_effective_signal_weights(SYSTEMATIC_CORE_OVERRIDES, {'ml_alpha': ic_result})
```

---

### Fix 3 — Replace Restricted FRED Series with Free Equivalent

**Problem**: `BAMLH0A0HYM2` (ICE BofA HY credit spread) is access-restricted on
FRED's free/anonymous endpoint. The `dropna()` across all 8 HMM feature columns
silently discarded the entire feature panel for all dates in the gap — the HMM
classifier was never actually fit on real data for most of the validation history.

**Fix**: Replace `BAMLH0A0HYM2` with the Moody's BAA–AAA spread, which is freely
accessible on FRED with history back to 1919.

```python
# decision/regime.py

# Before
credit_spread = self.fred.get('BAMLH0A0HYM2', date)

# After — Moody's BAA minus AAA: functionally equivalent, fully unrestricted
baa = self.fred.get('BAA', date)   # Moody's Seasoned Baa Corporate Bond Yield
aaa = self.fred.get('AAA', date)   # Moody's Seasoned Aaa Corporate Bond Yield
credit_spread = baa - aaa

# The signal derivation logic is unchanged:
spread_change = credit_spread - credit_spread_ma
credit_spread_signal = np.clip(-spread_change / 1.5, -1, 1)
```

Also fix the `dropna()` strategy to only drop rows for transient warmup gaps in
price-derived columns, not macro-derived ones:

```python
def build_regime_features(self, market_data: pd.DataFrame) -> np.ndarray:
    features = pd.DataFrame({
        'spy_return_1m':    market_data['SPY'].pct_change(21),
        'spy_return_3m':    market_data['SPY'].pct_change(63),
        'realized_vol_1m':  market_data['SPY'].pct_change().rolling(21).std() * np.sqrt(252),
        'vix_level':        market_data['VIXCLS'] / 100,
        'vix_change_1m':    market_data['VIXCLS'].pct_change(21),
        'yield_curve':      market_data['DGS10'] - market_data['DGS2'],
        'credit_spread':    market_data['BAA'] - market_data['AAA'],  # fixed
        'breadth_200ma':    market_data['pct_above_200ma'],
    })

    # Drop only for price-derived warmup gaps (first ~252 rows)
    # Macro columns get filled with 0 if missing — neutral signal, not dropped data
    macro_cols = ['yield_curve', 'credit_spread', 'breadth_200ma']
    price_cols = ['spy_return_1m', 'spy_return_3m', 'realized_vol_1m',
                  'vix_level', 'vix_change_1m']

    features[macro_cols] = features[macro_cols].fillna(0.0)
    features = features.dropna(subset=price_cols)   # only drop for price warmup

    return features.values
```

---

### Fix 4 — HMM Degeneracy Fallback

**Problem**: At small scale (few thousand rows, 8 features, 4 components),
`GaussianHMM` frequently produces degenerate fits — collapsed transition matrices,
near-zero emission variances, or NaN means — causing silent fallbacks to the
default label. The guide's illustrative code does not address this failure mode.

**Fix**: Add degeneracy detection after every `model.fit()` call and fall back to
a simple, robust VIX + 200-day MA crossover classifier when the HMM fit is bad.

```python
# decision/regime.py

import numpy as np
from hmmlearn import hmm

class RegimeClassifier:

    def __init__(self, n_regimes=4):
        self.n_regimes = n_regimes
        self.model = None
        self.use_hmm = False
        self.regime_labels = {}

    def fit(self, market_data: pd.DataFrame):
        X = self.build_regime_features(market_data)
        if len(X) < 252:
            self.use_hmm = False
            return

        model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type='full',
            n_iter=200,
            random_state=42
        )
        try:
            model.fit(X)
            if self._is_degenerate(model):
                raise ValueError("Degenerate HMM fit detected")
            self.model = model
            self.regime_labels = self._interpret_regimes(model)
            self.use_hmm = True
        except Exception as e:
            print(f"HMM fit failed ({e}); falling back to VIX+MA classifier.")
            self.use_hmm = False

    def _is_degenerate(self, model) -> bool:
        """
        Checks for common HMM degeneracy conditions.
        Returns True if the fit is unreliable and should be rejected.
        """
        # Any transition probability below 1% suggests collapsed state
        if (model.transmat_ < 0.01).any():
            return True
        # NaN or Inf in means or covariances
        if np.isnan(model.means_).any() or np.isinf(model.means_).any():
            return True
        # Any state has near-zero variance (collapsed to a point)
        for cov in model.covars_:
            if np.diag(cov).min() < 1e-6:
                return True
        return False

    def predict_current(self, market_data: pd.DataFrame) -> str:
        if not self.use_hmm:
            return self._simple_regime_fallback(market_data)

        X = self.build_regime_features(market_data)
        try:
            state_idx = self.model.predict(X)[-1]
            return self.regime_labels.get(state_idx, 'TRANSITION')
        except Exception:
            return self._simple_regime_fallback(market_data)

    def _simple_regime_fallback(self, market_data: pd.DataFrame) -> str:
        """
        Robust two-signal fallback when HMM is unavailable or degenerate.
        Uses only VIX level and SPY vs. 200-day MA — both always available.
        """
        vix = market_data['VIXCLS'].iloc[-1]
        spy = market_data['SPY']
        spy_vs_200ma = spy.iloc[-1] / spy.rolling(200).mean().iloc[-1] - 1

        if vix > 30 or spy_vs_200ma < -0.15:
            return 'CRISIS'
        elif spy_vs_200ma < -0.05 or vix > 22:
            return 'BEAR_TREND'
        elif vix < 15 and spy_vs_200ma > 0.05:
            return 'BULL_TREND'
        elif spy_vs_200ma > 0:
            return 'LOW_VOL_GRIND'
        else:
            return 'TRANSITION'

    def _interpret_regimes(self, model) -> dict:
        """
        Labels each HMM state by inspecting its mean feature vector.
        Every state gets a label — no fallback to a default.
        """
        means = model.means_
        # Feature order matches build_regime_features:
        # [spy_return_1m, spy_return_3m, realized_vol_1m, vix_level,
        #  vix_change_1m, yield_curve, credit_spread, breadth_200ma]
        regime_map = {}
        for i, mean in enumerate(means):
            ret_1m = mean[0]
            vol_1m = mean[2]
            vix    = mean[3]

            if ret_1m > 0.02 and vol_1m < 0.15 and vix < 0.20:
                regime_map[i] = 'BULL_TREND'
            elif ret_1m < -0.02 and vol_1m > 0.18:
                regime_map[i] = 'BEAR_TREND'
            elif vix > 0.28 or vol_1m > 0.25:
                regime_map[i] = 'CRISIS'
            elif ret_1m >= 0 and vol_1m < 0.13:
                regime_map[i] = 'LOW_VOL_GRIND'
            else:
                regime_map[i] = 'TRANSITION'   # safe conservative default

        return regime_map
```

---

### Fix 5 — Walk-Forward Tune `GROSS_EXPOSURE_BY_REGIME`

**Problem**: The exposure floors (1.0 / 0.9 / 0.75 / 0.60 / 0.35) were taken
directly from the guide's illustrative numbers and never validated against this
repository's universe. The 0.60 BEAR_TREND floor is the primary source of
underperformance during the 2022 sustained bear market.

**Fix**: Add the exposure floors as hyperparameters to the walk-forward sweep.
A coarse grid over the most impactful value (BEAR_TREND) is sufficient.

```python
# config/strategy.yaml — promote to tunable hyperparameters
systematic_core:
  gross_exposure_grid:
    BULL_TREND:     [1.00]           # rarely worth reducing
    LOW_VOL_GRIND:  [0.90]
    TRANSITION:     [0.65, 0.75]
    BEAR_TREND:     [0.30, 0.45, 0.60]   # highest-impact parameter to tune
    CRISIS:         [0.20, 0.35]
```

```python
# walk_forward.py — grid search over exposure combinations

from itertools import product

def tune_exposure_floors(strategy, data, validation_folds):
    """
    Runs a coarse grid search over GROSS_EXPOSURE_BY_REGIME values
    on the VALIDATION folds only. Never touches the test folds.
    """
    bear_options   = [0.30, 0.45, 0.60]
    crisis_options = [0.20, 0.35]
    results = []

    for bear_floor, crisis_floor in product(bear_options, crisis_options):
        exposure_config = {
            'BULL_TREND':    1.00,
            'LOW_VOL_GRIND': 0.90,
            'TRANSITION':    0.75,
            'BEAR_TREND':    bear_floor,
            'CRISIS':        crisis_floor,
        }
        strategy.set_exposure_config(exposure_config)

        fold_sharpes = []
        for fold in validation_folds:
            ret = strategy.backtest(data, fold['start'], fold['end'])
            fold_sharpes.append(compute_sharpe(ret))

        results.append({
            'bear_floor':   bear_floor,
            'crisis_floor': crisis_floor,
            'mean_sharpe':  np.mean(fold_sharpes),
            'min_sharpe':   np.min(fold_sharpes),   # robustness check
        })

    best = max(results, key=lambda r: r['mean_sharpe'])
    print(f"Best exposure config: BEAR={best['bear_floor']}, CRISIS={best['crisis_floor']}")
    print(f"Validation Sharpe: {best['mean_sharpe']:.3f} (worst fold: {best['min_sharpe']:.3f})")
    return best
```

---

### Fix 6 — Add REDUCE Tier to Risk Manager

**Problem**: The repo's risk manager has only two states (NORMAL / HALT). The
guide's design has three (NORMAL / REDUCE / HALT). The missing REDUCE tier means
the system stays fully invested until it hits the -15% HALT threshold, with no
gradual de-risking in between.

**Fix**: Add the REDUCE tier and a recovery condition that returns to NORMAL.

```python
# risk/manager.py

class RiskManager:

    LIMITS = {
        'max_drawdown_halt':    0.15,   # halt all trading at -15%
        'max_drawdown_reduce':  0.08,   # reduce exposure at -8%
        'reduce_exposure_scale': 0.50,  # halve gross exposure in REDUCE state
        'recovery_threshold':   0.03,   # return to NORMAL once DD recovers to -3%
    }

    def check_portfolio_drawdown(self, portfolio_value: pd.Series) -> str:
        peak = portfolio_value.cummax()
        current_dd = (portfolio_value.iloc[-1] / peak.iloc[-1]) - 1

        if current_dd <= -self.LIMITS['max_drawdown_halt']:
            return 'HALT'
        elif current_dd <= -self.LIMITS['max_drawdown_reduce']:
            return 'REDUCE'
        elif current_dd >= -self.LIMITS['recovery_threshold']:
            return 'NORMAL'
        else:
            # Between recovery threshold and reduce threshold: hold previous state
            return getattr(self, '_last_state', 'NORMAL')

    def apply_drawdown_action(self, target_weights: pd.Series,
                               action: str) -> pd.Series:
        if action == 'HALT':
            return pd.Series(0.0, index=target_weights.index)
        elif action == 'REDUCE':
            return target_weights * self.LIMITS['reduce_exposure_scale']
        return target_weights  # NORMAL
```

---

## Part II — Engineering Guide Corrections

The following gaps in `systematic_trading_engineering_guide.md` should be
addressed so future implementations avoid the same failure modes.

---

### Guide Gap A — HMM Fragility at Small Scale

Add the following warning block to **Section 3.6 (Regime Detection)**,
immediately before the `MarketRegimeClassifier` code block:

> **Production Warning — HMM Fragility**: `GaussianHMM` with 4 components
> frequently produces degenerate fits when the dataset is small (< 5,000 rows)
> or when features have high multicollinearity. Always check for degeneracy after
> every `model.fit()` call (collapsed transition matrices, near-zero emission
> variances, NaN means) and implement a robust fallback. A VIX-level + SPY-vs-200MA
> rule requires no fitting and is suitable as the fallback. Treat HMM as the
> advanced option; the rule-based fallback should be the default for new
> implementations. Note also that HMM state indices are **not stable across
> refits** — `_interpret_regimes()` must be re-called every time the model is
> refit to re-derive the BULL/BEAR/CRISIS label mapping.

---

### Guide Gap B — `n_long` Must Be a Ratio, Not an Absolute Count

Add the following note to **Section 3.7 (Portfolio Construction)**, directly
after the `PortfolioConstructor.__init__` signature:

> **Universe Scaling**: `n_long=50` in this example assumes a ~500-stock universe
> (the S&P 500 tier from Section 2). This represents ~10% of the available
> universe. **Do not use an absolute count** — always express this as a ratio of
> the live universe size. Holding 50 stocks from 500 is concentrated alpha;
> holding 50 stocks from 63 is near-index replication with noise. The correct
> implementation is:
> ```python
> n_long = max(5, int(len(universe) * 0.10))
> ```
> For smaller universes, apply a higher ratio (15–20%) and verify that position
> caps (5% max) still allow the target number of positions.

---

### Guide Gap C — IC Must Be a Hard Gate, Not a Checklist Item

In **Section 5.7 (Backtesting Checklist)**, the item "each signal has positive IC
on out-of-sample data (IC > 0.03)" should be strengthened from a checkbox to an
enforced gate. Add the following immediately after the checklist item:

> **IC Gate — Non-Negotiable**: A signal with unvalidated or sub-threshold IC
> must be assigned **weight zero** in the composite until validated. Do not
> include it in the walk-forward sweep with a non-zero weight on the assumption
> that it "probably works" — if its IC is below 0.03, it is adding noise to the
> ranking. Use `validate_ml_alpha_ic()` (see implementation guide) on a held-out
> validation period before assigning any weight to a new signal.

---

### Guide Gap D — Free Data Availability Is Overstated

Replace the Data Sources table in **Section 3.1** with the corrected version
below, which accurately marks which sources are restricted and provides
verified free fallbacks:

| Data Type | Verified Free Sources | Paid Sources | Notes |
|---|---|---|---|
| Daily OHLCV | Polygon.io (free tier), yfinance | Bloomberg, FactSet | yfinance has no SLA; use Polygon for production |
| Adjusted prices | Polygon.io, Tiingo ($10/mo) | CRSP | CRSP is gold standard for research |
| Fundamentals | SimFin (free tier), EDGAR XBRL | Compustat via WRDS | SimFin has ~3yr lag on some fields |
| Earnings estimates | **No reliable free source** | IBES, FactSet, Refinitiv | Use post-earnings price reaction as proxy |
| Credit spreads | FRED: `BAA`, `AAA` (unrestricted) | Bloomberg | `BAMLH0A0HYM2` is **access-restricted** on FRED free tier |
| Macro rates | FRED: `DGS10`, `DGS2`, `VIXCLS` (all unrestricted) | Bloomberg | All freely available |
| Short interest | FINRA bi-weekly file (free, 2-week lag) | S3 Partners, Ortex | Bi-weekly cadence limits signal freshness |
| Insider buying | SEC EDGAR Form 4 RSS feed (free) | FactSet | Parseable XML; requires a custom ingestion pipeline |
| Options flow/skew | **No reliable free source** | OptionMetrics, CBOE datashop | Use VIX term structure as a coarse proxy |

---

### Guide Gap E — Exposure Floor Values Are Illustrative Only

Add the following warning to **Section 3.6**, directly after the
`GROSS_EXPOSURE_BY_REGIME` table:

> **Calibration Required**: The values in this table (1.0 / 0.9 / 0.75 / 0.60 /
> 0.35) are **illustrative starting points only**. They must be tuned against your
> specific universe, cost structure, and regime classifier via walk-forward
> validation before deployment. Key sensitivities to test:
>
> - A 0.60 BEAR_TREND floor causes material underperformance vs. a binary
>   flatten during sustained (> 6 month) bear markets. If your classifier is
>   reliable during downturns, test lower floors (0.30–0.45).
> - A 0.35 CRISIS floor may be too high if your CRISIS classification is
>   accurate; too low if CRISIS is frequently mis-classified.
> - Run a coarse grid search over these values on validation folds and select
>   the combination with the best mean Sharpe and lowest worst-fold Sharpe.

---

### Guide Gap F — REDUCE Tier in Risk Management Is Non-Optional

In **Section 3.8 (Risk Management)**, add the following note after the
`check_portfolio_drawdown` method:

> **Implementation Note**: The three-tier response (NORMAL / REDUCE / HALT) is
> not optional. A binary NORMAL/HALT system remains fully invested until it hits
> the halt threshold, missing the gradual de-risking that limits drawdown
> accumulation in the -8% to -15% zone. The REDUCE state should scale gross
> exposure by 0.5 and maintain that scale until the portfolio recovers to within
> 3% of its prior peak — this prevents the system from immediately re-risking
> after a brief recovery that turns out to be a dead-cat bounce.

---

## Summary: Sequenced Execution Order

Execute in this order to avoid compounding changes obscuring which fix had impact:

```
Step 1: Fix 3 (FRED series replacement)
        → Run a single walk-forward to confirm HMM now fits on full history
        → Baseline: is OOS Sharpe better than before?

Step 2: Fix 4 (HMM degeneracy fallback)
        → Re-run. Verify no more degenerate fit warnings in logs.

Step 3: Fix 1 (n_long ratio)
        → Drop n_long to ~6 (10% of 63). Re-run walk-forward.
        → This is the change most likely to close the gap with Traditional.

Step 4: Fix 2 (IC validation of ml_alpha)
        → Run IC sweep on validation period before the walk-forward.
        → If mean IC < 0.03, set ml_alpha weight to 0 and re-run.
        → If mean IC ≥ 0.03, ml_alpha weight stays as configured.

Step 5: Fix 5 (tune GROSS_EXPOSURE_BY_REGIME)
        → Run coarse grid on validation folds. Lock in best BEAR/CRISIS floors.
        → Re-run full walk-forward with tuned floors on test folds.

Step 6: Fix 6 (REDUCE tier)
        → Add to risk manager. Re-run and verify drawdown profile improves.

Step 7: Guide updates (Gap A–F)
        → Update systematic_trading_engineering_guide.md with corrections.
```

---

*This document is derived from `systematic_core_postmortem.md` and should be
read alongside `systematic_trading_engineering_guide.md`.*
