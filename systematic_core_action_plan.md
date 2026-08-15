# Systematic Core — Revised Action Plan
### Target: ~200-stock universe + all outstanding fixes

---

> **Context**: Expanding to the full S&P 500 (~500 stocks) was judged too large
> a first step. This plan targets a ~200-stock universe — enough for
> cross-sectional signals to be statistically meaningful, small enough to
> maintain data quality without a full institutional data infrastructure build.
> All six implementation fixes from `systematic_core_improvements.md` are
> incorporated, with the `n_long` ratio re-calibrated for this universe size.

---

## 1. Target Universe: S&P 200 (Large-Cap Core)

### Design

Use the **top 200 stocks by market cap** from the current S&P 500, filtered for
liquidity. This gives:

- Enough names for cross-sectional signals to work (vs. 63 — too few)
- Clean, liquid, low-slippage stocks throughout (vs. full Russell 1000 — too many small-caps)
- Manageable data sourcing with mostly free/cheap tools
- `n_long = 10% of 200 = 20 positions` — concentrated enough to have alpha, diversified enough to avoid single-name blowups

| | Before | This Plan |
|---|---|---|
| **Universe size** | 63 | ~200 |
| **`n_long` at 10%** | 6 (too concentrated) | 20 (well-calibrated) |
| **`n_long=30` represents** | 48% of universe | 15% of universe |
| **Max position weight** | 0.06 (needs widening at n=6) | 0.05 (guide's original — works at n=20) |

### Universe Construction Logic

```python
# data/universe.py

import pandas as pd

# Tier-1 large caps: top 200 S&P 500 by market cap
# Updated quarterly; hand-maintained or sourced from a single Wikipedia/Polygon pull

LARGE_CAP_200 = [
    # Top 200 S&P 500 constituents by market cap — refresh quarterly
    # Sourced from: https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
    # sorted by market cap (largest first), take top 200
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'GOOG', 'TSLA', 'AVGO',
    'JPM',  'LLY',  'V',    'UNH',  'XOM',  'MA',    'JNJ',  'HD',   'PG',
    'COST', 'ABBV', 'WMT',  'BAC',  'NFLX', 'CRM',   'AMD',  'ORCL', 'MRK',
    'CVX',  'KO',   'PEP',  'TMO',  'ACN',  'ADBE',  'TXN',  'LIN',  'MCD',
    'QCOM', 'WFC',  'DHR',  'ABT',  'PM',   'GE',    'CAT',  'NEE',  'IBM',
    'SPGI', 'MS',   'GS',   'RTX',  'AMGN', 'INTU',  'ISRG', 'NOW',  'AXP',
    'T',    'BKNG', 'PLD',  'VRTX', 'SYK',  'MDLZ',  'BLK',  'REGN', 'ADI',
    'DE',   'PANW', 'LRCX', 'KLAC', 'CI',   'MMC',   'MU',   'BSX',  'SCHW',
    'SO',   'ZTS',  'CB',   'CME',  'DUK',  'ICE',   'SHW',  'AON',  'GILD',
    'TJX',  'PH',   'EOG',  'SLB',  'COP',  'HUM',   'ELV',  'USB',  'MCK',
    'EQIX', 'APD',  'WM',   'MCO',  'PSA',  'NSC',   'ITW',  'ETN',  'FDX',
    'FCX',  'CSX',  'ECL',  'EMR',  'HCA',  'ROP',   'CTAS', 'OXY',  'DXCM',
    'MNST', 'AIG',  'PNC',  'TFC',  'COF',  'ADP',   'FICO', 'ORLY', 'HLT',
    'GWW',  'FTNT', 'EW',   'MSI',  'AFL',  'BDX',   'MET',  'BIIB', 'CPRT',
    'MCHP', 'KMB',  'CCI',  'STZ',  'ALL',  'PPG',   'EXC',  'CARR', 'NUE',
    'IDXX', 'TEL',  'CTSH', 'GLW',  'VMC',  'IQV',   'KEYS', 'VRSK', 'TROW',
    'MLM',  'DOV',  'LHX',  'CINF', 'GPN',  'OTIS',  'XYL',  'MTD',  'BR',
    'ROK',  'CDNS', 'SNPS', 'ANSS', 'DG',   'DLTR',  'WAT',  'CBOE', 'EXPD',
    'FAST', 'WRB',  'TDY',  'LDOS', 'POOL', 'CHD',   'JKHY', 'NDSN', 'HRL',
    'CF',   'RPM',  'BRO',  'MKC',  'NVR',  'CSGP',  'SNA',  'AWK',  'ATO',
    'PKG',  'ERIE', 'LNT',  'WEC',  'CHRW', 'MKTX',  'MOH',  'TRV',  'PAYX',
    # Sector ETFs kept as macro overlay (not in cross-sectional ranking)
]

SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLU', 'XLB', 'XLY', 'XLP', 'XLRE', 'XLC']
MACRO_ETFS  = ['SPY', 'QQQ', 'IWM', 'TLT', 'IEF', 'GLD']

def get_tradeable_stocks(date: str = None) -> list[str]:
    """Returns the 200 large-cap stocks for cross-sectional ranking."""
    # Phase 1: static list, refreshed quarterly
    # Phase 2: replace with point-in-time membership DB (see Section 5)
    return LARGE_CAP_200

def get_macro_instruments() -> list[str]:
    """ETFs used for macro overlay and regime inputs — not cross-sectionally ranked."""
    return SECTOR_ETFS + MACRO_ETFS
```

### Quarterly Refresh Process (Manual, Phase 1)

Until a full point-in-time membership database is built, refresh the list
quarterly:

```python
import requests
import pandas as pd
from bs4 import BeautifulSoup

def fetch_sp500_by_marketcap(top_n: int = 200) -> list[str]:
    """
    Scrapes the current S&P 500 list from Wikipedia and returns the
    top N tickers by market cap (approximate — Wikipedia's table is
    sorted by market cap already).
    Run quarterly; commit the updated list to config.
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(requests.get(url).text)
    sp500 = tables[0]   # first table: current constituents
    # Wikipedia table columns: Symbol, Security, GICS Sector, ...
    tickers = sp500['Symbol'].str.replace('.', '-').tolist()
    return tickers[:top_n]
```

---

## 2. All Outstanding Fixes (Re-calibrated for 200-Stock Universe)

---

### Fix 1 — `n_long` Ratio (Re-calibrated)

The ratio-based approach is correct. The mistake was the universe size, not
the ratio. At 200 stocks, 10% = 20 positions — well within the healthy range.

```yaml
# config/strategy.yaml
systematic_core:
  top_n_pct:           0.10    # 10% of ~200 = ~20 positions
  top_n_min:           10      # floor
  top_n_max:           30      # ceiling — keep it selective
  max_position_weight: 0.05    # 5% — works correctly at n=20
  max_sector_weight:   0.25    # 25%
```

```python
# decision/portfolio.py
def resolve_top_n(universe_size: int, config: dict) -> int:
    top_n_pct = config.get('top_n_pct', 0.10)
    top_n_min = config.get('top_n_min', 10)
    top_n_max = config.get('top_n_max', 30)
    return int(np.clip(universe_size * top_n_pct, top_n_min, top_n_max))

# For 200-stock universe: clip(200 * 0.10, 10, 30) = clip(20, 10, 30) = 20 ✓
# For 63-stock universe:  clip(63  * 0.10, 10, 30) = clip(6,  10, 30) = 10
# (Even on 63 symbols, the floor of 10 is safer than the previous 30 or the broken 6)
```

---

### Fix 2 — IC Validation Gate for `ml_alpha`

Unchanged from `systematic_core_improvements.md` except the validation period
now runs against the 200-stock universe. With more names, expect IC to be
measurably higher than the -0.012 measured on 63 symbols (more cross-sectional
dispersion = better signal separation).

```python
# alpha_engine/ml/validation.py  (no changes to code — same gate, new universe)

# Target IC thresholds at 200 stocks:
#   IC >= 0.05: strong signal — use at full weight
#   IC 0.03–0.05: acceptable — use at half weight
#   IC < 0.03: zero weight until signal is improved

def get_weight_from_ic(mean_ic: float, base_weight: float = 0.35) -> float:
    if mean_ic >= 0.05:
        return base_weight
    elif mean_ic >= 0.03:
        return base_weight * 0.5
    else:
        return 0.0
```

Re-run IC measurement on the 200-stock universe before the walk-forward.
The 5 additional features that were unavailable on the 63-symbol universe
(earnings revision, earnings surprise, short interest, insider buying,
options skew) should also be sourced now — see Section 3.

---

### Fix 3 — Replace Restricted FRED Series

No change from `systematic_core_improvements.md`. `BAA - AAA` remains the
correct free replacement for `BAMLH0A0HYM2`.

```python
# data/macro.py

def get_credit_spread(date: str) -> float:
    """
    Moody's BAA minus AAA spread — unrestricted on FRED, history to 1919.
    Functionally equivalent to BAMLH0A0HYM2 for regime classification.
    """
    baa = fred.get_series('BAA', observation_start=date, observation_end=date)
    aaa = fred.get_series('AAA', observation_start=date, observation_end=date)
    if baa.empty or aaa.empty:
        return 0.0   # fallback: neutral signal
    return float(baa.iloc[-1] - aaa.iloc[-1])
```

Also fix `build_regime_features` `dropna()` behaviour (fill macro columns with
0.0, only drop for price-series warmup gaps) — identical to the prior fix.

---

### Fix 4 — HMM Degeneracy Fallback

Identical to `systematic_core_improvements.md`. The degeneracy check bug
(off-diagonal zeros in `covars_` being mis-flagged) is already fixed in tests.
No change to the implementation. Reproduce here for completeness:

```python
# decision/regime.py

def _is_degenerate(self, model) -> bool:
    """
    Check only diagonal entries of covars_ for near-zero variance.
    Off-diagonal entries are structurally zero for 'diag' covariance type
    and must NOT be included in this check.
    """
    diag_variances = np.diagonal(model.covars_, axis1=-2, axis2=-1)  # (n_components, n_features)
    if (diag_variances < 1e-6).any():
        return True
    if (model.transmat_ < 0.01).any():
        return True
    if np.isnan(model.means_).any():
        return True
    return False
```

---

### Fix 5 — Walk-Forward Tune `GROSS_EXPOSURE_BY_REGIME`

The grid search stays the same. The finding from the last run (grid
consistently preferred 0.60 for BEAR_TREND on the 63-symbol universe)
may reverse on 200 stocks where the portfolio has more genuine alpha to
protect. Re-run the grid on the new universe:

```python
# Grid to search on validation folds
param_grid = {
    'BEAR_TREND':    [0.30, 0.45, 0.60],
    'CRISIS':        [0.20, 0.35],
    'TRANSITION':    [0.65, 0.75],
    'BULL_TREND':    [1.00],
    'LOW_VOL_GRIND': [0.90],
}
```

**Hypothesis**: With 20 concentrated positions on 200 stocks, the signal
has enough alpha to justify reducing to 0.30–0.45 in BEAR_TREND. The 63-symbol
result (preferred 0.60) was likely because the strategy had so little genuine
alpha that staying invested was the only return source.

---

### Fix 6 — REDUCE Tier in Risk Manager

Identical to `systematic_core_improvements.md`. Add the intermediate tier:

```python
# backtest/engine.py and live/runner.py

def check_portfolio_drawdown(self, portfolio_value: pd.Series) -> str:
    peak = portfolio_value.cummax()
    current_dd = (portfolio_value.iloc[-1] / peak.iloc[-1]) - 1

    if current_dd <= -0.15:
        return 'HALT'
    elif current_dd <= -0.08:
        return 'REDUCE'   # halve gross exposure until recovery
    elif current_dd >= -0.03:
        return 'NORMAL'   # recovery threshold
    else:
        return getattr(self, '_last_dd_state', 'NORMAL')

def apply_drawdown_action(self, weights: pd.Series, action: str) -> pd.Series:
    if action == 'HALT':
        return pd.Series(0.0, index=weights.index)
    elif action == 'REDUCE':
        return weights * 0.50
    return weights
```

---

## 3. New Signals Unlocked by the Larger Universe

With 200 stocks, data sourcing for the 5 missing signals becomes practical.
These were unavailable or too noisy at 63 symbols.

### 3.1 Earnings Revision (Free Proxy via EDGAR)

A paid analyst estimate feed is ideal but a reliable free proxy can be built
from SEC EDGAR filings:

```python
# signals/earnings_revision.py

class EarningsRevisionProxy:
    """
    Free proxy for analyst earnings revision using post-earnings price
    reaction as a forward return predictor (PEAD — Post-Earnings Announcement Drift).

    Logic: stocks with strong positive earnings surprises continue to drift
    upward for 3–6 months as the market slowly incorporates the information.
    Use the 3-day abnormal return around the earnings date as the proxy.
    """

    def compute(self, ticker: str, earnings_date: pd.Timestamp,
                 prices: pd.Series, market_returns: pd.Series) -> float:
        """
        Returns the 3-day cumulative abnormal return (CAR) around earnings.
        Positive CAR = positive revision proxy.
        """
        t = prices.index.get_loc(earnings_date, method='nearest')
        if t < 2 or t + 2 >= len(prices):
            return 0.0

        window = slice(t - 1, t + 2)    # [-1, 0, +1] around announcement
        stock_ret  = prices.iloc[window].pct_change().sum()
        market_ret = market_returns.iloc[window].sum()
        car = stock_ret - market_ret     # abnormal return
        return np.clip(car / 0.10, -1, 1)   # normalize to [-1, +1]
```

**Paid alternative**: Nasdaq Data Link SHARADAR SF3 (analyst estimates) at
$50/mo gives the actual consensus EPS revision — use this if SHARADAR
fundamentals are already purchased for Fix 2.

### 3.2 Short Interest (FINRA Bi-Weekly, Free)

FINRA publishes the short interest file for all US equities twice a month
(settlement dates around the 15th and end of month).

```python
# data/short_interest.py

import requests
import pandas as pd
from io import StringIO

def fetch_finra_short_interest(settlement_date: str) -> pd.DataFrame:
    """
    Downloads FINRA's bi-weekly short interest file.
    Free, no API key required.
    settlement_date format: 'YYYYMMDD'
    """
    url = (
        f"https://cdn.finra.org/equity/regsho/monthly/"
        f"CNMSshvol{settlement_date}.txt"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    df = pd.read_csv(
        StringIO(resp.text),
        sep='|',
        usecols=['Symbol', 'ShortInterest', 'DaysTocover', 'SettlementDate']
    )
    df.rename(columns={'Symbol': 'ticker'}, inplace=True)
    df['SettlementDate'] = pd.to_datetime(df['SettlementDate'])
    return df.set_index('ticker')

class ShortInterestSignal:
    """
    High short interest = crowded short → mean reversion potential (positive signal)
    OR  high conviction bearish positioning (negative signal).
    In practice, high short interest is a NEGATIVE signal in the cross-section
    (shorted stocks tend to underperform; short squeeze events are unpredictable).
    """

    def compute(self, date: str, universe: list[str]) -> pd.Series:
        # Use most recent file on or before the signal date (bi-weekly lag)
        si_data = self.get_most_recent_file(date)
        days_to_cover = si_data.reindex(universe)['DaysTocover'].fillna(0)
        # Negate: high days-to-cover = negative signal
        return cross_section_zscore(-days_to_cover)
```

### 3.3 Insider Buying (SEC EDGAR Form 4, Free)

```python
# data/insider.py

import requests
import pandas as pd
from xml.etree import ElementTree as ET

class InsiderBuyingSignal:
    """
    SEC Form 4 filings: insider open-market purchases vs. sales.
    Net buying (purchases - sales) as % of shares outstanding = positive signal.
    EDGAR RSS feed is free and near-real-time.
    """

    EDGAR_FORM4_RSS = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4&dateb=&owner=include&count=40&search_text=&output=atom"

    def fetch_recent_transactions(self, ticker: str,
                                   lookback_days: int = 90) -> pd.DataFrame:
        """
        Fetches recent Form 4 filings for a ticker from EDGAR full-text search.
        """
        url = (
            f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
            f"&dateRange=custom&startdt="
            f"{(pd.Timestamp.now() - pd.Timedelta(days=lookback_days)).strftime('%Y-%m-%d')}"
            f"&forms=4"
        )
        resp = requests.get(url, headers={'User-Agent': 'research@example.com'})
        if resp.status_code != 200:
            return pd.DataFrame()

        hits = resp.json().get('hits', {}).get('hits', [])
        transactions = []
        for hit in hits:
            source = hit.get('_source', {})
            transactions.append({
                'ticker':        ticker,
                'filed_date':    source.get('file_date'),
                'transaction_type': source.get('period_of_report'),
            })
        return pd.DataFrame(transactions)

    def compute(self, date: str, universe: list[str]) -> pd.Series:
        """
        Net insider purchase ratio across the universe.
        Positive = net buying (positive signal), negative = net selling.
        """
        scores = {}
        for ticker in universe:
            df = self.fetch_recent_transactions(ticker)
            if df.empty:
                scores[ticker] = 0.0
                continue
            buys  = (df['transaction_type'] == 'P').sum()   # P = Purchase
            sells = (df['transaction_type'] == 'S').sum()   # S = Sale
            total = buys + sells
            scores[ticker] = (buys - sells) / total if total > 0 else 0.0
        return cross_section_zscore(pd.Series(scores))
```

---

## 4. Updated Signal Composite Weights

With the IC gate, expanded universe, and new signals, the composite weights
need re-validation. Run IC measurements on the 200-stock universe and set
weights proportional to measured IC. Use these as starting defaults:

```yaml
# config/strategy.yaml — systematic_core signal weights
# ALL weights set to 0.0 until IC is validated on 200-stock universe
# Estimated weights based on academic literature; override with measured IC

systematic_core:
  signal_weights:
    momentum_12_1:       0.20   # most robust — largest weight by default
    earnings_surprise:   0.15   # PEAD — strong empirical support
    earnings_revision:   0.15   # revision momentum — strong but needs data
    gross_profitability: 0.12   # Novy-Marx quality — low decay
    low_vol:             0.10   # defensive — regime-dependent
    ml_alpha:            0.00   # START AT ZERO — set after IC validation
    book_to_price:       0.08   # value — low IC in growth regimes
    short_interest:      0.10   # crowded short avoidance
    insider_buying:      0.10   # high signal-to-noise on individual names
```

After running IC validation on the 200-stock universe:

```python
# Adjust weights based on measured IC
def set_weights_from_ic(measured_ics: dict) -> dict:
    """
    Proportional weighting: higher IC = higher weight.
    Zero weight for any signal with IC < 0.03.
    """
    eligible = {k: max(v, 0) for k, v in measured_ics.items() if v >= 0.03}
    total_ic = sum(eligible.values())
    if total_ic == 0:
        return {k: 0.0 for k in measured_ics}
    weights = {k: v / total_ic for k, v in eligible.items()}
    return weights
```

---

## 5. Point-in-Time Membership: Lightweight Version for 200 Stocks

A full membership database (Section 3 of `universe_expansion.md`) is the
eventual goal. For Phase 1 with 200 stocks, a simplified approach works:

```python
# data/membership_lite.py

import pandas as pd

class LiteMembershipTracker:
    """
    Lightweight point-in-time tracker for the 200-stock universe.
    Records manually-verified quarterly snapshots rather than every
    daily add/remove event.

    Good enough for monthly rebalancing; upgrade to full membership
    DB if backtesting at weekly or higher frequency.
    """

    def __init__(self, snapshot_csv: str):
        """
        snapshot_csv has columns: ticker, quarter_start, quarter_end, sector
        Each row = a ticker was in the universe for that quarter.
        Build this from quarterly Wikipedia snapshots.
        """
        self.df = pd.read_csv(snapshot_csv, parse_dates=['quarter_start', 'quarter_end'])

    def get_members(self, as_of_date: str) -> list[str]:
        dt = pd.Timestamp(as_of_date)
        mask = (self.df['quarter_start'] <= dt) & (self.df['quarter_end'] >= dt)
        return self.df[mask]['ticker'].tolist()

    def get_sector(self, ticker: str, as_of_date: str) -> str:
        dt = pd.Timestamp(as_of_date)
        mask = (
            (self.df['ticker'] == ticker) &
            (self.df['quarter_start'] <= dt) &
            (self.df['quarter_end'] >= dt)
        )
        rows = self.df[mask]
        return rows['sector'].iloc[0] if not rows.empty else 'Unknown'
```

**Building the snapshot CSV**: Run this once, commit to the repo, and
update quarterly:

```python
def build_quarterly_snapshots(start_year: int, end_year: int,
                                output_path: str):
    """
    Fetches S&P 500 composition for each quarter-start and writes snapshot CSV.
    Uses Wikipedia for the list; no paid data required.
    """
    rows = []
    for year in range(start_year, end_year + 1):
        for quarter_month in [1, 4, 7, 10]:
            date_str = f"{year}-{quarter_month:02d}-01"
            try:
                tickers = fetch_sp500_by_marketcap(top_n=200)   # from Section 1
                quarter_end = pd.Timestamp(date_str) + pd.DateOffset(months=3) - pd.Timedelta(days=1)
                for t in tickers:
                    rows.append({
                        'ticker':        t,
                        'quarter_start': date_str,
                        'quarter_end':   quarter_end.strftime('%Y-%m-%d'),
                        'sector':        get_gics_sector(t),   # from Polygon ticker details
                    })
            except Exception as e:
                print(f"  Skipped {date_str}: {e}")
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved {len(rows)} rows to {output_path}")
```

---

## 6. Sequenced Implementation Plan

```
Week 1 — Universe & Data

  Day 1:
    [ ] Finalize LARGE_CAP_200 ticker list (top 200 S&P 500 by market cap)
    [ ] Build quarterly snapshot CSV back to 2005 (LiteMembershipTracker)
    [ ] Verify: len(get_members('2009-03-10')) ≈ 190–200

  Day 2–3:
    [ ] Download price data for all 200 tickers via Polygon
    [ ] Build Parquet store (daily, 2000–present)
    [ ] Run PriceDataValidator — fix all issues

  Day 4:
    [ ] Download SHARADAR SF1 fundamentals for 200 tickers (or SimFin free)
    [ ] Verify filing lag is applied (datekey + 45/60 days)
    [ ] Spot-check 5 tickers for point-in-time correctness

  Day 5:
    [ ] FINRA short interest: download all available bi-weekly files
    [ ] SEC EDGAR Form 4 pipeline: test on 10 tickers
    [ ] Replace BAMLH0A0HYM2 with BAA - AAA (Fix 3)

Week 2 — Signal Validation

  Day 6–7:
    [ ] Run IC validation for ALL signals on 200-stock universe
        Validation period: 2018-01-01 to 2020-12-31 (held out from walk-forward)
    [ ] Record mean_IC and IC_IR for each signal
    [ ] Set signal weights proportional to IC (zero any IC < 0.03)
    [ ] ml_alpha: starts at weight=0.00; re-enable only if IC >= 0.03

  Day 8:
    [ ] Confirm cross-sectional signal spread has improved vs. 63 symbols
        (IC should be meaningfully higher on 200 stocks)

Week 3 — System Fixes

  Day 9:
    [ ] Fix 1: top_n_pct=0.10 with min=10, max=30 — confirm resolves to 20
    [ ] Fix 4: HMM degeneracy check with corrected diagonal-only variance check
    [ ] Fix 4: VIX+MA fallback implemented and tested
    [ ] Fix 6: REDUCE tier added to risk manager

  Day 10:
    [ ] Fix 5: Run GROSS_EXPOSURE grid search on validation folds (2018–2020)
    [ ] Select best BEAR_TREND and CRISIS floors
    [ ] Lock config with tuned values

Week 4 — Walk-Forward & Decision

  Day 11–13:
    [ ] Run full walk-forward 2008–2024, 3 window configs
    [ ] Record per-window Sharpe, CAGR, MaxDD vs. Traditional (baseline: 1.227)

  Day 14–15:
    [ ] Interpret results:
        Sharpe >= 1.15: success — proceed to live paper trading
        Sharpe 1.00–1.15: meaningful improvement over post-fix 0.73
                          — tune signal weights further, recheck ml_alpha IC
        Sharpe < 1.00: universe size alone is not the fix
                       — revisit architecture (time-series momentum, sector rotation)
```

---

## 7. Success Criteria

| Metric | Minimum Bar | Target |
|---|---|---|
| Avg OOS Sharpe (3 window configs) | > 1.00 | > 1.15 |
| % walk-forward folds with positive Sharpe | > 70% | > 85% |
| Avg OOS CAGR | > 10% | > 14% |
| Avg Max Drawdown | < -15% | < -12% |
| Sharpe in 5yr/2yr window (contains 2022) | > 0.80 | > 1.00 |
| Cross-sectional IC improvement over 63-symbol baseline | > +0.01 | > +0.03 |

---

*Supersedes the full S&P 500 expansion plan in `universe_expansion.md`.*
*Read alongside `systematic_trading_engineering_guide.md` and*
*`systematic_core_improvements.md`.*
