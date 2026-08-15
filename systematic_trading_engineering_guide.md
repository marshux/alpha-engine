# Systematic Trading System Engineering Guide
### Building a Non-HFT Alpha-Generating System to Outperform SPY

---

> **Audience**: Software/quant engineers building a systematic, non-high-frequency trading system from scratch.  
> **Scope**: Equities-first, extendable to ETFs, futures, options. Holding periods: 1 day to 3 months.  
> **Goal**: Risk-adjusted outperformance of SPY (S&P 500 ETF) with Sharpe > 1.0, max drawdown < 20%.

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Securities Universe — What to Trade](#2-securities-universe--what-to-trade)
3. [Part I — Non-AI / ML-Assisted System](#part-i--non-ai--ml-assisted-system)
   - 3.1 [Data Infrastructure](#31-data-infrastructure)
   - 3.2 [Signal Engineering — Technical & Price-Based](#32-signal-engineering--technical--price-based)
   - 3.3 [Signal Engineering — Fundamental & Macro](#33-signal-engineering--fundamental--macro)
   - 3.4 [Signal Engineering — Sentiment & Options Flow](#34-signal-engineering--sentiment--options-flow)
   - 3.5 [ML-Based Signal Models](#35-ml-based-signal-models)
   - 3.6 [Regime Detection & Adaptation](#36-regime-detection--adaptation)
   - 3.7 [Portfolio Construction](#37-portfolio-construction)
   - 3.8 [Risk Management](#38-risk-management)
   - 3.9 [Execution Layer](#39-execution-layer)
4. [Part II — AI Augmentation Layer](#part-ii--ai-augmentation-layer)
   - 4.1 [LLM-Powered NLP Signals](#41-llm-powered-nlp-signals)
   - 4.2 [LLM for Research Automation](#42-llm-for-research-automation)
   - 4.3 [Reinforcement Learning for Portfolio Optimization](#43-reinforcement-learning-for-portfolio-optimization)
   - 4.4 [Adaptive Signal Weighting with Online Learning](#44-adaptive-signal-weighting-with-online-learning)
   - 4.5 [AI Infrastructure & Safety Rails](#45-ai-infrastructure--safety-rails)
5. [Part III — Effective Backtesting](#part-iii--effective-backtesting)
   - 5.1 [Backtesting Architecture](#51-backtesting-architecture)
   - 5.2 [Data Quality & Bias Elimination](#52-data-quality--bias-elimination)
   - 5.3 [Walk-Forward Validation](#53-walk-forward-validation)
   - 5.4 [Transaction Cost & Slippage Modeling](#54-transaction-cost--slippage-modeling)
   - 5.5 [Statistical Significance & Overfitting Detection](#55-statistical-significance--overfitting-detection)
   - 5.6 [Regime Robustness Testing](#56-regime-robustness-testing)
   - 5.7 [Backtesting Checklist](#57-backtesting-checklist)
6. [Appendix — Tech Stack Reference](#6-appendix--tech-stack-reference)

---

## 1. System Architecture Overview

The system is composed of six loosely-coupled layers. Each layer is independently testable and replaceable.

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│   Market Data · Fundamental · Macro · Alt Data · Options        │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      SIGNAL LAYER                               │
│   Technical Signals · Factor Signals · ML Models · LLM NLP     │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      REGIME LAYER                               │
│   Market Regime Classifier · Signal Weight Adjuster             │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                   PORTFOLIO CONSTRUCTION LAYER                  │
│   Alpha Combination · Position Sizing · Constraint Solver       │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    RISK MANAGEMENT LAYER                        │
│   Exposure Limits · Drawdown Monitor · Hedging Engine           │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      EXECUTION LAYER                            │
│   Order Router · Slippage Model · Broker API · Audit Log        │
└─────────────────────────────────────────────────────────────────┘
```

**Key design principles:**
- Every signal produces a **z-scored, rank-normalized alpha score** in `[-1, +1]`. This common currency allows signals to be combined, weighted, and switched without code changes.
- All data transformations are **point-in-time correct** — no future data can leak into any computation.
- The system runs in two modes: **research mode** (full history, fast vectorized) and **live mode** (incremental, event-driven, one bar at a time).

---

## 2. Securities Universe — What to Trade

Choosing the right universe is as important as the signals themselves. The universe must balance **liquidity**, **signal opportunity**, and **trading cost**.

### 2.1 Equity Universe Tiers

| Tier | Universe | Avg Daily Volume | Capacity | Best For |
|------|----------|-----------------|----------|----------|
| **Tier 1** | S&P 500 constituents (~500 stocks) | > $50M | $100M+ | Low slippage, clean data, institutional |
| **Tier 2** | Russell 1000 (add mid-cap) | > $10M | $20M–$100M | More alpha opportunity, moderate cost |
| **Tier 3** | Russell 2000 small-caps | > $2M | < $20M | High alpha potential, high friction |
| **ETF Layer** | Sector/factor ETFs (XLK, XLF, IWM, QQQ) | > $100M | Unlimited | Macro/regime plays, hedging |

**Recommendation for a new system**: Start with the **S&P 500 universe**. It provides the cleanest data, lowest slippage, and enough cross-sectional diversity for factor and momentum strategies. Expand to Russell 1000 once the system is validated.

### 2.2 Universe Construction Rules

```python
# Universe filter applied at each rebalance date (point-in-time)
def construct_universe(date, min_price=5.0, min_adv=10e6, min_market_cap=500e6):
    """
    Returns eligible tickers for a given date.
    Uses point-in-time index membership to avoid survivorship bias.
    """
    candidates = get_index_members(index='SP500', date=date)  # historical membership
    prices = get_adjusted_prices(candidates, date)
    adv = get_avg_daily_volume(candidates, date, lookback=20)  # 20-day ADV
    mktcap = get_market_cap(candidates, date)

    mask = (
        (prices >= min_price) &          # avoid penny stocks
        (adv >= min_adv) &               # liquidity filter
        (mktcap >= min_market_cap) &     # size filter
        (~is_recent_ipo(candidates, date, days=90))  # exclude new IPOs
    )
    return candidates[mask]
```

**Critical**: Always use a **point-in-time index membership database** (e.g., Compustat or FactSet historical constituents). Using today's S&P 500 membership to backtest 2010 introduces severe survivorship bias — you will only be trading companies that survived to today.

### 2.3 Extended Instruments

Once the equity core is running, consider adding:

- **Equity index futures** (ES, NQ, RTY): cheap, liquid macro/regime overlay
- **Sector ETFs**: XLK, XLF, XLE, XLV — for sector rotation strategies
- **VIX futures / UVXY / SVXY**: volatility regime plays
- **Treasury ETFs** (TLT, IEF, SHY): flight-to-quality signals and hedging
- **Gold / commodity ETFs** (GLD, USO): macro diversification

---

## Part I — Non-AI / ML-Assisted System

---

## 3.1 Data Infrastructure

### Data Sources and Priorities

> **Corrected 2026 — Free Data Availability Is Overstated**: The original
> version of this table listed "FRED (free)" for macro data without
> qualification. In practice, **not every FRED series is equally free**: some
> (e.g. ICE-provided commercial index data) restrict anonymous/free CSV access
> to a rolling recent window — confirmed directly against FRED's endpoint after
> `BAMLH0A0HYM2` (ICE BofA US High Yield Index OAS) silently returned only ~3
> years of history instead of its full multi-decade series, which in turn
> silently broke a regime classifier's entire feature panel for years of
> otherwise-valid data (a `dropna()` across all feature columns discarded every
> row touching the gap — see `systematic_core_postmortem.md`). The table below
> replaces the vague "FRED (free)" entry with specific, verified series IDs and
> flags which are actually unrestricted vs. which are not.

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

### Data Pipeline Architecture

```
Raw Sources ──► Ingestion ──► Validation ──► Normalization ──► Feature Store
                (async)       (schema,        (point-in-time    (columnar,
                              outlier,         adjustments,      partitioned
                              gap checks)      z-scoring)        by date)
```

**Storage recommendation**:  
- **Time-series data**: TimescaleDB (PostgreSQL extension) or ClickHouse for analytical queries  
- **Feature store**: Parquet files partitioned by date, stored locally or on S3  
- **Symbol master**: Relational DB mapping tickers ↔ CUSIPs ↔ company identifiers across time (critical for corporate actions, ticker changes, mergers)

### Point-in-Time Data Integrity

```python
class PointInTimeDataStore:
    """
    Every record has an 'as_of_date' (when the data was known to the market)
    and a 'valid_from'/'valid_to' for the value itself.
    Prevents look-ahead bias at the data level.
    """
    def get(self, ticker: str, field: str, as_of_date: date) -> float:
        return self.db.query(
            "SELECT value FROM data "
            "WHERE ticker = ? AND field = ? "
            "AND as_of_date <= ? "
            "ORDER BY as_of_date DESC LIMIT 1",
            (ticker, field, as_of_date)
        )
```

---

## 3.2 Signal Engineering — Technical & Price-Based

All signals are computed per-security and then **cross-sectionally ranked and z-scored** within the universe. This removes level effects and makes signals comparable across securities and time.

### Signal 1: Cross-Sectional Momentum (12-1 Month)

The most robust and academically validated signal. Buys the past 12-month winners (skipping the most recent month to avoid short-term reversal).

```python
def momentum_12_1(prices: pd.DataFrame, date: date) -> pd.Series:
    """
    Returns cross-sectional momentum signal for all tickers.
    prices: adjusted close, columns=tickers, index=dates
    """
    t = prices.index.get_loc(date)
    # 12-month return, excluding last month (reversal avoidance)
    ret_12m = prices.iloc[t - 252] / prices.iloc[t - 21] - 1  # ~252 trading days, ~21 days
    return cross_section_zscore(ret_12m)

def cross_section_zscore(series: pd.Series) -> pd.Series:
    """Rank-based z-score: robust to outliers."""
    ranked = series.rank(pct=True)          # 0 to 1
    return (ranked - 0.5) * 2              # -1 to +1
```

**Variants worth implementing**:
- **Short-term reversal** (1-week): opposite sign — recent losers outperform
- **Industry-adjusted momentum**: subtract industry mean return to get stock-specific momentum
- **Residual momentum**: regress returns against Fama-French 3 factors, use the residual

### Signal 2: Trend-Following (Time-Series Momentum)

Applied per-stock or per-ETF. Go long when the asset is above its trend, short or flat when below.

```python
def trend_signal(prices: pd.Series, fast=50, slow=200) -> float:
    """
    Returns +1 (long), 0 (flat), or -1 (short) based on MA crossover.
    Smoothed version: use the spread as a continuous signal.
    """
    sma_fast = prices.rolling(fast).mean().iloc[-1]
    sma_slow = prices.rolling(slow).mean().iloc[-1]
    spread = (sma_fast - sma_slow) / sma_slow
    return np.clip(spread * 10, -1, 1)  # scale to [-1, +1]

def time_series_momentum(prices: pd.Series, lookback=252) -> float:
    """TSMOM: sign of past 12-month return, scaled by inverse volatility."""
    ret = prices.iloc[-1] / prices.iloc[-lookback] - 1
    vol = prices.pct_change().rolling(60).std().iloc[-1] * np.sqrt(252)
    return np.sign(ret) * (0.40 / vol)  # volatility-scaled position
```

### Signal 3: Mean Reversion (Short-Term)

Effective at the weekly frequency within sectors — stocks that deviated sharply from peers tend to revert.

```python
def mean_reversion_signal(prices: pd.DataFrame, window=5) -> pd.Series:
    """
    Cross-sectional short-term reversal signal.
    Stocks with largest recent loss get highest positive score.
    """
    recent_return = prices.pct_change(window).iloc[-1]
    # Reversal: negate the momentum signal
    return cross_section_zscore(-recent_return)

def bollinger_band_signal(prices: pd.Series, window=20, n_std=2) -> float:
    """Continuous Bollinger Band position signal."""
    rolling_mean = prices.rolling(window).mean()
    rolling_std = prices.rolling(window).std()
    upper = rolling_mean + n_std * rolling_std
    lower = rolling_mean - n_std * rolling_std
    current = prices.iloc[-1]
    # Normalize: +1 at lower band, -1 at upper band
    band_range = upper.iloc[-1] - lower.iloc[-1]
    if band_range == 0:
        return 0.0
    return np.clip(2 * (rolling_mean.iloc[-1] - current) / band_range, -1, 1)
```

### Signal 4: Volatility & Quality Filters

Low-volatility anomaly — lower-beta stocks historically outperform on a risk-adjusted basis.

```python
def low_volatility_signal(returns: pd.DataFrame, window=60) -> pd.Series:
    """Rank stocks by inverse realized volatility (annualized)."""
    vol = returns.rolling(window).std() * np.sqrt(252)
    inv_vol = 1 / vol.iloc[-1]
    return cross_section_zscore(inv_vol)

def idiosyncratic_volatility(returns: pd.DataFrame, 
                              market_returns: pd.Series, window=60) -> pd.Series:
    """
    Idiosyncratic vol = residual vol after removing market beta.
    Low idiosyncratic vol is a positive signal.
    """
    results = {}
    for ticker in returns.columns:
        y = returns[ticker].dropna().tail(window)
        x = market_returns.reindex(y.index).dropna()
        aligned = pd.concat([y, x], axis=1).dropna()
        beta = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1] / np.var(aligned.iloc[:, 1])
        residuals = aligned.iloc[:, 0] - beta * aligned.iloc[:, 1]
        results[ticker] = residuals.std() * np.sqrt(252)
    return cross_section_zscore(pd.Series(results) * -1)  # negate: lower ivol = better
```

---

## 3.3 Signal Engineering — Fundamental & Macro

### Fundamental Signals (Require Point-in-Time Data)

```python
class FundamentalSignals:
    """
    All ratios computed using TTM (trailing twelve months) financials
    as known on the signal date — NOT restated figures.
    """

    def earnings_yield(self, date) -> pd.Series:
        """E/P ratio — value signal. High E/P = cheap stock."""
        eps_ttm = self.store.get_ttm('eps', date)
        price = self.store.get('adjusted_close', date)
        ep = eps_ttm / price
        return cross_section_zscore(ep)

    def earnings_revision(self, date, lookback_months=3) -> pd.Series:
        """
        Change in analyst consensus EPS estimate over past N months.
        Positive revision = positive signal (earnings momentum).
        """
        eps_now = self.store.get('consensus_eps_fwd12m', date)
        eps_prior = self.store.get('consensus_eps_fwd12m', 
                                    date - relativedelta(months=lookback_months))
        revision = (eps_now - eps_prior) / abs(eps_prior).clip(lower=0.01)
        return cross_section_zscore(revision)

    def earnings_surprise(self, date) -> pd.Series:
        """
        Post-earnings announcement drift (PEAD).
        Actual EPS vs consensus — buy surprises, sell misses.
        """
        actual = self.store.get('eps_actual_last_quarter', date)
        estimate = self.store.get('eps_estimate_last_quarter', date)
        surprise_pct = (actual - estimate) / abs(estimate).clip(lower=0.01)
        return cross_section_zscore(surprise_pct)

    def return_on_equity(self, date) -> pd.Series:
        """Profitability factor — high ROE stocks outperform."""
        net_income = self.store.get_ttm('net_income', date)
        equity = self.store.get('book_equity', date)
        roe = net_income / equity.clip(lower=1e6)
        return cross_section_zscore(roe)

    def gross_profitability(self, date) -> pd.Series:
        """Novy-Marx gross profitability — (Revenue - COGS) / Total Assets."""
        rev = self.store.get_ttm('revenue', date)
        cogs = self.store.get_ttm('cogs', date)
        assets = self.store.get('total_assets', date)
        gp = (rev - cogs) / assets.clip(lower=1e6)
        return cross_section_zscore(gp)

    def book_to_price(self, date) -> pd.Series:
        """Classic value signal. High B/P = cheap."""
        book = self.store.get('book_equity_per_share', date)
        price = self.store.get('adjusted_close', date)
        bp = book / price.clip(lower=0.01)
        return cross_section_zscore(bp)
```

### Macro / Market-Level Signals

These signals govern the **overall market exposure** (beta) of the portfolio — they are not cross-sectional but time-series.

```python
class MacroSignals:

    def yield_curve_slope(self, date) -> float:
        """10Y - 2Y Treasury spread. Inverted = risk-off signal."""
        t10 = self.fred.get('DGS10', date)
        t2  = self.fred.get('DGS2', date)
        slope = t10 - t2
        # Normalize: historically -0.5% to +3%
        return np.clip((slope - 1.25) / 1.0, -1, 1)

    def vix_regime(self, date) -> float:
        """
        Market fear gauge. 
        VIX < 15: risk-on (+1), 15-25: neutral (0), > 25: risk-off (-1).
        Use smoothed VIX (21-day MA) to avoid noise spikes.
        """
        vix = self.market_data.get('VIX', date, lookback=21).mean()
        if vix < 15:
            return 1.0
        elif vix < 25:
            return (25 - vix) / 10 * 2 - 1  # linear interpolation
        else:
            return -1.0

    def credit_spread(self, date) -> float:
        """
        HY-IG spread (or BAA-AAA). Widening spreads = risk-off.
        Use FRED series: BAMLH0A0HYM2 (HY spread).
        """
        spread = self.fred.get('BAMLH0A0HYM2', date)
        spread_ma = self.fred.get_rolling('BAMLH0A0HYM2', date, window=63)  # 3-month MA
        # Signal: is spread widening (bad) or tightening (good)?
        spread_change = spread - spread_ma
        return np.clip(-spread_change / 1.5, -1, 1)

    def market_breadth(self, date) -> float:
        """
        % of S&P 500 stocks above 200-day MA. 
        > 70% = strong trend (bullish), < 30% = oversold (mean reversion opportunity).
        """
        pct_above_200ma = self.compute_breadth(date, ma_period=200)
        return (pct_above_200ma - 0.5) * 2  # scale to [-1, +1]
```

---

## 3.4 Signal Engineering — Sentiment & Options Flow

### Sentiment Signals

```python
class SentimentSignals:

    def short_interest_signal(self, date) -> pd.Series:
        """
        High short interest = potential short squeeze OR bearish signal.
        Use as a NEGATIVE signal (crowded shorts underperform).
        FINRA publishes bi-weekly; interpolate for daily use.
        """
        short_ratio = self.store.get('short_interest_ratio', date)  # days to cover
        # High short ratio = negative alpha signal
        return cross_section_zscore(-short_ratio)

    def insider_buying(self, date, lookback_days=90) -> pd.Series:
        """
        Net insider purchases (buys - sells) as % of shares outstanding.
        SEC Form 4 filings — point-in-time available.
        Positive net buying = positive signal.
        """
        net_buys = self.store.get_insider_net_buys(date, lookback_days)
        shares = self.store.get('shares_outstanding', date)
        insider_ratio = net_buys / shares.clip(lower=1e6)
        return cross_section_zscore(insider_ratio)

    def put_call_ratio_signal(self, date, window=20) -> float:
        """
        Equity put/call ratio. CONTRARIAN signal.
        High P/C ratio (excessive fear) = bullish for market.
        """
        pc_ratio = self.options_data.get_equity_put_call_ratio(date)
        pc_ma = self.options_data.get_rolling_pc_ratio(date, window)
        # Deviation from mean: high P/C = buy signal
        deviation = pc_ratio - pc_ma
        return np.clip(-deviation / 0.2, -1, 1)  # normalized

    def options_implied_skew(self, date) -> pd.Series:
        """
        25-delta put skew (IV of OTM put - IV of ATM).
        High skew = market pricing in downside risk = bearish.
        """
        skew = self.options_data.get_skew_25d(date)
        return cross_section_zscore(-skew)
```

---

## 3.5 ML-Based Signal Models

Machine learning is used here as a **signal combiner and nonlinear feature extractor** — not a black-box replacement for the rules above. The output is still a normalized alpha score.

### Architecture: Two-Stage ML Pipeline

```
Stage 1 (Feature Engineering):
  Raw signals ──► Interaction terms ──► Rolling statistics ──► Lag features
                                                                    │
Stage 2 (Model):                                                    ▼
  XGBoost / LightGBM ──► Predict forward 1-month excess return ──► Alpha Score
```

### Feature Construction

```python
class MLFeatureBuilder:
    """
    Builds the feature matrix X for training and inference.
    Target y = forward 21-day excess return vs. SPY (point-in-time).
    """

    def build_features(self, ticker: str, date: date) -> dict:
        signals = self.signal_engine.compute_all(ticker, date)

        features = {
            # Raw signals
            'momentum_12_1':        signals['momentum_12_1'],
            'momentum_1m':          signals['momentum_1m'],
            'low_vol':              signals['low_vol'],
            'earnings_yield':       signals['earnings_yield'],
            'earnings_revision':    signals['earnings_revision'],
            'earnings_surprise':    signals['earnings_surprise'],
            'gross_profitability':  signals['gross_profitability'],
            'book_to_price':        signals['book_to_price'],
            'short_interest':       signals['short_interest'],
            'insider_buying':       signals['insider_buying'],

            # Interaction features (often predictive beyond individual signals)
            'momentum_x_quality':   signals['momentum_12_1'] * signals['gross_profitability'],
            'value_x_revision':     signals['book_to_price'] * signals['earnings_revision'],
            'momentum_x_low_vol':   signals['momentum_12_1'] * signals['low_vol'],

            # Temporal features
            'month_of_year':        date.month / 12.0,    # seasonality
            'is_january':           1.0 if date.month == 1 else 0.0,

            # Macro context features (same value for all stocks, but regime info)
            'vix_regime':           signals['vix_regime'],
            'yield_curve':          signals['yield_curve_slope'],
            'credit_spread_signal': signals['credit_spread'],
        }
        return features

    def build_training_set(self, start: date, end: date, 
                            forward_horizon: int = 21) -> tuple:
        X_rows, y_rows = [], []
        for dt in get_trading_days(start, end):
            universe = self.universe_builder.construct(dt)
            for ticker in universe:
                features = self.build_features(ticker, dt)
                # Target: forward excess return vs SPY (point-in-time)
                fwd_ret = self.get_forward_excess_return(ticker, dt, forward_horizon)
                if fwd_ret is not None:
                    X_rows.append(features)
                    y_rows.append(fwd_ret)
        return pd.DataFrame(X_rows), pd.Series(y_rows)
```

### Model Training

```python
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

class AlphaModel:
    """
    LightGBM model trained on cross-sectional features.
    Predicts forward excess return (rank-based target for robustness).
    """

    def __init__(self, n_estimators=500, learning_rate=0.05,
                 max_depth=4, min_child_samples=50):
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,             # shallow = less overfit
            min_child_samples=min_child_samples,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=1.0,                  # L2 regularization
            n_jobs=-1
        )

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        IMPORTANT: Use rank-transformed target (IC optimization).
        Predicting ranks is more robust than predicting raw returns.
        """
        y_rank = y.rank(pct=True)  # 0 to 1 cross-sectional rank
        self.model.fit(
            X, y_rank,
            eval_set=[(X.tail(5000), y_rank.tail(5000))],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Returns normalized alpha scores."""
        raw_pred = pd.Series(self.model.predict(X), index=X.index)
        return cross_section_zscore(raw_pred)

    def feature_importance(self) -> pd.DataFrame:
        """Useful for signal attribution and debugging."""
        return pd.DataFrame({
            'feature': self.model.feature_name_,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
```

### Signal Combination

```python
class SignalCombiner:
    """
    Combines individual signals into a composite alpha score.
    Weights are regime-dependent and updated periodically.
    """

    DEFAULT_WEIGHTS = {
        'momentum_12_1':        0.20,
        'earnings_revision':    0.15,
        'earnings_surprise':    0.15,
        'gross_profitability':  0.12,
        'low_vol':              0.10,
        'ml_alpha':             0.15,
        'book_to_price':        0.08,
        'short_interest':       0.05,
    }

    def combine(self, signals: dict, weights: dict = None) -> pd.Series:
        if weights is None:
            weights = self.DEFAULT_WEIGHTS
        composite = pd.Series(0.0, index=next(iter(signals.values())).index)
        for signal_name, weight in weights.items():
            if signal_name in signals:
                composite += weight * signals[signal_name]
        return cross_section_zscore(composite)
```

---

## 3.6 Regime Detection & Adaptation

A single static signal weighting will fail during regime changes. The system must detect the current market regime and adapt signal weights and portfolio exposure accordingly.

### Regime Classifier

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
>
> Confirmed in practice (alpha-engine's `decision/regime.py`): on real 2011 and
> 2022 market data, `GaussianHMM.fit()` produced collapsed transition matrices
> (a component the Viterbi path never visits) that only surfaced as a crash on
> the *next* `.predict()` call (`ValueError: startprob_ must sum to 1 (got nan)`),
> not at fit time. A proactive degeneracy check immediately after `fit()` — not
> reliance on hmmlearn's own internal validation — is what actually catches this
> before it reaches production. See `systematic_core_postmortem.md` and
> `systematic_core_improvements.md` for the full incident writeup.

```python
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
import numpy as np

class MarketRegimeClassifier:
    """
    Identifies market regime using a Hidden Markov Model on market features.
    Regimes: BULL_TREND, BEAR_TREND, HIGH_VOL, LOW_VOL_GRIND, CRISIS
    """

    def __init__(self, n_regimes=4):
        self.model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type='full',
            n_iter=200
        )

    def build_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        Market-level features for regime classification.
        All features are normalized and point-in-time safe.
        """
        features = pd.DataFrame({
            'spy_return_1m':    market_data['SPY'].pct_change(21),
            'spy_return_3m':    market_data['SPY'].pct_change(63),
            'realized_vol_1m':  market_data['SPY'].pct_change().rolling(21).std() * np.sqrt(252),
            'vix_level':        market_data['VIX'] / 100,
            'vix_change_1m':    market_data['VIX'].pct_change(21),
            'yield_curve':      market_data['T10Y2Y'],         # FRED series
            'credit_spread':    market_data['BAMLH0A0HYM2'],
            'breadth_200ma':    market_data['pct_above_200ma'],
        }).dropna()
        return features.values

    def fit(self, market_data: pd.DataFrame):
        X = self.build_features(market_data)
        self.model.fit(X)
        self.regime_labels = self.interpret_regimes(X)

    def predict_current(self, market_data: pd.DataFrame) -> str:
        X = self.build_features(market_data)
        regime_idx = self.model.predict(X)[-1]
        return self.regime_labels[regime_idx]

    def interpret_regimes(self, X: np.ndarray) -> dict:
        """
        After fitting, label each regime by inspecting its mean feature vector.
        Returns mapping from regime index to descriptive label.
        """
        means = self.model.means_
        regime_map = {}
        for i, mean in enumerate(means):
            ret_1m, _, vol_1m, vix, _, yc, cs, breadth = mean
            if ret_1m > 0.02 and vol_1m < 0.15:
                regime_map[i] = 'BULL_TREND'
            elif ret_1m < -0.02 and vol_1m > 0.20:
                regime_map[i] = 'BEAR_TREND'
            elif vix > 0.30 or vol_1m > 0.25:
                regime_map[i] = 'CRISIS'
            elif ret_1m > 0 and vol_1m < 0.12:
                regime_map[i] = 'LOW_VOL_GRIND'
            else:
                regime_map[i] = 'TRANSITION'
        return regime_map
```

### Regime-Adaptive Signal Weights

```python
REGIME_WEIGHTS = {
    'BULL_TREND': {
        'momentum_12_1':       0.30,   # momentum works best in trends
        'earnings_revision':   0.20,
        'gross_profitability': 0.15,
        'ml_alpha':            0.20,
        'book_to_price':       0.05,
        'low_vol':             0.10,
    },
    'BEAR_TREND': {
        'momentum_12_1':       0.10,   # momentum fragile in bear markets
        'earnings_revision':   0.15,
        'gross_profitability': 0.20,
        'ml_alpha':            0.15,
        'book_to_price':       0.20,   # value more defensive
        'low_vol':             0.20,   # low vol shines defensively
    },
    'CRISIS': {
        'momentum_12_1':       0.05,
        'earnings_revision':   0.05,
        'gross_profitability': 0.15,
        'ml_alpha':            0.05,
        'book_to_price':       0.10,
        'low_vol':             0.30,   # defense dominates
        'cash_buffer':         0.30,   # reduce gross exposure
    },
    'LOW_VOL_GRIND': {
        'momentum_12_1':       0.20,
        'earnings_revision':   0.20,
        'gross_profitability': 0.15,
        'ml_alpha':            0.25,
        'book_to_price':       0.10,
        'low_vol':             0.10,
    }
}
```

### Portfolio Beta Overlay

```python
class RegimeOverlay:
    """
    Scales total portfolio gross exposure based on regime.
    In crisis, reduces beta to SPY significantly.
    """

    GROSS_EXPOSURE_BY_REGIME = {
        'BULL_TREND':   1.0,    # fully invested
        'LOW_VOL_GRIND': 0.9,
        'TRANSITION':   0.75,
        'BEAR_TREND':   0.60,
        'CRISIS':       0.35,   # heavily defensive
    }

    def apply(self, target_positions: pd.Series, regime: str) -> pd.Series:
        scale = self.GROSS_EXPOSURE_BY_REGIME.get(regime, 0.75)
        return target_positions * scale
```

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
>
> Confirmed in practice: alpha-engine's Systematic Core strategy, running these
> exact illustrative floors unmodified, held 60% gross exposure through the
> entirety of the 2022 bear market once its regime classifier was fixed to
> actually detect it — a real, material drag vs. a binary all-cash flatten for
> that period. The floors themselves were never independently re-validated for
> that strategy's universe before this was discovered. See
> `systematic_core_postmortem.md`.

---

## 3.7 Portfolio Construction

### From Alpha Scores to Position Weights

```python
class PortfolioConstructor:
    """
    Converts composite alpha scores into target portfolio weights.
    Uses mean-variance optimization with alpha tilt and constraints.
    """

    def __init__(self, n_long=50, n_short=0, target_vol=0.10,
                 max_position=0.05, sector_max=0.25):
        self.n_long = n_long
        self.target_vol = target_vol     # annualized target vol
        self.max_position = max_position  # 5% max per stock
        self.sector_max = sector_max      # 25% max per sector
```

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
>
> Confirmed in practice: alpha-engine's Systematic Core strategy originally used
> a hardcoded `top_n=30` against a 63-symbol universe — holding ~48% of every
> available name, not a comparable level of selectivity to this section's own
> 10% ratio. Switching to a ratio-based `top_n_pct`/`top_n_min` (with the same
> 5-position floor and 10% default) was one of the higher-leverage fixes applied
> after this was flagged. See `systematic_core_postmortem.md`.

```python

    def construct(self, alpha: pd.Series, 
                  cov_matrix: pd.DataFrame,
                  sector_map: dict) -> pd.Series:
        """
        Simple long-only alpha-weighted portfolio with constraints.
        For long/short, alpha signs drive long vs. short.
        """
        # Select top N stocks by alpha
        top_stocks = alpha.nlargest(self.n_long).index
        alpha_selected = alpha[top_stocks]

        # Initial weights: proportional to alpha rank
        weights = alpha_selected.rank() / alpha_selected.rank().sum()

        # Volatility targeting: scale weights so portfolio vol ≈ target
        port_vol = self._compute_portfolio_vol(weights, cov_matrix)
        weights = weights * (self.target_vol / port_vol)

        # Apply constraints
        weights = weights.clip(upper=self.max_position)
        weights = self._apply_sector_cap(weights, sector_map)
        weights = weights / weights.sum()  # renormalize

        return weights

    def _compute_portfolio_vol(self, weights: pd.Series, 
                                cov: pd.DataFrame) -> float:
        w = weights.reindex(cov.index).fillna(0).values
        return np.sqrt(w @ cov.values @ w) * np.sqrt(252)

    def _apply_sector_cap(self, weights: pd.Series, 
                           sector_map: dict) -> pd.Series:
        sector_weights = {}
        for ticker, w in weights.items():
            sector = sector_map.get(ticker, 'Unknown')
            sector_weights[sector] = sector_weights.get(sector, 0) + w

        # Scale down sectors over cap
        for sector, total_w in sector_weights.items():
            if total_w > self.sector_max:
                scale = self.sector_max / total_w
                for ticker in weights.index:
                    if sector_map.get(ticker) == sector:
                        weights[ticker] *= scale
        return weights

    def estimate_covariance(self, returns: pd.DataFrame, 
                             method='ledoit_wolf') -> pd.DataFrame:
        """
        Ledoit-Wolf shrinkage estimator — robust for high-dim covariance.
        """
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(returns.dropna())
        cov = pd.DataFrame(lw.covariance_, 
                           index=returns.columns, 
                           columns=returns.columns)
        return cov * 252  # annualize
```

### Rebalancing Frequency & Turnover Control

```python
class RebalanceController:
    """
    Controls when and how much to rebalance to limit turnover costs.
    """

    def should_rebalance(self, current_date: date, 
                          last_rebalance: date,
                          current_weights: pd.Series,
                          target_weights: pd.Series,
                          drift_threshold: float = 0.02) -> bool:
        """
        Rebalance if: (a) scheduled time has passed, OR
                      (b) any position has drifted > threshold.
        """
        # Time-based: rebalance monthly (21 trading days)
        days_elapsed = (current_date - last_rebalance).days
        if days_elapsed >= 21:
            return True

        # Drift-based: check if any position has drifted significantly
        drift = (current_weights - target_weights).abs()
        if drift.max() > drift_threshold:
            return True

        return False

    def compute_trades(self, current_weights: pd.Series,
                        target_weights: pd.Series,
                        min_trade_size: float = 0.005) -> pd.Series:
        """
        Returns required trades (positive = buy, negative = sell).
        Filters out tiny trades to reduce costs.
        """
        trades = target_weights.subtract(current_weights, fill_value=0)
        trades[trades.abs() < min_trade_size] = 0  # ignore small rebalances
        return trades
```

---

## 3.8 Risk Management

### Multi-Level Risk Framework

```python
class RiskManager:
    """
    Enforces risk limits across three levels:
    1. Position level (individual stock)
    2. Factor level (sector, style factor)
    3. Portfolio level (total drawdown, volatility)
    """

    LIMITS = {
        'max_position_size':        0.05,   # 5% per stock
        'max_sector_exposure':      0.25,   # 25% per GICS sector
        'max_factor_beta':          0.30,   # 0.3 vs any style factor
        'max_portfolio_vol':        0.15,   # 15% annualized
        'max_drawdown_halt':        0.15,   # halt trading if DD > 15%
        'max_drawdown_reduce':      0.08,   # reduce exposure if DD > 8%
        'stop_loss_per_position':   0.10,   # cut position if -10% since entry
    }

    def check_portfolio_drawdown(self, portfolio_value: pd.Series) -> str:
        """
        Returns action: 'NORMAL', 'REDUCE', or 'HALT'.
        """
        peak = portfolio_value.cummax()
        current_dd = (portfolio_value.iloc[-1] / peak.iloc[-1]) - 1
        
        if current_dd < -self.LIMITS['max_drawdown_halt']:
            return 'HALT'
        elif current_dd < -self.LIMITS['max_drawdown_reduce']:
            return 'REDUCE'
        return 'NORMAL'
```

> **Implementation Note**: The three-tier response (NORMAL / REDUCE / HALT) is
> not optional. A binary NORMAL/HALT system remains fully invested until it hits
> the halt threshold, missing the gradual de-risking that limits drawdown
> accumulation in the -8% to -15% zone. The REDUCE state should scale gross
> exposure by 0.5 and maintain that scale until the portfolio recovers to within
> 3% of its prior peak — this prevents the system from immediately re-risking
> after a brief recovery that turns out to be a dead-cat bounce.
>
> Implemented in alpha-engine as an opt-in `risk.reduce_drawdown_pct`/
> `risk.reduce_exposure_scale` config pair (default disabled, so existing
> strategies are unaffected unless they explicitly opt in) — see
> `decision/portfolio.py` and `backtest/engine.py`/`live/runner.py`'s REDUCE-tier
> handling. When combined with a regime-based exposure overlay (Section 3.6),
> the two independent exposure signals should combine via `min()` (whichever is
> more cautious wins), not multiplicatively — compounding two independent
> partial cuts can produce an unintentionally extreme combined exposure level.

```python
    def apply_stop_losses(self, positions: dict, 
                           entry_prices: dict,
                           current_prices: dict) -> list:
        """Returns list of tickers to exit due to stop-loss trigger."""
        to_exit = []
        for ticker, shares in positions.items():
            if ticker not in entry_prices:
                continue
            ret = current_prices[ticker] / entry_prices[ticker] - 1
            if ret < -self.LIMITS['stop_loss_per_position']:
                to_exit.append(ticker)
        return to_exit
```

### Factor Exposure Monitoring

```python
class FactorExposureMonitor:
    """
    Monitors portfolio exposure to Fama-French / Barra-style risk factors.
    Prevents unintended concentrated bets on any single factor.
    """

    def compute_factor_exposures(self, weights: pd.Series,
                                  factor_loadings: pd.DataFrame) -> pd.Series:
        """
        factor_loadings: DataFrame with tickers as rows, factors as columns
        Returns portfolio-level factor exposure (beta to each factor).
        """
        portfolio_loadings = factor_loadings.T.dot(weights.reindex(factor_loadings.index).fillna(0))
        return portfolio_loadings

    def check_limits(self, exposures: pd.Series) -> dict:
        violations = {}
        for factor, exposure in exposures.items():
            if abs(exposure) > self.LIMITS['max_factor_beta']:
                violations[factor] = exposure
        return violations
```

---

## 3.9 Execution Layer

### Order Management

```python
class ExecutionEngine:
    """
    Translates target portfolio weights into executed orders.
    Handles order splitting, timing, and broker integration.
    """

    def __init__(self, broker_api, slippage_model, portfolio_value: float):
        self.broker = broker_api
        self.slippage = slippage_model
        self.portfolio_value = portfolio_value

    def execute_rebalance(self, current_weights: pd.Series,
                           target_weights: pd.Series,
                           current_prices: dict) -> list:
        """
        Executes trades to move from current to target weights.
        Sells first (to free up capital), then buys.
        """
        trades = self.compute_trades(current_weights, target_weights)
        
        # Separate sells and buys
        sells = trades[trades < 0].sort_values()    # largest sells first
        buys  = trades[trades > 0].sort_values(ascending=False)
        
        executed = []
        for ticker, weight_delta in pd.concat([sells, buys]).items():
            dollar_amount = weight_delta * self.portfolio_value
            shares = dollar_amount / current_prices[ticker]
            
            # Estimate market impact before sending
            impact_bps = self.slippage.estimate(
                ticker=ticker,
                shares=abs(shares),
                adv=self.get_adv(ticker)
            )
            
            order = self.broker.place_order(
                ticker=ticker,
                shares=round(shares),
                order_type='VWAP',  # VWAP orders reduce market impact
                time_in_force='DAY'
            )
            executed.append({**order, 'impact_bps': impact_bps})

        return executed

class SlippageModel:
    """
    Estimates expected slippage based on trade size relative to ADV.
    Uses square-root market impact model (standard in practice).
    """

    def estimate(self, ticker: str, shares: float, adv: float,
                  volatility: float = 0.015) -> float:
        """
        Returns expected slippage in basis points.
        Model: impact = sigma * sqrt(Q / ADV) * participation_factor
        where sigma = daily vol, Q = trade size, ADV = avg daily volume.
        """
        participation = shares / adv
        impact_bps = volatility * np.sqrt(participation) * 10000
        return impact_bps
```

---

## Part II — AI Augmentation Layer

---

## 4.1 LLM-Powered NLP Signals

LLMs add the most value where traditional ML struggles: **understanding the semantic content of unstructured financial text**. The goal is always to produce a normalized alpha signal identical in format to the quantitative signals above.

### Pipeline Architecture

```
Text Sources ──► Chunking ──► LLM Extraction ──► Score ──► Signal Store
(10-K, 10-Q,     (split       (structured         (0 to 1)  (joins quant
 earnings calls,  long docs)   output via          normalized  signal flow)
 news, Fed mins)               function calling)
```

### Signal A: Earnings Call Sentiment

```python
from openai import OpenAI
import json

class EarningsCallSignal:
    """
    Analyzes earnings call transcripts for management tone and guidance.
    Returns a sentiment score per ticker per earnings date.
    """
    
    SYSTEM_PROMPT = """You are a financial analyst extracting structured signals 
    from earnings call transcripts. Analyze the provided transcript excerpt and 
    return a JSON object with the following fields:
    - management_tone: float [-1, +1], where -1=very negative/defensive, +1=very confident/positive
    - guidance_direction: float [-1, +1], where -1=lowered guidance, 0=maintained, +1=raised guidance  
    - uncertainty_level: float [0, 1], where 0=very certain, 1=very uncertain (hedging language)
    - capex_signal: float [-1, +1], investment acceleration vs. reduction
    - demand_signal: float [-1, +1], demand acceleration vs. deceleration mentioned
    - key_risks: list of up to 3 specific risk factors mentioned
    - key_positives: list of up to 3 specific positive factors mentioned
    Return ONLY valid JSON."""

    def __init__(self, client: OpenAI, model: str = "gpt-4o"):
        self.client = client
        self.model = model

    def analyze_transcript(self, transcript: str, 
                            ticker: str, 
                            date: str) -> dict:
        """
        Chunk transcript and analyze in sections, then aggregate.
        """
        chunks = self._chunk_transcript(transcript, max_tokens=3000)
        chunk_results = []

        for chunk in chunks:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Analyze this transcript excerpt:\n\n{chunk}"}
                ],
                temperature=0.1,    # low temperature for consistency
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            chunk_results.append(result)

        # Aggregate across chunks (average numeric scores)
        return self._aggregate_results(chunk_results, ticker, date)

    def _aggregate_results(self, results: list, ticker: str, date: str) -> dict:
        numeric_fields = ['management_tone', 'guidance_direction', 
                          'uncertainty_level', 'capex_signal', 'demand_signal']
        aggregated = {
            'ticker': ticker,
            'date': date,
        }
        for field in numeric_fields:
            values = [r[field] for r in results if field in r]
            aggregated[field] = np.mean(values) if values else 0.0

        # Composite signal: weighted combination
        aggregated['composite_score'] = (
            0.35 * aggregated['management_tone'] +
            0.35 * aggregated['guidance_direction'] +
            -0.20 * aggregated['uncertainty_level'] +
            0.10 * aggregated['demand_signal']
        )
        return aggregated

    def _chunk_transcript(self, text: str, max_tokens: int = 3000) -> list:
        """Split transcript preserving speaker boundaries."""
        # Split by speaker turns (Q&A vs. prepared remarks)
        sentences = text.split('\n\n')
        chunks, current_chunk, current_len = [], [], 0
        for sentence in sentences:
            est_tokens = len(sentence.split()) * 1.3  # rough token estimate
            if current_len + est_tokens > max_tokens and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk, current_len = [sentence], est_tokens
            else:
                current_chunk.append(sentence)
                current_len += est_tokens
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        return chunks
```

### Signal B: SEC Filing Analysis (10-K / 10-Q)

```python
class SECFilingSignal:
    """
    Extracts financial health and risk signals from SEC filings.
    Focus on: MD&A section, Risk Factors, Going Concern language,
              Liquidity discussion, guidance-like language.
    """

    EXTRACTION_PROMPT = """Analyze this section from an SEC filing (10-K or 10-Q).
    Return a JSON object with:
    - liquidity_signal: float [-1,+1]. Is the company describing improving (+1) or deteriorating (-1) liquidity?
    - business_momentum: float [-1,+1]. Are operating trends described as accelerating (+1) or decelerating (-1)?
    - risk_change: float [-1,+1]. Are new/worsening risks being disclosed (-1) or risks reducing (+1)?
    - disclosure_quality: float [0,1]. Is management being transparent (1) or vague/hedging (0)?
    - going_concern_flag: bool. Any going concern or material doubt language?
    - new_risk_factors: list of up to 3 new or significantly changed risk factors
    Return ONLY valid JSON."""

    def analyze_filing(self, filing_text: str, 
                        section: str = 'MD&A') -> dict:
        """
        Analyze specific sections. MD&A is highest signal-to-noise.
        """
        section_text = self._extract_section(filing_text, section)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.EXTRACTION_PROMPT},
                {"role": "user", "content": section_text[:6000]}  # trim to context limit
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
```

### Signal C: News Event Classification

```python
class NewsSignalPipeline:
    """
    Classifies news events in real-time and assesses expected price impact.
    Operates on raw news feed items (headline + body).
    """

    CLASSIFICATION_PROMPT = """You are a financial news analyst. Classify this news item.
    Return JSON with:
    - relevance: float [0,1]. How relevant is this to the stock's fundamental value?
    - sentiment: float [-1,+1]. Negative (-1) to positive (+1) for the primary company.
    - event_type: one of [EARNINGS_GUIDE, M&A, LEGAL, MACRO, PRODUCT, MANAGEMENT, SECTOR, IRRELEVANT]
    - impact_duration: one of [TRANSIENT_1D, SHORT_3D, MEDIUM_2W, LONG_PERMANENT]
    - confidence: float [0,1]. How confident are you in this classification?
    - affected_tickers: list of ticker symbols primarily affected
    Return ONLY valid JSON."""

    def process_news_item(self, headline: str, body: str, 
                           publish_time: datetime) -> dict:
        content = f"Headline: {headline}\n\nBody: {body[:2000]}"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.CLASSIFICATION_PROMPT},
                {"role": "user", "content": content}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        result['publish_time'] = publish_time.isoformat()
        result['raw_headline'] = headline
        return result

    def build_news_alpha(self, ticker: str, date: date, 
                          lookback_days: int = 5) -> float:
        """
        Aggregates recent news items into a single alpha score.
        Decays older news items exponentially.
        """
        news_items = self.news_store.get_ticker_news(ticker, date, lookback_days)
        if not news_items:
            return 0.0

        weighted_sentiment = 0.0
        total_weight = 0.0
        for item in news_items:
            days_old = (date - item['publish_date']).days
            decay = np.exp(-0.5 * days_old)     # half-life ~1.4 days
            relevance_weight = item.get('relevance', 0.5)
            confidence = item.get('confidence', 0.5)
            weight = decay * relevance_weight * confidence
            weighted_sentiment += weight * item['sentiment']
            total_weight += weight

        return weighted_sentiment / total_weight if total_weight > 0 else 0.0
```

### Signal D: Fed / Macro Language Analysis

```python
class MacroNarrativeSignal:
    """
    Analyzes Fed minutes, FOMC statements, and macro reports to extract
    the current monetary policy regime and its equity implications.
    """

    MACRO_PROMPT = """Analyze this Federal Reserve document excerpt.
    Return JSON with:
    - rate_direction: float [-1,+1]. Toward more easing (+1) or tightening (-1)?
    - economic_outlook: float [-1,+1]. Pessimistic (-1) to optimistic (+1)?
    - labor_market_assessment: float [-1,+1]. Weakening (-1) to strengthening (+1)?
    - inflation_concern_level: float [0,1]. Low (0) to high (1) inflation concern?
    - equity_implication: float [-1,+1]. Bearish (-1) to bullish (+1) for equities overall?
    - key_phrases: list of 3-5 most policy-relevant phrases (verbatim from text)
    Return ONLY valid JSON."""

    def analyze_fomc(self, document_text: str) -> dict:
        chunks = self._chunk_document(document_text, max_chars=4000)
        results = []
        for chunk in chunks:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.MACRO_PROMPT},
                    {"role": "user", "content": chunk}
                ],
                temperature=0.05,
                response_format={"type": "json_object"}
            )
            results.append(json.loads(response.choices[0].message.content))
        return self._aggregate_macro_results(results)
```

---

## 4.2 LLM for Research Automation

Beyond signals, LLMs can accelerate the entire research pipeline.

### Automated Hypothesis Generation

```python
class HypothesisEngine:
    """
    Uses LLM to generate and evaluate new signal hypotheses,
    then queues them for systematic backtesting.
    """

    HYPOTHESIS_PROMPT = """You are a systematic trading researcher.
    Based on the following backtest results of existing signals, suggest
    3 new testable signal hypotheses that could improve the strategy.
    
    Current signals and their Information Coefficients (IC):
    {signal_ics}
    
    Current regime performance:
    {regime_performance}
    
    For each hypothesis, provide:
    - signal_name: descriptive name
    - hypothesis: one-sentence description of the alpha thesis
    - data_required: list of data series needed
    - implementation_sketch: Python pseudocode (5-10 lines)
    - academic_basis: relevant academic literature or economic intuition
    - expected_decay: how quickly this signal may decay (months/years)
    - risks: main risks or failure modes
    
    Return as JSON array."""

    def generate_hypotheses(self, signal_ics: dict, 
                             regime_performance: dict) -> list:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": self.HYPOTHESIS_PROMPT.format(
                    signal_ics=json.dumps(signal_ics, indent=2),
                    regime_performance=json.dumps(regime_performance, indent=2)
                )
            }],
            temperature=0.7,    # higher temperature for creativity
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content).get('hypotheses', [])
```

### Automated Backtest Interpretation

```python
class BacktestInterpreter:
    """
    Feeds backtest results to an LLM for natural-language interpretation
    and failure mode diagnosis.
    """

    INTERPRETATION_PROMPT = """You are a quantitative portfolio manager reviewing 
    backtest results. Analyze these metrics and provide:
    1. A plain-English summary of the strategy's performance (3-4 sentences)
    2. The top 3 strengths
    3. The top 3 weaknesses or red flags
    4. Specific recommendations to improve Sharpe ratio
    5. Evidence of overfitting (if any)
    6. Transaction cost sensitivity analysis interpretation
    
    Backtest metrics:
    {metrics}
    
    Return as structured JSON."""

    def interpret(self, backtest_results: dict) -> str:
        # Summarize key metrics (avoid sending raw time series to LLM)
        metrics_summary = {
            'sharpe_ratio': backtest_results['sharpe'],
            'max_drawdown': backtest_results['max_dd'],
            'annualized_return': backtest_results['ann_ret'],
            'information_ratio': backtest_results['ir'],
            'regime_sharpes': backtest_results['regime_breakdown'],
            'signal_ics': backtest_results['signal_ics'],
            'turnover_pct': backtest_results['monthly_turnover'],
            'hit_rate': backtest_results['hit_rate'],
            'avg_holding_period_days': backtest_results['avg_holding_days'],
        }
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": self.INTERPRETATION_PROMPT.format(
                    metrics=json.dumps(metrics_summary, indent=2)
                )
            }],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
```

---

## 4.3 Reinforcement Learning for Portfolio Optimization

RL treats portfolio construction as a sequential decision problem, optimizing long-term risk-adjusted returns directly.

```python
import gymnasium as gym
import numpy as np

class PortfolioEnv(gym.Env):
    """
    Custom RL environment for portfolio management.
    State: alpha scores + current weights + macro features + regime
    Action: target portfolio weights (continuous)
    Reward: Sharpe-scaled return minus transaction costs
    """

    def __init__(self, returns: pd.DataFrame, 
                  alpha_scores: pd.DataFrame,
                  n_assets: int = 50):
        super().__init__()
        self.returns = returns
        self.alpha_scores = alpha_scores
        self.n_assets = n_assets
        self.current_step = 0
        self.current_weights = np.zeros(n_assets)

        # State: [alpha_scores, current_weights, macro_features]
        state_dim = n_assets * 2 + 5  # 2 per asset + 5 macro features
        self.observation_space = gym.spaces.Box(
            low=-3.0, high=3.0, shape=(state_dim,), dtype=np.float32
        )
        # Action: weight adjustments (continuous, summing to 0 = no rebalance)
        self.action_space = gym.spaces.Box(
            low=-0.1, high=0.1, shape=(n_assets,), dtype=np.float32
        )

    def step(self, action: np.ndarray):
        # Apply action (weight adjustments)
        new_weights = np.clip(self.current_weights + action, 0, 0.1)
        new_weights /= new_weights.sum() + 1e-8

        # Compute reward
        returns_today = self.returns.iloc[self.current_step].values[:self.n_assets]
        portfolio_return = np.dot(new_weights, returns_today)
        
        # Transaction cost penalty
        turnover = np.sum(np.abs(new_weights - self.current_weights))
        cost_penalty = turnover * 0.001  # 10bps round-trip

        # Reward: risk-adjusted return with cost penalty
        reward = portfolio_return - cost_penalty
        
        self.current_weights = new_weights
        self.current_step += 1
        
        obs = self._get_obs()
        done = self.current_step >= len(self.returns) - 1
        return obs, reward, done, False, {}

    def _get_obs(self) -> np.ndarray:
        alphas = self.alpha_scores.iloc[self.current_step].values[:self.n_assets]
        macro = self._get_macro_features(self.current_step)
        return np.concatenate([alphas, self.current_weights, macro]).astype(np.float32)
```

**Recommendation**: Use **Proximal Policy Optimization (PPO)** from Stable-Baselines3 for training. RL should be treated as an **experimental signal** and run alongside the rules-based system, not as a replacement.

---

## 4.4 Adaptive Signal Weighting with Online Learning

Rather than fixed signal weights, use an online learning algorithm to adapt weights based on recent signal performance.

```python
class AdaptiveWeightLearner:
    """
    Uses online learning (Exponential Gradient / EWA) to adapt signal 
    weights based on rolling Information Coefficient (IC).
    
    The key insight: weight signals proportional to their recent IC,
    with exponential decay on older observations.
    """

    def __init__(self, signal_names: list, 
                  learning_rate: float = 0.05,
                  ic_window: int = 63):   # 3-month rolling IC
        self.signal_names = signal_names
        self.lr = learning_rate
        self.ic_window = ic_window
        self.weights = {s: 1.0 / len(signal_names) for s in signal_names}

    def update(self, realized_ics: dict) -> dict:
        """
        Update signal weights using Exponential Gradient algorithm.
        realized_ics: {signal_name: rolling_63d_ic} for this period
        """
        # Convert ICs to positive weights (clip negative IC signals)
        ic_scores = {s: max(realized_ics.get(s, 0), 0) 
                     for s in self.signal_names}
        
        # Exponential gradient update
        for signal in self.signal_names:
            self.weights[signal] *= np.exp(self.lr * ic_scores[signal])

        # Normalize to sum to 1
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {s: w / total for s, w in self.weights.items()}
        
        return self.weights

    def compute_ic(self, signal_scores: pd.Series, 
                    forward_returns: pd.Series) -> float:
        """
        Information Coefficient: Spearman rank correlation between 
        signal scores and realized forward returns.
        IC > 0.05 is considered useful in practice.
        """
        from scipy.stats import spearmanr
        aligned = pd.concat([signal_scores, forward_returns], axis=1).dropna()
        if len(aligned) < 20:
            return 0.0
        ic, _ = spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
        return ic
```

---

## 4.5 AI Infrastructure & Safety Rails

LLMs in a live trading pipeline require strict guardrails. A hallucination can generate a catastrophically wrong signal.

### Confidence Scoring and Consistency Checks

```python
class LLMSignalValidator:
    """
    Validates LLM output before it enters the signal pipeline.
    Runs multiple-call consistency checks and bounds-checks.
    """

    def validate_signal(self, signal_dict: dict, 
                         signal_name: str) -> tuple[bool, str]:
        # 1. Bounds check
        for field, (lo, hi) in self.EXPECTED_BOUNDS[signal_name].items():
            val = signal_dict.get(field)
            if val is None or not (lo <= val <= hi):
                return False, f"Field {field}={val} out of bounds [{lo}, {hi}]"

        # 2. Consistency check: re-run with higher temperature and compare
        if abs(signal_dict.get('composite_score', 0)) > 0.7:
            # High-conviction signals get a second pass
            second_pass = self.rerun_with_temp(signal_dict['raw_input'], temp=0.3)
            sign_match = (np.sign(signal_dict['composite_score']) == 
                          np.sign(second_pass.get('composite_score', 0)))
            if not sign_match:
                return False, "Inconsistent high-conviction signal — rejected"

        # 3. Confidence threshold
        if signal_dict.get('confidence', 1.0) < 0.3:
            return False, f"Low confidence: {signal_dict.get('confidence')}"

        return True, "OK"

    EXPECTED_BOUNDS = {
        'earnings_call': {
            'management_tone':      (-1, 1),
            'guidance_direction':   (-1, 1),
            'uncertainty_level':    (0, 1),
            'composite_score':      (-1, 1),
        }
    }
```

### LLM Pipeline Rate Limiting & Caching

```python
class LLMSignalCache:
    """
    Caches LLM signal results to avoid re-processing the same documents.
    Uses content hash as cache key to handle updates.
    """

    def __init__(self, redis_client):
        self.cache = redis_client
        self.ttl_seconds = 86400 * 30  # 30-day cache for historical

    def get_or_compute(self, doc_hash: str, 
                        compute_fn: callable, 
                        *args, **kwargs) -> dict:
        cached = self.cache.get(f"llm_signal:{doc_hash}")
        if cached:
            return json.loads(cached)
        result = compute_fn(*args, **kwargs)
        self.cache.setex(f"llm_signal:{doc_hash}", 
                          self.ttl_seconds, 
                          json.dumps(result))
        return result
```

---

## Part III — Effective Backtesting

Backtesting is where most systematic strategies fail before they ever reach live trading. Biased backtests produce fictitiously high Sharpe ratios. This section is the **most important safeguard in the entire system**.

---

## 5.1 Backtesting Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    BACKTESTING ENGINE                      │
│                                                            │
│  ┌──────────────┐    ┌──────────────┐    ┌─────────────┐  │
│  │ Event Loop   │    │ Portfolio    │    │ Performance │  │
│  │ (bar-by-bar  │───►│ State Engine │───►│ Accounting  │  │
│  │  simulation) │    │ (PnL, fills, │    │ (returns,   │  │
│  │              │    │  slippage)   │    │  drawdown)  │  │
│  └──────────────┘    └──────────────┘    └─────────────┘  │
│         │                                                  │
│  ┌──────▼───────┐    ┌──────────────┐                      │
│  │ Data Feed    │    │ Signal       │                      │
│  │ (PIT-safe    │───►│ Computation  │                      │
│  │  OHLCV +     │    │ (all signals │                      │
│  │  features)   │    │  per bar)    │                      │
│  └──────────────┘    └──────────────┘                      │
└────────────────────────────────────────────────────────────┘
```

**Framework choice**: Build on top of **Zipline (Quantopian's open-source engine)**, **Backtrader**, or **VectorBT** for speed.  
For production-grade research: **Nasdaq Alphalens** for factor analysis, **PyFolio** for portfolio tear sheets.

---

## 5.2 Data Quality & Bias Elimination

### The Four Deadly Biases

| Bias | Description | Fix |
|------|-------------|-----|
| **Survivorship Bias** | Only using stocks that survived to today | Use point-in-time index membership; include delisted stocks |
| **Look-Ahead Bias** | Using data not yet known on the signal date | Strict `as_of_date` discipline on every data access |
| **Snooping Bias** | Testing many parameters until one works | Use out-of-sample period; apply multiple testing correction |
| **Selection Bias** | Cherry-picking backtest start/end dates | Always include multiple market cycles including 2000-02, 2007-09, 2020 |

### Survivorship Bias Elimination

```python
class SurvivorshipBiasFreeUniverse:
    """
    Proper universe construction requires both:
    1. Historical index membership (who WAS in the index then)
    2. Delisting data (what happened to stocks that left)
    """

    def get_universe_with_delistings(self, start: date, end: date) -> pd.DataFrame:
        """
        Returns a panel of all stocks that were tradeable on each date,
        including stocks that were later delisted.
        """
        # Active stocks
        active_stocks = self.get_historical_index_members(start, end)

        # Delisted stocks (critical! omitting these inflates returns)
        delisted = self.get_delisted_stocks(start, end)
        delisting_returns = self.get_delisting_returns(delisted)
        # When a stock delists, apply the delisting return and close position

        return pd.concat([active_stocks, delisting_returns], axis=0)

    def apply_delisting_return(self, portfolio: dict, 
                                delisted_ticker: str,
                                delisting_return: float) -> dict:
        """
        Force-close any position in a delisted stock at the delisting return.
        Ignore this and you overstate returns by ~1-3% per year.
        """
        if delisted_ticker in portfolio:
            portfolio['cash'] += portfolio[delisted_ticker] * (1 + delisting_return)
            del portfolio[delisted_ticker]
        return portfolio
```

### Look-Ahead Bias Checklist

```python
class LookAheadBiasChecker:
    """
    Automated checks to detect potential look-ahead bias in signal computation.
    Run this on every new signal before adding it to the system.
    """

    def check_fundamental_data(self, signal_fn, ticker: str) -> bool:
        """
        Test: does the signal produce DIFFERENT values when computed at
        earnings announcement date vs. 1 day before?
        If identical, data may be improperly forward-filled.
        """
        t_announce = self.get_earnings_date(ticker)
        t_before = t_announce - timedelta(days=1)
        
        val_on_day = signal_fn(ticker, t_announce)
        val_before = signal_fn(ticker, t_before)
        
        # Earnings revision signal SHOULD be different after announcement
        if val_on_day == val_before:
            print(f"WARNING: Signal identical before/after earnings for {ticker}")
            return False
        return True

    def check_return_contamination(self, features: pd.DataFrame, 
                                    forward_returns: pd.Series) -> float:
        """
        Test: correlation between features and SAME-DAY returns should be low.
        High same-day correlation = look-ahead contamination.
        """
        from scipy.stats import spearmanr
        same_day_ic, _ = spearmanr(features.mean(axis=1), 
                                    forward_returns.shift(1))  # returns on signal date
        if abs(same_day_ic) > 0.1:
            print(f"WARNING: High same-day IC={same_day_ic:.3f}. Possible look-ahead bias.")
        return same_day_ic
```

---

## 5.3 Walk-Forward Validation

Walk-forward validation is **the gold standard for systematic strategy validation**. Never report backtest results without it.

```python
class WalkForwardValidator:
    """
    Implements expanding window and rolling window walk-forward validation.
    
    Timeline:
    |---- Train 1 ----|-- Val 1 --|-- Test 1 --|
                      |---- Train 2 ----|-- Val 2 --|-- Test 2 --|
                                        ...
    
    - Train: fits the model / optimizes signal weights
    - Val: selects hyperparameters (never optimizes on test!)
    - Test: strictly held-out, results reported once only
    """

    def __init__(self, train_years=5, val_years=1, test_years=1, 
                  step_years=1, mode='expanding'):
        self.train_years = train_years
        self.val_years = val_years
        self.test_years = test_years
        self.step_years = step_years
        self.mode = mode  # 'expanding' or 'rolling'

    def generate_folds(self, start: date, end: date) -> list:
        folds = []
        current = start + relativedelta(years=self.train_years + self.val_years)

        while current + relativedelta(years=self.test_years) <= end:
            if self.mode == 'expanding':
                train_start = start
            else:  # rolling
                train_start = current - relativedelta(
                    years=self.train_years + self.val_years
                )

            val_start = current - relativedelta(years=self.val_years)
            test_start = current
            test_end = current + relativedelta(years=self.test_years)

            folds.append({
                'train': (train_start, val_start - timedelta(days=1)),
                'val':   (val_start, test_start - timedelta(days=1)),
                'test':  (test_start, test_end)
            })
            current += relativedelta(years=self.step_years)
        return folds

    def run(self, strategy, data: dict) -> pd.DataFrame:
        """
        Runs walk-forward validation. Returns per-fold performance.
        """
        folds = self.generate_folds(data['start'], data['end'])
        results = []

        for i, fold in enumerate(folds):
            print(f"Fold {i+1}/{len(folds)}: Train {fold['train']}, Test {fold['test']}")

            # 1. Train on training window
            strategy.fit(data, fold['train'][0], fold['train'][1])

            # 2. Select hyperparameters on validation window (never test!)
            best_params = strategy.select_params(
                data, fold['val'][0], fold['val'][1]
            )
            strategy.set_params(best_params)

            # 3. Evaluate on test window (use ONCE only)
            test_returns = strategy.backtest(
                data, fold['test'][0], fold['test'][1]
            )
            fold_metrics = self.compute_metrics(test_returns)
            fold_metrics['fold'] = i + 1
            fold_metrics['test_start'] = fold['test'][0]
            fold_metrics['test_end'] = fold['test'][1]
            results.append(fold_metrics)

        return pd.DataFrame(results)

    def compute_metrics(self, returns: pd.Series) -> dict:
        ann_return = (1 + returns).prod() ** (252 / len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        drawdown = (returns.cumsum() - returns.cumsum().cummax()).min()
        return {
            'ann_return': ann_return,
            'ann_vol':    ann_vol,
            'sharpe':     sharpe,
            'max_dd':     drawdown,
            'hit_rate':   (returns > 0).mean(),
            'calmar':     ann_return / abs(drawdown) if drawdown != 0 else 0,
        }
```

### Reporting Walk-Forward Results

```python
def report_walk_forward_results(wf_results: pd.DataFrame):
    """
    A strategy is credible only if it shows:
    1. Positive Sharpe in MOST folds (not just the average)
    2. No single fold is dramatically better than others (no lucky period)
    3. Performance is consistent across different market regimes
    """
    print("=== Walk-Forward Validation Results ===\n")
    print(wf_results[['test_start', 'test_end', 'sharpe', 'ann_return', 
                        'max_dd', 'hit_rate']].to_string(index=False))
    print(f"\nMean Sharpe: {wf_results['sharpe'].mean():.2f}")
    print(f"Sharpe Std:  {wf_results['sharpe'].std():.2f}")
    print(f"% Positive Sharpe Folds: {(wf_results['sharpe'] > 0).mean()*100:.0f}%")
    print(f"Worst Fold Sharpe: {wf_results['sharpe'].min():.2f}")

    # Red flags
    if wf_results['sharpe'].std() > 1.0:
        print("\n⚠️  HIGH VARIANCE ACROSS FOLDS — possible overfitting or regime sensitivity")
    if (wf_results['sharpe'] > 0).mean() < 0.7:
        print("\n⚠️  FEWER THAN 70% POSITIVE FOLDS — strategy may not be robust")
```

---

## 5.4 Transaction Cost & Slippage Modeling

Poor cost modeling is the #1 reason backtests look better than live performance.

```python
class RealisticCostModel:
    """
    Models all-in trading costs: commission + bid-ask spread + market impact.
    """

    # Approximate one-way costs by liquidity tier
    COST_TABLE = {
        'SP500_large_cap':   {'commission': 0.0001, 'spread': 0.0005, 'impact': 0.0010},
        'SP500_mid':         {'commission': 0.0001, 'spread': 0.0010, 'impact': 0.0020},
        'Russell1000_small': {'commission': 0.0002, 'spread': 0.0020, 'impact': 0.0050},
        'Russell2000':       {'commission': 0.0005, 'spread': 0.0050, 'impact': 0.0100},
    }

    def estimate_round_trip_cost(self, ticker: str, 
                                  trade_value: float,
                                  adv: float) -> float:
        """
        Returns round-trip cost in basis points.
        Includes: commission (both legs) + bid-ask (both legs) + market impact.
        """
        tier = self.classify_ticker(ticker)
        costs = self.COST_TABLE[tier]
        
        # Market impact scales with sqrt of participation rate
        participation = trade_value / adv
        impact_bps = costs['impact'] * np.sqrt(participation / 0.01) * 10000
        
        # Round-trip: multiply one-way costs by 2
        total_bps = 2 * (
            costs['commission'] * 10000 +
            costs['spread'] * 10000 +
            impact_bps
        )
        return total_bps

    def apply_to_backtest(self, trades: pd.DataFrame, 
                           prices: pd.DataFrame) -> pd.Series:
        """
        Applies realistic costs to a series of trades in the backtest.
        Returns daily PnL after costs.
        """
        cost_pnl = pd.Series(0.0, index=prices.index)
        for _, trade in trades.iterrows():
            cost_bps = self.estimate_round_trip_cost(
                trade['ticker'], 
                abs(trade['value']),
                trade['adv']
            )
            cost_pnl[trade['date']] -= abs(trade['value']) * cost_bps / 10000
        return cost_pnl

def cost_sensitivity_analysis(strategy_returns_gross: pd.Series,
                               turnover_monthly: float,
                               cost_bps_range: list = [0, 10, 20, 30, 50]) -> pd.DataFrame:
    """
    Shows how net Sharpe varies with assumed transaction cost level.
    If Sharpe collapses below 0.5 at 20bps, the strategy is too fragile.
    """
    results = []
    daily_cost = turnover_monthly / 21  # approximate daily cost drag
    for cost_bps in cost_bps_range:
        daily_drag = daily_cost * cost_bps / 10000
        net_returns = strategy_returns_gross - daily_drag
        ann_return = (1 + net_returns).prod() ** (252 / len(net_returns)) - 1
        ann_vol = net_returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        results.append({'cost_bps': cost_bps, 'sharpe': sharpe, 
                         'ann_return': ann_return})
    return pd.DataFrame(results)
```

---

## 5.5 Statistical Significance & Overfitting Detection

```python
class StatisticalValidator:
    """
    Applies rigorous statistical tests before declaring a strategy valid.
    """

    def compute_t_stat(self, returns: pd.Series) -> float:
        """
        t-statistic for Sharpe ratio.
        Minimum threshold: t > 2.0 (p < 0.05) for published strategies.
        In practice, aim for t > 3.0 to account for multiple testing.
        """
        n = len(returns)
        sr = returns.mean() / returns.std()  # daily Sharpe
        t_stat = sr * np.sqrt(n)
        return t_stat

    def deflated_sharpe_ratio(self, observed_sharpe: float,
                               n_trials: int,
                               backtest_length: int,
                               skewness: float = 0,
                               kurtosis: float = 3) -> float:
        """
        Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio.
        Adjusts for the number of strategies tested (multiple testing).
        
        DSR < 0.95 means the strategy is likely due to data snooping.
        """
        from scipy.stats import norm
        
        # Expected maximum Sharpe under null (from testing n_trials strategies)
        sr_0 = (1 - 0.5772) * norm.ppf(1 - 1 / n_trials) + \
               0.5772 * norm.ppf(1 - 1 / (n_trials * np.e))
        
        # Variance adjustment for non-normal returns
        var_sr = (1 - skewness * observed_sharpe + 
                  (kurtosis - 1) / 4 * observed_sharpe**2) / (backtest_length - 1)
        
        dsr = norm.cdf((observed_sharpe - sr_0) / np.sqrt(var_sr))
        return dsr

    def minimum_backtest_length(self, target_sharpe: float,
                                 n_parameters: int) -> int:
        """
        Minimum number of years needed to statistically validate a Sharpe ratio,
        given the number of parameters tested.
        Based on minimum track record length (Bailey & de Prado).
        """
        from scipy.stats import norm
        z_alpha = norm.ppf(0.95)
        min_obs = (z_alpha + 0.5 * target_sharpe) ** 2 / target_sharpe ** 2
        # Inflate for multiple testing
        min_obs_adjusted = min_obs * np.log(n_parameters)
        years = min_obs_adjusted / 252
        return max(int(np.ceil(years)), 5)  # minimum 5 years

def permutation_test(strategy_fn, returns: pd.Series, 
                      n_permutations: int = 1000) -> float:
    """
    Tests if the strategy's Sharpe is better than random by permuting returns.
    p-value < 0.05 required. This is the most robust test for genuine alpha.
    """
    observed_sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    null_sharpes = []
    
    for _ in range(n_permutations):
        permuted_returns = returns.sample(frac=1).values  # shuffle dates
        null_sharpe = (np.mean(permuted_returns) / np.std(permuted_returns)) * np.sqrt(252)
        null_sharpes.append(null_sharpe)
    
    p_value = np.mean(np.array(null_sharpes) >= observed_sharpe)
    return p_value
```

---

## 5.6 Regime Robustness Testing

A strategy that only works in bull markets is not a strategy — it is leveraged beta.

```python
class RegimeRobustnessTest:
    """
    Breaks backtest into market regimes and evaluates per-regime performance.
    Strategy must show positive alpha in AT LEAST 3 of 4 regimes.
    """

    HISTORICAL_REGIMES = {
        'dotcom_boom':    ('1997-01-01', '2000-03-09'),
        'dotcom_bust':    ('2000-03-10', '2002-10-09'),
        'bull_2003_07':   ('2002-10-10', '2007-10-09'),
        'gfc_crisis':     ('2007-10-10', '2009-03-09'),
        'post_gfc_bull':  ('2009-03-10', '2020-02-19'),
        'covid_crash':    ('2020-02-20', '2020-03-23'),
        'covid_recovery': ('2020-03-24', '2021-12-31'),
        'rate_hike_2022': ('2022-01-01', '2022-12-31'),
        'ai_bull_2023':   ('2023-01-01', '2024-12-31'),
    }

    def run_regime_analysis(self, backtest_returns: pd.Series,
                             spy_returns: pd.Series) -> pd.DataFrame:
        results = []
        for regime_name, (start, end) in self.HISTORICAL_REGIMES.items():
            period_returns = backtest_returns[start:end]
            period_spy = spy_returns[start:end]
            if len(period_returns) < 20:
                continue
            
            excess = period_returns - period_spy
            results.append({
                'regime': regime_name,
                'start': start,
                'end': end,
                'strategy_ann_ret': (1 + period_returns).prod() ** (252 / len(period_returns)) - 1,
                'spy_ann_ret': (1 + period_spy).prod() ** (252 / len(period_spy)) - 1,
                'excess_ann_ret': (1 + excess).prod() ** (252 / len(excess)) - 1,
                'sharpe': period_returns.mean() / period_returns.std() * np.sqrt(252),
                'max_dd': (period_returns.cumsum() - period_returns.cumsum().cummax()).min(),
                'n_days': len(period_returns),
            })
        return pd.DataFrame(results)
```

---

## 5.7 Backtesting Checklist

Use this checklist before declaring any backtest result valid:

```
DATA INTEGRITY
 ☐ Using point-in-time fundamental data (not restated)
 ☐ Universe includes delisted stocks (no survivorship bias)
 ☐ Corporate actions (splits, dividends) properly adjusted
 ☐ Historical index membership used (not current membership)
 ☐ No same-day data used in signal (signals formed at prior day's close)
 ☐ Fundamental data delayed by filing lag (10-K: +60 days, 10-Q: +45 days)

SIGNAL VALIDATION
 ☐ Each signal has positive IC on out-of-sample data (IC > 0.03)
 ☐ Signals are cross-sectionally z-scored (no level effects)
 ☐ Signal decay analysis performed (IC vs. holding period horizon)
 ☐ No signal selected by testing >50 candidates without deflation
```

> **IC Gate — Non-Negotiable**: A signal with unvalidated or sub-threshold IC
> must be assigned **weight zero** in the composite until validated. Do not
> include it in the walk-forward sweep with a non-zero weight on the assumption
> that it "probably works" — if its IC is below 0.03, it is adding noise to the
> ranking. Validate on a held-out period before assigning any weight to a new
> signal.
>
> Confirmed in practice: alpha-engine's `ml_alpha` signal was assigned a 0.35
> composite weight for an entire session of walk-forward validation before its
> IC was ever actually measured. When finally checked (`alpha_engine.ml.
> validation.compute_mean_ic`, held-out 2021–2022 period), its mean IC was
> **-0.0120** — below the 0.03 threshold, meaning it had been contributing noise
> to the ranking, not signal, for every backtest run up to that point. This is
> exactly the failure mode this gate exists to prevent. See
> `systematic_core_postmortem.md`.

```
BACKTEST METHODOLOGY  
 ☐ Walk-forward validation with held-out test periods
 ☐ In-sample and out-of-sample periods clearly separated
 ☐ Hyperparameters selected on VALIDATION set, not test set
 ☐ Backtest covers at least one full market cycle (bull + bear)
 ☐ Backtest includes 2000-2002 or 2007-2009 crisis period

COST MODELING
 ☐ Bid-ask spread estimated per liquidity tier
 ☐ Market impact modeled (sqrt model) for large positions
 ☐ Commission costs included
 ☐ Cost sensitivity analysis shows Sharpe > 0.5 at 25 bps
 ☐ Turnover measured and reasonable (monthly < 30% for daily-rebalance)

STATISTICAL VALIDITY
 ☐ t-statistic of Sharpe > 2.0 (preferably > 3.0)
 ☐ Deflated Sharpe Ratio > 0.95
 ☐ Permutation test p-value < 0.05
 ☐ Positive Sharpe in ≥ 70% of walk-forward folds
 ☐ Strategy profitable in both crisis AND bull market regimes

LIVE TRADING READINESS
 ☐ Paper traded for minimum 3 months with matching live performance
 ☐ Slippage in paper trading matches modeled slippage
 ☐ All data sources confirmed available with acceptable latency in live
 ☐ Model retrain schedule defined (monthly ML refit recommended)
 ☐ Kill switch and position limits tested
```

---

## 6. Appendix — Tech Stack Reference

### Core Libraries

| Component | Library | Notes |
|-----------|---------|-------|
| Data ingestion | `polygon-api-client`, `fredapi`, `yfinance` | Polygon for high-quality OHLCV |
| Data storage | `TimescaleDB`, `Parquet + PyArrow` | TimescaleDB for live; Parquet for research |
| Signal computation | `pandas`, `numpy`, `scipy` | Vectorized for speed |
| ML models | `lightgbm`, `scikit-learn`, `xgboost` | LightGBM preferred for tabular |
| Portfolio optimization | `cvxpy`, `scipy.optimize` | CVXPY for constrained optimization |
| Backtesting | `vectorbt`, `zipline-reloaded`, `pyfolio` | VectorBT for speed; Zipline for event-driven |
| Factor analysis | `alphalens-reloaded` | IC, turnover, factor return analysis |
| Regime detection | `hmmlearn`, `sklearn.mixture` | HMM for regime, GMM for clustering |
| RL training | `stable-baselines3`, `gymnasium` | PPO for portfolio RL |
| LLM integration | `openai`, `anthropic`, `langchain` | Direct APIs preferred over LangChain for production |
| NLP preprocessing | `spacy`, `transformers` | FinBERT for financial sentiment |
| Execution | `alpaca-trade-api`, `ibapi` (Interactive Brokers) | IB for institutional-grade |
| Monitoring | `prometheus`, `grafana` | Signal health, live performance tracking |
| Orchestration | `prefect`, `airflow` | Daily pipeline scheduling |

### Minimum Data Requirements for Live Deployment

```
Daily prices (adjusted):   2+ years live, 20+ years historical
Fundamental data:          10+ years (Compustat or SimFin)
Macro data:                20+ years (FRED API, free)
Short interest:            3+ years (FINRA, bi-weekly)
Options data:              2+ years (CBOE or OptionMetrics)
News / transcripts:        As far back as available (Refinitiv, SEC EDGAR)
```

### Recommended Development Timeline

| Phase | Duration | Milestone |
|-------|----------|-----------|
| Data infrastructure | 4–6 weeks | Clean PIT data store, universe builder |
| Signal library (non-AI) | 6–8 weeks | 8–10 signals with IC > 0.03 validated |
| ML models | 4–6 weeks | Walk-forward Sharpe > 0.8 out-of-sample |
| Regime detection | 2–3 weeks | 4-regime classifier with adaptive weights |
| Portfolio construction | 2–3 weeks | Constrained optimizer + risk management |
| Backtesting validation | 4 weeks | Full checklist passed |
| LLM signal integration | 6–8 weeks | Earnings + news signals validated |
| Paper trading | 8–12 weeks | Live vs. backtest divergence < 0.3 Sharpe |
| Live trading (small) | Ongoing | Scale up as live track record builds |

---

*This document is a living engineering reference. Signal edge decays over time — plan for a quarterly signal review cycle, monthly ML retraining, and annual full strategy audit.*
