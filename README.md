# alpha-engine

Systematic swing-trading research engine, aiming to beat SPY/VOO/VTI over
days-to-weeks holding periods. Quant signals (momentum, mean-reversion,
volatility, value, quality) plus an AI-powered news research signal feed
into a fully deterministic, rules-based decision engine — the AI researches
and scores, it never decides. See `CLAUDE.md` for the full architecture
writeup.

Currently backtest-only: no live or paper order execution yet.

## Setup

```
python3 -m venv .venv   # see CLAUDE.md if this fails (no system pip/venv)
.venv/bin/pip install -e ".[dev]"
cp .env.example .env    # fill in ANTHROPIC_API_KEY (only needed for the live news signal)
```

## Run a backtest

```
.venv/bin/python3 -m alpha_engine.cli backtest --start 2019-01-01 --end 2024-12-31
```

Add `--fundamentals` to include the value/quality signals, `--symbols` to
override the universe. Writes `output/equity_curve.csv` and
`output/tearsheet.png`.

## Run walk-forward validation

```
.venv/bin/python3 -m alpha_engine.cli walkforward --start 2019-01-01 --end 2024-12-31 \
    --train-years 3 --test-years 1 --step-years 1
```

Add `--param-grid path/to/grid.yaml` to grid-search strategy config on each
train window instead of just checking out-of-sample consistency.

## Run tests

```
.venv/bin/python3 -m pytest tests/ -v
```

## How It Works

This section explains the system in depth: the financial theory behind
each piece, why it's built the way it is, and — just as important — what
it doesn't claim to do. If you're new to investing terminology, the
**Glossary** near the bottom defines every term used here the first time
it comes up in bold.

### The big picture

Most people invest passively: buy an index fund like SPY (tracks the S&P
500) and hold it. That's a strong baseline — historically ~10%/year before
inflation — because it captures the average return of the whole market
with minimal effort and minimal cost. Beating it consistently is hard,
and most professional fund managers don't manage it net of fees over long
periods.

alpha-engine's bet is narrower and more specific than "beat the market
somehow": rank a universe of large-cap stocks and sector ETFs every week
by a handful of measurable characteristics that have historically been
associated with higher risk-adjusted returns, hold the top-ranked names,
and manage risk mechanically rather than emotionally. This is called
**factor investing** — the idea that stock returns aren't random, they're
partly explained by measurable factors (momentum, value, quality, low
volatility, size, and a few others) that persist across decades and
markets. It's the same intellectual tradition behind Fama-French factor
models (Nobel Prize in Economics, 2013) and most quantitative hedge funds.

The system is explicitly a **research platform first**, not a proven
money-maker yet — see "Honest limitations" at the end before you read
too much confidence into anything above.

### The signals — what each one measures and why it might work

Every signal produces a score per symbol per rebalance date, standardized
via a **z-score** (how many standard deviations above or below the
average a value is — a z-score of +2 means "unusually high vs. the rest
of the universe today," -2 means "unusually low"). Standardizing this way
means signals measured in wildly different units (a percentage return vs.
a dollar ratio) can be added together meaningfully.

**Momentum** (`signals/momentum.py`) — measures a stock's total return
over the past ~12 months, excluding the most recent month. Stocks that
have gone up tend to keep going up over the next few months, and vice
versa. This is one of the most replicated findings in finance (Jegadeesh
& Titman, 1993) and violates the strict "efficient market hypothesis"
(the theory that prices instantly reflect all information, so past
returns shouldn't predict future ones). The leading explanation is
behavioral: investors underreact to news at first, then a trend attracts
more buyers as it becomes visible, feeding on itself for a while before
reversing. The most recent month is excluded because *very* short-term
performance tends to reverse (see next signal), so blending it in would
work against the effect.

**Mean reversion** (`signals/mean_reversion.py`) — the opposite intuition,
applied over a much shorter window (20 trading days, about a month): a
stock trading far above its own recent average price is more likely to
pull back, and one trading far below is more likely to bounce. This
exploits short-term overreaction and liquidity effects — a stock that
moved sharply on a burst of selling (or buying) often overshoots fair
value briefly, and gets pulled back as the imbalance clears. Momentum and
mean reversion sound contradictory but they operate on different time
horizons and are a well-documented pairing: trend over the medium term,
reversion over the short term.

**Volatility** (`signals/volatility.py`) — favors calmer stocks over
choppier ones (annualized standard deviation of daily returns, lower is
better here). This is the **low-volatility anomaly**: standard financial
theory (CAPM) says taking more risk should be rewarded with more return,
but empirically, low-volatility stocks have historically delivered
*better* risk-adjusted returns than high-volatility ones — the extra risk
of volatile stocks hasn't historically been compensated. Leading
explanations include leverage constraints (investors who want more
upside can't always use leverage, so they reach for volatile stocks
directly, bidding up their price and lowering their future returns) and
lottery-seeking behavior (people overpay for stocks that could have a
big, exciting payoff, the same way people overpay for lottery tickets
relative to their odds).

**Value** (`signals/value.py`) — favors stocks that are cheap relative to
their accounting book value (**price-to-book ratio**, or **P/B**: the
stock's market price divided by the company's net worth per share on its
balance sheet — a P/B of 1 means the market values the company at exactly
its stated net worth; below 1 means the market thinks it's worth less
than the accountants say). This is the classic value-investing factor
(Fama & French, 1992) — cheap stocks have historically outperformed
expensive ones over long periods. There are two competing explanations,
and honestly nobody has fully settled which dominates: (1) cheap stocks
are cheap because they're genuinely riskier (distressed companies,
cyclical businesses), so the extra return is fair compensation for risk
investors are bearing; (2) markets are irrationally pessimistic about
out-of-favor companies and eventually correct, which is closer to what
Warren Buffett-style value investors believe they're exploiting.

**Quality** (`signals/quality.py`) — favors profitable, conservatively
financed companies: high **return on equity** (**ROE** — net income
divided by shareholder equity, essentially "how much profit does the
company generate per dollar shareholders have invested") and low **debt-
to-equity** (how much the company has borrowed relative to what
shareholders own — more debt means more fragility if business slows
down). This factor is newer to mainstream finance (Asness, Frazzini &
Pedersen's "Quality Minus Junk", 2013) but the logic is intuitive:
well-run, financially sound businesses are less likely to blow up, and
the market has historically underpriced that safety somewhat.

**News/AI sentiment** (`signals/news_ai.py`, live-only) — Claude searches
recent news per symbol and returns a structured sentiment score. This
isn't a classic academic factor; it's a bet that a careful reader of
breaking news (earnings surprises, guidance changes, lawsuits, management
changes) can extract a faster or more nuanced signal than pure historical
price/fundamentals data provides. It's the newest, least-validated piece
of the system by design — see the Core Design Constraint below for why
it's kept on a short leash.

**Why combine several signals instead of picking the "best" one?** The
same reason a diversified stock portfolio beats betting on one stock:
each factor has periods where it does poorly (momentum gets crushed in
sharp reversals; value can underperform for years at a stretch, as it did
for most of the 2010s), and they're not perfectly correlated with each
other. Blending signals whose good and bad periods don't fully overlap
smooths the ride, even if no single signal is dominant. `config/
strategy.yaml`'s `signal_weights` control how much each one counts.

### From signals to a portfolio

Every score gets combined into one ranked list (`decision/scorer.py`,
plain weighted sum — see the Core Design Constraint below for why this
step is deliberately *not* run through an LLM). The top-N ranked symbols
become the portfolio (`decision/portfolio.py`); everything else is
excluded that week.

**Position sizing** — how much money goes into each of the N picks — has
three modes:
- `equal_weight`: every position gets 1/N of the portfolio. Simple,
  robust, no assumptions about risk.
- `volatility_scaled`: calmer stocks get a bigger allocation, shakier
  ones get less — a simple form of risk parity (each position contributes
  roughly similar *risk*, not similar dollars).
- `min_variance`: solves for the mathematically risk-minimizing
  combination of the selected stocks, accounting for how they move
  *together* (their **covariance** — vol-scaling only looks at each
  stock in isolation, min-variance also cares whether two stocks tend to
  rise and fall together, in which case holding both adds less
  diversification than it looks like). This is a simplified version of
  Harry Markowitz's Modern Portfolio Theory (Nobel Prize, 1990) — the
  foundational math of "diversification reduces risk for the same
  expected return."

A **sector cap** (`config/sectors.yaml`, e.g. no more than 30% in
Technology) prevents the portfolio from accidentally becoming a
concentrated bet on one part of the economy just because several
top-ranked stocks happen to share a sector.

### Risk management

Two mechanical, unemotional rules sit on top of the portfolio:

- **Stop-loss**: any position that's fallen more than a configured
  percentage (default 15%) from where it was bought is force-sold at the
  next rebalance, regardless of what the signals currently say about it.
- **Circuit breaker**: if the whole portfolio's **drawdown** (the drop
  from its highest-ever value so far — see Glossary) exceeds a threshold
  (default 20%), the entire portfolio is sold to cash for a cooldown
  period, then re-enters.

These exist because the single biggest threat to a systematic strategy
isn't a bad month — it's the human operating it panicking and abandoning
it at exactly the wrong time, or a strategy quietly digging a hole so
deep it can't mathematically recover (a 50% loss requires a 100% gain
just to break even). Mechanical rules remove that judgment call.

### Core design constraint: AI researches, it never decides

This is the one architectural rule that shapes everything else. The
Claude-powered news signal is allowed to be messy, subjective, and
non-deterministic — that's fine for research. But the moment signals
become target portfolio weights, the code is a plain weighted sum and a
handful of `if` statements (`decision/scorer.py`, `decision/portfolio.py`,
`decision/risk.py`) — no LLM call anywhere in that path. Given the exact
same inputs, it always produces the exact same trade. That's what makes
a backtest *mean* anything: if the decision logic could vary run to run,
"this strategy returned 15% last year" would be unfalsifiable.

### Backtesting discipline: no cheating with the future

A backtest is trustworthy only if, at each simulated point in time, it
only used information that would have actually been available then. The
single most common way amateur backtests silently lie to themselves is
**lookahead bias** — accidentally letting the simulation "see" future
data. Two examples handled explicitly in this codebase:

- The AI news signal is **never used in backtests at all**, because
  Claude's web search returns *today's* internet, not what was on the
  internet on some date in 2021. There's no reliable way to ask it "what
  did the news look like as of March 2021" — so rather than pretend
  otherwise, that signal is live-only.
- The fundamentals signals (value/quality) select, for each historical
  date, the most recent *quarterly report whose date is on or before*
  that day — never a report published after it, even if that means the
  signal is simply unavailable for older dates (documented as a known
  gap rather than papered over).

**Transaction costs** are modeled too (commission + slippage in **basis
points**, hundredths of a percent — 5 bps = 0.05%), since a strategy that
looks great on paper but trades so often that fees eat the returns isn't
actually a good strategy.

### Validating robustness: why walk-forward, not just one backtest

Running one backtest over one historical period and looking at the final
number is the single easiest way to fool yourself in this field — it's
trivially easy to tune a strategy's parameters until they fit that
*specific* period well (called **overfitting** or **curve-fitting**), the
same way you could always draw a curve that passes exactly through a
scattered set of past data points but that curve tells you nothing about
the next point.

`backtest/walkforward.py` guards against this by splitting history into
rolling **train** and **test** windows (e.g. train on 2019-2021, test on
2022; then train on 2020-2022, test on 2023; and so on). Any parameter
tuning happens only on the train window; performance is only ever
*reported* on the following test window, which the tuning step never
saw. If a strategy only looks good because of the exact combination of
parameters "we happened to pick," it will visibly fall apart out-of-
sample across these windows — which is exactly what a single full-period
backtest can hide. Our first real walk-forward run surfaced this
concretely: the 2022 window (a genuine bear market for growth/momentum
strategies) came back with a *negative* Sharpe ratio while 2023 was
strong — information a single 2019-2024 backtest would have averaged away
and hidden.

### Glossary — performance metrics

| Term | Meaning |
|---|---|
| **CAGR** | Compound Annual Growth Rate — the smoothed, constant annual return that would take you from the starting value to the ending value. "14.8% CAGR" means "grew at an average of 14.8%/year, compounding." |
| **Alpha** | Return in excess of a benchmark. "+2% alpha vs. SPY" means the strategy beat SPY by 2 percentage points of CAGR over the period measured. |
| **Drawdown** | How far below its own all-time-high (so far) a portfolio has fallen, as a percentage. "-25% drawdown" means it's currently worth 25% less than its peak value. |
| **Max drawdown** | The worst drawdown experienced at any point during the period — a measure of "how bad did it get," independent of how things ended up. |
| **Sharpe ratio** | Return earned per unit of volatility (total ups-and-downs) taken on, relative to a risk-free rate. Higher is better; above 1.0 is generally considered good, above 2.0 is very good. Penalizes upside and downside volatility equally. |
| **Sortino ratio** | Like Sharpe, but only penalizes *downside* volatility — a strategy with big upside swings and no downside ones scores better here than under Sharpe. |
| **Calmar ratio** | CAGR divided by max drawdown — "return earned per unit of the single worst loss endured," a more drawdown-focused cousin of Sharpe. |
| **Win rate** | The percentage of rebalance periods with a positive return. |
| **Basis point (bp)** | 1/100th of a percent. 100 bps = 1%. |
| **Z-score** | How many standard deviations a value is from the average of the group — the standardization method used to make different signals comparable (see "The signals" above). |

### What makes this approach different (and what doesn't)

Being honest about this matters more than sounding impressive. Compared
to a typical hobbyist backtest (and even a fair number of professional
ones built under time pressure), the things that are genuinely more
rigorous here:

- **The AI/deterministic boundary** is enforced structurally, not just by
  intention — it's physically impossible for an LLM call to end up in the
  decision path without someone deliberately rewiring the architecture.
- **Point-in-time discipline** is explicit and tested (see
  `tests/test_fundamentals_signals.py`'s point-in-time assertions), not
  assumed.
- **Walk-forward validation exists as real, runnable infrastructure**,
  not just a phrase in a pitch deck — most retail algo-trading projects
  never build this and never find out their strategy was overfit until
  it's live and losing money.
- **Every known limitation is documented in `CLAUDE.md`** rather than
  silently glossed over — the circuit breaker's cooldown logic, the
  shallow depth of fundamentals history, the long-only approximation in
  min-variance sizing, all called out explicitly.

What this system is **not**: a proven source of market-beating returns.
The full-universe backtest run so far tracks close to SPY, sometimes
slightly ahead, sometimes slightly behind, depending on the exact period
and parameters — see "Honest limitations" below. Rigor in the *process*
is not the same thing as a validated edge in the *result*.

### Honest limitations

- **No live track record.** Everything here is backtested. Backtests are
  necessary but not sufficient — a strategy can look good in backtesting
  and still fail live, because of overfitting, regime change, or costs
  that are harder to model accurately in practice (real slippage, market
  impact of actually placing orders, liquidity drying up exactly when you
  need it).
- **Factor premiums can disappear or shrink.** Momentum, value, and
  quality have all gone through multi-year stretches of underperforming
  a simple index, and it's an open debate in finance how much of their
  historical edge gets arbitraged away once a factor becomes widely known
  and traded (value's weak 2010s are a commonly cited example).
- **Small universe, survivorship-bias-prone by construction.** The
  ~50 large-cap tickers in `config/universe.yaml` are today's well-known
  winners, hand-picked rather than pulled from a point-in-time index
  membership list — see `CLAUDE.md`'s "Known gaps" for the full list of
  what this system does and doesn't try to control for.
- **This is not investment advice.** It's a research and engineering
  project. Past performance — backtested or otherwise — does not
  guarantee future results, and nothing in this repository should be
  read as a recommendation to buy or sell any security.
