"""Walk-forward backtest loop.

At each rebalance date, signals are computed only from price history up to
that date (no lookahead), combined via the deterministic decision engine
(decision/scorer.py, decision/portfolio.py), and the simulated portfolio is
updated.

Positions are tracked as portfolio *weights*, not share counts -- this
keeps the engine simple and avoids share-rounding/cash-drag bookkeeping
that isn't the point of a v1 research backtester. One consequence: risk
checks (circuit breaker, stop-loss) are only evaluated at rebalance
granularity (weekly by default), not intraday/daily. That's a known
limitation of this simplification, not an oversight -- a stop-loss can be
breached and only get acted on at the next rebalance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from alpha_engine.backtest.costs import turnover_cost
from alpha_engine.decision.portfolio import (
    build_portfolio,
    capping_diagnostics,
    resolve_top_n,
    scale_gross_exposure,
    select_top_n,
)
from alpha_engine.decision.risk import regime_is_risk_on, stopped_out_positions
from alpha_engine.decision.scorer import combine_scores
from alpha_engine.ml.features import build_ml_features
from alpha_engine.signals import (
    ai_theme,
    earnings_revision,
    insider_buying,
    mean_reversion,
    momentum,
    quality,
    short_interest,
    value,
    volatility,
)

if TYPE_CHECKING:
    # Deferred at runtime (see the regime.mode == "hmm" branch below) so a
    # plain backtest-only install (no `ml` extra: lightgbm/hmmlearn) never
    # pays for this import unless a strategy actually opts into it.
    from alpha_engine.ml.alpha_model import AlphaModel

SIGNAL_MODULES = {
    "momentum": momentum,
    "mean_reversion": mean_reversion,
    "volatility": volatility,
}

_CADENCE_FREQ = {"daily": "B", "weekly": "W-FRI", "biweekly": "2W-FRI", "monthly": "ME"}

PERIODS_PER_YEAR = {"daily": 252, "weekly": 52, "biweekly": 26, "monthly": 12}

DEFAULT_CIRCUIT_BREAKER_COOLDOWN_PERIODS = 8


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    weights_history: dict[pd.Timestamp, pd.Series] = field(default_factory=dict)
    turnover_history: dict[pd.Timestamp, float] = field(default_factory=dict)
    regime_history: dict[pd.Timestamp, str] = field(default_factory=dict)
    capping_history: dict[pd.Timestamp, dict] = field(default_factory=dict)


def rebalance_dates(all_dates: pd.DatetimeIndex, cadence: str) -> pd.DatetimeIndex:
    """Snap each period-end (per `cadence`) to the nearest available
    trading date on or before it, so rebalances always land on a real
    market day even when the calendar period-end is a weekend/holiday.
    """
    if cadence not in _CADENCE_FREQ:
        raise ValueError(f"unknown cadence: {cadence}")
    period_ends = pd.date_range(all_dates.min(), all_dates.max(), freq=_CADENCE_FREQ[cadence])
    snapped = []
    for d in period_ends:
        available = all_dates[all_dates <= d]
        if len(available):
            snapped.append(available[-1])
    return pd.DatetimeIndex(sorted(set(snapped)))


def _prices_at(prices: dict[str, pd.DataFrame], as_of: pd.Timestamp) -> dict[str, float]:
    out: dict[str, float] = {}
    for symbol, df in prices.items():
        hist = df.loc[:as_of]
        if not hist.empty:
            out[symbol] = float(hist["close"].iloc[-1])
    return out


def run_backtest(
    prices: dict[str, pd.DataFrame],
    signal_weights: dict[str, float],
    strategy_config: dict,
    sector_map: dict[str, str] | None = None,
    fundamentals: dict[str, pd.DataFrame] | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    benchmark_prices: pd.DataFrame | None = None,
    theme_weights: dict[str, float] | None = None,
    macro_data: pd.DataFrame | None = None,
    ml_model: "AlphaModel | None" = None,
    short_interest_data: dict[str, pd.DataFrame] | None = None,
    insider_transactions: dict[str, pd.DataFrame] | None = None,
) -> BacktestResult:
    """`sector_map`, if provided: symbol -> sector name (config/sectors.yaml),
    used for the portfolio's sector-exposure cap.

    `fundamentals`, if provided: symbol -> quarterly fundamentals DataFrame
    (data.fundamentals.get_universe_fundamentals) -- when given, adds the
    `value`/`quality` signals to the signal table for this run. Symbols
    without fundamentals data (e.g. ETFs) simply score neutral on these,
    via the same missing-signal handling as everywhere else.

    `start`/`end`, if provided: restrict rebalancing to this sub-range of
    dates. Deliberately does NOT slice `prices` -- signals still see full
    lookback history from before `start` (momentum needs ~273 trading days
    of it). This is what backtest/walkforward.py uses to evaluate a single
    train/test window without starving signals of history at the window's
    start; slicing `prices` per-window would do exactly that.

    `benchmark_prices`, if provided together with `strategy_config["regime"]
    ["enabled"]`: OHLCV for the regime filter. `strategy_config["regime"]
    ["mode"]` selects which filter: `"trend_filter"` (default -- unset is
    treated the same) is the original Faber-style binary filter
    (decision.risk.regime_is_risk_on) -- when the benchmark is below its
    own trailing moving average, the portfolio holds cash regardless of
    what individual stock signals say, until the benchmark recovers above
    it again. `"hmm"` uses decision.regime's 4-state HMM classifier
    instead: a continuous gross-exposure scalar (decision.portfolio.
    scale_gross_exposure) rather than an all-or-nothing flatten, plus
    regime-adaptive signal-weight overrides for that period only -- see
    decision/regime.py's module docstring. `"hmm"` mode additionally needs
    `macro_data`; without it (or without `benchmark_prices`), it silently
    degrades to full exposure/no override rather than failing, same
    "missing data degrades gracefully" posture as everywhere else.
    `"trend_filter_scaled"` is the same binary above/below-moving-average
    signal as `"trend_filter"`, but scales exposure to
    `regime.risk_off_exposure` (default 0.6) instead of fully flattening --
    for a strategy whose regime overlay is meant to stay independent of
    another strategy's own regime mechanism (e.g. a momentum strategy
    blended alongside an HMM-driven one). Records `"RISK_ON"`/`"RISK_OFF"`
    into `regime_history`. Every date `evaluate_regime` (or the
    `trend_filter_scaled` check) actually runs on is recorded in the
    returned `BacktestResult.regime_history` (label only, not the
    exposure/weight values -- those are recoverable from `weights_history`
    instead) --
    diagnostic use, e.g. auditing mean gross exposure per regime. The
    circuit breaker and stop-loss below are unaffected by `regime.mode` --
    both remain in force regardless.

    `strategy_config["risk"]["reduce_drawdown_pct"]`, if set: an opt-in
    intermediate drawdown tier between "normal" and the circuit breaker's
    binary HALT (`risk.max_drawdown_pct`) -- once drawdown from the
    running peak crosses this (shallower) threshold, exposure is scaled
    by `risk.reduce_exposure_scale` (default 0.5) instead of doing
    nothing until HALT fires. Unset by default (`None`), which is a
    complete no-op -- existing strategies that never set it get
    byte-identical behavior. If `regime.mode == "hmm"` is also active in
    the same period, the two exposure signals combine via `min()` (the
    more cautious of the two wins), not multiplicatively.

    `theme_weights`, if provided: symbol -> AI-exposure intensity
    (config/themes.yaml's `ai` mapping) -- adds the `ai_theme` signal
    (signals/ai_theme.py) to the signal table, and the resulting
    concentration in tagged names is capped by
    `strategy_config["portfolio"]["max_theme_weight"]` if set (see
    decision/portfolio.apply_theme_cap).

    `macro_data`, if provided: alpha_engine.data.macro.get_macro_data()'s
    combined yield-curve/vix/credit-spread panel -- required for
    `regime.mode == "hmm"` (see above) and forwarded into the `ml_alpha`
    feature builder below when `ml_model` is also given.

    `ml_model`, if provided: a fitted `alpha_engine.ml.alpha_model.
    AlphaModel` -- adds an `ml_alpha` signal to the signal table for this
    run (`signal_weights.ml_alpha` in strategy.yaml), gated exactly like
    `value`/`quality` are gated on `fundamentals`. Callers are responsible
    for fitting/loading the model appropriately for point-in-time safety
    (backtest/walkforward.py fits one model per train window and reuses it
    for that window's test evaluation; the web backend loads a model
    retrained on a periodic schedule for live use) -- `run_backtest` itself
    has no awareness of train/test windows or retrain schedules.

    `short_interest_data`, if provided: symbol -> bi-weekly short-interest
    DataFrame (data.short_interest.get_universe_short_interest) -- adds
    the `short_interest` signal to the signal table. `insider_transactions`,
    if provided: symbol -> open-market Form 4 transaction DataFrame
    (data.insider.get_universe_insider_transactions) -- adds the
    `insider_buying` signal. `earnings_revision` (PEAD proxy) is added
    whenever both `fundamentals` and `benchmark_prices` are given, no
    separate param needed. All three also feed the `ml_alpha` feature
    builder when `ml_model` is given. None of the three are wired into
    live/runner.py's `run_live_rebalance` (see that function's docstring)
    -- a strategy weighting them will simply have those signals silently
    score neutral (0) in live mode, same missing-signal handling as
    everywhere else, not a special case.
    """
    sector_map = sector_map or {}
    regime_cfg = strategy_config.get("regime", {})
    regime_mode = regime_cfg.get("mode", "trend_filter")
    all_dates = pd.DatetimeIndex(sorted(set().union(*(df.index for df in prices.values()))))
    dates = rebalance_dates(all_dates, strategy_config["rebalance"])
    if start is not None:
        dates = dates[dates >= start]
    if end is not None:
        dates = dates[dates <= end]

    portfolio_cfg = strategy_config["portfolio"]
    risk_cfg = strategy_config["risk"]
    cost_cfg = strategy_config["costs"]

    equity = float(strategy_config["backtest"]["initial_capital"])
    weights: pd.Series = pd.Series(dtype=float)
    entry_prices: dict[str, float] = {}

    equity_points: dict[pd.Timestamp, float] = {}
    weights_history: dict[pd.Timestamp, pd.Series] = {}
    turnover_history: dict[pd.Timestamp, float] = {}
    regime_history: dict[pd.Timestamp, str] = {}
    capping_history: dict[pd.Timestamp, dict] = {}

    prev_prices: dict[str, float] = {}
    has_prev = False

    # Circuit-breaker state: `breaker_peak` is the reference high-water mark
    # used for the drawdown check, tracked separately from the full equity
    # curve. This matters once the breaker has fired: while flattened to
    # cash, equity is frozen, so drawdown-from-the-pre-crash-peak would
    # never recover on its own and the strategy would be locked out of the
    # market forever. Instead: stay in cash for a fixed cooldown, then
    # re-enter and reset `breaker_peak` to the post-cooldown equity level,
    # so the drawdown clock restarts from where the strategy actually is
    # rather than an unrecoverable historical high.
    breaker_peak = equity
    cooldown_remaining = 0
    cooldown_periods = risk_cfg.get(
        "circuit_breaker_cooldown_periods", DEFAULT_CIRCUIT_BREAKER_COOLDOWN_PERIODS
    )

    for as_of in dates:
        current_prices = _prices_at(prices, as_of)

        # Accrue return over the period since the last rebalance using the
        # PRIOR period's target weights -- weights were decided last period
        # using only data available then, applied here to this period's
        # realized return. No lookahead.
        if has_prev and not weights.empty:
            period_return = sum(
                w * (current_prices[s] / prev_prices[s] - 1.0)
                for s, w in weights.items()
                if s in current_prices and s in prev_prices
            )
            equity *= 1.0 + period_return

        equity_points[as_of] = equity

        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            weights = pd.Series(dtype=float)
            entry_prices = {}
            weights_history[as_of] = weights
            turnover_history[as_of] = 0.0
            if cooldown_remaining == 0:
                breaker_peak = equity  # drawdown clock restarts from here
            prev_prices, has_prev = current_prices, True
            continue

        breaker_peak = max(breaker_peak, equity)
        drawdown = equity / breaker_peak - 1.0
        if drawdown <= -risk_cfg["max_drawdown_pct"]:
            turnover = weights.abs().sum()
            equity -= turnover_cost(turnover, equity, cost_cfg)
            equity_points[as_of] = equity
            weights = pd.Series(dtype=float)
            entry_prices = {}
            weights_history[as_of] = weights
            turnover_history[as_of] = turnover
            cooldown_remaining = cooldown_periods
            prev_prices, has_prev = current_prices, True
            continue

        # REDUCE tier: opt-in (risk.reduce_drawdown_pct unset -> stays 1.0,
        # byte-identical to before this existed). An intermediate drawdown
        # zone between "fine" and the HALT threshold above scales exposure
        # down instead of doing nothing until HALT fires -- unlike HALT,
        # no cooldown/flatten-to-cash: this just dampens the otherwise-
        # normal portfolio, re-evaluated fresh every rebalance the same
        # way the trend filter is.
        risk_reduce_scale = 1.0
        reduce_threshold = risk_cfg.get("reduce_drawdown_pct")
        if reduce_threshold is not None and drawdown <= -reduce_threshold:
            risk_reduce_scale = risk_cfg.get("reduce_exposure_scale", 0.5)

        gross_exposure = 1.0
        effective_signal_weights = signal_weights

        if (
            regime_cfg.get("enabled", False)
            and regime_mode == "trend_filter"
            and benchmark_prices is not None
            and not regime_is_risk_on(benchmark_prices, as_of, regime_cfg["ma_window_days"])
        ):
            # Unlike the circuit breaker, no cooldown here -- re-evaluated
            # fresh every rebalance, so the portfolio resumes investing the
            # moment the benchmark crosses back above its moving average.
            turnover = weights.abs().sum()
            equity -= turnover_cost(turnover, equity, cost_cfg)
            equity_points[as_of] = equity
            weights = pd.Series(dtype=float)
            entry_prices = {}
            weights_history[as_of] = weights
            turnover_history[as_of] = turnover
            prev_prices, has_prev = current_prices, True
            continue
        elif regime_cfg.get("enabled", False) and regime_mode == "hmm":
            # Continuous alternative to the binary trend filter above: no
            # early `continue` here at all, even at low exposure -- the
            # portfolio still gets built normally below, just scaled down
            # (or, for a recognized weight override, tilted) rather than
            # fully flattened. See decision/regime.py's module docstring.
            if benchmark_prices is not None and macro_data is not None:
                from alpha_engine.decision.regime import evaluate_regime, resolve_exposure_table

                label, gross_exposure, weight_overrides = evaluate_regime(
                    prices,
                    benchmark_prices,
                    macro_data,
                    as_of,
                    exposure_table=resolve_exposure_table(regime_cfg),
                )
                regime_history[as_of] = label
                if weight_overrides is not None:
                    effective_signal_weights = weight_overrides
        elif regime_cfg.get("enabled", False) and regime_mode == "trend_filter_scaled":
            # Same Faber-style binary signal as "trend_filter" (benchmark
            # above/below its own trailing moving average), but continuous
            # exposure scaling instead of a full flatten -- for a strategy
            # whose regime overlay is deliberately meant to not move in
            # lockstep with another strategy's own regime mechanism (e.g.
            # a momentum strategy blended alongside an HMM-driven one:
            # both retreating for the same reason at the same time gives
            # the blend no diversification benefit). `regime.risk_off_exposure`
            # (default 0.6) sets the reduced-but-not-zero exposure level
            # used when the trend signal is risk-off.
            if benchmark_prices is not None:
                risk_on = regime_is_risk_on(benchmark_prices, as_of, regime_cfg["ma_window_days"])
                gross_exposure = 1.0 if risk_on else regime_cfg.get("risk_off_exposure", 0.6)
                regime_history[as_of] = "RISK_ON" if risk_on else "RISK_OFF"

        stopped = stopped_out_positions(entry_prices, current_prices, risk_cfg["stop_loss_pct"])

        signal_table = pd.DataFrame(
            {name: mod.compute(prices, as_of) for name, mod in SIGNAL_MODULES.items()}
        )
        if fundamentals is not None:
            signal_table["value"] = value.compute(prices, fundamentals, as_of)
            signal_table["quality"] = quality.compute(fundamentals, as_of)
            if benchmark_prices is not None:
                signal_table["earnings_revision"] = earnings_revision.compute(
                    prices, fundamentals, benchmark_prices, as_of
                )
        if theme_weights is not None:
            signal_table["ai_theme"] = ai_theme.compute(prices, as_of, theme_weights)
        if short_interest_data is not None:
            signal_table["short_interest"] = short_interest.compute(short_interest_data, as_of)
        if insider_transactions is not None:
            signal_table["insider_buying"] = insider_buying.compute(insider_transactions, as_of)
        if ml_model is not None:
            ml_features = build_ml_features(
                prices,
                fundamentals,
                macro_data,
                as_of,
                list(prices.keys()),
                benchmark_prices=benchmark_prices,
                short_interest_data=short_interest_data,
                insider_transactions=insider_transactions,
            )
            signal_table["ml_alpha"] = ml_model.predict(ml_features)

        scores = combine_scores(signal_table, effective_signal_weights)
        if stopped:
            scores = scores.drop(index=[s for s in stopped if s in scores.index])

        sizing_data = None
        if portfolio_cfg["sizing"] == "volatility_scaled":
            sizing_data = volatility.raw_annualized_vol(prices, as_of)
            # Only candidates with enough trailing history to be sized are
            # eligible for selection at all. Without this, a signal that
            # doesn't need price history itself (e.g. ai_theme, which only
            # checks dict membership) can populate `scores` before any
            # symbol has the ~20 days volatility_scaled sizing needs --
            # size_positions would then correctly refuse to size a
            # selected-but-unsizeable symbol (a real invariant it should
            # keep enforcing), so the filtering has to happen up here,
            # before selection, not by loosening that guard.
            scores = scores[scores.index.isin(sizing_data.index)]
        elif portfolio_cfg["sizing"] == "min_variance":
            sizing_data = volatility.trailing_returns_matrix(prices, list(prices.keys()), as_of)
            scores = scores[scores.index.isin(sizing_data.columns)]

        selected = select_top_n(scores, resolve_top_n(scores, portfolio_cfg))
        capping_history[as_of] = capping_diagnostics(
            selected, sizing_data, portfolio_cfg["sizing"], portfolio_cfg["max_position_weight"]
        )
        target_weights = build_portfolio(
            scores, sizing_data, sector_map=sector_map, config=portfolio_cfg, theme_map=theme_weights
        )
        # Whichever of the two independent exposure signals (regime read,
        # REDUCE-tier drawdown check) is more cautious wins -- taking the
        # min avoids the two compounding multiplicatively into an
        # unintentionally extreme combined cut.
        effective_exposure = min(gross_exposure, risk_reduce_scale)
        if effective_exposure != 1.0:
            target_weights = scale_gross_exposure(target_weights, effective_exposure)

        turnover = weights.add(-target_weights, fill_value=0.0).abs().sum()
        equity -= turnover_cost(turnover, equity, cost_cfg)
        equity_points[as_of] = equity
        turnover_history[as_of] = turnover

        for s in target_weights.index:
            if weights.get(s, 0.0) == 0.0:
                entry_prices[s] = current_prices.get(s, entry_prices.get(s, 0.0))
        for s in list(entry_prices.keys()):
            if s not in target_weights.index:
                entry_prices.pop(s, None)

        weights = target_weights
        weights_history[as_of] = weights
        prev_prices, has_prev = current_prices, True

    return BacktestResult(
        equity_curve=pd.Series(equity_points).sort_index(),
        weights_history=weights_history,
        turnover_history=turnover_history,
        regime_history=regime_history,
        capping_history=capping_history,
    )
