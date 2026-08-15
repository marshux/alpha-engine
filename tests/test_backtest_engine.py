from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_engine.backtest import metrics
from alpha_engine.backtest.engine import rebalance_dates, run_backtest


def _series_df(closes: np.ndarray, n_days: int, start="2020-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n_days)
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1_000_000},
        index=dates,
    )


def _base_strategy_config(**overrides) -> dict:
    cfg = {
        "rebalance": "weekly",
        "signal_weights": {"momentum": 1.0, "mean_reversion": 0.0, "volatility": 0.0},
        "portfolio": {
            "top_n": 2,
            "max_position_weight": 0.6,
            "max_sector_weight": 1.0,
            "sizing": "equal_weight",
        },
        "risk": {"stop_loss_pct": 0.5, "max_drawdown_pct": 0.99},
        "costs": {"commission_bps": 0.0, "slippage_bps": 0.0},
        "backtest": {"initial_capital": 100_000, "benchmark": "SPY"},
    }
    cfg.update(overrides)
    return cfg


def test_rebalance_dates_weekly_snaps_to_trading_days():
    all_dates = pd.bdate_range("2020-01-01", periods=60)
    dates = rebalance_dates(all_dates, "weekly")
    assert len(dates) > 0
    assert dates.isin(all_dates).all()
    assert dates.is_monotonic_increasing


def test_rebalance_dates_daily_matches_every_trading_day():
    all_dates = pd.bdate_range("2020-01-01", periods=60)
    dates = rebalance_dates(all_dates, "daily")
    assert dates.isin(all_dates).all()
    assert dates.is_monotonic_increasing
    assert len(dates) == len(all_dates)


def test_rebalance_dates_rejects_unknown_cadence():
    all_dates = pd.bdate_range("2020-01-01", periods=10)
    with pytest.raises(ValueError):
        rebalance_dates(all_dates, "hourly")


def test_backtest_grows_equity_when_all_assets_trend_up():
    n = 300
    up_a = _series_df(100 * (1.001 ** np.arange(n)), n)
    up_b = _series_df(50 * (1.0015 ** np.arange(n)), n)
    prices = {"A": up_a, "B": up_b}
    config = _base_strategy_config()

    result = run_backtest(prices, config["signal_weights"], config)

    assert len(result.equity_curve) > 1
    assert result.equity_curve.iloc[-1] > result.equity_curve.iloc[0]


def test_backtest_no_lookahead_first_period_flat():
    # Equity should not change between the very first rebalance point and
    # the initial capital, since there's no prior-period weight to have
    # accrued a return with yet.
    n = 300
    prices = {"A": _series_df(100 * (1.001 ** np.arange(n)), n)}
    config = _base_strategy_config()
    config["portfolio"]["top_n"] = 1

    result = run_backtest(prices, config["signal_weights"], config)

    first_equity = result.equity_curve.iloc[0]
    assert first_equity == pytest.approx(config["backtest"]["initial_capital"])


def test_backtest_costs_reduce_final_equity():
    n = 300
    rng = np.random.default_rng(7)
    a = _series_df(100 * np.cumprod(1 + rng.normal(0.001, 0.01, n)), n)
    b = _series_df(80 * np.cumprod(1 + rng.normal(0.001, 0.01, n)), n)
    prices = {"A": a, "B": b}

    cheap = _base_strategy_config()
    expensive = _base_strategy_config(costs={"commission_bps": 50.0, "slippage_bps": 100.0})

    cheap_result = run_backtest(prices, cheap["signal_weights"], cheap)
    expensive_result = run_backtest(prices, expensive["signal_weights"], expensive)

    assert expensive_result.equity_curve.iloc[-1] < cheap_result.equity_curve.iloc[-1]


def test_circuit_breaker_flattens_and_halts_losses():
    n = 300
    # sharp crash partway through
    closes = np.concatenate([np.full(150, 100.0), np.linspace(100.0, 20.0, n - 150)])
    prices = {"A": _series_df(closes, n)}
    config = _base_strategy_config(
        # cooldown far longer than the remaining test window, so this test
        # isolates the "halts losses immediately" behavior from re-entry
        # (covered separately in test_circuit_breaker_reenters_after_cooldown)
        risk={"stop_loss_pct": 0.99, "max_drawdown_pct": 0.10, "circuit_breaker_cooldown_periods": 999},
        portfolio={
            "top_n": 1,
            "max_position_weight": 1.0,
            "max_sector_weight": 1.0,
            "sizing": "equal_weight",
        },
    )

    result = run_backtest(prices, config["signal_weights"], config)

    # once flattened, equity should stay constant (no further losses accrue)
    tail = result.equity_curve.iloc[-5:]
    assert tail.max() == pytest.approx(tail.min(), rel=1e-9)
    # some weights_history entries should be empty (flattened to cash)
    assert any(w.empty for w in result.weights_history.values())


def test_circuit_breaker_reenters_after_cooldown():
    n = 400
    # flat, then a sharp one-period crash (triggers the breaker immediately),
    # then flat through the cooldown window, then a strong recovery -- the
    # strategy should re-enter and grow from the post-cooldown equity level
    # rather than staying locked in cash for the rest of the run
    closes = np.concatenate(
        [
            np.full(90, 100.0),
            np.full(1, 50.0),  # crash
            np.full(59, 50.0),  # flat while cooldown elapses
            np.linspace(50.0, 200.0, n - 150),  # recovery
        ]
    )
    prices = {"A": _series_df(closes, n)}
    config = _base_strategy_config(
        risk={"stop_loss_pct": 0.99, "max_drawdown_pct": 0.10, "circuit_breaker_cooldown_periods": 4},
        portfolio={
            "top_n": 1,
            "max_position_weight": 1.0,
            "max_sector_weight": 1.0,
            "sizing": "equal_weight",
        },
    )

    result = run_backtest(prices, config["signal_weights"], config)

    dates = sorted(result.weights_history.keys())
    # find the first rebalance after the crash where the strategy is holding
    # a position again (re-entered)
    reentry_idx = next(
        i for i, d in enumerate(dates) if not result.weights_history[d].empty and i > 5
    )
    reentry_date = dates[reentry_idx]
    equity_at_reentry = result.equity_curve.loc[reentry_date]

    assert result.equity_curve.iloc[-1] > equity_at_reentry  # grew after re-entering
    assert any(w.empty for w in result.weights_history.values())  # was flattened at some point


def test_start_end_restrict_dates_without_starving_signal_lookback():
    n = 400
    up = _series_df(100 * (1.002 ** np.arange(n)), n)  # strong uptrend
    flat = _series_df(np.full(n, 100.0), n)
    prices = {"A": up, "B": flat}
    config = _base_strategy_config(
        portfolio={
            "top_n": 1,
            "max_position_weight": 1.0,
            "max_sector_weight": 1.0,
            "sizing": "equal_weight",
        }
    )
    all_dates = pd.DatetimeIndex(sorted(set(up.index) | set(flat.index)))
    start = all_dates[300]  # well past momentum's 273-day lookback requirement

    result = run_backtest(prices, config["signal_weights"], config, start=start)

    assert result.equity_curve.index.min() >= start
    first_date = sorted(result.weights_history.keys())[0]
    # if `prices` had been sliced to `start` instead of kept full, momentum
    # would have no lookback history yet and this would tie/neutral instead
    # of clearly picking the uptrend symbol
    assert list(result.weights_history[first_date].index) == ["A"]


def test_start_end_produces_fewer_dates_than_unrestricted():
    n = 300
    prices = {"A": _series_df(np.full(n, 100.0), n)}
    config = _base_strategy_config()
    all_dates = pd.DatetimeIndex(prices["A"].index)
    midpoint = all_dates[len(all_dates) // 2]

    full_result = run_backtest(prices, config["signal_weights"], config)
    restricted_result = run_backtest(prices, config["signal_weights"], config, start=midpoint)

    assert len(restricted_result.equity_curve) < len(full_result.equity_curve)


def test_regime_filter_holds_cash_during_benchmark_downtrend():
    n = 300
    # stock A trends up steadily the whole time -- signals alone would
    # always want to hold it
    a = _series_df(100 * (1.001 ** np.arange(n)), n)
    prices = {"A": a}

    # benchmark: up (0-99), sharp sustained decline (100-199, clearly below
    # its own trailing MA), recovery (200-299)
    benchmark_closes = np.concatenate(
        [
            np.linspace(100.0, 150.0, 100),
            np.linspace(150.0, 80.0, 100),
            np.linspace(80.0, 160.0, 100),
        ]
    )
    benchmark_prices = _series_df(benchmark_closes, n)

    config = _base_strategy_config(
        portfolio={
            "top_n": 1,
            "max_position_weight": 1.0,
            "max_sector_weight": 1.0,
            "sizing": "equal_weight",
        },
        regime={"enabled": True, "ma_window_days": 20},
    )

    result = run_backtest(prices, config["signal_weights"], config, benchmark_prices=benchmark_prices)

    dates = sorted(result.weights_history.keys())
    downtrend_dates = [d for d in dates if benchmark_prices.index[130] <= d <= benchmark_prices.index[190]]
    recovery_dates = [d for d in dates if d >= benchmark_prices.index[260]]

    assert downtrend_dates, "test setup issue: no rebalance dates fell in the downtrend window"
    assert recovery_dates, "test setup issue: no rebalance dates fell in the recovery window"
    assert all(result.weights_history[d].empty for d in downtrend_dates)
    assert all(not result.weights_history[d].empty for d in recovery_dates)


def test_regime_filter_disabled_by_default_ignores_benchmark():
    n = 200
    a = _series_df(100 * (1.001 ** np.arange(n)), n)
    prices = {"A": a}
    # benchmark deep in a "downtrend" the whole time, but regime is not enabled
    benchmark_prices = _series_df(np.linspace(150.0, 50.0, n), n)

    config = _base_strategy_config(
        portfolio={
            "top_n": 1,
            "max_position_weight": 1.0,
            "max_sector_weight": 1.0,
            "sizing": "equal_weight",
        },
    )  # no "regime" key at all

    result = run_backtest(prices, config["signal_weights"], config, benchmark_prices=benchmark_prices)

    dates = sorted(result.weights_history.keys())
    assert any(not result.weights_history[d].empty for d in dates[5:])


def test_theme_weights_break_ties_toward_tagged_symbol():
    n = 60
    # two symbols with IDENTICAL price behavior -- every price-based signal
    # scores them identically, so any selection difference must come from
    # the ai_theme signal specifically
    closes = 100 * (1.001 ** np.arange(n))
    prices = {"TAGGED": _series_df(closes, n), "OTHER": _series_df(closes.copy(), n)}

    config = _base_strategy_config(
        signal_weights={"momentum": 0.0, "mean_reversion": 0.0, "volatility": 0.0, "ai_theme": 1.0},
        portfolio={
            "top_n": 1,
            "max_position_weight": 1.0,
            "max_sector_weight": 1.0,
            "sizing": "equal_weight",
        },
    )

    result = run_backtest(
        prices, config["signal_weights"], config, theme_weights={"TAGGED": 1.0}
    )

    dates = sorted(result.weights_history.keys())
    last_nonempty = [d for d in dates if not result.weights_history[d].empty][-1]
    assert list(result.weights_history[last_nonempty].index) == ["TAGGED"]


def test_volatility_scaled_sizing_survives_cold_start_with_history_free_signal():
    # Regression test: ai_theme (and any other signal that doesn't need
    # price history) can populate `scores` before any symbol has the ~20
    # days volatility_scaled sizing requires. Selection must be filtered
    # down to sizeable candidates before size_positions runs, not crash.
    n = 60
    closes = 100 * (1.001 ** np.arange(n))
    prices = {"TAGGED": _series_df(closes, n), "OTHER": _series_df(closes.copy(), n)}

    config = _base_strategy_config(
        signal_weights={"momentum": 0.0, "mean_reversion": 0.0, "volatility": 0.0, "ai_theme": 1.0},
        portfolio={
            "top_n": 2,
            "max_position_weight": 1.0,
            "max_sector_weight": 1.0,
            "sizing": "volatility_scaled",
        },
    )

    # must not raise, even though at the first rebalance neither symbol has
    # 20 days of history for volatility_scaled sizing yet
    result = run_backtest(prices, config["signal_weights"], config, theme_weights={"TAGGED": 1.0})

    dates = sorted(result.weights_history.keys())
    assert result.weights_history[dates[0]].empty  # cold start: nothing sizeable yet
    assert not result.weights_history[dates[-1]].empty  # eventually invests once history exists


def test_volatility_scaled_sizing_survives_cold_start_with_fundamentals_signal():
    # Same root cause as the test above, but this is the actual failure
    # mode a real --fundamentals + volatility_scaled walk-forward run hit:
    # quality/value are populated from fundamentals reports, which don't
    # depend on price history either, so they can populate `scores` at a
    # cold start too.
    n = 30
    closes = 100 * (1.001 ** np.arange(n))
    prices = {"A": _series_df(closes, n), "B": _series_df(closes.copy(), n)}
    fundamentals = {
        "A": pd.DataFrame(
            {"Stockholders Equity": [100.0], "Net Income": [20.0], "Total Debt": [10.0]},
            index=pd.to_datetime(["2019-12-31"]),  # before the first rebalance date
        ),
    }

    config = _base_strategy_config(
        signal_weights={"momentum": 0.0, "mean_reversion": 0.0, "volatility": 0.0, "quality": 1.0},
        portfolio={
            "top_n": 2,
            "max_position_weight": 1.0,
            "max_sector_weight": 1.0,
            "sizing": "volatility_scaled",
        },
    )

    result = run_backtest(prices, config["signal_weights"], config, fundamentals=fundamentals)

    dates = sorted(result.weights_history.keys())
    assert result.weights_history[dates[0]].empty
    assert not result.weights_history[dates[-1]].empty


def test_metrics_cagr_and_drawdown_on_known_curve():
    dates = pd.bdate_range("2020-01-01", periods=3)
    curve = pd.Series([100.0, 150.0, 90.0], index=dates)

    dd = metrics.max_drawdown(curve)
    assert dd == pytest.approx(90.0 / 150.0 - 1.0)

    flat = pd.Series([100.0, 100.0, 100.0], index=dates)
    assert metrics.sharpe_ratio(flat, periods_per_year=52) == 0.0


def test_hmm_regime_mode_scales_exposure_continuously_not_binary(monkeypatch):
    n = 300
    a = _series_df(100 * (1.001 ** np.arange(n)), n)
    prices = {"A": a}
    benchmark_prices = _series_df(100 * (1.0005 ** np.arange(n)), n)
    macro_data = pd.DataFrame(
        {"yield_curve": 1.0, "vix": 15.0, "credit_spread": 3.0},
        index=pd.bdate_range("2020-01-01", periods=n),
    )

    def fake_evaluate_regime(prices_, benchmark_prices_, macro_data_, as_of, **kwargs):
        return ("BEAR_TREND", 0.4, None)

    monkeypatch.setattr("alpha_engine.decision.regime.evaluate_regime", fake_evaluate_regime)

    config = _base_strategy_config(
        portfolio={"top_n": 1, "max_position_weight": 1.0, "max_sector_weight": 1.0, "sizing": "equal_weight"},
        regime={"enabled": True, "mode": "hmm", "ma_window_days": 20},
    )

    result = run_backtest(
        prices, config["signal_weights"], config, benchmark_prices=benchmark_prices, macro_data=macro_data
    )

    dates = sorted(result.weights_history.keys())
    last_nonempty = [d for d in dates if not result.weights_history[d].empty][-1]
    # Scaled to the fake regime's 0.4 exposure -- not fully invested (1.0)
    # and not flattened to cash (0.0), unlike the binary trend filter.
    assert result.weights_history[last_nonempty]["A"] == pytest.approx(0.4)


def test_hmm_regime_mode_applies_regime_weight_overrides(monkeypatch):
    n = 300
    # UP: strong uptrend (momentum favors it). FLAT: dead flat (mean
    # reversion is indifferent, but momentum scores it far worse than UP).
    up = _series_df(100 * (1.003 ** np.arange(n)), n)
    flat = _series_df(np.full(n, 100.0), n)
    prices = {"UP": up, "FLAT": flat}
    benchmark_prices = _series_df(100 * (1.0005 ** np.arange(n)), n)
    macro_data = pd.DataFrame(
        {"yield_curve": 1.0, "vix": 15.0, "credit_spread": 3.0},
        index=pd.bdate_range("2020-01-01", periods=n),
    )

    def fake_evaluate_regime(prices_, benchmark_prices_, macro_data_, as_of, **kwargs):
        # Force weights entirely onto mean_reversion (momentum key absent
        # -> excluded from combine_scores entirely, see decision/scorer.py).
        return ("CRISIS", 1.0, {"mean_reversion": 1.0})

    monkeypatch.setattr("alpha_engine.decision.regime.evaluate_regime", fake_evaluate_regime)

    config = _base_strategy_config(
        signal_weights={"momentum": 1.0, "mean_reversion": 0.0, "volatility": 0.0},
        portfolio={"top_n": 1, "max_position_weight": 1.0, "max_sector_weight": 1.0, "sizing": "equal_weight"},
        regime={"enabled": True, "mode": "hmm", "ma_window_days": 20},
    )

    result = run_backtest(
        prices, config["signal_weights"], config, benchmark_prices=benchmark_prices, macro_data=macro_data
    )

    dates = sorted(result.weights_history.keys())
    last_nonempty = [d for d in dates if not result.weights_history[d].empty][-1]
    # Under the base (momentum-only) weights UP would always win -- the
    # override swapping in mean_reversion-only proves it was actually used.
    assert list(result.weights_history[last_nonempty].index) != ["UP"]


def test_hmm_regime_mode_records_regime_label_per_date(monkeypatch):
    n = 300
    a = _series_df(100 * (1.001 ** np.arange(n)), n)
    prices = {"A": a}
    benchmark_prices = _series_df(100 * (1.0005 ** np.arange(n)), n)
    macro_data = pd.DataFrame(
        {"yield_curve": 1.0, "vix": 15.0, "credit_spread": 3.0},
        index=pd.bdate_range("2020-01-01", periods=n),
    )

    def fake_evaluate_regime(prices_, benchmark_prices_, macro_data_, as_of, **kwargs):
        return ("BEAR_TREND", 0.4, None)

    monkeypatch.setattr("alpha_engine.decision.regime.evaluate_regime", fake_evaluate_regime)

    config = _base_strategy_config(
        portfolio={"top_n": 1, "max_position_weight": 1.0, "max_sector_weight": 1.0, "sizing": "equal_weight"},
        regime={"enabled": True, "mode": "hmm", "ma_window_days": 20},
    )

    result = run_backtest(
        prices, config["signal_weights"], config, benchmark_prices=benchmark_prices, macro_data=macro_data
    )

    assert result.regime_history
    assert set(result.regime_history.values()) == {"BEAR_TREND"}


def test_trend_filter_mode_leaves_regime_history_empty():
    # regime_history is only populated in "hmm" mode -- the binary trend
    # filter has no regime label concept, and this must stay a true no-op
    # for strategies that never touch regime.mode at all (Traditional,
    # Contrarian Volatility).
    n = 300
    a = _series_df(100 * (1.001 ** np.arange(n)), n)
    prices = {"A": a}
    benchmark_prices = _series_df(100 * (1.0005 ** np.arange(n)), n)

    config = _base_strategy_config(
        portfolio={"top_n": 1, "max_position_weight": 1.0, "max_sector_weight": 1.0, "sizing": "equal_weight"},
        regime={"enabled": True, "mode": "trend_filter", "ma_window_days": 20},
    )

    result = run_backtest(prices, config["signal_weights"], config, benchmark_prices=benchmark_prices)

    assert result.regime_history == {}


def test_hmm_regime_mode_without_macro_data_degrades_to_full_exposure():
    n = 300
    a = _series_df(100 * (1.001 ** np.arange(n)), n)
    prices = {"A": a}
    benchmark_prices = _series_df(100 * (1.0005 ** np.arange(n)), n)

    config = _base_strategy_config(
        portfolio={"top_n": 1, "max_position_weight": 1.0, "max_sector_weight": 1.0, "sizing": "equal_weight"},
        regime={"enabled": True, "mode": "hmm", "ma_window_days": 20},
    )

    # No macro_data passed -- hmm mode should degrade gracefully rather
    # than error, same posture as every other missing-data case.
    result = run_backtest(prices, config["signal_weights"], config, benchmark_prices=benchmark_prices)

    dates = sorted(result.weights_history.keys())
    last_nonempty = [d for d in dates if not result.weights_history[d].empty][-1]
    assert result.weights_history[last_nonempty]["A"] == pytest.approx(1.0)


def test_ml_model_signal_participates_in_scoring():
    n = 300
    up = _series_df(100 * (1.001 ** np.arange(n)), n)
    down = _series_df(100 * (0.999 ** np.arange(n)), n)
    prices = {"UP": up, "DOWN": down}

    class _FakeMlModel:
        def predict(self, X):
            return pd.Series({sym: (1.0 if sym == "DOWN" else -1.0) for sym in X.index})

    config = _base_strategy_config(
        signal_weights={"momentum": 0.0, "mean_reversion": 0.0, "volatility": 0.0, "ml_alpha": 1.0},
        portfolio={"top_n": 1, "max_position_weight": 1.0, "max_sector_weight": 1.0, "sizing": "equal_weight"},
    )

    result = run_backtest(prices, config["signal_weights"], config, ml_model=_FakeMlModel())

    dates = sorted(result.weights_history.keys())
    last_nonempty = [d for d in dates if not result.weights_history[d].empty][-1]
    # DOWN only wins because the fake model favors it -- momentum/mean
    # reversion would both favor UP, so this proves ml_alpha actually
    # participated in scoring, not just got computed and ignored.
    assert list(result.weights_history[last_nonempty].index) == ["DOWN"]


def test_reduce_tier_scales_exposure_at_intermediate_drawdown():
    n = 300
    # Flat, then a moderate ~10% decline -- crosses a 5% REDUCE threshold
    # but stays well under the (deliberately very high) HALT threshold.
    closes = np.concatenate([np.full(150, 100.0), np.linspace(100.0, 90.0, n - 150)])
    prices = {"A": _series_df(closes, n)}
    config = _base_strategy_config(
        risk={
            "stop_loss_pct": 0.99,
            "max_drawdown_pct": 0.99,
            "reduce_drawdown_pct": 0.05,
            "reduce_exposure_scale": 0.4,
        },
        portfolio={"top_n": 1, "max_position_weight": 1.0, "max_sector_weight": 1.0, "sizing": "equal_weight"},
    )

    result = run_backtest(prices, config["signal_weights"], config)

    dates = sorted(result.weights_history.keys())
    assert result.weights_history[dates[-1]]["A"] == pytest.approx(0.4)


def test_reduce_tier_disabled_by_default_ignores_drawdown():
    n = 300
    closes = np.concatenate([np.full(150, 100.0), np.linspace(100.0, 90.0, n - 150)])
    prices = {"A": _series_df(closes, n)}
    config = _base_strategy_config(
        risk={"stop_loss_pct": 0.99, "max_drawdown_pct": 0.99},  # no reduce_drawdown_pct at all
        portfolio={"top_n": 1, "max_position_weight": 1.0, "max_sector_weight": 1.0, "sizing": "equal_weight"},
    )

    result = run_backtest(prices, config["signal_weights"], config)

    dates = sorted(result.weights_history.keys())
    assert result.weights_history[dates[-1]]["A"] == pytest.approx(1.0)


def test_reduce_tier_combines_with_hmm_regime_via_min(monkeypatch):
    n = 300
    closes = np.concatenate([np.full(150, 100.0), np.linspace(100.0, 90.0, n - 150)])
    prices = {"A": _series_df(closes, n)}
    benchmark_prices = _series_df(100 * (1.0005 ** np.arange(n)), n)
    macro_data = pd.DataFrame(
        {"yield_curve": 1.0, "vix": 15.0, "credit_spread": 3.0}, index=pd.bdate_range("2020-01-01", periods=n)
    )

    def fake_evaluate_regime(prices_, benchmark_prices_, macro_data_, as_of, **kwargs):
        return ("BULL_TREND", 0.9, None)  # regime alone would only cut to 0.9

    monkeypatch.setattr("alpha_engine.decision.regime.evaluate_regime", fake_evaluate_regime)

    config = _base_strategy_config(
        risk={
            "stop_loss_pct": 0.99,
            "max_drawdown_pct": 0.99,
            "reduce_drawdown_pct": 0.05,
            "reduce_exposure_scale": 0.3,  # REDUCE is the more cautious of the two
        },
        portfolio={"top_n": 1, "max_position_weight": 1.0, "max_sector_weight": 1.0, "sizing": "equal_weight"},
        regime={"enabled": True, "mode": "hmm", "ma_window_days": 20},
    )

    result = run_backtest(
        prices, config["signal_weights"], config, benchmark_prices=benchmark_prices, macro_data=macro_data
    )

    dates = sorted(result.weights_history.keys())
    # min(0.9, 0.3) == 0.3 -- REDUCE wins, not a 0.9*0.3 compound.
    assert result.weights_history[dates[-1]]["A"] == pytest.approx(0.3)


def test_short_interest_signal_participates_in_scoring(monkeypatch):
    n = 300
    prices = {"UP": _series_df(100 * (1.001 ** np.arange(n)), n), "DOWN": _series_df(100 * (0.999 ** np.arange(n)), n)}

    monkeypatch.setattr(
        "alpha_engine.signals.short_interest.compute",
        lambda short_interest_data, as_of: pd.Series({"UP": -1.0, "DOWN": 1.0}),
    )

    config = _base_strategy_config(
        signal_weights={"momentum": 0.0, "mean_reversion": 0.0, "volatility": 0.0, "short_interest": 1.0},
        portfolio={"top_n": 1, "max_position_weight": 1.0, "max_sector_weight": 1.0, "sizing": "equal_weight"},
    )

    result = run_backtest(prices, config["signal_weights"], config, short_interest_data={"UP": pd.DataFrame(), "DOWN": pd.DataFrame()})

    dates = sorted(result.weights_history.keys())
    last_nonempty = [d for d in dates if not result.weights_history[d].empty][-1]
    # DOWN only wins because the fake short_interest signal favors it --
    # momentum favors UP, proving short_interest actually drove selection.
    assert list(result.weights_history[last_nonempty].index) == ["DOWN"]


def test_insider_buying_signal_participates_in_scoring(monkeypatch):
    n = 300
    prices = {"UP": _series_df(100 * (1.001 ** np.arange(n)), n), "DOWN": _series_df(100 * (0.999 ** np.arange(n)), n)}

    monkeypatch.setattr(
        "alpha_engine.signals.insider_buying.compute",
        lambda insider_transactions, as_of: pd.Series({"UP": -1.0, "DOWN": 1.0}),
    )

    config = _base_strategy_config(
        signal_weights={"momentum": 0.0, "mean_reversion": 0.0, "volatility": 0.0, "insider_buying": 1.0},
        portfolio={"top_n": 1, "max_position_weight": 1.0, "max_sector_weight": 1.0, "sizing": "equal_weight"},
    )

    result = run_backtest(prices, config["signal_weights"], config, insider_transactions={"UP": pd.DataFrame(), "DOWN": pd.DataFrame()})

    dates = sorted(result.weights_history.keys())
    last_nonempty = [d for d in dates if not result.weights_history[d].empty][-1]
    assert list(result.weights_history[last_nonempty].index) == ["DOWN"]


def test_earnings_revision_signal_added_when_fundamentals_and_benchmark_given(monkeypatch):
    n = 300
    prices = {"UP": _series_df(100 * (1.001 ** np.arange(n)), n), "DOWN": _series_df(100 * (0.999 ** np.arange(n)), n)}
    benchmark_prices = _series_df(100 * (1.0005 ** np.arange(n)), n)
    fundamentals = {
        "UP": pd.DataFrame({"Stockholders Equity": [100.0]}, index=pd.to_datetime(["2020-01-01"])),
        "DOWN": pd.DataFrame({"Stockholders Equity": [100.0]}, index=pd.to_datetime(["2020-01-01"])),
    }

    monkeypatch.setattr(
        "alpha_engine.signals.earnings_revision.compute",
        lambda prices_, fundamentals_, benchmark_prices_, as_of: pd.Series({"UP": -1.0, "DOWN": 1.0}),
    )

    config = _base_strategy_config(
        signal_weights={"momentum": 0.0, "mean_reversion": 0.0, "volatility": 0.0, "earnings_revision": 1.0},
        portfolio={"top_n": 1, "max_position_weight": 1.0, "max_sector_weight": 1.0, "sizing": "equal_weight"},
    )

    result = run_backtest(
        prices, config["signal_weights"], config, fundamentals=fundamentals, benchmark_prices=benchmark_prices
    )

    dates = sorted(result.weights_history.keys())
    last_nonempty = [d for d in dates if not result.weights_history[d].empty][-1]
    assert list(result.weights_history[last_nonempty].index) == ["DOWN"]


def test_earnings_revision_not_added_without_benchmark_prices():
    n = 300
    prices = {"A": _series_df(100 * (1.001 ** np.arange(n)), n)}
    fundamentals = {"A": pd.DataFrame({"Stockholders Equity": [100.0]}, index=pd.to_datetime(["2020-01-01"]))}

    config = _base_strategy_config(
        signal_weights={"momentum": 0.0, "mean_reversion": 0.0, "volatility": 0.0, "earnings_revision": 1.0},
        portfolio={"top_n": 1, "max_position_weight": 1.0, "max_sector_weight": 1.0, "sizing": "equal_weight"},
    )

    # No benchmark_prices given -- earnings_revision weight is nonzero but
    # the signal never gets added to signal_table, so it's silently
    # excluded from scoring (combine_scores' missing-signal handling)
    # rather than raising.
    result = run_backtest(prices, config["signal_weights"], config, fundamentals=fundamentals)

    assert result.equity_curve is not None  # completes without error


def test_ml_features_receive_new_signal_inputs_when_ml_model_given():
    n = 300
    prices = {"A": _series_df(100 * (1.001 ** np.arange(n)), n)}
    benchmark_prices = _series_df(100 * (1.0005 ** np.arange(n)), n)
    captured_columns = []

    class _CapturingMlModel:
        def predict(self, X):
            captured_columns.append(set(X.columns))
            return pd.Series({s: 0.0 for s in X.index})

    config = _base_strategy_config(
        signal_weights={"momentum": 1.0, "mean_reversion": 0.0, "volatility": 0.0},
        portfolio={"top_n": 1, "max_position_weight": 1.0, "max_sector_weight": 1.0, "sizing": "equal_weight"},
    )

    run_backtest(
        prices,
        config["signal_weights"],
        config,
        benchmark_prices=benchmark_prices,
        ml_model=_CapturingMlModel(),
        short_interest_data={"A": pd.DataFrame()},
        insider_transactions={"A": pd.DataFrame()},
    )

    assert len(captured_columns) > 0
    assert {"earnings_revision", "short_interest", "insider_buying"} <= captured_columns[0]
