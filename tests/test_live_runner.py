"""Tests for live/runner.py's orchestration logic -- config/price/broker
fetches are all monkeypatched (no network calls, no real Alpaca account),
and state.py's JSON file is redirected to tmp_path so tests never touch
the real data/live_state.json.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from alpha_engine.live import runner
from alpha_engine.live.state import LiveState, save_state

_SIGNAL_MODS = (
    "momentum",
    "mean_reversion",
    "volatility",
    "value",
    "quality",
    "ai_theme",
)


def _base_strategy_config(**overrides) -> dict:
    cfg = {
        "rebalance": "weekly",
        "signal_weights": {
            "momentum": 0.2,
            "mean_reversion": 0.2,
            "volatility": 0.1,
            "value": 0.15,
            "quality": 0.15,
            "ai_theme": 0.2,
        },
        "portfolio": {
            "top_n": 2,
            "max_position_weight": 0.6,
            "max_sector_weight": 1.0,
            "max_theme_weight": 0.8,
            "sizing": "equal_weight",
        },
        "risk": {"stop_loss_pct": 0.5, "max_drawdown_pct": 0.2, "circuit_breaker_cooldown_periods": 4},
        "regime": {"enabled": False, "ma_window_days": 200},
        "backtest": {"initial_capital": 100_000, "benchmark": "SPY"},
    }
    cfg.update(overrides)
    return cfg


def _series_df(close: float, n_days: int = 5, start: str = "2026-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n_days)
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1_000_000}, index=dates
    )


def _patch_config(monkeypatch, strategy_overrides=None):
    monkeypatch.setattr(runner, "load_strategy", lambda: _base_strategy_config(**(strategy_overrides or {})))
    monkeypatch.setattr(runner, "load_universe", lambda: {"large_caps": ["AAPL", "MSFT"], "benchmarks": ["SPY"]})
    monkeypatch.setattr(runner, "load_sector_map", lambda: {"AAPL": "Technology", "MSFT": "Technology"})
    monkeypatch.setattr(runner, "load_themes", lambda: {"ai": {"AAPL": 1.0}})
    monkeypatch.setattr(runner, "universe_symbols", lambda universe: list(universe["large_caps"]))


def _patch_data_fetches(monkeypatch):
    monkeypatch.setattr(
        runner,
        "get_universe_prices",
        lambda symbols, start, end: {s: _series_df(100.0) for s in symbols} | {"SPY": _series_df(400.0)},
    )
    monkeypatch.setattr(runner, "get_universe_fundamentals", lambda symbols: {})


def _patch_neutral_sentiment(monkeypatch, calls=None):
    def fake(as_of):
        if calls is not None:
            calls.append(as_of)
        return {"sentiment": 0.0, "confidence": 0.0, "event_tags": [], "summary": ""}

    monkeypatch.setattr(runner.news_ai, "compute_ai_sector_sentiment", fake)


def test_skips_if_already_ran_today(tmp_path, monkeypatch):
    state_path = tmp_path / "live_state.json"
    save_state(LiveState(last_rebalance_date=pd.Timestamp.now().normalize().date().isoformat()), state_path)

    calls = []
    monkeypatch.setattr(runner.broker, "get_account", lambda client: calls.append("get_account"))

    result = runner.run_live_rebalance(state_path=state_path, broker_client=object())

    assert result.skipped_reason == "already ran today"
    assert calls == []  # never touched the broker at all


def test_full_signal_table_is_assembled(tmp_path, monkeypatch):
    _patch_config(monkeypatch)
    _patch_data_fetches(monkeypatch)

    called = []

    def _tracked(name):
        def _fake(*args, **kwargs):
            called.append(name)
            return pd.Series({"AAPL": 1.0, "MSFT": 0.0})
        return _fake

    for name in _SIGNAL_MODS:
        monkeypatch.setattr(getattr(runner, name), "compute", _tracked(name))

    sentiment_calls = []
    _patch_neutral_sentiment(monkeypatch, sentiment_calls)

    monkeypatch.setattr(runner.broker, "get_account", lambda client: {"equity": 100_000.0, "cash": 100_000.0, "buying_power": 100_000.0})
    monkeypatch.setattr(runner.broker, "get_positions", lambda client: {})

    result = runner.run_live_rebalance(
        dry_run=True, state_path=tmp_path / "live_state.json", broker_client=object()
    )

    # All six per-stock signal modules must be called -- not just the
    # 3-entry SIGNAL_MODULES dict (momentum/mean_reversion/volatility).
    # Missing value/quality/ai_theme here would silently under-signal in
    # live trading despite strategy.yaml weighting them.
    assert set(called) == set(_SIGNAL_MODS)
    # AI sentiment is a single portfolio-level call, not a per-symbol
    # signal -- exactly one call, not one per symbol.
    assert len(sentiment_calls) == 1
    assert not result.target_weights.empty
    assert result.dry_run is True


def test_ai_sentiment_modulates_theme_cap(tmp_path, monkeypatch):
    _patch_config(monkeypatch)
    _patch_data_fetches(monkeypatch)

    for name in _SIGNAL_MODS:
        monkeypatch.setattr(getattr(runner, name), "compute", lambda *a, **k: pd.Series({"AAPL": 1.0, "MSFT": 0.5}))

    monkeypatch.setattr(
        runner.news_ai,
        "compute_ai_sector_sentiment",
        lambda as_of: {"sentiment": 0.8, "confidence": 0.9, "event_tags": [], "summary": ""},
    )

    modulate_calls = []
    real_modulate = runner.modulate_theme_cap

    def _tracked_modulate(base, sentiment, confidence, *a, **k):
        modulate_calls.append((base, sentiment, confidence))
        return real_modulate(base, sentiment, confidence, *a, **k)

    monkeypatch.setattr(runner, "modulate_theme_cap", _tracked_modulate)

    monkeypatch.setattr(runner.broker, "get_account", lambda client: {"equity": 100_000.0, "cash": 100_000.0, "buying_power": 100_000.0})
    monkeypatch.setattr(runner.broker, "get_positions", lambda client: {})

    runner.run_live_rebalance(dry_run=True, state_path=tmp_path / "live_state.json", broker_client=object())

    assert modulate_calls == [(0.8, 0.8, 0.9)]  # base_max_theme_weight=0.8 from _base_strategy_config


def test_ai_usage_propagates_from_sentiment_call(tmp_path, monkeypatch):
    _patch_config(monkeypatch)
    _patch_data_fetches(monkeypatch)

    for name in _SIGNAL_MODS:
        monkeypatch.setattr(getattr(runner, name), "compute", lambda *a, **k: pd.Series({"AAPL": 1.0, "MSFT": 0.5}))

    usage = {
        "input_tokens": 1200,
        "output_tokens": 300,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "web_search_requests": 2,
    }
    monkeypatch.setattr(
        runner.news_ai,
        "compute_ai_sector_sentiment",
        lambda as_of: {"sentiment": 0.0, "confidence": 0.0, "event_tags": [], "summary": "", "usage": usage},
    )

    monkeypatch.setattr(runner.broker, "get_account", lambda client: {"equity": 100_000.0, "cash": 100_000.0, "buying_power": 100_000.0})
    monkeypatch.setattr(runner.broker, "get_positions", lambda client: {})

    result = runner.run_live_rebalance(dry_run=True, state_path=tmp_path / "live_state.json", broker_client=object())

    assert result.ai_usage == usage


def test_ai_usage_is_none_when_sentiment_call_is_a_cache_hit(tmp_path, monkeypatch):
    _patch_config(monkeypatch)
    _patch_data_fetches(monkeypatch)

    for name in _SIGNAL_MODS:
        monkeypatch.setattr(getattr(runner, name), "compute", lambda *a, **k: pd.Series({"AAPL": 1.0, "MSFT": 0.5}))

    monkeypatch.setattr(
        runner.news_ai,
        "compute_ai_sector_sentiment",
        lambda as_of: {"sentiment": 0.0, "confidence": 0.0, "event_tags": [], "summary": "", "usage": None},
    )

    monkeypatch.setattr(runner.broker, "get_account", lambda client: {"equity": 100_000.0, "cash": 100_000.0, "buying_power": 100_000.0})
    monkeypatch.setattr(runner.broker, "get_positions", lambda client: {})

    result = runner.run_live_rebalance(dry_run=True, state_path=tmp_path / "live_state.json", broker_client=object())

    assert result.ai_usage is None


def test_strategy_overrides_change_portfolio_config(tmp_path, monkeypatch):
    _patch_config(monkeypatch)
    _patch_data_fetches(monkeypatch)
    for name in _SIGNAL_MODS:
        monkeypatch.setattr(getattr(runner, name), "compute", lambda *a, **k: pd.Series({"AAPL": 1.0, "MSFT": 0.5}))
    _patch_neutral_sentiment(monkeypatch)

    captured_configs = []
    real_build_portfolio = runner.build_portfolio

    def _tracked_build_portfolio(*args, **kwargs):
        captured_configs.append(kwargs.get("config"))
        return real_build_portfolio(*args, **kwargs)

    monkeypatch.setattr(runner, "build_portfolio", _tracked_build_portfolio)
    monkeypatch.setattr(runner.broker, "get_account", lambda client: {"equity": 100_000.0, "cash": 100_000.0, "buying_power": 100_000.0})
    monkeypatch.setattr(runner.broker, "get_positions", lambda client: {})

    runner.run_live_rebalance(
        dry_run=True,
        state_path=tmp_path / "live_state.json",
        broker_client=object(),
        strategy_overrides={"portfolio.top_n": 1},
    )

    assert captured_configs[0]["top_n"] == 1  # overridden from the fixture's default of 2


def test_symbols_override_restricts_universe_and_skips_irrelevant_sentiment_call(tmp_path, monkeypatch):
    # Fixture universe is AAPL+MSFT, and only AAPL is theme-tagged
    # (_patch_config's load_themes stub). Restricting to just MSFT should
    # both narrow the price fetch and skip the (real-money) AI sentiment
    # call entirely, since no theme-tagged symbol remains in scope.
    _patch_config(monkeypatch)

    price_calls = []

    def fake_prices(symbols, start, end):
        price_calls.append(list(symbols))
        return {s: _series_df(100.0) for s in symbols} | {"SPY": _series_df(400.0)}

    monkeypatch.setattr(runner, "get_universe_prices", fake_prices)
    monkeypatch.setattr(runner, "get_universe_fundamentals", lambda symbols: {})

    for name in _SIGNAL_MODS:
        monkeypatch.setattr(getattr(runner, name), "compute", lambda *a, **k: pd.Series({"MSFT": 1.0}))

    sentiment_calls = []
    _patch_neutral_sentiment(monkeypatch, sentiment_calls)

    monkeypatch.setattr(runner.broker, "get_account", lambda client: {"equity": 100_000.0, "cash": 100_000.0, "buying_power": 100_000.0})
    monkeypatch.setattr(runner.broker, "get_positions", lambda client: {})

    runner.run_live_rebalance(
        dry_run=True,
        state_path=tmp_path / "live_state.json",
        broker_client=object(),
        symbols_override=["MSFT"],
    )

    assert price_calls[0] == ["MSFT"]
    assert sentiment_calls == []


def test_dry_run_never_submits_orders(tmp_path, monkeypatch):
    _patch_config(monkeypatch)
    _patch_data_fetches(monkeypatch)

    for name in _SIGNAL_MODS:
        monkeypatch.setattr(getattr(runner, name), "compute", lambda *a, **k: pd.Series({"AAPL": 1.0, "MSFT": 0.5}))
    _patch_neutral_sentiment(monkeypatch)

    monkeypatch.setattr(runner.broker, "get_account", lambda client: {"equity": 100_000.0, "cash": 100_000.0, "buying_power": 100_000.0})
    monkeypatch.setattr(runner.broker, "get_positions", lambda client: {})

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("submit_orders must never be called in dry-run mode")

    monkeypatch.setattr(runner.broker, "submit_orders", _fail_if_called)

    result = runner.run_live_rebalance(
        dry_run=True, state_path=tmp_path / "live_state.json", broker_client=object()
    )

    assert result.dry_run is True
    assert all(o["status"] == "DRY_RUN" for o in result.orders)


def test_circuit_breaker_flattens_without_computing_signals(tmp_path, monkeypatch):
    _patch_config(monkeypatch)
    _patch_data_fetches(monkeypatch)

    called = []
    for name in _SIGNAL_MODS:
        monkeypatch.setattr(
            getattr(runner, name), "compute", lambda *a, **k: called.append("signal") or pd.Series(dtype=float)
        )
    sentiment_calls = []
    _patch_neutral_sentiment(monkeypatch, sentiment_calls)

    # Equity has dropped 30% from the persisted peak -- past the 20%
    # max_drawdown_pct configured above.
    state_path = tmp_path / "live_state.json"
    save_state(LiveState(breaker_peak=100_000.0, cooldown_remaining=0, last_rebalance_date=None), state_path)

    monkeypatch.setattr(runner.broker, "get_account", lambda client: {"equity": 70_000.0, "cash": 70_000.0, "buying_power": 70_000.0})
    monkeypatch.setattr(runner.broker, "get_positions", lambda client: {"AAPL": {"qty": 100.0, "avg_entry_price": 100.0, "market_value": 70_000.0}})

    result = runner.run_live_rebalance(dry_run=True, state_path=state_path, broker_client=object())

    assert called == []  # no signal computation when the breaker trips
    assert sentiment_calls == []  # no AI sentiment call either -- nothing would use it
    assert result.target_weights.empty
    assert any(o["symbol"] == "AAPL" and o["side"] == "sell" for o in result.orders)

    saved = json.loads(state_path.read_text())
    assert saved["cooldown_remaining"] == 4  # risk.circuit_breaker_cooldown_periods


def test_hmm_regime_mode_scales_exposure_instead_of_flattening(tmp_path, monkeypatch):
    _patch_config(
        monkeypatch, strategy_overrides={"regime": {"enabled": True, "mode": "hmm", "ma_window_days": 200}}
    )
    _patch_data_fetches(monkeypatch)
    _patch_neutral_sentiment(monkeypatch)

    for name in _SIGNAL_MODS:
        monkeypatch.setattr(
            getattr(runner, name), "compute", lambda *a, **k: pd.Series({"AAPL": 1.0, "MSFT": 0.0})
        )

    monkeypatch.setattr(
        runner.broker,
        "get_account",
        lambda client: {"equity": 100_000.0, "cash": 100_000.0, "buying_power": 100_000.0},
    )
    monkeypatch.setattr(runner.broker, "get_positions", lambda client: {})

    def fake_evaluate_regime(prices, benchmark_prices, macro_data, as_of, **kwargs):
        return ("BEAR_TREND", 0.4, None)

    monkeypatch.setattr("alpha_engine.decision.regime.evaluate_regime", fake_evaluate_regime)
    macro_data = pd.DataFrame(
        {"yield_curve": [1.0], "vix": [15.0], "credit_spread": [3.0]}, index=[pd.Timestamp("2026-01-05")]
    )

    result = runner.run_live_rebalance(
        dry_run=True, state_path=tmp_path / "live_state.json", broker_client=object(), macro_data=macro_data
    )

    # top_n=2 selects both AAPL/MSFT equal-weight (0.5 each, summing to
    # 1.0) before the fake regime's 0.4 exposure scales it down -- not
    # flattened to empty the way the binary trend filter would.
    assert not result.target_weights.empty
    assert result.target_weights.sum() == pytest.approx(0.4)


def test_hmm_regime_mode_without_macro_data_degrades_to_full_exposure(tmp_path, monkeypatch):
    _patch_config(
        monkeypatch, strategy_overrides={"regime": {"enabled": True, "mode": "hmm", "ma_window_days": 200}}
    )
    _patch_data_fetches(monkeypatch)
    _patch_neutral_sentiment(monkeypatch)

    for name in _SIGNAL_MODS:
        monkeypatch.setattr(
            getattr(runner, name), "compute", lambda *a, **k: pd.Series({"AAPL": 1.0, "MSFT": 0.0})
        )

    monkeypatch.setattr(
        runner.broker,
        "get_account",
        lambda client: {"equity": 100_000.0, "cash": 100_000.0, "buying_power": 100_000.0},
    )
    monkeypatch.setattr(runner.broker, "get_positions", lambda client: {})

    # No macro_data passed -- hmm mode should degrade to full exposure
    # rather than error.
    result = runner.run_live_rebalance(
        dry_run=True, state_path=tmp_path / "live_state.json", broker_client=object()
    )

    assert result.target_weights.sum() == pytest.approx(1.0)


def test_ml_model_signal_participates_in_scoring(tmp_path, monkeypatch):
    _patch_config(
        monkeypatch,
        strategy_overrides={
            "signal_weights": {
                "momentum": 0.0,
                "mean_reversion": 0.0,
                "volatility": 0.0,
                "value": 0.0,
                "quality": 0.0,
                "ai_theme": 0.0,
                "ml_alpha": 1.0,
            },
            "portfolio": {
                "top_n": 1,
                "max_position_weight": 1.0,
                "max_sector_weight": 1.0,
                "max_theme_weight": 1.0,
                "sizing": "equal_weight",
            },
        },
    )
    _patch_data_fetches(monkeypatch)
    _patch_neutral_sentiment(monkeypatch)

    for name in _SIGNAL_MODS:
        monkeypatch.setattr(
            getattr(runner, name), "compute", lambda *a, **k: pd.Series({"AAPL": 0.0, "MSFT": 0.0})
        )

    monkeypatch.setattr(
        runner.broker,
        "get_account",
        lambda client: {"equity": 100_000.0, "cash": 100_000.0, "buying_power": 100_000.0},
    )
    monkeypatch.setattr(runner.broker, "get_positions", lambda client: {})

    class _FakeMlModel:
        def predict(self, X):
            return pd.Series({sym: (1.0 if sym == "MSFT" else -1.0) for sym in X.index})

    result = runner.run_live_rebalance(
        dry_run=True, state_path=tmp_path / "live_state.json", broker_client=object(), ml_model=_FakeMlModel()
    )

    # AAPL/MSFT score identically on every other signal (all zeroed above)
    # -- MSFT winning proves ml_alpha actually drove the selection.
    assert list(result.target_weights.index) == ["MSFT"]


def test_reduce_tier_scales_exposure_at_intermediate_drawdown(tmp_path, monkeypatch):
    _patch_config(
        monkeypatch,
        strategy_overrides={
            "risk": {
                "stop_loss_pct": 0.99,
                "max_drawdown_pct": 0.99,
                "reduce_drawdown_pct": 0.05,
                "reduce_exposure_scale": 0.4,
                "circuit_breaker_cooldown_periods": 4,
            },
            "portfolio": {
                "top_n": 2,
                "max_position_weight": 1.0,
                "max_sector_weight": 1.0,
                "sizing": "equal_weight",
            },
        },
    )
    _patch_data_fetches(monkeypatch)
    _patch_neutral_sentiment(monkeypatch)

    for name in _SIGNAL_MODS:
        monkeypatch.setattr(
            getattr(runner, name), "compute", lambda *a, **k: pd.Series({"AAPL": 1.0, "MSFT": 0.0})
        )

    state_path = tmp_path / "live_state.json"
    # 8% drawdown from the persisted peak -- past the 5% REDUCE threshold,
    # well under the (deliberately very high) HALT threshold.
    save_state(LiveState(breaker_peak=100_000.0, cooldown_remaining=0, last_rebalance_date=None), state_path)

    monkeypatch.setattr(
        runner.broker,
        "get_account",
        lambda client: {"equity": 92_000.0, "cash": 92_000.0, "buying_power": 92_000.0},
    )
    monkeypatch.setattr(runner.broker, "get_positions", lambda client: {})

    result = runner.run_live_rebalance(dry_run=True, state_path=state_path, broker_client=object())

    # top_n=2, equal_weight -> 0.5 each (sum 1.0) before REDUCE scales to 0.4.
    assert result.target_weights.sum() == pytest.approx(0.4)


def test_reduce_tier_disabled_by_default_ignores_drawdown(tmp_path, monkeypatch):
    _patch_config(
        monkeypatch,
        strategy_overrides={
            "risk": {"stop_loss_pct": 0.99, "max_drawdown_pct": 0.99, "circuit_breaker_cooldown_periods": 4},
            "portfolio": {
                "top_n": 2,
                "max_position_weight": 1.0,
                "max_sector_weight": 1.0,
                "sizing": "equal_weight",
            },
        },
    )
    _patch_data_fetches(monkeypatch)
    _patch_neutral_sentiment(monkeypatch)

    for name in _SIGNAL_MODS:
        monkeypatch.setattr(
            getattr(runner, name), "compute", lambda *a, **k: pd.Series({"AAPL": 1.0, "MSFT": 0.0})
        )

    state_path = tmp_path / "live_state.json"
    save_state(LiveState(breaker_peak=100_000.0, cooldown_remaining=0, last_rebalance_date=None), state_path)

    monkeypatch.setattr(
        runner.broker,
        "get_account",
        lambda client: {"equity": 92_000.0, "cash": 92_000.0, "buying_power": 92_000.0},
    )
    monkeypatch.setattr(runner.broker, "get_positions", lambda client: {})

    result = runner.run_live_rebalance(dry_run=True, state_path=state_path, broker_client=object())

    assert result.target_weights.sum() == pytest.approx(1.0)
