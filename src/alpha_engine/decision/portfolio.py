"""Deterministic portfolio construction: rank -> top-N -> position sizing ->
exposure caps. Given the same scores/volatility inputs, always produces the
same target weights.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def select_top_n(scores: pd.Series, top_n: int) -> pd.Series:
    return scores.sort_values(ascending=False).head(top_n)


def resolve_top_n(scores: pd.Series, config: dict) -> int:
    """Resolves the position count for this rebalance -- either a fixed
    `config["top_n"]` (Traditional/Contrarian Volatility's mode,
    unchanged) or, when `config["top_n_pct"]` is set instead, a ratio of
    the live scoreable universe (`len(scores)`, i.e. however many symbols
    actually have a score this rebalance) that period, clamped between
    `config.get("top_n_min", 1)` and `config.get("top_n_max")` (no ceiling
    if unset).

    Ratio-based sizing keeps a strategy's selectivity constant regardless
    of universe size -- a fixed count on a small universe can end up
    holding a much larger fraction of "everything" than the same count
    would on a larger one, diluting whatever edge the ranking signal has
    (the guide's own portfolio example holds 50 of a ~500-name universe,
    ~10%). The `top_n_max` ceiling exists because the ratio alone isn't
    self-correcting in the other direction: on this repo's original
    63-symbol universe, a 10% ratio resolved to just 6 positions --
    concentrated enough to add material single-name risk (confirmed via
    walk-forward: Sharpe dropped from 1.13 pre-fix to 0.73 post-fix, see
    systematic_core_findings.md) even though the *ratio* itself was
    right. `top_n_min`/`top_n_max` bound the resolved count to a sane
    range regardless of how small or large the live universe is.
    """
    if "top_n_pct" in config:
        top_n_min = config.get("top_n_min", 1)
        top_n_max = config.get("top_n_max")
        raw = int(len(scores) * config["top_n_pct"])
        clamped = max(top_n_min, raw)
        if top_n_max is not None:
            clamped = min(top_n_max, clamped)
        return clamped
    return config["top_n"]


def _min_variance_weights(returns: pd.DataFrame) -> pd.Series:
    """Closed-form global minimum-variance weights:
    w = inv(Sigma) @ 1 / (1^T @ inv(Sigma) @ 1).

    Uses the pseudo-inverse rather than a strict inverse for numerical
    safety when the sample covariance matrix is near-singular (common with
    few observations relative to the number of symbols). Long-only is
    approximated by clipping negative weights to zero and renormalizing,
    rather than solving a full constrained QP -- keeps this dependency-free
    (no scipy needed) at the cost of not being the *exact* long-only
    optimum. Falls back to equal-weight if every raw weight comes out
    non-positive (degenerate covariance).
    """
    symbols = list(returns.columns)
    equal_weight_fallback = pd.Series(1.0 / len(symbols), index=symbols)

    cov = returns.cov().values
    ones = np.ones(len(symbols))
    inv_cov = np.linalg.pinv(cov)
    raw = inv_cov @ ones
    raw_sum = raw.sum()
    if not np.isfinite(raw_sum) or raw_sum == 0:
        return equal_weight_fallback
    raw = raw / raw_sum

    weights = pd.Series(raw, index=symbols).clip(lower=0.0)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return equal_weight_fallback
    return weights / total


def _compute_raw_weights(
    selected: pd.Series,
    volatility: pd.Series | pd.DataFrame | None,
    sizing: str,
) -> pd.Series:
    """The pre-cap sizing pass shared by `size_positions` and
    `capping_diagnostics` -- factored out so the diagnostic can inspect
    what capping actually threw away without duplicating the sizing math
    (and risking the two drifting out of sync).
    """
    n = len(selected)
    if n == 0:
        return pd.Series(dtype=float)

    if sizing == "equal_weight":
        return pd.Series(1.0 / n, index=selected.index)
    elif sizing == "volatility_scaled":
        if volatility is None or not isinstance(volatility, pd.Series):
            raise ValueError("volatility series required for volatility_scaled sizing")
        vol = volatility.reindex(selected.index)
        if vol.isna().any() or (vol <= 0).any():
            raise ValueError(
                "volatility_scaled sizing requires positive volatility for every selected symbol"
            )
        inv_vol = 1.0 / vol
        return inv_vol / inv_vol.sum()
    elif sizing == "min_variance":
        if volatility is None or not isinstance(volatility, pd.DataFrame):
            raise ValueError("returns DataFrame required for min_variance sizing")
        returns = volatility.reindex(columns=selected.index)
        if returns.isna().any().any():
            raise ValueError(
                "min_variance sizing requires return history for every selected symbol"
            )
        return _min_variance_weights(returns)
    else:
        raise ValueError(f"unknown sizing method: {sizing}")


def _redistribute_excess(raw: pd.Series, max_position_weight: float, max_iterations: int = 10) -> pd.Series:
    """Iteratively push each capped position's excess weight onto the
    positions that haven't been capped yet, proportional to their current
    weight, instead of leaving it as cash. A single redistribution pass
    can push a previously-uncapped position over the cap itself, so this
    repeats until nothing new gets capped (in practice 2-3 rounds) or
    `max_iterations` is hit, at which point whatever's left is clipped
    defensively.

    Once a position is capped it's *permanently* excluded from receiving
    further redistribution, even if a later round's math would otherwise
    land more weight on it -- capped means "at its ceiling," not "eligible
    for more." Without this, a position sitting exactly at the cap could
    be treated as fair game again next round, immediately breach the cap,
    and need re-capping in an unstable cycle instead of monotonically
    converging.

    If every position ends up capped, there's nowhere left to redistribute
    into and any final residual stays as cash -- a real structural ceiling
    (`n * max_position_weight < 1.0`), not a bug.
    """
    weights = raw.copy()
    permanently_capped = pd.Series(False, index=weights.index)
    for _ in range(max_iterations):
        newly_capped = (weights > max_position_weight) & ~permanently_capped
        if not newly_capped.any():
            break
        excess = (weights[newly_capped] - max_position_weight).sum()
        weights[newly_capped] = max_position_weight
        permanently_capped |= newly_capped
        eligible = ~permanently_capped
        eligible_total = weights[eligible].sum()
        if eligible_total <= 0:
            break
        weights[eligible] += excess * (weights[eligible] / eligible_total)
    else:
        weights = weights.clip(upper=max_position_weight)
    return weights


def size_positions(
    selected: pd.Series,
    volatility: pd.Series | pd.DataFrame | None,
    sizing: str,
    max_position_weight: float,
    redistribute_capped_excess: bool = False,
) -> pd.Series:
    """Target weights (summing to <=1.0) for the selected symbols.

    - equal_weight: 1/N each
    - volatility_scaled: inverse-volatility weighted (lower-vol names get
      more capital) -- `volatility` must be a `pd.Series` with a positive
      value for every selected symbol
    - min_variance: closed-form global minimum-variance portfolio (see
      `_min_variance_weights`) -- `volatility` must instead be a
      `pd.DataFrame` of trailing daily returns (symbols as columns), with
      no missing values for any selected symbol. The same parameter name
      carries a different shape depending on `sizing` rather than adding a
      second parameter, so the call site (backtest/engine.py) stays simple;
      see `signals/volatility.trailing_returns_matrix` for how it's built.
    All three are capped at `max_position_weight`. By default, since raw
    weights already sum to 1.0 before capping, capping can only reduce the
    total -- capital freed up by the cap is left as cash rather than
    redistributed, which keeps sizing simple and fully deterministic at
    the cost of sometimes being under-invested (see `capping_diagnostics`
    to measure how much this actually costs -- on Systematic Core's
    config, ~14% of the portfolio on average, ~15% in BULL_TREND
    specifically; see systematic_core_findings_v4.md's Action plan v4,
    Finding 1). `redistribute_capped_excess=True` opts into
    `_redistribute_excess` instead: capped excess flows to the other
    selected positions rather than sitting idle. Off by default so
    existing strategies (Traditional, Contrarian Volatility) get zero
    behavior change unless they explicitly turn it on via
    `portfolio.redistribute_capped_excess: true`.
    """
    if len(selected) == 0:
        return pd.Series(dtype=float)
    raw = _compute_raw_weights(selected, volatility, sizing)
    if redistribute_capped_excess:
        return _redistribute_excess(raw, max_position_weight)
    return raw.clip(upper=max_position_weight)


def capping_diagnostics(
    selected: pd.Series,
    volatility: pd.Series | pd.DataFrame | None,
    sizing: str,
    max_position_weight: float,
) -> dict:
    """How much of `size_positions`'s cap-driven cash drag (see its
    docstring) is happening for this one selection -- diagnostic only,
    not used anywhere in the live decision path. Returns
    `{"n_capped": int, "excess_weight": float}`: `n_capped` is how many
    selected symbols' raw (pre-cap) weight exceeded `max_position_weight`;
    `excess_weight` is the total weight trapped by those caps (raw sum
    minus clipped sum) -- exactly the amount left as cash by this step
    alone, before any downstream sector/theme/regime adjustment.
    """
    if len(selected) == 0:
        return {"n_capped": 0, "excess_weight": 0.0}
    raw = _compute_raw_weights(selected, volatility, sizing)
    clipped = raw.clip(upper=max_position_weight)
    return {
        "n_capped": int((raw > max_position_weight).sum()),
        "excess_weight": float(raw.sum() - clipped.sum()),
    }


def apply_sector_cap(
    weights: pd.Series, sector_map: dict[str, str], max_sector_weight: float
) -> pd.Series:
    """Scale down any sector's aggregate weight that exceeds the cap,
    proportionally within that sector.

    Symbols absent from `sector_map` are left as-is at the sector level --
    single-position caps still apply to them via `size_positions`. In
    normal use every symbol in config/universe.yaml has an entry in
    config/sectors.yaml (enforced by tests/test_config.py); this only
    matters for ad-hoc symbols passed via `--symbols`.
    """
    if not sector_map:
        return weights
    adjusted = weights.copy()
    known = pd.Series({s: sector_map[s] for s in weights.index if s in sector_map})
    if known.empty:
        return adjusted
    for _sector, symbols in known.groupby(known).groups.items():
        sector_total = adjusted.loc[symbols].sum()
        if sector_total > max_sector_weight:
            adjusted.loc[symbols] = adjusted.loc[symbols] * (max_sector_weight / sector_total)
    return adjusted


def apply_theme_cap(
    weights: pd.Series, theme_symbols: dict[str, float] | set[str], max_theme_weight: float
) -> pd.Series:
    """Scale down the combined weight of every theme-tagged position if it
    exceeds `max_theme_weight`, proportionally within the group -- same
    mechanism as `apply_sector_cap`, but one aggregate cap across a single
    named theme (e.g. signals/ai_theme.py's AI tilt) rather than per-GICS-
    sector groups.

    This is the "hedge" for a deliberate directional tilt: it bounds how
    much the portfolio can concentrate in the theme, it does not offset
    the theme's risk with an opposing position. `theme_symbols` can be the
    theme's intensity dict directly (only membership matters here, not the
    intensity value) or a plain set of symbols.
    """
    if not theme_symbols:
        return weights
    members = [s for s in weights.index if s in theme_symbols]
    if not members:
        return weights
    adjusted = weights.copy()
    theme_total = adjusted.loc[members].sum()
    if theme_total > max_theme_weight:
        adjusted.loc[members] = adjusted.loc[members] * (max_theme_weight / theme_total)
    return adjusted


def modulate_theme_cap(
    base_max_theme_weight: float,
    sentiment: float,
    confidence: float,
    sensitivity: float = 0.5,
    min_weight: float = 0.15,
    max_weight: float = 0.50,
) -> float:
    """Scales the AI-theme concentration cap by a live AI-sector sentiment
    read (signals/news_ai.py::compute_ai_sector_sentiment) -- positive
    sentiment loosens the cap (more room for the tilt), negative sentiment
    tightens it (more defensive), weighted by the research's own stated
    confidence so a low-confidence read barely moves the cap at all.

    Purely deterministic given `(sentiment, confidence)`: the AI supplies
    two numbers, this formula decides what they mean for the cap. Live-only
    by construction -- there's no historical sentiment feed for the
    backtester to call this with, so `backtest/engine.py` never does; it
    always uses the static `max_theme_weight` from strategy.yaml unmodified.

    `sentiment`/`confidence` are each expected in [-1, 1] / [0, 1] (the
    schema `signals/news_ai.py` forces), but this isn't enforced here --
    the final `min_weight`/`max_weight` clip is what actually bounds the
    result regardless of input range.
    """
    adjustment = 1.0 + sentiment * confidence * sensitivity
    return float(min(max(base_max_theme_weight * adjustment, min_weight), max_weight))


def scale_gross_exposure(weights: pd.Series, exposure_fraction: float) -> pd.Series:
    """Scales every target weight by `exposure_fraction`, leaving the
    remainder as cash -- the HMM regime mode's continuous analogue of the
    Faber trend filter's binary flatten-to-cash (decision/risk.py's
    `regime_is_risk_on`). Purely deterministic given the two inputs, same
    determinism pattern as `modulate_theme_cap`. `exposure_fraction` is
    expected in [0, 1] (decision/regime.py's `GROSS_EXPOSURE_BY_REGIME`
    values are), but this isn't enforced here -- a caller passing 1.0
    (full exposure) is a no-op, which is what backtest/engine.py and
    live/runner.py both do when regime.mode is "trend_filter".
    """
    return weights * exposure_fraction


def build_portfolio(
    scores: pd.Series,
    volatility: pd.Series | pd.DataFrame | None,
    sector_map: dict[str, str],
    config: dict,
    theme_map: dict[str, float] | None = None,
) -> pd.Series:
    selected = select_top_n(scores, resolve_top_n(scores, config))
    weights = size_positions(
        selected,
        volatility,
        config["sizing"],
        config["max_position_weight"],
        redistribute_capped_excess=config.get("redistribute_capped_excess", False),
    )
    weights = apply_sector_cap(weights, sector_map, config["max_sector_weight"])
    if theme_map and "max_theme_weight" in config:
        weights = apply_theme_cap(weights, theme_map, config["max_theme_weight"])
    return weights
