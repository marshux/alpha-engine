"""Deterministic portfolio construction: rank -> top-N -> position sizing ->
exposure caps. Given the same scores/volatility inputs, always produces the
same target weights.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def select_top_n(scores: pd.Series, top_n: int) -> pd.Series:
    return scores.sort_values(ascending=False).head(top_n)


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


def size_positions(
    selected: pd.Series,
    volatility: pd.Series | pd.DataFrame | None,
    sizing: str,
    max_position_weight: float,
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
    All three are capped at `max_position_weight`. Since raw weights already
    sum to 1.0 before capping, capping can only reduce the total -- any
    capital freed up by the cap is deliberately left as cash rather than
    redistributed to other positions, which keeps sizing simple and fully
    deterministic at the cost of sometimes being under-invested.
    """
    n = len(selected)
    if n == 0:
        return pd.Series(dtype=float)

    if sizing == "equal_weight":
        raw = pd.Series(1.0 / n, index=selected.index)
    elif sizing == "volatility_scaled":
        if volatility is None or not isinstance(volatility, pd.Series):
            raise ValueError("volatility series required for volatility_scaled sizing")
        vol = volatility.reindex(selected.index)
        if vol.isna().any() or (vol <= 0).any():
            raise ValueError(
                "volatility_scaled sizing requires positive volatility for every selected symbol"
            )
        inv_vol = 1.0 / vol
        raw = inv_vol / inv_vol.sum()
    elif sizing == "min_variance":
        if volatility is None or not isinstance(volatility, pd.DataFrame):
            raise ValueError("returns DataFrame required for min_variance sizing")
        returns = volatility.reindex(columns=selected.index)
        if returns.isna().any().any():
            raise ValueError(
                "min_variance sizing requires return history for every selected symbol"
            )
        raw = _min_variance_weights(returns)
    else:
        raise ValueError(f"unknown sizing method: {sizing}")

    return raw.clip(upper=max_position_weight)


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


def build_portfolio(
    scores: pd.Series,
    volatility: pd.Series | pd.DataFrame | None,
    sector_map: dict[str, str],
    config: dict,
    theme_map: dict[str, float] | None = None,
) -> pd.Series:
    selected = select_top_n(scores, config["top_n"])
    weights = size_positions(
        selected, volatility, config["sizing"], config["max_position_weight"]
    )
    weights = apply_sector_cap(weights, sector_map, config["max_sector_weight"])
    if theme_map and "max_theme_weight" in config:
        weights = apply_theme_cap(weights, theme_map, config["max_theme_weight"])
    return weights
