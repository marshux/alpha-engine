"""Deterministic portfolio construction: rank -> top-N -> position sizing ->
exposure caps. Given the same scores/volatility inputs, always produces the
same target weights.
"""
from __future__ import annotations

import pandas as pd


def select_top_n(scores: pd.Series, top_n: int) -> pd.Series:
    return scores.sort_values(ascending=False).head(top_n)


def size_positions(
    selected: pd.Series,
    volatility: pd.Series | None,
    sizing: str,
    max_position_weight: float,
) -> pd.Series:
    """Target weights (summing to <=1.0) for the selected symbols.

    - equal_weight: 1/N each
    - volatility_scaled: inverse-volatility weighted (lower-vol names get
      more capital) -- `volatility` must have a positive value for every
      selected symbol
    Both are capped at `max_position_weight`. Since raw weights already sum
    to 1.0 before capping, capping can only reduce the total -- any capital
    freed up by the cap is deliberately left as cash rather than
    redistributed to other positions, which keeps sizing simple and fully
    deterministic at the cost of sometimes being under-invested.
    """
    n = len(selected)
    if n == 0:
        return pd.Series(dtype=float)

    if sizing == "equal_weight":
        raw = pd.Series(1.0 / n, index=selected.index)
    elif sizing == "volatility_scaled":
        if volatility is None:
            raise ValueError("volatility series required for volatility_scaled sizing")
        vol = volatility.reindex(selected.index)
        if vol.isna().any() or (vol <= 0).any():
            raise ValueError(
                "volatility_scaled sizing requires positive volatility for every selected symbol"
            )
        inv_vol = 1.0 / vol
        raw = inv_vol / inv_vol.sum()
    else:
        raise ValueError(f"unknown sizing method: {sizing}")

    return raw.clip(upper=max_position_weight)


def apply_sector_cap(
    weights: pd.Series, sector_map: dict[str, str], max_sector_weight: float
) -> pd.Series:
    """Scale down any sector's aggregate weight that exceeds the cap,
    proportionally within that sector.

    Symbols absent from `sector_map` are left as-is at the sector level --
    v1's large-cap universe has no GICS sector classification wired in yet
    (a known data gap, not a design choice); single-position caps still
    apply to them via `size_positions`.
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


def build_portfolio(
    scores: pd.Series,
    volatility: pd.Series | None,
    sector_map: dict[str, str],
    config: dict,
) -> pd.Series:
    selected = select_top_n(scores, config["top_n"])
    weights = size_positions(
        selected, volatility, config["sizing"], config["max_position_weight"]
    )
    return apply_sector_cap(weights, sector_map, config["max_sector_weight"])
