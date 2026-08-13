"""Simple linear transaction cost model: commission + slippage in basis
points, applied to dollar turnover at each rebalance.
"""
from __future__ import annotations


def cost_fraction(cost_cfg: dict) -> float:
    return (cost_cfg["commission_bps"] + cost_cfg["slippage_bps"]) / 10_000.0


def turnover_cost(turnover_fraction: float, equity: float, cost_cfg: dict) -> float:
    """`turnover_fraction`: sum of |weight change| across all symbols at
    this rebalance (e.g. 0.5 == 50% of portfolio notional traded).
    """
    return turnover_fraction * equity * cost_fraction(cost_cfg)
