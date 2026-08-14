"""Loaders for the YAML config files under config/ (universe, strategy, sectors, themes)."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def load_universe(path: Path = CONFIG_DIR / "universe.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_strategy(path: Path = CONFIG_DIR / "strategy.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_sector_map(path: Path = CONFIG_DIR / "sectors.yaml") -> dict[str, str]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_themes(path: Path = CONFIG_DIR / "themes.yaml") -> dict[str, dict[str, float]]:
    """theme name -> {symbol: intensity}, e.g. themes["ai"]["NVDA"] == 1.0."""
    with open(path) as f:
        return yaml.safe_load(f)


def universe_symbols(universe: dict, include_benchmarks: bool = False) -> list[str]:
    symbols = list(universe.get("large_caps", [])) + list(universe.get("sector_etfs", []))
    if include_benchmarks:
        symbols += list(universe.get("benchmarks", []))
    return symbols


def apply_overrides(base_config: dict, overrides: dict[str, object]) -> dict:
    """`overrides`: dotted-path -> value, e.g. {"portfolio.top_n": 10}.
    Returns a deep copy of `base_config` with each path set; never mutates
    `base_config`. Shared by backtest/walkforward.py's grid search and
    live/runner.py's per-strategy config overrides (see live/runner.py and
    web/backend/app/models.py::Strategy.config_overrides_json) -- one
    dotted-override mechanism for both backtest tuning and live strategy
    variants.
    """
    config = copy.deepcopy(base_config)
    for path, value in overrides.items():
        *parents, leaf = path.split(".")
        node = config
        for key in parents:
            node = node[key]
        node[leaf] = value
    return config
