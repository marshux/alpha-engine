"""Loaders for the YAML config files under config/ (universe, strategy)."""
from __future__ import annotations

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


def universe_symbols(universe: dict, include_benchmarks: bool = False) -> list[str]:
    symbols = list(universe.get("large_caps", [])) + list(universe.get("sector_etfs", []))
    if include_benchmarks:
        symbols += list(universe.get("benchmarks", []))
    return symbols
