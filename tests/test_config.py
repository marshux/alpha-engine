from __future__ import annotations

from alpha_engine.config import load_sector_map, load_universe, universe_symbols


def test_sector_map_covers_every_universe_symbol():
    universe = load_universe()
    sector_map = load_sector_map()

    symbols = universe_symbols(universe, include_benchmarks=False)
    missing = [s for s in symbols if s not in sector_map]

    assert missing == [], f"symbols missing from config/sectors.yaml: {missing}"


def test_sector_map_has_no_orphan_entries():
    universe = load_universe()
    sector_map = load_sector_map()

    symbols = set(universe_symbols(universe, include_benchmarks=False))
    orphans = [s for s in sector_map if s not in symbols]

    assert orphans == [], f"sectors.yaml has entries not in universe.yaml: {orphans}"
