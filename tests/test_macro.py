"""Tests for data/macro.py using a mocked pd.read_csv -- no real network
call to FRED. Covers the per-series cache round-trip and the combined
yield-curve/vix/credit-spread panel construction.
"""
from __future__ import annotations

import io
from unittest.mock import patch

import pandas as pd
import pytest

from alpha_engine.data import macro


def _fred_csv(series_id: str, dates: list[str], values: list[str]) -> str:
    lines = [f"DATE,{series_id}"] + [f"{d},{v}" for d, v in zip(dates, values)]
    return "\n".join(lines)


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(macro, "CACHE_DIR", tmp_path / "macro")


def test_get_fred_series_parses_dates_and_coerces_missing_to_nan():
    csv_text = _fred_csv("DGS10", ["2024-01-01", "2024-01-02", "2024-01-03"], ["4.0", ".", "4.2"])
    with patch("alpha_engine.data.macro.pd.read_csv", return_value=pd.read_csv(io.StringIO(csv_text))) as mock_read:
        series = macro.get_fred_series("DGS10")

    mock_read.assert_called_once_with("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10")
    assert series.name == "DGS10"
    assert series.loc["2024-01-01"] == pytest.approx(4.0)
    assert pd.isna(series.loc["2024-01-02"])
    assert series.loc["2024-01-03"] == pytest.approx(4.2)


def test_get_fred_series_uses_cache_on_second_call():
    csv_text = _fred_csv("VIXCLS", ["2024-01-01"], ["15.0"])
    with patch("alpha_engine.data.macro.pd.read_csv", return_value=pd.read_csv(io.StringIO(csv_text))) as mock_read:
        macro.get_fred_series("VIXCLS")
        macro.get_fred_series("VIXCLS")

    assert mock_read.call_count == 1


def test_get_fred_series_force_refresh_bypasses_cache():
    csv_text = _fred_csv("VIXCLS", ["2024-01-01"], ["15.0"])
    with patch("alpha_engine.data.macro.pd.read_csv", return_value=pd.read_csv(io.StringIO(csv_text))) as mock_read:
        macro.get_fred_series("VIXCLS")
        macro.get_fred_series("VIXCLS", force_refresh=True)

    assert mock_read.call_count == 2


def test_get_macro_data_combines_and_forward_fills():
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    real_read_csv = pd.read_csv

    def fake_read_csv(url):
        if "DGS10" in url:
            return real_read_csv(io.StringIO(_fred_csv("DGS10", dates, ["4.5", "4.5", "4.6"])))
        if "DGS2" in url:
            return real_read_csv(io.StringIO(_fred_csv("DGS2", dates, ["4.0", "4.0", "4.0"])))
        if "VIXCLS" in url:
            return real_read_csv(io.StringIO(_fred_csv("VIXCLS", dates, ["14.0", ".", "16.0"])))
        if "BAA" in url:
            return real_read_csv(io.StringIO(_fred_csv("BAA", dates, ["6.0", "6.0", "6.1"])))
        if "AAA" in url:
            return real_read_csv(io.StringIO(_fred_csv("AAA", dates, ["4.5", "4.5", "4.5"])))
        raise AssertionError(f"unexpected url {url}")

    with patch("alpha_engine.data.macro.pd.read_csv", side_effect=fake_read_csv):
        data = macro.get_macro_data()

    assert list(data.columns) == ["yield_curve", "vix", "credit_spread"]
    assert data["yield_curve"].loc["2024-01-01"] == pytest.approx(0.5)
    # VIX's missing 2024-01-02 value should be forward-filled from 01-01
    assert data["vix"].loc["2024-01-02"] == pytest.approx(14.0)
    assert data["vix"].loc["2024-01-03"] == pytest.approx(16.0)
    assert data["credit_spread"].loc["2024-01-01"] == pytest.approx(1.5)
    assert data["credit_spread"].loc["2024-01-03"] == pytest.approx(1.6)
