"""Unit tests for EventStudyBuilder."""
import numpy as np
import pandas as pd
import pytest

from stats_transformer import EventStudyBuilder


@pytest.fixture
def fx_values():
    """Simple daily FX series for two countries."""
    rows = []
    for country in ["US", "GB"]:
        for i in range(20):
            rows.append({
                "country": country,
                "date": pd.Timestamp("2023-01-02") + pd.Timedelta(days=i),
                "rate": 1.0 + i * 0.01,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def events():
    return pd.DataFrame([
        {"country": "US", "event_date": pd.Timestamp("2023-01-10")},
        {"country": "GB", "event_date": pd.Timestamp("2023-01-12")},
    ])


def test_logpct_horizons(fx_values, events):
    builder = EventStudyBuilder()
    result = builder.build(
        fx_values, events,
        value_col="rate",
        horizons=[1, 5],
        change="logpct",
        event_date_col="event_date",
    )
    assert len(result) == 2
    assert "rate_d0_d1_change" in result.columns
    assert "rate_d0_d5_change" in result.columns
    assert result["rate_d0_d1_change"].notna().all()


def test_raw_change_bps(fx_values, events):
    builder = EventStudyBuilder()
    result = builder.build(
        fx_values, events,
        value_col="rate",
        horizons=[1],
        change="raw",
        scale=100.0,
        event_date_col="event_date",
    )
    assert "rate_d0_d1_change_bps" in result.columns


def test_event_before_series_dropped(fx_values):
    early_event = pd.DataFrame([{"country": "US", "event_date": pd.Timestamp("2022-01-01")}])
    builder = EventStudyBuilder()
    result = builder.build(
        fx_values, early_event,
        value_col="rate",
        horizons=[1],
        event_date_col="event_date",
    )
    assert len(result) == 0


def test_custom_prefix(fx_values, events):
    builder = EventStudyBuilder()
    result = builder.build(
        fx_values, events,
        value_col="rate",
        horizons=[2],
        col_prefix="usd_fx",
        event_date_col="event_date",
    )
    assert "usd_fx_d0_d2_change" in result.columns


def test_preserves_event_columns(fx_values):
    events_with_extras = pd.DataFrame([
        {"country": "US", "event_date": pd.Timestamp("2023-01-10"), "move": "hike"},
    ])
    builder = EventStudyBuilder()
    result = builder.build(
        fx_values, events_with_extras,
        value_col="rate",
        horizons=[1],
        event_date_col="event_date",
    )
    assert "move" in result.columns
    assert result["move"].iloc[0] == "hike"


def test_top_level_import():
    from stats_transformer import EventStudyBuilder as ESB
    assert ESB is EventStudyBuilder
