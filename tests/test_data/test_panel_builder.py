import pandas as pd
import pytest
from stats_transformer.data.panel_builder import PanelDataBuilder


@pytest.fixture
def base_panel():
    return pd.DataFrame({
        "country": ["USA", "USA", "CAN", "CAN"],
        "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-01-01", "2020-02-01"]),
        "gdp": [1.0, 1.1, 0.9, 0.95],
    })


def test_join_merges_matching_entity_and_period(base_panel):
    builder = PanelDataBuilder()
    cpi_df = pd.DataFrame({
        "country": ["USA", "CAN"],
        "date": pd.to_datetime(["2020-01-15", "2020-01-15"]),
        "cpi": [2.0, 1.5],
    })

    merged = builder.join(base_panel, cpi_df, name="CPI", on_freq="M")

    assert "cpi" in merged.columns
    usa_jan = merged[(merged["country"] == "USA") & (merged["date"] == "2020-01-01")]
    assert usa_jan["cpi"].iloc[0] == 2.0
    usa_feb = merged[(merged["country"] == "USA") & (merged["date"] == "2020-02-01")]
    assert pd.isna(usa_feb["cpi"].iloc[0])


def test_join_returns_original_panel_for_none_or_empty_df(base_panel):
    builder = PanelDataBuilder()
    assert builder.join(base_panel, None, name="x") is base_panel
    assert builder.join(base_panel, pd.DataFrame(), name="x") is base_panel


def test_join_returns_original_panel_when_index_cols_missing(base_panel):
    builder = PanelDataBuilder()
    df = pd.DataFrame({"not_country": ["USA"], "not_date": [pd.Timestamp("2020-01-01")], "val": [1.0]})
    result = builder.join(base_panel, df, name="bad")
    pd.testing.assert_frame_equal(result, base_panel)


def test_join_drops_overlapping_columns_before_merge(base_panel):
    builder = PanelDataBuilder()
    df = pd.DataFrame({
        "country": ["USA"],
        "date": pd.to_datetime(["2020-01-01"]),
        "gdp": [999.0],  # overlaps with base_panel's gdp
        "new_col": [5.0],
    })
    merged = builder.join(base_panel, df, name="overlap")
    assert "new_col" in merged.columns
    # original gdp values preserved, not overwritten by the incoming df's gdp
    assert merged.loc[merged["country"] == "USA", "gdp"].iloc[0] == 1.0


def test_join_returns_original_panel_when_no_new_columns(base_panel):
    builder = PanelDataBuilder()
    df = pd.DataFrame({"country": ["USA"], "date": pd.to_datetime(["2020-01-01"])})
    result = builder.join(base_panel, df, name="no_new_cols")
    pd.testing.assert_frame_equal(result, base_panel)


def test_join_handles_multiindex_panel(base_panel):
    builder = PanelDataBuilder()
    indexed_panel = base_panel.set_index(["country", "date"])
    cpi_df = pd.DataFrame({
        "country": ["USA"],
        "date": pd.to_datetime(["2020-01-01"]),
        "cpi": [2.0],
    })
    merged = builder.join(indexed_panel, cpi_df, name="CPI")
    assert isinstance(merged.index, pd.MultiIndex)
    assert "cpi" in merged.columns


def test_join_warns_and_returns_panel_when_time_col_missing():
    builder = PanelDataBuilder()
    panel_no_date = pd.DataFrame({"country": ["USA"], "gdp": [1.0]})
    df = pd.DataFrame({"country": ["USA"], "date": pd.to_datetime(["2020-01-01"]), "cpi": [2.0]})
    result = builder.join(panel_no_date, df, name="x")
    pd.testing.assert_frame_equal(result, panel_no_date)


def test_join_respects_cols_filter(base_panel):
    builder = PanelDataBuilder()
    df = pd.DataFrame({
        "country": ["USA"],
        "date": pd.to_datetime(["2020-01-01"]),
        "cpi": [2.0],
        "unwanted": [999.0],
    })
    merged = builder.join(base_panel, df, name="CPI", cols=["cpi"])
    assert "cpi" in merged.columns
    assert "unwanted" not in merged.columns


def test_join_global_broadcasts_onto_all_entities(base_panel):
    builder = PanelDataBuilder()
    vix_df = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-02-01"]), "vix_close": [15.0, 18.0]})

    merged = builder.join_global(base_panel, vix_df, name="VIX")

    assert "vix_close" in merged.columns
    jan_rows = merged[merged["date"] == "2020-01-01"]
    assert (jan_rows["vix_close"] == 15.0).all()


def test_join_global_returns_original_panel_for_none_or_empty_df(base_panel):
    builder = PanelDataBuilder()
    assert builder.join_global(base_panel, None, name="x") is base_panel
    assert builder.join_global(base_panel, pd.DataFrame(), name="x") is base_panel


def test_join_global_returns_original_panel_when_time_col_missing(base_panel):
    builder = PanelDataBuilder()
    df = pd.DataFrame({"not_date": [1], "vix_close": [15.0]})
    result = builder.join_global(base_panel, df, name="x")
    pd.testing.assert_frame_equal(result, base_panel)


def test_join_global_handles_multiindex_panel(base_panel):
    builder = PanelDataBuilder()
    indexed_panel = base_panel.set_index(["country", "date"])
    vix_df = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "vix_close": [15.0]})
    merged = builder.join_global(indexed_panel, vix_df, name="VIX")
    assert isinstance(merged.index, pd.MultiIndex)
    assert "vix_close" in merged.columns


def test_join_global_renames_differently_named_time_column(base_panel):
    builder = PanelDataBuilder()
    vix_df = pd.DataFrame({"obs_date": pd.to_datetime(["2020-01-01"]), "vix_close": [15.0]})
    merged = builder.join_global(base_panel, vix_df, name="VIX", time_col="obs_date")
    assert "vix_close" in merged.columns
    assert "obs_date" not in merged.columns


def test_build_chains_entity_and_global_joins(base_panel):
    builder = PanelDataBuilder()
    cpi_df = pd.DataFrame({"country": ["USA", "CAN"], "date": pd.to_datetime(["2020-01-01", "2020-01-01"]), "cpi": [2.0, 1.5]})
    vix_df = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "vix_close": [15.0]})

    result = builder.build(base_panel, [
        {"type": "entity", "df": cpi_df, "name": "CPI", "cols": ["cpi"]},
        {"type": "global", "df": vix_df, "name": "VIX", "cols": ["vix_close"]},
    ])

    assert "cpi" in result.columns
    assert "vix_close" in result.columns
    # base_panel itself must be untouched (build() copies before joining)
    assert "cpi" not in base_panel.columns


def test_builder_uses_custom_entity_and_time_columns():
    builder = PanelDataBuilder(entity_col="iso3", time_col="period")
    panel = pd.DataFrame({"iso3": ["USA"], "period": pd.to_datetime(["2020-01-01"]), "gdp": [1.0]})
    df = pd.DataFrame({"iso3": ["USA"], "period": pd.to_datetime(["2020-01-01"]), "cpi": [2.0]})

    merged = builder.join(panel, df, name="CPI")
    assert "cpi" in merged.columns
