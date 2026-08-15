import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from stats_transformer.visualization.tables.descriptive_stats import DescriptiveStatsTable


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        "x1": [10.0, 20.0, np.nan, 40.0, 50.0],
    })


def test_build_produces_section_header_and_variable_rows(sample_df):
    table = DescriptiveStatsTable(table_generator=MagicMock())
    result = table.build([
        {"label": "Outcomes", "vars": [("y", "Y Variable", sample_df)]},
    ])

    assert len(result) == 2
    header_row = result.iloc[0]
    assert "Outcomes" in header_row["Variable"]
    assert header_row["N"] == ""

    var_row = result.iloc[1]
    assert var_row["Variable"] == "Y Variable"
    assert var_row["N"] == "5"
    assert var_row["Mean"] == "3.000"


def test_build_handles_missing_values_correctly(sample_df):
    table = DescriptiveStatsTable(table_generator=MagicMock())
    result = table.build([
        {"label": "Section", "vars": [("x1", "X1", sample_df)]},
    ])

    row = result.iloc[1]
    assert row["N"] == "4"  # one NaN dropped


def test_build_skips_column_not_in_dataframe(sample_df):
    table = DescriptiveStatsTable(table_generator=MagicMock())
    result = table.build([
        {"label": "Section", "vars": [("nonexistent", "Missing Var", sample_df)]},
    ])

    # Only the section header row remains; the missing variable is skipped.
    assert len(result) == 1


def test_build_skips_malformed_variable_tuple(sample_df):
    table = DescriptiveStatsTable(table_generator=MagicMock())
    result = table.build([
        {"label": "Section", "vars": [("y", "Y")]},  # missing the df element
    ])
    assert len(result) == 1


def test_build_multiple_sections(sample_df):
    table = DescriptiveStatsTable(table_generator=MagicMock())
    result = table.build([
        {"label": "Section A", "vars": [("y", "Y", sample_df)]},
        {"label": "Section B", "vars": [("x1", "X1", sample_df)]},
    ])
    assert len(result) == 4  # 2 headers + 2 variable rows


def test_compute_stats_returns_nan_row_for_empty_series():
    table = DescriptiveStatsTable(table_generator=MagicMock())
    stats = table._compute_stats(pd.Series([np.nan, np.nan]))
    assert stats["N"] == 0
    assert pd.isna(stats["Mean"])


def test_compute_stats_returns_correct_percentiles():
    table = DescriptiveStatsTable(table_generator=MagicMock())
    stats = table._compute_stats(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert stats["N"] == 5
    assert stats["p50"] == 3.0
    assert stats["Min"] == 1.0
    assert stats["Max"] == 5.0


def test_export_calls_table_generator_export_all(sample_df):
    mock_gen = MagicMock()
    table = DescriptiveStatsTable(table_generator=mock_gen)

    table.export([{"label": "Section", "vars": [("y", "Y", sample_df)]}], filename="out.tex")

    mock_gen.export_all.assert_called_once()
    call_args = mock_gen.export_all.call_args
    assert call_args[0][1] == "out.tex"


def test_export_returns_none_and_skips_when_table_empty():
    mock_gen = MagicMock()
    table = DescriptiveStatsTable(table_generator=mock_gen)

    result = table.export([], filename="out.tex")

    assert result is None
    mock_gen.export_all.assert_not_called()
