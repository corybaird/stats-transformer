import numpy as np
import pandas as pd
import pytest
from stats_transformer.models.regression.panel_diagnostics import PanelDiagnostics


@pytest.fixture
def panel():
    return pd.DataFrame({
        "country": ["USA"] * 4 + ["CAN"] * 4,
        "date": [2020, 2021, 2022, 2023] * 2,
        "x1": [1.0, 2.0, 3.0, 4.0, 1.0, np.nan, np.nan, np.nan],
        "x2": [5.0, 6.0, 7.0, 8.0, 4.0, 5.0, 6.0, 7.0],
    })


def test_missing_coverage_flags_entities_over_threshold(panel):
    diag = PanelDiagnostics()
    summary = diag.missing_coverage(panel, cols=["x1", "x2"], threshold=0.3)

    assert "USA" in summary.index
    assert "CAN" in summary.index
    assert summary.loc["USA", "flagged_cols"] == "None"
    assert "x1" in summary.loc["CAN", "flagged_cols"]


def test_missing_coverage_handles_multiindex_panel(panel):
    diag = PanelDiagnostics()
    indexed_panel = panel.set_index(["country", "date"])
    summary = diag.missing_coverage(indexed_panel, cols=["x1", "x2"])
    assert "USA" in summary.index
    assert summary.loc["USA", "n_obs"] == 4


def test_missing_coverage_reports_n_obs_per_entity(panel):
    diag = PanelDiagnostics()
    summary = diag.missing_coverage(panel, cols=["x1"])
    assert summary.loc["USA", "n_obs"] == 4
    assert summary.loc["CAN", "n_obs"] == 4


def test_vif_table_returns_variables_excluding_constant(panel):
    diag = PanelDiagnostics()
    vif = diag.vif_table(panel, cols=["x1", "x2"])

    assert "const" not in vif["Variable"].values
    assert set(vif["Variable"]) <= {"x1", "x2"}
    assert "VIF" in vif.columns


def test_vif_table_returns_empty_when_all_rows_dropped():
    diag = PanelDiagnostics()
    all_nan_panel = pd.DataFrame({"x1": [np.nan, np.nan], "x2": [np.nan, np.nan]})
    vif = diag.vif_table(all_nan_panel, cols=["x1", "x2"])
    assert vif.empty


def test_vif_table_does_not_duplicate_existing_constant(panel):
    diag = PanelDiagnostics()
    panel_with_const = panel.copy()
    panel_with_const["const"] = 1.0
    vif = diag.vif_table(panel_with_const, cols=["x1", "x2", "const"])
    assert "const" not in vif["Variable"].values


def test_unit_root_summary_is_a_documented_placeholder():
    # unit_root_summary is an explicit placeholder (see module docstring);
    # this pins its documented shape rather than asserting it does nothing.
    diag = PanelDiagnostics()
    result = diag.unit_root_summary(pd.DataFrame(), cols=["x1"])
    assert list(result.columns) == ["col", "statistic", "pval", "verdict"]
    assert result.empty


def test_cross_section_dependence_is_a_documented_placeholder():
    diag = PanelDiagnostics()
    result = diag.cross_section_dependence(pd.DataFrame(), cols=["x1"])
    assert list(result.columns) == ["col", "CD_stat", "pval"]
    assert result.empty


def test_run_returns_all_four_diagnostic_results(panel):
    diag = PanelDiagnostics()
    results = diag.run(panel, cols=["x1", "x2"])

    assert set(results.keys()) == {"missing_coverage", "vif", "unit_roots", "cross_section_dependence"}
    assert isinstance(results["missing_coverage"], pd.DataFrame)
    assert isinstance(results["vif"], pd.DataFrame)


def test_diagnostics_uses_custom_entity_and_time_columns():
    diag = PanelDiagnostics(entity_col="iso3", time_col="period")
    panel = pd.DataFrame({"iso3": ["USA", "USA"], "period": [2020, 2021], "x1": [1.0, 2.0]})
    summary = diag.missing_coverage(panel, cols=["x1"])
    assert "USA" in summary.index
