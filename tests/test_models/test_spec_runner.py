import pandas as pd
import pytest
from stats_transformer.models.regression.spec_runner import SpecificationRunner
from stats_transformer.models.regression.regression import RegressionModel


@pytest.fixture
def panel():
    return pd.DataFrame({
        "country": ["USA", "USA", "USA", "CAN", "CAN", "CAN"],
        "date": ["2020", "2021", "2022", "2020", "2021", "2022"],
        "y": [10.0, 12.0, 14.0, 9.0, 11.0, 13.0],
        "x1": [2.0, 3.0, 4.0, 1.0, 2.0, 3.0],
        "x2": [5.0, 6.0, 7.0, 4.0, 5.0, 6.0],
    })


def test_run_single_spec_single_key_variable(panel):
    runner = SpecificationRunner(RegressionModel)
    runner.add_spec(name="baseline", target="y", key_variables=["x1"])

    result = runner.run(panel)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["spec_name"] == "baseline"
    assert row["key_variable"] == "x1"
    assert row["coef"] is not None
    assert row["n_obs"] == 6


def test_run_loops_over_multiple_key_variables(panel):
    runner = SpecificationRunner(RegressionModel)
    runner.add_spec(name="baseline", target="y", key_variables=["x1", "x2"])

    result = runner.run(panel)

    assert len(result) == 2
    assert set(result["key_variable"]) == {"x1", "x2"}


def test_run_includes_control_coefficients(panel):
    runner = SpecificationRunner(RegressionModel)
    runner.add_spec(name="with_controls", target="y", key_variables=["x1"], controls=["x2"])

    result = runner.run(panel)

    assert "x2_coef" in result.columns
    assert "x2_se" in result.columns
    assert "x2_pval" in result.columns
    assert result.iloc[0]["x2_coef"] is not None


def test_run_multiple_specs_accumulate_in_results(panel):
    runner = SpecificationRunner(RegressionModel)
    runner.add_spec(name="spec1", target="y", key_variables=["x1"])
    runner.add_spec(name="spec2", target="y", key_variables=["x2"])

    result = runner.run(panel)

    assert len(result) == 2
    assert set(result["spec_name"]) == {"spec1", "spec2"}


def test_run_skips_key_variable_with_missing_columns(panel):
    runner = SpecificationRunner(RegressionModel)
    runner.add_spec(name="baseline", target="y", key_variables=["x1", "nonexistent_col"])

    result = runner.run(panel)

    assert len(result) == 1
    assert result.iloc[0]["key_variable"] == "x1"


def test_run_applies_subset_mask(panel):
    runner = SpecificationRunner(RegressionModel)
    runner.add_spec(name="usa_only", target="y", key_variables=["x1"], subset_mask_func=lambda df: df["country"] == "USA")

    result = runner.run(panel)

    assert len(result) == 1
    assert result.iloc[0]["n_obs"] == 3


def test_run_handles_multiindex_panel(panel):
    runner = SpecificationRunner(RegressionModel)
    runner.add_spec(name="baseline", target="y", key_variables=["x1"])

    indexed_panel = panel.set_index(["country", "date"])
    result = runner.run(indexed_panel)

    assert len(result) == 1
    assert result.iloc[0]["n_obs"] == 6


def test_run_continues_after_model_failure(panel):
    runner = SpecificationRunner(RegressionModel)
    # x1 is a valid spec; a second spec targeting a constant column
    # (zero variance) should fail to fit without stopping the whole run.
    bad_panel = panel.copy()
    bad_panel["x_const"] = 1.0
    runner.add_spec(name="good", target="y", key_variables=["x1"])
    runner.add_spec(name="bad_target", target="x_const", key_variables=["x1"])

    result = runner.run(bad_panel)

    # Regardless of whether the constant-target spec errors or fits
    # trivially, the well-specified run must still be present.
    assert "good" in set(result["spec_name"])


def test_to_dataframe_reflects_results_after_run(panel):
    runner = SpecificationRunner(RegressionModel)
    runner.add_spec(name="baseline", target="y", key_variables=["x1"])
    runner.run(panel)

    df = runner.to_dataframe()
    assert len(df) == 1


def test_to_dataframe_empty_before_run():
    runner = SpecificationRunner(RegressionModel)
    df = runner.to_dataframe()
    assert df.empty


def test_add_spec_stores_registered_specs():
    runner = SpecificationRunner(RegressionModel)
    runner.add_spec(name="a", target="y", key_variables=["x1"], controls=["x2"], entity_effects=True)

    assert len(runner.specs) == 1
    assert runner.specs[0]["name"] == "a"
    assert runner.specs[0]["controls"] == ["x2"]
    assert runner.specs[0]["model_kwargs"] == {"entity_effects": True}
