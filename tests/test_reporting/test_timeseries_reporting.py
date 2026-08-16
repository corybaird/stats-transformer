import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import xarray as xr

from stats_transformer.models.timeseries.identification.blanchard_quah import BlanchardQuahModel
from stats_transformer.models.timeseries.reduced_form.local_projections import LocalProjectionsModel
from stats_transformer.models.timeseries.reduced_form.local_projections_iv import LocalProjectionsIVModel
from stats_transformer.models.timeseries.reduced_form.var import VARModel
from stats_transformer.reporting.timeseries import BlanchardQuahResultAdapter, LocalProjectionsIVResultAdapter, LocalProjectionsResultAdapter, TimeSeriesReporter, TimeSeriesResultAdapter, VARResultAdapter


def _synthetic_data(observations=120):
    generator = np.random.default_rng(42)
    shocks = generator.normal(size=(observations, 2))
    values = np.zeros((observations, 2))
    transition = np.array([[0.55, 0.10], [0.15, 0.40]])
    for index in range(1, observations):
        values[index] = transition @ values[index - 1] + shocks[index]
    data = pd.DataFrame(values, columns=["output", "inflation"])
    data["instrument"] = shocks[:, 0] + generator.normal(scale=0.2, size=observations)
    data["date"] = pd.date_range("1990-01-01", periods=observations, freq="QE")
    return data


def test_time_series_result_adapter_is_abstract():
    with pytest.raises(TypeError):
        TimeSeriesResultAdapter(None)


def test_var_adapter_normalizes_reporting_results():
    data = _synthetic_data()
    model = VARModel(target_variables=["output", "inflation"], date_column="date", maxlags=2)
    model.fit(data)

    report = VARResultAdapter(model, horizons=6).build()

    assert isinstance(report.irfs, xr.Dataset)
    assert set(report.irfs.coords.keys()) == {"horizon", "response", "shock"}
    assert report.irfs.coords["horizon"].max().item() == 6
    assert report.fevd.coords["horizon"].max().item() == 6
    
    fevd_df = report.fevd.to_dataframe().reset_index()
    sums = fevd_df.groupby(["horizon", "response"])["share"].sum()
    np.testing.assert_allclose(sums.to_numpy(), 1.0)
    
    assert len(report.historical_decomposition.data_vars) > 0
    assert len(report.structural_shocks.data_vars) > 0


def test_blanchard_quah_adapter_uses_long_run_impact_matrix():
    data = _synthetic_data()
    model = BlanchardQuahModel(target_variables=["output", "inflation"], date_column="date", maxlags=2)
    model.fit(data)

    report = BlanchardQuahResultAdapter(model, horizons=4).build()
    impact = report.irfs["estimate"].sel(horizon=0).to_pandas().reindex(index=model.target_variables, columns=model.target_variables)

    np.testing.assert_allclose(impact.to_numpy(), model.B_0)
    assert report.specification.loc[report.specification["statistic"] == "identification", "value"].iloc[0] == "long-run restrictions"


def test_local_projection_adapters_preserve_inference():
    data = _synthetic_data()
    lp_model = LocalProjectionsModel(target="output", shock_var="inflation", horizon=4)
    lp_model.fit(data)
    lp_report = LocalProjectionsResultAdapter(lp_model).build()

    iv_model = LocalProjectionsIVModel(target_variable="output", shock_variable="inflation", instrument_variable="instrument", horizons=4, date_column="date")
    iv_model.fit(data)
    iv_report = LocalProjectionsIVResultAdapter(iv_model).build()

    assert {"std_error", "lower", "upper"}.issubset(set(lp_report.irfs.data_vars.keys()))
    assert {"std_error", "lower", "upper"}.issubset(set(iv_report.irfs.data_vars.keys()))
    assert len(lp_report.irfs.coords["horizon"]) == 5
    assert len(iv_report.irfs.coords["horizon"]) == 5


def test_reporter_exports_var_figures_and_tables(tmp_path):
    data = _synthetic_data()
    model = VARModel(target_variables=["output", "inflation"], date_column="date", maxlags=2)
    model.fit(data)
    reporter = TimeSeriesReporter.from_var(model, horizons=4, output_dir=tmp_path)

    result = reporter.run(figure_outputs=["irfs", "fevd", "historical_decomposition"], table_formats=["csv"])

    assert len(result.figures["irfs"]) == 2
    assert all(Path(path).exists() for path in result.figures["irfs"])
    assert result.figures["fevd"]
    assert all(Path(path).exists() for path in result.figures["fevd"])
    assert result.figures["historical_decomposition"]
    assert all(Path(path).exists() for path in result.figures["historical_decomposition"])
    assert result.tables["irfs"]["csv"].exists()
    assert result.tables["fevd"]["csv"].exists()
