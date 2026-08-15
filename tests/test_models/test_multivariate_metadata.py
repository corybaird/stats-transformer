import json
import numpy as np
import pandas as pd
import pytest
from stats_transformer.models.timeseries.reduced_form.var import VARModel
from stats_transformer.models.timeseries.reduced_form.vecm import VECMModel
from stats_transformer.models.timeseries.identification.blanchard_quah import BlanchardQuahModel


def _var_frame(n=140):
    gen = np.random.default_rng(5)
    shocks = gen.normal(size=(n, 3))
    values = np.zeros((n, 3))
    transition = 0.4 * np.eye(3) + 0.1 * (np.ones((3, 3)) - np.eye(3))
    for i in range(1, n):
        values[i] = transition @ values[i - 1] + shocks[i]
    return pd.DataFrame(values, columns=["gdp", "cpi", "rate"])


def _integrated_frame(n=140):
    gen = np.random.default_rng(5)
    y1 = np.cumsum(gen.normal(size=n))
    y2 = y1 * 0.5 + gen.normal(size=n)
    return pd.DataFrame({"y1": y1, "y2": y2})


def test_var_metadata_has_no_dummy():
    model = VARModel(target_variables=["gdp", "cpi", "rate"], maxlags=1)
    model.fit(_var_frame())
    assert "dummy" not in json.dumps(model.get_model_metadata())


def test_vecm_metadata_has_no_dummy():
    model = VECMModel(target_variables=["y1", "y2"], k_ar_diff=1)
    model.fit(_integrated_frame())
    assert "dummy" not in json.dumps(model.get_model_metadata())


def test_svar_metadata_has_no_dummy():
    model = BlanchardQuahModel(target_variables=["gdp", "cpi"], maxlags=1)
    model.fit(_var_frame()[["gdp", "cpi"]])
    assert "dummy" not in json.dumps(model.get_model_metadata())


def test_var_metadata_reports_variables_not_fake_split():
    # A VAR is symmetric: there is no dependent/independent split to report.
    model = VARModel(target_variables=["gdp", "cpi", "rate"], maxlags=1)
    model.fit(_var_frame())
    summary = model.get_model_metadata()["summary"]

    assert summary.get("variables") == ["gdp", "cpi", "rate"]
    assert "dependent_variable" not in summary
    assert "independent_variables" not in summary


def test_single_equation_metadata_still_reports_dependent_and_independent():
    # No-op guarantee for genuinely single-equation models.
    from stats_transformer.models.regression.regression import RegressionModel

    gen = np.random.default_rng(1)
    df = pd.DataFrame({"x1": gen.normal(size=60), "x2": gen.normal(size=60)})
    df["y"] = 2.0 * df["x1"] + gen.normal(size=60)

    model = RegressionModel(target="y", independent_variables=["x1", "x2"])
    model.fit(df)
    summary = model.get_model_metadata()["summary"]

    assert summary["dependent_variable"] == "y"
    assert summary["independent_variables"] == ["x1", "x2"]
    assert "variables" not in summary


def test_multivariate_models_construct_without_target_arguments():
    # The "dummy" sentinel existed only to satisfy ModelBase's validation.
    # Multivariate models should no longer need to pass fake values at all.
    model = VARModel(target_variables=["gdp", "cpi", "rate"], maxlags=1)
    assert model.target is None
    assert model.independent_variables == []
