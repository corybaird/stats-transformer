import numpy as np
import pandas as pd
import pytest
from stats_transformer.models.timeseries.reduced_form.var import VARModel
from stats_transformer.models.regression.regression import RegressionModel


def _var_frame(n=140):
    gen = np.random.default_rng(5)
    shocks = gen.normal(size=(n, 3))
    values = np.zeros((n, 3))
    transition = 0.4 * np.eye(3) + 0.1 * (np.ones((3, 3)) - np.eye(3))
    for i in range(1, n):
        values[i] = transition @ values[i - 1] + shocks[i]
    return pd.DataFrame(values, columns=["gdp", "cpi", "rate"])


def test_var_coefficients_are_not_empty():
    model = VARModel(target_variables=["gdp", "cpi", "rate"], maxlags=1)
    model.fit(_var_frame())
    coefficients = model.get_model_metadata()["coefficients"]
    assert coefficients != {}


def test_var_coefficients_are_nested_by_equation():
    model = VARModel(target_variables=["gdp", "cpi", "rate"], maxlags=1)
    model.fit(_var_frame())
    coefficients = model.get_model_metadata()["coefficients"]

    assert set(coefficients) == {"gdp", "cpi", "rate"}
    gdp_terms = coefficients["gdp"]
    assert "L1.gdp" in gdp_terms
    entry = gdp_terms["L1.gdp"]
    assert isinstance(entry["value"], float)
    assert isinstance(entry["std_err"], float)
    assert isinstance(entry["p_value"], float)


def test_var_coefficients_match_underlying_params():
    model = VARModel(target_variables=["gdp", "cpi", "rate"], maxlags=1)
    model.fit(_var_frame())
    coefficients = model.get_model_metadata()["coefficients"]

    expected = float(model.model.params.loc["L1.gdp", "gdp"])
    assert coefficients["gdp"]["L1.gdp"]["value"] == pytest.approx(expected)


def test_single_equation_coefficients_stay_flat():
    # No-op guarantee: OLS keeps the flat {term: {...}} shape.
    gen = np.random.default_rng(1)
    df = pd.DataFrame({"x1": gen.normal(size=60), "x2": gen.normal(size=60)})
    df["y"] = 2.0 * df["x1"] + gen.normal(size=60)

    model = RegressionModel(target="y", independent_variables=["x1", "x2"])
    model.fit(df)
    coefficients = model.get_model_metadata()["coefficients"]

    assert "x1" in coefficients
    assert isinstance(coefficients["x1"]["value"], float)
    assert coefficients["x1"]["ci_lower"] is not None
