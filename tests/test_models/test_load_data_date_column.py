import numpy as np
import pandas as pd
import pytest
import tempfile
import os
import yaml
from stats_transformer.models.regression.regression import RegressionModel
from stats_transformer.models.timeseries.identification.blanchard_quah import BlanchardQuahModel
from stats_transformer.models.timeseries.identification.sign_zero import SignZeroSVARModel


def _var_frame(n=140, freq="QS", date_name="date"):
    gen = np.random.default_rng(7)
    shocks = gen.normal(size=(n, 2))
    values = np.zeros((n, 2))
    transition = np.array([[0.5, 0.1], [0.1, 0.4]])
    for i in range(1, n):
        values[i] = transition @ values[i - 1] + shocks[i]
    df = pd.DataFrame(values, columns=["y1", "y2"])
    df[date_name] = pd.date_range("2000-01-01", periods=n, freq=freq)
    return df


def test_single_equation_model_still_indexes_on_date():
    # No-op guarantee: models that never set time_column (regression, panel,
    # IV, discrete) must keep the pre-existing behavior of indexing on a
    # column literally named "date".
    gen = np.random.default_rng(3)
    df = pd.DataFrame({"x1": gen.normal(size=50), "y": gen.normal(size=50)})
    df["date"] = pd.date_range("2010-01-01", periods=50, freq="MS")

    model = RegressionModel(target="y", independent_variables=["x1"])
    model.load_data(df)

    assert model.df_clean.index.name == "date"
    assert "date" not in model.df_clean.columns


def test_svar_keeps_date_column_accessible_when_time_column_set():
    # blanchard_quah sets self.time_column, so its date column must survive
    # load_data as a column rather than being consumed into the index.
    df = _var_frame()
    model = BlanchardQuahModel(target_variables=["y1", "y2"], date_column="date", maxlags=1)
    model.fit(df)

    assert "date" in model.df_clean.columns


def test_svar_sorts_by_date_when_input_is_shuffled():
    df = _var_frame()
    shuffled = df.sample(frac=1.0, random_state=11).reset_index(drop=True)

    ordered = BlanchardQuahModel(target_variables=["y1", "y2"], date_column="date", maxlags=1)
    ordered.fit(df)
    scrambled = BlanchardQuahModel(target_variables=["y1", "y2"], date_column="date", maxlags=1)
    scrambled.fit(shuffled)

    assert scrambled.df_clean["date"].is_monotonic_increasing
    np.testing.assert_allclose(ordered.var_result.sigma_u, scrambled.var_result.sigma_u, rtol=1e-10)


def test_narrative_restrictions_work_with_date_column():
    # Regression test: _check_narrative reads self.df_clean[self.date_column].
    # When the column is literally named "date", load_data used to move it to
    # the index, so this raised KeyError: 'date'.
    df = _var_frame()
    target_date = str(df["date"].iloc[60].date())

    config = {
        "variables": ["y1", "y2"],
        "shocks": ["a", "b"],
        "restrictions": [{"shock": "a", "response": "y1", "type": "sign", "value": "+", "horizon": 0}],
        "narrative_restrictions": [{"shock": "a", "type": "sign", "value": "+", "date": target_date}],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name

    try:
        model = SignZeroSVARModel(target_variables=["y1", "y2"], config_path=config_path, date_column="date", maxlags=1, max_draws=200, required_accepts=3)
        model.fit(df)
        assert model.total_draws > 0
    finally:
        os.remove(config_path)


def test_narrative_restrictions_work_with_non_date_column_name():
    # The same path with a differently-named time column, which never hit the bug.
    df = _var_frame(date_name="period")
    target_date = str(df["period"].iloc[60].date())

    config = {
        "variables": ["y1", "y2"],
        "shocks": ["a", "b"],
        "restrictions": [{"shock": "a", "response": "y1", "type": "sign", "value": "+", "horizon": 0}],
        "narrative_restrictions": [{"shock": "a", "type": "sign", "value": "+", "date": target_date}],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name

    try:
        model = SignZeroSVARModel(target_variables=["y1", "y2"], config_path=config_path, date_column="period", maxlags=1, max_draws=200, required_accepts=3)
        model.fit(df)
        assert model.total_draws > 0
    finally:
        os.remove(config_path)
