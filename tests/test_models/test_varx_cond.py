import numpy as np
import pandas as pd
import pytest
from stats_transformer.models.timeseries.reduced_form.var import VARModel
from stats_transformer.models.timeseries.reduced_form.varx import VARXModel, ConditionalForecaster

A1_DGP = np.array([[0.5, 0.3], [-0.2, 0.6]])
B0_DGP = np.array([[0.8], [-0.5]])
C_DGP = np.array([0.2, -0.1])

def make_varx_df(n=400, seed=7, warmup=200):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n + warmup)
    u = rng.normal(scale=0.05, size=(n + warmup, 2))
    y = np.zeros((n + warmup, 2))
    for t in range(1, n + warmup):
        y[t] = C_DGP + A1_DGP @ y[t - 1] + B0_DGP[:, 0] * x[t] + u[t]
    df = pd.DataFrame({"date": pd.date_range("2000-01-01", periods=n, freq="ME"), "y1": y[warmup:, 0], "y2": y[warmup:, 1], "x1": x[warmup:]})
    return df

def fit_varx(df=None, **kwargs):
    if df is None:
        df = make_varx_df()
    model = VARXModel(target_variables=["y1", "y2"], exog_variables=["x1"], date_column="date", **kwargs)
    model.fit(df)
    return model

def fit_varx_lag1(df=None):
    return fit_varx(df, maxlags=1)

def test_varx_fit_recovers_dgp():
    model = fit_varx()
    assert np.allclose(model.A_matrices[0], A1_DGP, atol=0.06)
    assert np.allclose(model.B_matrices[0], B0_DGP, atol=0.06)
    assert np.allclose(model.intercept, C_DGP, atol=0.06)
    assert np.allclose(model.sigma_u, np.eye(2) * 0.05 ** 2, atol=0.01)

def test_varx_default_maxlags_is_two():
    df = make_varx_df()
    model = fit_varx(df)
    assert model.A_matrices.shape == (2, 2, 2)
    assert len(model.B_matrices) == 1
    assert abs(model.A_matrices[1]).max() < 0.2

def test_varx_metrics_and_summary():
    model = fit_varx()
    metrics = model.get_model_metrics()
    assert np.isfinite(metrics["aic"]) and np.isfinite(metrics["bic"])
    assert metrics["num_observations"] == 398
    assert metrics["maxlags"] == 2 and metrics["exog_lags"] == 0
    summary = model.get_summary()
    assert isinstance(summary, str)
    assert "y1" in summary and "y2" in summary and "AIC" in summary

def test_varx_exog_lags():
    n = 500
    rng = np.random.default_rng(11)
    x = rng.normal(size=n)
    u = rng.normal(scale=0.05, size=n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.4 * y[t - 1] + 0.9 * x[t] + 0.3 * x[t - 1] + u[t]
    df = pd.DataFrame({"y1": y[1:], "x1": x[1:]})
    model = VARXModel(target_variables=["y1"], exog_variables=["x1"], maxlags=1, exog_lags=1)
    model.fit(df)
    assert np.allclose(model.A_matrices[0], [[0.4]], atol=0.06)
    assert np.allclose(model.B_matrices[0], [[0.9]], atol=0.06)
    assert np.allclose(model.B_matrices[1], [[0.3]], atol=0.06)

def test_varx_dynamic_multipliers():
    model = fit_varx(maxlags=1)
    mults = model.dynamic_multipliers(steps=4)
    assert model.A_matrices.shape == (1, 2, 2) and len(model.B_matrices) == 1
    assert list(mults.columns) == ["y1", "y2"]
    assert mults.shape == (4, 2)
    B0 = model.B_matrices[0]
    assert np.allclose(mults.iloc[0], B0[:, 0])
    rolling = B0[:, 0].copy()
    for step in pd.RangeIndex(4):
        expected = rolling.copy()
        rolling = model.A_matrices[0] @ rolling
        assert np.allclose(mults.iloc[step], expected)

def test_varx_forecast_matches_recursion():
    model = fit_varx(maxlags=1)
    steps = 3
    last_y = model.y.values[-1]
    exog_future = pd.DataFrame({"x1": [1.1, -2.0, 0.5]})
    forecast = model.forecast(steps, exog_future=exog_future)
    manual = []
    prev = last_y
    for h in range(steps):
        val = model.intercept + model.A_matrices[0] @ (prev if h == 0 else manual[-1]) + model.B_matrices[0][:, 0] * exog_future.values[h]
        manual.append(val)
    assert np.allclose(forecast.values, manual, atol=1e-10)
    assert list(forecast.columns) == ["y1", "y2"]

def test_varx_forecast_default_holds_last_exog():
    model = fit_varx(maxlags=1)
    last_exog = model.df_clean["x1"].values[-1]
    with_exog = model.forecast(2, exog_future=pd.DataFrame({"x1": [last_exog] * 2}))
    default = model.forecast(2)
    assert np.allclose(with_exog.values, default.values, atol=1e-10)

def test_varx_forecast_bad_shape_raises():
    model = fit_varx(maxlags=1)
    with pytest.raises(ValueError):
        model.forecast(3, exog_future=np.zeros((2, 1)))

def test_varx_model_metadata():
    model = fit_varx()
    metadata = model.get_model_metadata()
    assert "aic" in metadata["metrics"]
    assert "y1" in metadata["coefficients"]
    assert "L1.y2" in metadata["coefficients"]["y1"]
    assert metadata["summary"]["variables"] == ["y1", "y2"]

def test_conditional_forecaster_single_step_min_norm():
    A = np.array([[0.5, 0.2], [0.0, 0.3]])
    history = np.array([[1.0, 2.0]])
    forecaster = ConditionalForecaster((A, np.zeros(2), np.eye(2), history), var_names=["cy1", "cy2"])
    forecast, shocks = forecaster.run({"cy1": {1: 3.0}})
    assert forecast.loc[1, "cy1"] == pytest.approx(3.0)
    assert forecast.loc[1, "cy2"] == pytest.approx(0.6)
    assert shocks.loc[1, "cy1_shock"] == pytest.approx(3.0 - 0.9)
    assert shocks.loc[1, "cy2_shock"] == pytest.approx(0.0)
    assert list(forecast.columns) == ["cy1", "cy2"]

def test_conditional_forecaster_multistep_min_norm_ratio():
    A = np.array([[0.5, 0.2], [0.0, 0.3]])[:, :, None].transpose(2, 0, 1)
    history = np.array([[1.0, 2.0]])
    forecaster = ConditionalForecaster((A, np.zeros(2), np.eye(2), history), var_names=["y1", "y2"])
    base_y1_h2 = (A[0] @ (A[0] @ history[-1]))[0]
    forecast, shocks = forecaster.run({"y1": {2: base_y1_h2 + 1.0}})
    assert forecast.loc[2, "y1"] == pytest.approx(base_y1_h2 + 1.0)
    assert shocks.loc[2, "y2_shock"] == pytest.approx(0.0)
    assert shocks.loc[1, "y1_shock"] / shocks.loc[1, "y2_shock"] == pytest.approx(A[0][0, 0] / A[0][0, 1])
    lam = shocks.loc[2, "y1_shock"]
    assert shocks.loc[1, "y1_shock"] == pytest.approx(A[0][0, 0] * lam)
    assert shocks.loc[1, "y2_shock"] == pytest.approx(A[0][0, 1] * lam)

def test_conditional_forecaster_active_shocks():
    A = np.array([[0.5, 0.2], [0.0, 0.3]])
    history = np.array([[1.0, 2.0]])
    forecaster = ConditionalForecaster((A, np.zeros(2), np.eye(2), history), var_names=["y1", "y2"], active_shocks=["y2"])
    forecast, shocks = forecaster.run({"y2": {1: 1.2}})
    assert forecast.loc[1, "y2"] == pytest.approx(1.2)
    assert forecast.loc[1, "y1"] == pytest.approx(0.9)
    assert shocks.loc[1, "y2_shock"] == pytest.approx(0.6)
    assert shocks.loc[1, "y1_shock"] == pytest.approx(0.0)

def test_conditional_forecaster_dataframe_constraints():
    A = np.array([[0.5, 0.2], [0.0, 0.3]])
    history = np.array([[1.0, 2.0]])
    forecaster = ConditionalForecaster((A, np.zeros(2), np.eye(2), history), var_names=["y1", "y2"])
    matrix = pd.DataFrame({1: [3.0]}, index=["y1"])
    forecast, shocks = forecaster.run(matrix)
    assert forecast.loc[1, "y1"] == pytest.approx(3.0)
    assert shocks.loc[1, "y1_shock"] == pytest.approx(2.1)

def test_conditional_forecaster_horizon_extend_and_baseline_only():
    A = np.array([[0.5, 0.2], [0.0, 0.3]])
    history = np.array([[1.0, 2.0]])
    forecaster = ConditionalForecaster((A, np.zeros(2), np.eye(2), history), var_names=["y1", "y2"])
    forecast, shocks = forecaster.run({}, horizon=3)
    assert forecast.shape == (3, 2) and shocks.shape == (3, 2)
    assert np.allclose(shocks.values, 0.0)
    assert forecast.loc[1, "y2"] == pytest.approx(0.6)

def test_conditional_forecaster_from_fitted_var_model():
    df = make_varx_df()
    var_df = df[["date", "y1", "y2"]]
    model = VARModel(target_variables=["y1", "y2"], maxlags=1)
    model.fit(var_df)
    forecaster = ConditionalForecaster(model)
    baseline, shocks = forecaster.run({}, horizon=4)
    baseline_manual = model.model.forecast(model.y.values[-1:], 4)
    assert np.allclose(baseline.values, baseline_manual, atol=1e-8)
    assert np.allclose(shocks.values, 0.0)
    target = baseline.loc[2, "y1"] + 1.0
    forecast, _ = forecaster.run({"y1": {2: target}})
    assert forecast.loc[2, "y1"] == pytest.approx(target)

def test_conditional_forecaster_varx_model_exact_constraint():
    model = fit_varx()
    forecaster = ConditionalForecaster(model)
    baseline, _ = forecaster.run({}, horizon=2)
    target = baseline.loc[2, "y1"] + 0.5
    forecast, shocks = forecaster.run({"y1": {2: target}})
    assert forecast.loc[2, "y1"] == pytest.approx(target)
    assert shocks.shape == (2, 2)

def test_conditional_forecaster_rejects_bad_input():
    with pytest.raises(TypeError):
        ConditionalForecaster("not a model")
