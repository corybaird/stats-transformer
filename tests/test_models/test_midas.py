import numpy as np
import pandas as pd
import pytest

from stats_transformer.models.timeseries.reduced_form.midas import (
    MIDASModel,
    UMIDASModel,
    _beta_weights,
    _exp_almon_weights,
)


def _simulate(weights, n_blocks=500, freq_ratio=3, ar=0.0, noise=0.1, seed=42):
    rng = np.random.default_rng(seed)
    K = len(weights)
    n = n_blocks * freq_ratio
    x = rng.normal(size=n)
    y = np.full(n, np.nan)
    prev = 0.0
    for end in [(t + 1) * freq_ratio - 1 for t in range(n_blocks)]:
        if end - K < 0:
            continue
        lags = [x[end - j] for j in range(1, K + 1)]
        y[end] = ar * prev + float(np.dot(weights, lags)) + rng.normal(scale=noise)
        prev = y[end]
    return pd.DataFrame({"x": x, "y": y})


def test_umidas_recovers_dgp_with_ar_lag():
    K, m = 4, 3
    weights = np.array([0.5, -0.25, 0.15, 0.1])
    df = _simulate(weights, n_blocks=500, freq_ratio=m, ar=0.3, noise=0.2, seed=1)
    model = UMIDASModel(target="y", independent_variables=["x"], freq_ratio=m, hf_lags=K, y_lags=1)
    metrics = model.fit(df)
    params = model.model.params
    assert metrics["r_squared"] > 0.8
    for j, w in enumerate(weights):
        assert abs(params[f"x_hf_lag{j + 1}"] - w) < 0.08
    assert abs(params["y_lag1"] - 0.3) < 0.06
    assert abs(params["const"]) < 0.1
    metadata = model.get_model_metadata()
    assert metadata["coefficients"]["x_hf_lag1"]["std_err"] > 0
    assert "x_hf_lag1" in model.get_summary()


def test_umidas_predicts_holdout():
    K, m = 4, 3
    weights = np.array([0.6, 0.2, -0.1, 0.05])
    df = _simulate(weights, n_blocks=600, freq_ratio=m, noise=0.1, seed=7)
    split = 500 * m
    model = UMIDASModel(target="y", independent_variables=["x"], freq_ratio=m, hf_lags=K)
    model.fit(df.iloc[:split])
    test = df.iloc[split - int(np.ceil(K / m)) * m:]
    preds = model.predict(test)
    actual = test["y"].dropna()
    common = preds.index.intersection(actual.index)
    assert len(common) == len(preds)
    mse = float(np.mean((preds.loc[common] - actual.loc[common]) ** 2))
    assert mse < 0.03


def test_umidas_prealigned_columns_match_reshape():
    K, m = 4, 3
    weights = np.array([0.5, -0.3, 0.2, 0.1])
    df = _simulate(weights, n_blocks=400, freq_ratio=m, noise=0.1, seed=3)
    reshaped = UMIDASModel(target="y", independent_variables=["x"], freq_ratio=m, hf_lags=K)
    reshaped.fit(df)
    prealigned = UMIDASModel(target="y", independent_variables=list(reshaped.X.columns))
    prealigned.fit(reshaped.X.join(reshaped.y.rename("y")))
    assert np.allclose(prealigned.model.params.to_numpy(), reshaped.model.params.to_numpy(), atol=1e-8)


def test_midas_almon_recovers_weights_and_polynomial():
    K, m = 6, 3
    b = np.array([1.0, -0.05, 0.005])
    k = np.arange(1, K + 1)
    weights = b[0] + b[1] * k + b[2] * k ** 2
    df = _simulate(weights, n_blocks=400, freq_ratio=m, noise=0.01, seed=5)
    model = MIDASModel(target="y", independent_variables=["x"], freq_ratio=m, hf_lags=K, weight_function="almon", poly_degree=2)
    metrics = model.fit(df)
    assert np.allclose(model.weights, weights, atol=0.005)
    Q = np.vander(k, 3, increasing=True)
    assert np.allclose(np.linalg.lstsq(Q, model.weights, rcond=None)[0], b, atol=0.005)
    assert np.all(model.weight_se > 0)
    assert metrics["r_squared"] > 0.9
    assert metrics["weight_function"] == "almon"


def test_midas_almon_predicts_holdout():
    K, m = 6, 3
    b = np.array([1.0, -0.15, 0.01])
    weights = b[0] + b[1] * np.arange(1, K + 1) + b[2] * np.arange(1, K + 1) ** 2
    df = _simulate(weights, n_blocks=500, freq_ratio=m, noise=0.02, seed=9)
    split = 400 * m
    model = MIDASModel(target="y", independent_variables=["x"], freq_ratio=m, hf_lags=K, weight_function="almon")
    model.fit(df.iloc[:split])
    test = df.iloc[split - int(np.ceil(K / m)) * m:]
    preds = model.predict(test)
    actual = test["y"].dropna()
    common = preds.index.intersection(actual.index)
    mse = float(np.mean((preds.loc[common] - actual.loc[common]) ** 2))
    assert mse < 0.002
    assert len(model.get_summary().splitlines()) > 3


def test_midas_exp_almon_recovers_weights():
    K, m = 8, 3
    theta = np.array([-0.2, -0.05])
    weights = _exp_almon_weights(theta, K)
    df = _simulate(weights, n_blocks=500, freq_ratio=m, noise=0.05, seed=11)
    model = MIDASModel(target="y", independent_variables=["x"], freq_ratio=m, hf_lags=K, weight_function="exp_almon")
    metrics = model.fit(df)
    assert np.allclose(model.weights, weights, atol=0.02)
    assert np.allclose(model.theta, theta, atol=0.15)
    assert metrics["r_squared"] > 0.8
    assert np.all(model.weight_se > 0)
    assert abs(model.theta[0]) + abs(model.theta[1]) > 0


def test_midas_beta_recovers_weights():
    K, m = 8, 3
    theta = np.array([2.0, 3.0])
    weights = _beta_weights(theta, K)
    df = _simulate(weights, n_blocks=500, freq_ratio=m, noise=0.05, seed=13)
    model = MIDASModel(target="y", independent_variables=["x"], freq_ratio=m, hf_lags=K, weight_function="beta")
    metrics = model.fit(df)
    assert np.allclose(model.weights, weights, atol=0.02)
    assert metrics["weight_function"] == "beta"
    assert np.all(model.weight_se > 0)


def test_predict_before_fit_raises():
    model = UMIDASModel(target="y", independent_variables=["x"])
    with pytest.raises(ValueError):
        model.predict(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))


def test_invalid_weight_function_raises():
    with pytest.raises(ValueError):
        MIDASModel(target="y", independent_variables=["x"], weight_function="bogus")
