import numpy as np
import pandas as pd
import pytest

from stats_transformer.models.timeseries.identification.max_share import MaxShareSVARModel

N = 300
LAGS = 1


def _var_frame(variances=(4.0, 1.0, 1.0), seed=11):
    rng = np.random.default_rng(seed)
    shocks = rng.normal(size=(N, 3)) * np.sqrt(variances)
    y = np.zeros((N, 3))
    for t in range(1, N):
        y[t] = 0.5 * y[t - 1] + shocks[t]
    df = pd.DataFrame(y, columns=["a", "b", "c"])
    df["date"] = pd.date_range("2000-01-01", periods=N, freq="MS")
    return df


def _fit(df, **kwargs):
    model = MaxShareSVARModel(target_variables=["a", "b", "c"], date_column="date", maxlags=LAGS, **kwargs)
    model.fit(df)
    return model


def test_fit_identifies_dominant_shock():
    model = _fit(_var_frame(), target_variable="a")
    k = 3
    Q = np.asarray(model.Q)
    B = np.asarray(model.impact_matrix)
    # orthonormal rotation and impact factorization
    np.testing.assert_allclose(Q @ Q.T, np.eye(k), atol=1e-10)
    sigma = np.asarray(model.var_result.sigma_u)
    np.testing.assert_allclose(B @ B.T, sigma, atol=1e-10)
    # with independent errors and a variance-4 driving shock, the max-share
    # shock recovers most of variable a's forecast error variance
    assert model.max_share > 0.6
    # sign convention: non-negative response of a at max_horizon
    irfs = model.irf(steps=model.max_horizon + 1)
    value = irfs.loc[(model.max_horizon, "a"), "shock_1"]
    assert value >= 0
    metadata = model.get_model_metadata()
    assert np.asarray(metadata["impact_matrix"]).shape == (k, k)
    assert metadata["max_share"] == model.max_share


def test_fevd_rows_sum_to_one_and_matches_share():
    model = _fit(_var_frame(), target_index=0)
    shares = model.fevd(steps=41)
    np.testing.assert_allclose(shares.sum(axis=1), 1.0, atol=1e-10)
    assert list(shares.columns) == ["shock_1", "shock_2", "shock_3"]
    assert list(shares.index) == ["a", "b", "c"]
    # fevd at the configured horizon must reproduce the maximized share for
    # the target variable
    window = 41
    expected = float(shares.loc["a", "shock_1"])
    assert expected == pytest.approx(model.max_share, abs=1e-8)


def test_zero_impact_gives_zero_contemporaneous_response():
    model = _fit(_var_frame(), target_variable="a", zero_impact=True)
    B = np.asarray(model.impact_matrix)
    # news shock: null contemporaneous impact on the target variable
    assert B[0, 0] == pytest.approx(0.0, abs=1e-10)
    Q = np.asarray(model.Q)
    np.testing.assert_allclose(Q @ Q.T, np.eye(3), atol=1e-10)
    sigma = np.asarray(model.var_result.sigma_u)
    np.testing.assert_allclose(B @ B.T, sigma, atol=1e-10)
    assert model.max_share > 0.0
    assert model.max_share <= 1.0 + 1e-10


def test_min_horizon_offset():
    model = MaxShareSVARModel(target_variables=["a", "b", "c"], date_column="date", maxlags=LAGS, min_horizon=2, max_horizon=8, target_variable="a")
    model.fit(_var_frame())
    assert 0.0 <= model.max_share <= 1.0 + 1e-10


def test_untrained_raises_and_missing_columns_raise():
    model = MaxShareSVARModel(target_variables=["a", "b", "c"], maxlags=LAGS)
    with pytest.raises(ValueError):
        model.irf()
    with pytest.raises(ValueError):
        model.get_summary()
    with pytest.raises(ValueError):
        model.fit(pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, 2.0]}))
    with pytest.raises(ValueError):
        MaxShareSVARModel(target_variables=["a", "b"], target_variable="z", maxlags=LAGS).fit(_var_frame())


def test_run_returns_metadata():
    model = MaxShareSVARModel(target_variables=["a", "b", "c"], date_column="date", maxlags=LAGS, target_variable="a")
    metadata = model.run(_var_frame())
    assert "impact_matrix" in metadata
    assert metadata["metrics"]["nobs"] == N - LAGS
