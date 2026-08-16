import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.api import VAR
from stats_transformer.models.timeseries.identification.blanchard_quah import BlanchardQuahModel


def _bq_frame(n=200):
    gen = np.random.default_rng(21)
    shocks = gen.normal(size=(n, 2))
    values = np.zeros((n, 2))
    transition = np.array([[0.5, 0.1], [0.2, 0.4]])
    for i in range(1, n):
        values[i] = transition @ values[i - 1] + shocks[i]
    return pd.DataFrame(values, columns=["y1", "y2"])


def _expected_long_run(var_result, k):
    # Recompute the long-run impact matrix independently, slicing lag
    # coefficients by k_trend, which is correct for every trend spec.
    params = var_result.params
    sum_A = np.zeros((k, k))
    for lag in range(var_result.k_ar):
        start = var_result.k_trend + lag * k
        sum_A += params.iloc[start:start + k, :].values.T
    inv_phi = np.linalg.inv(np.eye(k) - sum_A)
    sigma_u = var_result.sigma_u
    sigma_u = sigma_u.values if isinstance(sigma_u, pd.DataFrame) else sigma_u
    long_run_cov = inv_phi @ sigma_u @ inv_phi.T
    return np.linalg.inv(inv_phi) @ np.linalg.cholesky(long_run_cov)


def test_blanchard_quah_matches_reference_with_constant():
    df = _bq_frame()
    model = BlanchardQuahModel(target_variables=["y1", "y2"], maxlags=1)
    model.fit(df)

    expected = _expected_long_run(model.var_result, 2)
    np.testing.assert_allclose(model.B_0, expected, rtol=1e-8)


def test_blanchard_quah_correct_under_trend_ct():
    # With trend="ct" there are two deterministic rows (const, trend). The old
    # `1 if 'const' in params.index else 0` offset returned 1, slicing the
    # trend row as if it were a lag coefficient and corrupting B_0.
    df = _bq_frame()
    model = BlanchardQuahModel(target_variables=["y1", "y2"], maxlags=1)
    model.load_data(df)
    model.y = model.df_clean[["y1", "y2"]].astype(float)
    model.var_result = VAR(model.y).fit(maxlags=1, trend="ct")
    model.model = model.var_result
    model._compute_blanchard_quah()

    assert model.var_result.k_trend == 2
    expected = _expected_long_run(model.var_result, 2)
    np.testing.assert_allclose(model.B_0, expected, rtol=1e-8)


def test_blanchard_quah_correct_without_trend():
    df = _bq_frame()
    model = BlanchardQuahModel(target_variables=["y1", "y2"], maxlags=1)
    model.load_data(df)
    model.y = model.df_clean[["y1", "y2"]].astype(float)
    model.var_result = VAR(model.y).fit(maxlags=1, trend="n")
    model.model = model.var_result
    model._compute_blanchard_quah()

    assert model.var_result.k_trend == 0
    expected = _expected_long_run(model.var_result, 2)
    np.testing.assert_allclose(model.B_0, expected, rtol=1e-8)
