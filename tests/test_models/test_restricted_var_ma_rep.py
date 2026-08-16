import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.api import VAR
from stats_transformer.models.timeseries.reduced_form.restrictions import RestrictedVAR
from stats_transformer.models.timeseries.reduced_form.var import VARModel
from stats_transformer.models.timeseries.decompositions import TimeSeriesDecompositions


def _var_frame(n=160):
    gen = np.random.default_rng(13)
    shocks = gen.normal(size=(n, 2))
    values = np.zeros((n, 2))
    transition = np.array([[0.5, 0.1], [0.2, 0.4]])
    for i in range(1, n):
        values[i] = transition @ values[i - 1] + shocks[i]
    return pd.DataFrame(values, columns=["y1", "y2"])


def test_k_trend_is_one_with_constant():
    data = _var_frame()
    mask = np.ones((3, 2), dtype=bool)
    result = RestrictedVAR(data, mask=mask, maxlags=1, trend="c").fit()
    assert result.k_trend == 1


def test_k_trend_is_zero_without_constant():
    data = _var_frame()
    mask = np.ones((2, 2), dtype=bool)
    result = RestrictedVAR(data, mask=mask, maxlags=1, trend="n").fit()
    assert result.k_trend == 0


def test_exog_names_are_real_not_placeholders():
    data = _var_frame()
    mask = np.ones((3, 2), dtype=bool)
    result = RestrictedVAR(data, mask=mask, maxlags=1, trend="c").fit()
    assert result.exog_names == ["const", "L1.y1", "L1.y2"]


def test_ma_rep_matches_statsmodels_for_unrestricted_mask():
    # An all-True mask is mathematically an unrestricted OLS VAR, so the MA
    # representation must match statsmodels exactly. This is the strongest
    # available check that the k_trend slicing is correct.
    # Note statsmodels' maxn=n returns n+1 matrices (horizons 0..n).
    data = _var_frame()
    mask = np.ones((3, 2), dtype=bool)
    restricted = RestrictedVAR(data, mask=mask, maxlags=1, trend="c").fit()
    reference = VAR(data).fit(maxlags=1)

    np.testing.assert_allclose(restricted.ma_rep(maxn=8), reference.ma_rep(maxn=8), rtol=1e-8, atol=1e-10)


def test_ma_rep_respects_zero_restrictions():
    data = _var_frame()
    mask = np.array([[True, True], [True, True], [False, True]])
    restricted = RestrictedVAR(data, mask=mask, maxlags=1, trend="c").fit()

    phis = restricted.ma_rep(maxn=4)
    assert phis.shape == (5, 2, 2)  # horizons 0..4, matching statsmodels
    np.testing.assert_allclose(phis[0], np.eye(2))
    # The restricted coefficient is exactly zero, so y2 has no first-order
    # impulse effect on y1.
    assert phis[1][0, 1] == pytest.approx(0.0, abs=1e-12)


def test_decompositions_work_for_restricted_var():
    # TimeSeriesDecompositions calls result.ma_rep(...), which
    # RestrictedVARResults did not implement.
    data = _var_frame()
    mask = np.ones((3, 2), dtype=bool)
    model = VARModel(target_variables=["y1", "y2"], maxlags=1, mask=mask)
    model.fit(data)

    fevd = TimeSeriesDecompositions(model.model).compute_fevd(steps=6)
    assert fevd.shape == (6, 2, 2)
    # Each response row's variance shares sum to 1.
    np.testing.assert_allclose(fevd.sum(axis=2), np.ones((6, 2)), rtol=1e-6)


def test_forecaster_still_matches_after_shared_extraction():
    # VARForecaster._compute_ma_rep and RestrictedVARResults.ma_rep now share
    # one implementation; confirm the forecaster path is unchanged.
    # _compute_ma_rep(steps) returns `steps` matrices, whereas statsmodels'
    # ma_rep(maxn=n) returns n+1 (horizons 0..n) -- compare the overlap.
    from stats_transformer.models.timeseries.reduced_form.forecasting import VARForecaster

    data = _var_frame()
    reference = VAR(data).fit(maxlags=1)
    forecaster = VARForecaster(reference)

    np.testing.assert_allclose(forecaster._compute_ma_rep(6), reference.ma_rep(maxn=6)[:6], rtol=1e-8, atol=1e-10)
