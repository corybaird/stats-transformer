"""Unit and regression tests for academic paper replications."""
import pytest
from examples.academic.vanderwel_2026 import Vanderwel2026Replication
from examples.academic.nakamura_steinsson_2018b_jorda import NakamuraSteinsson2018bJordaReplication
from examples.academic.bauer_swanson_2023 import BauerSwanson2023Replication
from stats_transformer.models.regression.regression import RegressionModel
from stats_transformer.models.unsupervised.unsupervised import PCAModel
from stats_transformer.models.timeseries.reduced_form.local_projections import LocalProjectionsModel
from stats_transformer.models.timeseries.reduced_form.forecasting import VARForecaster
import pandas as pd
import numpy as np


def test_vanderwel_2026_replication():
    """Verify Vanderwel et al. (2026) Table 2 Diebold-Li row is replicated within tolerance."""
    rep = Vanderwel2026Replication()
    res = rep.run()
    assert res["passed"], f"Vanderwel replication failed with max error {res['max_error']}"
    for replicated, published in zip(res["r2_oos_replicated"], res["r2_oos_published"], strict=False):
        assert abs(replicated - published) <= rep.TOLERANCE


def test_nakamura_steinsson_2018b_jorda_replication():
    """Verify Nakamura & Steinsson (2018b) Appendix Jordà multiplier and IRFs match published values."""
    rep = NakamuraSteinsson2018bJordaReplication()
    res = rep.run()
    assert res["passed"], f"Nakamura & Steinsson multiplier error {res['multiplier_error']} or IRF diff {res['max_irf_diff']}"
    assert abs(res["multiplier_h36"] - 1.95) < 0.01


def test_bauer_swanson_2023_replication():
    """Verify Bauer & Swanson (2023) macro news regression replicates with RobustOLSModel HC3.

    Published coefficient tables are not available in the source replication
    repository (shipped folder is confidential/empty), so the coefficients and
    R^2 are asserted against a direct statsmodels HC3 computation on the
    identical sample (BauerSwanson2023Replication.TOLERANCE = 1e-4).
    """
    rep = BauerSwanson2023Replication()
    res = rep.run()
    assert res["passed"], f"Bauer & Swanson replication failed: coef error {res['max_coef_error']}, r2 error {res['r2_error']}"
    assert res["n_observations"] > 0
    assert res["r_squared"] >= 0.0
    for name in rep.NEWS:
        assert abs(res["coefficients"][name] - res["reference_coefficients"][name]) <= rep.TOLERANCE


def test_regression_predict_accessor():
    """Verify RegressionModel.predict() generates accurate predictions."""
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = 2.5 * x + 1.0 + np.random.normal(0, 0.1, 50)
    df = pd.DataFrame({"x": x, "y": y})

    model = RegressionModel(target="y", independent_variables=["x"])
    model.fit(df)

    new_df = pd.DataFrame({"x": [2.0, 4.0, 6.0]})
    preds = model.predict(new_df)
    assert len(preds) == 3
    assert np.allclose(preds.values, [6.0, 11.0, 16.0], atol=0.5)


def test_pca_standardize_toggle():
    """Verify PCAModel(standardize=False) preserves variance differences (demean only)."""
    # X has variance 1 on col1 and variance 100 on col2
    col1 = np.array([-1.0, 0.0, 1.0])
    col2 = np.array([-10.0, 0.0, 10.0])
    df = pd.DataFrame({"c1": col1, "c2": col2})

    pca_raw = PCAModel(features=["c1", "c2"], n_components=2, standardize=False)
    pca_raw.fit(df)
    # The first component of unscaled PCA should align almost entirely with c2
    assert abs(pca_raw.model.components_[0, 1]) > 0.99

    pca_std = PCAModel(features=["c1", "c2"], n_components=2, standardize=True)
    pca_std.fit(df)
    # Scaled PCA treats c1 and c2 equally (loadings ~ 1/sqrt(2) = 0.7071)
    assert np.allclose(abs(pca_std.model.components_[0, 0]), abs(pca_std.model.components_[0, 1]), atol=1e-3)


def test_local_projections_cumulative_and_hac():
    """Verify LocalProjectionsModel supports cumulative differences, lags, and HAC standard errors."""
    np.random.seed(123)
    T = 150
    shock = np.random.normal(0, 1, T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.7 * y[t - 1] + 1.2 * shock[t] + np.random.normal(0, 0.2)
    df = pd.DataFrame({"y": y, "shock": shock})

    # Fit cumulative LP with HAC standard errors and 1 lag
    lp = LocalProjectionsModel(
        target="y",
        shock_var="shock",
        horizon=4,
        cumulative=True,
        lags=1,
        cov_type="HAC",
    )
    lp.fit(df)
    irf = lp.compute_irf()
    assert len(irf) == 5
    assert "horizon" in irf.columns
    assert "effect" in irf.columns
    assert "stderr" in irf.columns
    assert all(irf["stderr"] > 0)


def test_var_forecaster_direct():
    """Verify VARForecaster.forecast_direct generates direct multi-step forecasts."""
    np.random.seed(99)
    T = 100
    y1 = np.cumsum(np.random.randn(T))
    y2 = np.cumsum(np.random.randn(T))
    df = pd.DataFrame({"y1": y1, "y2": y2})

    fc = VARForecaster.forecast_direct(df, target_variables=["y1", "y2"], lags=2, steps=4)
    assert fc.shape == (4, 2)
    assert list(fc.columns) == ["y1", "y2"]
    assert list(fc.index) == [1, 2, 3, 4]
    assert not fc.isna().any().any()
