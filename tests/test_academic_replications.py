"""Unit and regression tests for academic paper replications."""
import pytest
from examples.academic.acosta_brennan_jacobson_2024 import AcostaBrennanJacobson2024Replication
from examples.academic.vanderwel_2026 import Vanderwel2026Replication
from examples.academic.nakamura_steinsson_2018b_jorda import NakamuraSteinsson2018bJordaReplication
from examples.academic.nakamura_steinsson import NakamuraSteinssonExample
from examples.academic.bauer_swanson_2023 import BauerSwanson2023Replication
from examples.academic.king_plosser_stock_watson_1991 import KingPlosserStockWatson1991Replication
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


def test_king_plosser_stock_watson_1991_svec():
    """Verify KPSW (1991) three-variable SVEC estimation succeeds and is just-identified.

    Spec: two transitory shocks have zero long-run effects (LR columns 1-2 of
    Xi B, King et al. 1991 eq. (7)) plus one contemporaneous zero B[1,2] to
    separate the transitory shocks (Brueggemann 2006). Published tables are
    not directly comparable (different sample than KPSW's 1949-1988 Citibase
    data), so success plus the identification structure is asserted.
    """
    rep = KingPlosserStockWatson1991Replication()
    out = rep.run()
    B = out["SR"]
    assert np.isfinite(B).all()
    assert np.isclose(B[1, 2], 0.0, atol=1e-6)
    assert np.isclose(np.asarray(rep.model.svec_fit.LR_est)[0, 1], 0.0, atol=1e-6)
    fit = rep.model.svec_fit
    assert fit.max_restriction_violation <= fit.restr_tol
    # No over-identifying restrictions: LR test statistic must be ~ 0.
    assert abs(out["metrics"]["lr_stat"]) < 1e-6


def test_nakamura_steinsson_transform_check():
    """Verify FeatureEngineer changeraw on SVENY01 matches master.dta DNY1 (n = 4,786)."""
    res = NakamuraSteinssonExample().run()
    assert res["passed"], f"max error {res['max_error']}"
    assert res["n"] == 4786
    assert res["max_error"] < 1e-6


def test_nakamura_steinsson_2018b_jorda_replication():
    """Verify Nakamura & Steinsson (2018b) Appendix Jordà multiplier and IRFs match published values."""
    rep = NakamuraSteinsson2018bJordaReplication()
    res = rep.run()
    assert res["passed"], f"Nakamura & Steinsson multiplier error {res['multiplier_error']} or IRF diff {res['max_irf_diff']}"
    assert abs(res["multiplier_h36"] - 1.95) < 0.01


def test_bauer_swanson_2023_news_predictability():
    """Verify Bauer & Swanson (2023) raw mps is predictable from pre-FOMC news.

    Published coefficient tables are not available in the source replication
    repository (shipped folder is confidential/empty), so two things are
    asserted instead: (1) RobustOLSModel HC3 coefficients and R^2 match a
    direct statsmodels HC3 computation on the identical sample (tolerance
    1e-4), and (2) the paper's economic finding: raw mps has materially
    positive R^2 with at least one news coefficient significant at 5%, while
    the mps_orth placebo regression has near-zero R^2.
    """
    rep = BauerSwanson2023Replication()
    res = rep.run()
    assert res["passed"], f"Bauer & Swanson replication failed: coef error {res['max_coef_error']}, r2 {res['r_squared']}, placebo r2 {res['placebo_r_squared']}, significant news {res['significant_news']}"
    assert res["n_observations"] > 0
    assert res["r_squared"] >= rep.MIN_NEWS_R2
    assert res["placebo_r_squared"] < rep.MAX_PLACEBO_R2
    assert len(res["significant_news"]) >= 1


def test_acosta_brennan_jacobson_2024_gss_rotation():
    """Verify the GSS rotation properties of the shipped surprise factors.

    ns (the policy-rate surprise) is mechanically built from gss_target and
    gss_path, so no OLS of ns on the factors is fitted; instead the GSS
    rotation is checked: target and path are orthogonal (correlation near
    zero) and the orthogonal factor variance shares of ns sum to one.
    """
    out = AcostaBrennanJacobson2024Replication().run()
    rotation = out["rotation_metrics"]
    assert abs(rotation["target_path_corr"]) < 0.05
    assert np.isfinite(rotation["variance_share_target"])
    assert np.isfinite(rotation["variance_share_path"])
    assert rotation["variance_share_target"] + rotation["variance_share_path"] > 0.99
    assert rotation["num_observations"] > 0
    assert "var_metrics" in out


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
