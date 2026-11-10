import numpy as np
import pandas as pd
import pytest
from stats_transformer.models.timeseries.reduced_form.lp_extensions import (
    QuantileLPModel,
    SmoothLPModel,
    StateDependentLPModel,
)

def create_regime_data(n=300):
    np.random.seed(42)
    x = np.random.normal(size=n)
    z = np.random.normal(size=n)
    c = np.random.normal(size=n)
    y = np.where(z > 0, 1.2, 0.3) * x + 0.5 * c + 0.1 * np.random.normal(size=n)
    return pd.DataFrame({"y": y, "x": x, "z": z, "c": c, "date": pd.date_range("2000-01-01", periods=n, freq="D")})

def create_simple_data(n=300):
    np.random.seed(7)
    x = np.random.normal(size=n)
    c = np.random.normal(size=n)
    y = 1.0 * x + 0.5 * c + np.random.normal(size=n)
    return pd.DataFrame({"y": y, "x": x, "c": c, "date": pd.date_range("2000-01-01", periods=n, freq="D")})

def test_state_dependent_threshold_recovers_regime_responses():
    df = create_regime_data()
    model = StateDependentLPModel(target="y", shock_var="x", horizon=2, controls=["c"], transition_variable="z", transition="threshold", threshold=0.0, cov_type="HC3", date_column="date")
    metrics = model.fit(df)
    assert metrics["irf_r1_h0"] > metrics["irf_r2_h0"]
    assert abs(model.irf_regime1_df["effect"].iloc[0] - 1.2) < 0.1
    assert abs(model.irf_regime2_df["effect"].iloc[0] - 0.3) < 0.1
    assert len(model.irf_regime1_df) == 3 and len(model.irf_regime2_df) == 3
    assert (model.irf_regime1_df["stderr"] >= 0).all()
    assert list(model.irf_regime1_df.columns) == ["horizon", "effect", "stderr", "lower_ci", "upper_ci", "pvalue"]

def test_state_dependent_logistic_threshold_and_hac():
    df = create_regime_data()
    model = StateDependentLPModel(target="y", shock_var="x", horizon=2, controls=["c"], transition_variable="z", transition="logistic", gamma=5.0, center=0.0, cov_type="HAC", date_column="date")
    metrics = model.fit(df)
    assert np.isfinite(model.irf_regime1_df["effect"]).all()
    assert np.isfinite(model.irf_regime2_df["effect"]).all()
    assert model.irf_regime1_df["effect"].iloc[0] > model.irf_regime2_df["effect"].iloc[0]
    assert abs(model.irf_regime1_df["effect"].iloc[0] - 1.2) < 0.2
    assert model.irf_regime2_df["effect"].iloc[0] > 0.1
    assert (model.irf_regime1_df["stderr"] > 0).all()
    irf = model.compute_irf()
    assert set(irf["regime"]) == {1, 2}

def test_state_dependent_empty_regime_raises():
    df = create_regime_data()
    model = StateDependentLPModel(target="y", shock_var="x", horizon=2, transition_variable="z", transition="threshold", threshold=100.0, date_column="date")
    with pytest.raises(ValueError, match="empty regime"):
        model.fit(df)

def test_smooth_lp_with_explicit_penalty():
    df = create_simple_data()
    model = SmoothLPModel(target="y", shock_var="x", horizon=4, controls=["c"], penalty_lambda=1.0, cov_type="HC3", date_column="date")
    metrics = model.fit(df)
    irf = model.irf_df
    assert list(irf.columns) == ["horizon", "coef", "std_err", "ci_lower", "ci_upper"]
    assert (irf["horizon"] == range(5)).all()
    assert np.isfinite(irf["coef"]).all()
    assert (irf["std_err"] > 0).all()
    assert (irf["ci_lower"] < irf["ci_upper"]).all()
    assert abs(irf["coef"].iloc[0] - 1.0) < 0.15
    assert np.isfinite(metrics["irf_h0"]) and np.isfinite(metrics["irf_h_max"])

def test_smooth_lp_gcv_selects_penalty():
    df = create_simple_data()
    model = SmoothLPModel(target="y", shock_var="x", horizon=4, controls=["c"], penalty_lambda=None, date_column="date")
    model.fit(df)
    assert model.selected_lambda is not None and model.selected_lambda > 0
    assert len(model.irf_df) == 5
    assert np.isfinite(model.irf_paths.values).all()

def test_quantile_lp_estimates_median_close_to_ols():
    df = create_simple_data()
    model = QuantileLPModel(target="y", shock_var="x", horizon=2, controls=["c"], quantiles=(0.1, 0.5, 0.9), date_column="date")
    metrics = model.fit(df)
    assert set(model.irf_by_quantile.keys()) == {0.1, 0.5, 0.9}
    assert abs(model.irf_by_quantile[0.5]["coef"].iloc[0] - 1.0) < 0.15
    irf = model.irf_df
    assert list(irf.columns) == ["horizon", "quantile", "coef", "std_err", "ci_lower", "ci_upper"]
    assert len(irf) == 3 * 3
    assert (irf["ci_lower"] < irf["ci_upper"]).all()
    assert np.isfinite(irf["coef"]).all()
