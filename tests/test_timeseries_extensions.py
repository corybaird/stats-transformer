import numpy as np
import pandas as pd
from stats_transformer.models.timeseries.identification.blanchard_quah import BlanchardQuahModel
from stats_transformer.models.timeseries.identification.proxy_svar import ProxySVARModel
from stats_transformer.models.timeseries.identification.sign_zero import SignZeroSVARModel
from stats_transformer.models.timeseries.reduced_form.local_projections import LocalProjectionsModel
from stats_transformer.models.timeseries.reduced_form.local_projections_iv import LocalProjectionsIVModel
from stats_transformer.models.timeseries.reduced_form.dynamic_factor import DynamicFactorModel
from stats_transformer.models.timeseries.decompositions import TimeSeriesDecompositions
from examples.academic.var.stock_watson_2001 import StockWatson2001Replication
from examples.academic.var.blanchard_quah_1989 import BlanchardQuah1989Replication
from examples.academic.var.gertler_karadi_2015 import GertlerKaradi2015Replication
from examples.academic.var.jorda_taylor_2025 import JordaTaylor2025Replication
from examples.academic.miranda_agrippino_rey_2020 import MirandaAgrippinoRey2020Replication

def create_synthetic_data():
    np.random.seed(42)
    n = 100
    y1 = np.cumsum(np.random.normal(size=n))
    y2 = np.cumsum(np.random.normal(size=n)) + 0.5 * y1
    z = np.random.normal(size=n) + 0.3 * y1
    df = pd.DataFrame({"y1": y1, "y2": y2, "z": z, "date": pd.date_range("2000-01-01", periods=n, freq="QE")})
    return df

def create_factor_panel_data(seed=42, n=300, n_series=8, phi=0.7, factor_var=0.25, noise_var=0.3):
    np.random.seed(seed)
    factor = np.zeros(n)
    for t in range(1, n):
        factor[t] = phi * factor[t - 1] + np.random.normal(scale=np.sqrt(factor_var))
    loadings = np.random.uniform(0.5, 1.5, size=n_series) * np.random.choice([-1, 1], size=n_series)
    X = factor[:, None] @ loadings[None, :] + np.random.normal(scale=np.sqrt(noise_var), size=(n, n_series))
    columns = [f"x{i}" for i in range(n_series)]
    df = pd.DataFrame(X, columns=columns)
    df["date"] = pd.date_range("2000-01-01", periods=n, freq="D")
    return df, factor, columns

def test_blanchard_quah_model():
    df = create_synthetic_data()
    model = BlanchardQuahModel(target_variables=["y1", "y2"], date_column="date", maxlags=2)
    metrics = model.fit(df)
    assert metrics["nobs"] > 0
    assert model.B_0 is not None
    assert model.B_0.shape == (2, 2)

def test_proxy_svar_model():
    df = create_synthetic_data()
    model = ProxySVARModel(target_variables=["y1", "y2"], instrument_variable="z", date_column="date", maxlags=2)
    metrics = model.fit(df)
    assert metrics["nobs"] > 0
    assert model.impact_column is not None
    assert len(model.impact_column) == 2

def test_sign_restrictions_model():
    df = create_synthetic_data()
    model = SignZeroSVARModel(target_variables=["y1", "y2"], date_column="date", maxlags=2, max_draws=100)
    metrics = model.fit(df)
    assert "nobs" in metrics

def test_local_projections_iv_model():
    df = create_synthetic_data()
    model = LocalProjectionsIVModel(target_variable="y1", shock_variable="y2", instrument_variable="z", horizons=3, date_column="date")
    metrics = model.fit(df)
    assert metrics["horizons"] == 3
    assert len(model.irf_coefficients) == 4

def test_local_projections_model():
    df = create_synthetic_data()
    model = LocalProjectionsModel(target="y1", shock_var="y2", controls=["z"], horizon=3, date_column="date")
    metrics = model.fit(df)
    assert metrics["horizon"] == 3
    assert metrics["shock_var"] == "y2"
    irf = model.compute_irf()
    assert len(irf) == 4
    assert list(irf.columns) == ["horizon", "effect", "stderr", "lower_ci", "upper_ci", "pvalue"]
    assert irf["horizon"].tolist() == [0, 1, 2, 3]

def test_local_projections_model_keeps_date_column():
    df = create_synthetic_data()
    model = LocalProjectionsModel(target="y1", shock_var="y2", horizon=2, date_column="date")
    model.fit(df)
    assert "date" in model.df_clean.columns
    assert model.df_clean["date"].is_monotonic_increasing

def test_local_projections_model_without_date_column_indexes_date():
    df = create_synthetic_data()
    model = LocalProjectionsModel(target="y1", shock_var="y2", horizon=2)
    model.fit(df)
    assert "date" not in model.df_clean.columns
    assert model.df_clean.index.name == "date"

def test_local_projections_variance_decomposition():
    df = create_synthetic_data()
    model = LocalProjectionsModel(target="y1", shock_var="y2", controls=["z"], horizon=3)
    model.fit(df)
    vd = model.compute_vd()
    assert len(vd) == 4
    assert (vd["variance_explained"] >= 0).all()

def test_local_projections_run_returns_metadata():
    df = create_synthetic_data()
    model = LocalProjectionsModel(target="y1", shock_var="y2", horizon=2, date_column="date")
    metadata = model.run(df)
    assert "metrics" in metadata
    assert metadata["metrics"]["horizon"] == 2

def test_dynamic_factor_model_recovers_known_factor():
    df, true_factor, columns = create_factor_panel_data()
    model = DynamicFactorModel(target_variables=columns, date_column="date", n_factors=1, factor_lags=1, max_iter=100, tol=1e-9)
    metrics = model.fit(df)
    assert metrics["converged"] or metrics["n_iter"] == model.max_iter
    correlation = np.corrcoef(model.factors_[:, 0], true_factor)[0, 1]
    assert abs(correlation) > 0.9

def test_dynamic_factor_model_loglikelihood_increases_monotonically():
    df, _, columns = create_factor_panel_data(seed=7, n=200, n_series=6)
    model = DynamicFactorModel(target_variables=columns, date_column="date", n_factors=1, factor_lags=1, max_iter=50, tol=1e-9)
    model.fit(df)
    ll = np.array(model.loglikelihood_history)
    assert len(ll) > 1
    assert np.all(np.diff(ll) >= -1e-6)

def test_dynamic_factor_model_two_factors():
    np.random.seed(3)
    n, n_series, k = 250, 10, 2
    transition = np.array([[0.6, 0.1], [0.0, 0.5]])
    factors = np.zeros((n, k))
    for t in range(1, n):
        factors[t] = transition @ factors[t - 1] + np.random.multivariate_normal(mean=[0, 0], cov=np.eye(k) * 0.25)
    loadings = np.random.uniform(0.4, 1.2, size=(n_series, k)) * np.random.choice([-1, 1], size=(n_series, k))
    X = factors @ loadings.T + np.random.normal(scale=np.sqrt(0.3), size=(n, n_series))
    columns = [f"x{i}" for i in range(n_series)]
    df = pd.DataFrame(X, columns=columns)
    df["date"] = pd.date_range("2000-01-01", periods=n, freq="D")

    model = DynamicFactorModel(target_variables=columns, date_column="date", n_factors=2, factor_lags=1, max_iter=100, tol=1e-9)
    model.fit(df)
    ll = np.array(model.loglikelihood_history)
    assert np.all(np.diff(ll) >= -1e-6)
    design = np.column_stack([model.factors_, np.ones(n)])
    coef, *_ = np.linalg.lstsq(design, factors, rcond=None)
    residual = factors - design @ coef
    r2 = 1 - np.sum(residual ** 2) / np.sum((factors - factors.mean(axis=0)) ** 2)
    assert r2 > 0.8

def test_dynamic_factor_model_compute_factors_and_run():
    df, _, columns = create_factor_panel_data(seed=11, n=150, n_series=6)
    model = DynamicFactorModel(target_variables=columns, date_column="date", n_factors=1, max_iter=50)
    metadata = model.run(df)
    assert "metrics" in metadata
    factors_df = model.compute_factors()
    assert list(factors_df.columns) == ["factor_1", "date"]
    assert len(factors_df) == len(model.df_clean)

def test_miranda_agrippino_rey_2020_replication():
    result = MirandaAgrippinoRey2020Replication().run()
    assert "metrics" in result
    assert result["metrics"]["converged"] or result["metrics"]["n_iter"] > 0
    assert abs(result["true_factor_correlation"]) > 0.9
    assert not result["reference_series"].empty

def test_decompositions():
    df = create_synthetic_data()
    model = BlanchardQuahModel(target_variables=["y1", "y2"], date_column="date", maxlags=2)
    model.fit(df)
    decomp = TimeSeriesDecompositions(model.var_result, B_0=model.B_0)
    res = decomp.run(steps=5)
    assert res["fevd"].shape == (5, 2, 2)

def test_replications():
    sw = StockWatson2001Replication()
    res_sw = sw.run()
    assert "metrics" in res_sw
    bq = BlanchardQuah1989Replication()
    res_bq = bq.run()
    assert "metrics" in res_bq
    gk = GertlerKaradi2015Replication()
    res_gk = gk.run()
    assert "metrics" in res_gk
    jt = JordaTaylor2025Replication()
    res_jt = jt.run()
    assert "metrics" in res_jt
