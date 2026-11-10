import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from stats_transformer.models.timeseries.reduced_form.favar import FAVARModel

def make_favar_data(n=400, seed=42):
    rng = np.random.default_rng(seed)
    f1 = np.zeros(n)
    f2 = np.zeros(n)
    for t in range(1, n):
        f1[t] = 0.6 * f1[t - 1] + rng.normal(scale=0.8)
        f2[t] = 0.3 * f2[t - 1] + rng.normal(scale=0.8)
    policy = 0.5 * f1 - 0.4 * f2 + rng.normal(scale=0.2, size=n)
    panel = {}
    for i in range(8):
        panel[f"x{i}"] = 0.9 * f1 + rng.normal(scale=0.3, size=n)
    for i in range(8, 16):
        panel[f"x{i}"] = 0.9 * f2 + rng.normal(scale=0.3, size=n)
    for i in range(4):
        panel[f"s{i}"] = 0.8 * f2 + 0.1 * f1 + rng.normal(scale=0.3, size=n)
    df = pd.DataFrame(panel)
    df["policy"] = policy
    df["date"] = pd.date_range("2000-01-01", periods=n, freq="ME")
    return df

def make_model():
    return FAVARModel(
        observed_variables=["policy"],
        panel_variables=[f"x{i}" for i in range(16)] + [f"s{i}" for i in range(4)],
        slow_variables=[f"s{i}" for i in range(4)],
        n_factors=2,
        maxlags=2,
        date_column="date",
    )

def test_fit_metrics():
    metrics = make_model().fit(make_favar_data())
    assert "aic" in metrics and "bic" in metrics
    assert metrics["num_observations"] > 0
    assert metrics["n_factors"] == 2
    assert metrics["n_panel_variables"] == 20
    assert metrics["n_slow_variables"] == 4

def test_irf_shape_and_index():
    model = make_model()
    model.fit(make_favar_data())
    irf = model.irf(steps=6)
    assert irf.shape == (7, 20 * 3)
    assert irf.index.names == ["horizon"]
    assert list(irf.index) == list(range(7))
    assert irf.columns.names == ["response", "shock"]
    shocks = list(irf.columns.get_level_values("shock").unique())
    assert shocks == ["factor_1", "factor_2", "policy"]
    responses = list(irf.columns.get_level_values("response").unique())
    assert responses == [f"x{i}" for i in range(16)] + [f"s{i}" for i in range(4)]

def test_irf_matches_loading_map():
    model = make_model()
    model.fit(make_favar_data())
    steps = 5
    irf = model.irf(steps=steps)
    orth = model.model.irf(steps).orth_irfs
    panel_irfs = np.einsum("ik,hkj->hij", model.loadings, orth)
    for h in [0, steps]:
        expected = panel_irfs[h] * model.scaler.scale_[:, None]
        got = irf.loc[h].to_numpy().reshape(20, 3)
        np.testing.assert_allclose(got, expected, rtol=1e-10)

def test_fevd_rows_sum_to_one():
    model = make_model()
    model.fit(make_favar_data())
    fevd = model.fevd(steps=12)
    assert fevd.shape == (20, 3)
    assert list(fevd.index) == [f"x{i}" for i in range(16)] + [f"s{i}" for i in range(4)]
    assert list(fevd.columns) == ["factor_1", "factor_2", "policy"]
    np.testing.assert_allclose(fevd.sum(axis=1).to_numpy(), np.ones(20), atol=1e-8)
    assert (fevd.to_numpy() >= 0).all() and (fevd.to_numpy() <= 1).all()

def test_slow_variables_purge_factors_from_observed():
    model = make_model()
    model.fit(make_favar_data())
    F_hat = model.var_spec.endog[:, :2]
    Y = model.var_spec.endog[:, 2]
    slow = model.df_clean[[f"s{i}" for i in range(4)]].to_numpy()
    slow_pc = PCA(n_components=1, svd_solver="full").fit_transform(slow)
    for i in range(2):
        design = np.column_stack([np.ones(len(Y)), slow_pc])
        y_res = Y - design @ np.linalg.lstsq(design, Y, rcond=None)[0]
        f_res = F_hat[:, i] - design @ np.linalg.lstsq(design, F_hat[:, i], rcond=None)[0]
        assert abs(np.corrcoef(y_res, f_res)[0, 1]) < 0.15

def test_without_slow_variables_runs():
    df = make_favar_data()
    model = FAVARModel(observed_variables=["policy"], panel_variables=[f"x{i}" for i in range(16)] + [f"s{i}" for i in range(4)], n_factors=2, maxlags=2)
    metrics = model.fit(df)
    assert metrics["n_slow_variables"] == 0
    irf = model.irf(steps=4)
    assert np.isfinite(irf.to_numpy()).all()

def test_missing_column_raises():
    df = make_favar_data().drop(columns=["x0"])
    with pytest.raises(ValueError, match="Missing columns"):
        make_model().fit(df)

def test_slow_variables_not_in_panel_raises():
    df = make_favar_data()
    model = FAVARModel(observed_variables=["policy"], panel_variables=["x0", "x1", "x2"], slow_variables=["s3"], n_factors=2, maxlags=2)
    with pytest.raises(ValueError, match="Slow variables not in panel"):
        model.fit(df)

def test_too_many_factors_raises():
    df = make_favar_data()
    model = FAVARModel(observed_variables=["policy"], panel_variables=["x0", "x1", "x2"], n_factors=3, maxlags=2)
    with pytest.raises(ValueError, match="n_factors"):
        model.fit(df)

def test_unfitted_raises():
    model = make_model()
    with pytest.raises(ValueError, match="Model not trained"):
        model.irf()
    with pytest.raises(ValueError, match="Model not trained"):
        model.get_summary()
    with pytest.raises(ValueError, match="Model not trained"):
        model.get_model_metrics()

def test_summary_and_metadata():
    model = make_model()
    model.fit(make_favar_data())
    summary = model.get_summary()
    assert "FAVAR Model" in summary
    assert "policy" in summary
    metadata = model.get_model_metadata()
    assert metadata["summary"]["model_type"].startswith("VAR")
    assert metadata["summary"]["variables"] == [f"factor_{i}" for i in range(1, 3)] + ["policy"]
    assert "aic" in metadata["metrics"]
    assert set(metadata["coefficients"].keys()) == set(model.state_names)

def test_save_load_roundtrip():
    import tempfile, os
    from stats_transformer.models.base import ModelBase
    model = make_model()
    model.fit(make_favar_data())
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "favar.joblib")
        model.save(path)
        loaded = ModelBase.load(path)
        pd.testing.assert_frame_equal(model.irf(steps=5), loaded.irf(steps=5))
