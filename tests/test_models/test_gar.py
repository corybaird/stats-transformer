import numpy as np
import pandas as pd
import pytest
from stats_transformer.models.timeseries.reduced_form.gar import GaRModel

QUANTILE_COLUMNS = ["q05", "q25", "q50", "q75", "q95"]


def create_data(n=300, seed=11):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    eps = (0.1 + 0.4 * np.abs(x)) * rng.standard_normal(size=n)
    y = 0.7 * x + eps
    return pd.DataFrame({"y": y, "x": x, "date": pd.date_range("2000-01-01", periods=n, freq="D")})


@pytest.fixture(scope="module")
def fitted():
    model = GaRModel(target="y", independent_variables=["x"], quantiles=(0.05, 0.25, 0.5, 0.75, 0.95), horizon=1, date_column="date")
    model.fit(create_data())
    return model


def test_predict_quantiles_columns_and_order(fitted):
    qf = fitted.predict_quantiles()
    assert list(qf.columns) == QUANTILE_COLUMNS
    assert (np.diff(qf.to_numpy(), axis=1) >= 0).all()


def test_rearrange_sorts_crossed_quantiles():
    crossed = pd.DataFrame({"q50": [0.5, 0.7], "q05": [0.6, 0.1], "q95": [0.2, 0.9]})
    out = GaRModel(target="y", independent_variables=["x"])._rearrange_quantiles(crossed)
    expected = pd.DataFrame({"q50": [0.2, 0.1], "q05": [0.5, 0.7], "q95": [0.6, 0.9]})
    pd.testing.assert_frame_equal(out, expected)


def test_predict_quantiles_new_data(fitted):
    new = pd.DataFrame({"x": np.linspace(-2, 2, 20)})
    qf = fitted.predict_quantiles(new)
    assert qf.shape == (20, 5)
    assert list(qf.columns) == QUANTILE_COLUMNS
    assert (np.diff(qf.to_numpy(), axis=1) >= 0).all()
    wider = (qf["q95"] - qf["q05"]).to_numpy()
    # Heteroskedastic DGP: predictive spread varies with x
    assert wider.max() > wider.min()


def test_fit_aligns_target_at_horizon():
    df = create_data(n=150)
    model = GaRModel(target="y", independent_variables=["x"], horizon=2)
    metrics = model.fit(df)
    assert metrics["nobs"] == len(df) - 2


def test_horizon_zero_prealigned(fitted):
    df = create_data(n=150)
    model = GaRModel(target="y", independent_variables=["x"], horizon=0)
    model.fit(df)
    assert model.get_model_metrics()["nobs"] == len(df)


def test_skewed_t_params_valid(fitted):
    p = fitted.skew_t_params
    assert set(p.columns) == {"mu", "sigma", "alpha", "nu"}
    assert (p["sigma"] > 0).all()
    assert (p["nu"] > 2).all()


def test_compute_gar_returns_finite_series(fitted):
    gar = fitted.compute_gar(alpha=0.05)
    assert isinstance(gar, pd.Series)
    assert (gar.index == fitted.skew_t_params.index).all()
    assert np.isfinite(gar.to_numpy()).all()


def test_expected_shortfall_below_gar(fitted):
    gar = fitted.compute_gar(alpha=0.05)
    es = fitted.compute_expected_shortfall(alpha=0.05)
    assert (es < gar).all()


def test_gar_median_matches_quantile_prediction(fitted):
    q50 = fitted.predict_quantiles()["q50"].to_numpy()
    gar = fitted.compute_gar(alpha=0.5).to_numpy()
    assert np.allclose(gar, q50, atol=0.3)


def test_metadata_and_metrics(fitted):
    metrics = fitted.get_model_metrics()
    assert metrics["horizon"] == 1
    assert metrics["skewed_t_df"] > 2
    metadata = fitted.get_model_metadata(metrics)
    assert metadata["coefficients"]["const"]["value"] is not None
    assert "x" in metadata["coefficients"]


def test_summary_mentions_quantiles(fitted):
    summary = fitted.get_summary()
    assert "0.05" in summary and "0.95" in summary


def test_not_fitted_raises():
    model = GaRModel(target="y", independent_variables=["x"])
    with pytest.raises(ValueError, match="fitted"):
        model.compute_gar()
    with pytest.raises(ValueError, match="fitted"):
        model.predict_quantiles()
    with pytest.raises(ValueError, match="fitted"):
        model.get_summary()


def test_missing_columns_rejected():
    model = GaRModel(target="y", independent_variables=["x", "z"])
    with pytest.raises(ValueError, match="missing"):
        model.fit(create_data(n=60))
