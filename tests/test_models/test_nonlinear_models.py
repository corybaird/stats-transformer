import numpy as np
import pandas as pd
import pytest
from stats_transformer.models.timeseries.nonlinear.tvar import TVARModel
from stats_transformer.models.timeseries.nonlinear.tvecm import TVECMModel
from stats_transformer.models.timeseries.nonlinear.stvar import STVARModel
from stats_transformer.models.timeseries.nonlinear.girf import GIRFEngine


@pytest.fixture
def sample_tvar_data():
    rng = np.random.default_rng(42)
    T = 150
    y1 = np.zeros(T)
    y2 = np.zeros(T)
    th = rng.standard_normal(T)

    for t in range(1, T):
        if th[t - 1] <= 0:
            y1[t] = 0.5 * y1[t - 1] + 0.1 * y2[t - 1] + rng.standard_normal()
            y2[t] = 0.2 * y1[t - 1] + 0.4 * y2[t - 1] + rng.standard_normal()
        else:
            y1[t] = -0.3 * y1[t - 1] + 0.6 * y2[t - 1] + rng.standard_normal() * 1.5
            y2[t] = 0.1 * y1[t - 1] - 0.5 * y2[t - 1] + rng.standard_normal() * 1.5

    df = pd.DataFrame({"y1": y1, "y2": y2, "threshold": th})
    return df


def test_tvar_model_estimation(sample_tvar_data):
    model = TVARModel(target_variables=["y1", "y2"], threshold_variable="threshold", lags=1, delay=1)
    metrics = model.fit(sample_tvar_data)

    assert "gamma" in metrics
    assert "ssr" in metrics
    assert model.regime_1_params.shape == (3, 2)
    assert model.regime_2_params.shape == (3, 2)
    assert 0.0 < metrics["regime_1_share"] < 1.0
    assert "TVAR" in model.get_summary()


def test_tvecm_model_estimation():
    rng = np.random.default_rng(123)
    T = 150
    e1 = rng.standard_normal(T)
    e2 = rng.standard_normal(T)
    y1 = np.cumsum(e1)
    y2 = 0.8 * y1 + np.cumsum(e2 * 0.2)
    df = pd.DataFrame({"y1": y1, "y2": y2})

    model = TVECMModel(target_variables=["y1", "y2"], k_ar_diff=1, coint_rank=1, delay=1)
    metrics = model.fit(df)

    assert "gamma" in metrics
    assert "coint_rank" in metrics
    assert model.beta is not None
    assert model.regime_1_params is not None
    assert model.regime_2_params is not None
    assert "TVECM" in model.get_summary()


def test_stvar_model_estimation(sample_tvar_data):
    model = STVARModel(target_variables=["y1", "y2"], transition_variable="threshold", lags=1, delay=1)
    metrics = model.fit(sample_tvar_data)

    assert "gamma" in metrics
    assert "c" in metrics
    assert "mean_weight" in metrics
    assert model.transition_weights is not None
    assert len(model.transition_weights) == metrics["num_observations"]
    assert "STVAR" in model.get_summary()


def test_girf_engine_execution(sample_tvar_data):
    model = TVARModel(target_variables=["y1", "y2"], threshold_variable="threshold", lags=1, delay=1)
    model.fit(sample_tvar_data)

    girf_engine = GIRFEngine(model=model, horizon=10, shock_size=1.0, num_histories=10, num_bootstrap=20, seed=1)
    girf_df = girf_engine.compute_girf(shock_variable="y1")

    assert isinstance(girf_df, pd.DataFrame)
    assert set(girf_df.columns) == {"horizon", "variable", "shock", "girf", "ci_lower", "ci_upper"}
    assert len(girf_df) == 10 * 2  # 10 horizons * 2 variables
