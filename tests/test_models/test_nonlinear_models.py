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
    assert metrics["regime_1_share"] + metrics["regime_2_share"] == pytest.approx(1.0)
    assert "TVAR" in model.get_summary()

def test_tvar_user_gamma_empty_regime_raises(sample_tvar_data):
    # gamma far above the threshold series leaves regime 2 empty
    model = TVARModel(target_variables=["y1", "y2"], threshold_variable="threshold", lags=1, delay=1, gamma=1e6)
    with pytest.raises(ValueError, match="gamma=1e\+06"):
        model.fit(sample_tvar_data)

    # gamma near the upper tail leaves regime 2 below the minimum sample size
    th_sorted = np.sort(sample_tvar_data["threshold"].values)
    model = TVARModel(target_variables=["y1", "y2"], threshold_variable="threshold", lags=1, delay=1, gamma=th_sorted[-2])
    with pytest.raises(ValueError, match="regime sizes"):
        model.fit(sample_tvar_data)


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
    assert 0.0 < metrics["regime_1_share"] < 1.0
    assert metrics["regime_1_share"] + metrics["regime_2_share"] == pytest.approx(1.0)
    assert "TVECM" in model.get_summary()


def test_tvecm_design_matrix_alignment_d2():
    """Hand-built example: the ECT regressor is ECT_{t-1} regardless of the
    delay, while the threshold variable is ECT_{t-d} (d = 2 here)."""
    model = TVECMModel(target_variables=["y1", "y2"], k_ar_diff=1, delay=2)
    y_levels = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 2.0], [4.0, 2.0], [5.0, 3.0], [6.0, 3.0]])
    ect_series = np.array([0.0, 0.5, 1.0, -0.2, 0.3, -0.1])

    dY_eff, X_eff, ect_regressor, ect_threshold = model._construct_tvecm_design_matrices(y_levels, ect_series)

    # effective rows predict Δy_t for t = 2..5 (T = 6, max_lag = 2)
    np.testing.assert_allclose(dY_eff, np.column_stack([np.full(4, 1.0), [1.0, 0.0, 1.0, 0.0]]))
    # rows: [1, ECT_{t-1}, Δy_{t-1}]; the ECT column is ECT shifted by 1
    expected_X = np.array([
        [1.0, 0.5, 1.0, 0.0],   # t=2: ECT_1, Δy_1
        [1.0, 1.0, 1.0, 1.0],   # t=3: ECT_2, Δy_2
        [1.0, -0.2, 1.0, 0.0],  # t=4: ECT_3, Δy_3
        [1.0, 0.3, 1.0, 1.0],   # t=5: ECT_4, Δy_4
    ])
    np.testing.assert_allclose(X_eff, expected_X)
    # threshold column is ECT shifted by 2 (ECT_{t-d})
    np.testing.assert_allclose(ect_threshold, ect_series[0:4])

    # d = 1: regressor unchanged (always ECT_{t-1}), threshold becomes ECT_{t-1}
    model_d1 = TVECMModel(target_variables=["y1", "y2"], k_ar_diff=1, delay=1)
    dY_eff1, X_eff1, ect_reg1, ect_th1 = model_d1._construct_tvecm_design_matrices(y_levels, ect_series)
    np.testing.assert_allclose(X_eff1, expected_X)
    np.testing.assert_allclose(ect_th1, ect_series[1:5])

def test_tvecm_d1_matches_statsmodels_vecm_design():
    """d = 1: the TVECM design matches the standard VECM (statsmodels) design,
    which pairs ECT_{t-1} with Δy_{t-1} (Lütkepohl pp. 286, _endog_matrices)."""
    rng = np.random.default_rng(123)
    T = 150
    e1 = rng.standard_normal(T)
    e2 = rng.standard_normal(T)
    y1 = np.cumsum(e1)
    y2 = 0.8 * y1 + np.cumsum(e2 * 0.2)
    df = pd.DataFrame({"y1": y1, "y2": y2})

    model = TVECMModel(target_variables=["y1", "y2"], k_ar_diff=1, coint_rank=1, delay=1, gamma=0.0)
    model.fit(df)

    from statsmodels.tsa.vector_ar.vecm import VECM
    y = df[["y1", "y2"]].to_numpy(dtype=float)
    res = VECM(df[["y1", "y2"]], k_ar_diff=1, coint_rank=1, deterministic="n").fit()
    ect = y @ res.beta[:, 0]
    dY = np.diff(y, axis=0)
    dY_eff = dY[1:]                       # Δy_t for t = 2..T-1
    X_ref = np.column_stack([np.ones(len(dY_eff)), ect[1:-1], dY[:-1]])
    mask1 = ect[1:-1] <= model.gamma
    p1 = np.linalg.lstsq(X_ref[mask1], dY_eff[mask1], rcond=None)[0]
    p2 = np.linalg.lstsq(X_ref[~mask1], dY_eff[~mask1], rcond=None)[0]
    np.testing.assert_allclose(model.regime_1_params, p1, atol=1e-10)
    np.testing.assert_allclose(model.regime_2_params, p2, atol=1e-10)

def test_tvecm_user_gamma_empty_regime_raises():
    rng = np.random.default_rng(123)
    T = 150
    e1 = rng.standard_normal(T)
    e2 = rng.standard_normal(T)
    y1 = np.cumsum(e1)
    y2 = 0.8 * y1 + np.cumsum(e2 * 0.2)
    df = pd.DataFrame({"y1": y1, "y2": y2})

    # gamma far above the ECT series leaves regime 2 empty
    model = TVECMModel(target_variables=["y1", "y2"], k_ar_diff=1, coint_rank=1, delay=1, gamma=1e6)
    with pytest.raises(ValueError, match="gamma=1e\+06"):
        model.fit(df)

    # gamma far below the ECT series leaves regime 1 empty
    model = TVECMModel(target_variables=["y1", "y2"], k_ar_diff=1, coint_rank=1, delay=1, gamma=-1e6)
    with pytest.raises(ValueError, match="at least"):
        model.fit(df)

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
