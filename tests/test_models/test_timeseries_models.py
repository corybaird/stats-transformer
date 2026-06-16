import pytest
import pandas as pd
import numpy as np
from stats_transformer.models.timeseries.var import VARModel
from stats_transformer.models.timeseries.vecm import VECMModel
from stats_transformer.models.timeseries import GrangerCausalityTester

def test_var_model():
    # Simple VAR setup
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start="2000-01-01", periods=n, freq="ME")
    
    # Generate some correlated random walks
    e1 = np.random.normal(size=n)
    e2 = np.random.normal(size=n)
    y1 = np.cumsum(e1)
    y2 = 0.5 * y1 + np.cumsum(e2)
    
    df = pd.DataFrame({
        "date": dates,
        "y1": y1,
        "y2": y2
    })
    
    model = VARModel(
        target_variables=["y1", "y2"],
        date_column="date",
        maxlags=2
    )
    
    metrics = model.fit(df)
    
    assert "aic" in metrics
    assert metrics["num_observations"] > 0
    assert model.model is not None

def test_vecm_model():
    # Simple VECM setup
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start="2000-01-01", periods=n, freq="ME")
    
    # Cointegrated system
    # y1_t = y2_t + e1_t
    # y2_t = y2_{t-1} + e2_t
    e2 = np.random.normal(size=n)
    y2 = np.cumsum(e2)
    y1 = y2 + np.random.normal(scale=0.1, size=n)
    
    df = pd.DataFrame({
        "date": dates,
        "y1": y1,
        "y2": y2
    })
    
    model = VECMModel(
        target_variables=["y1", "y2"],
        date_column="date",
        k_ar_diff=1
    )
    
    metrics = model.fit(df)
    
    assert "num_observations" in metrics
    assert metrics["num_observations"] == 100
    assert model.model is not None

def test_granger_causality_tester_single_series():
    np.random.seed(42)
    n = 120
    x = np.random.normal(size=n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.8 * x[t - 1] + 0.2 * y[t - 1] + np.random.normal(scale=0.2)

    df = pd.DataFrame({
        "date": pd.date_range(start="2000-01-01", periods=n, freq="ME"),
        "x": x,
        "y": y,
    })

    tester = GrangerCausalityTester(date_column="date", max_lag=2)
    results = tester.fit(df, caused="y", causing="x")
    metadata = tester.get_model_metadata()

    assert not results.empty
    assert {"caused", "causing", "lag", "test", "statistic", "p_value"}.issubset(results.columns)
    assert metadata["best_lag"] in [1, 2]
    assert metadata["min_p_value"] < 0.05

def test_granger_causality_tester_panel():
    np.random.seed(7)
    rows = []
    for entity in ["USA", "CAN"]:
        n = 80
        x = np.random.normal(size=n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.7 * x[t - 1] + np.random.normal(scale=0.3)
        for date, x_val, y_val in zip(pd.date_range("2001-01-01", periods=n, freq="ME"), x, y):
            rows.append({"country": entity, "date": date, "x": x_val, "y": y_val})

    tester = GrangerCausalityTester(
        entity_column="country",
        date_column="date",
        max_lag=1,
        standardize=True,
    )
    results = tester.fit(pd.DataFrame(rows), caused="y", causing="x")
    metadata = tester.get_model_metadata()

    assert set(results["entity"]) == {"USA", "CAN"}
    assert metadata["num_tests"] == len(results)
    assert metadata["min_p_value"] < 0.05
