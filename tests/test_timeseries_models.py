import pytest
import pandas as pd
import numpy as np
from stats_transformer.models.timeseries.var import VARModel
from stats_transformer.models.timeseries.vecm import VECMModel

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
