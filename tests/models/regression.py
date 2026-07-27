import pytest
import pandas as pd
import numpy as np
import os
from stats_transformer.models.regression.robust_ols import RobustOLSModel
from stats_transformer.models.regression.panel import PanelRegressionModel
from stats_transformer.models.regression.iv import IV2SLSModel

def test_robust_ols_model():
    df = pd.DataFrame({
        "target": [1, 2, 3, 4, 5],
        "feature1": [2, 4, 6, 8, 10],
        "feature2": [1, 1, 2, 2, 3]
    })
    
    model = RobustOLSModel(
        target="target",
        independent_variables=["feature1", "feature2"],
        cov_type="HC3"
    )
    
    metrics = model.fit(df)
    
    assert "r_squared" in metrics
    assert metrics["num_observations"] == 5
    assert model.model is not None
    assert hasattr(model.model, 'cov_type')
    assert model.model.cov_type == "HC3"

def test_panel_regression_model():
    # linearmodels PanelOLS expects a MultiIndex or entity/time columns
    df = pd.DataFrame({
        "entity": ["A", "A", "A", "B", "B", "B"],
        "date": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01", "2020-01-01", "2021-01-01", "2022-01-01"]),
        "target": [1.1, 2.1, 3.1, 1.2, 2.2, 3.2],
        "feature": [0.5, 1.5, 2.5, 0.6, 1.6, 2.6]
    })
    
    model = PanelRegressionModel(
        target="target",
        independent_variables=["feature"],
        entity_column="entity",
        time_column="date",
        entity_effects=True
    )
    
    metrics = model.fit(df)
    
    assert "rsquared" in metrics
    assert metrics["nobs"] == 6
    assert model.model is not None

def test_iv_2sls_model():
    # Simple IV setup
    # y = b0 + b1*x + e
    # x = a0 + a1*z + u
    np.random.seed(42)
    n = 100
    z = np.random.normal(size=n)
    u = np.random.normal(size=n)
    x = 0.5 * z + u + np.random.normal(size=n) * 0.1
    y = 2.0 * x + u + np.random.normal(size=n) * 0.1
    
    df = pd.DataFrame({
        "y": y,
        "x": x,
        "z": z,
        "w": np.random.normal(size=n) # exogenous control
    })
    
    model = IV2SLSModel(
        target="y",
        independent_variables=["w"],
        endogenous=["x"],
        instruments=["z"]
    )
    
    metrics = model.fit(df)
    
    assert "r_squared" in metrics
    assert metrics["num_observations"] == 100
    assert model.model is not None

def test_regression_diagnostics():
    from stats_transformer.models.regression.diagnostics import RegressionDiagnostics
    
    df = pd.DataFrame({
        "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "x1": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        "x2": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    })
    
    model = RobustOLSModel(target="y", independent_variables=["x1", "x2"])
    model.fit(df)
    
    diag = RegressionDiagnostics(model.model)
    bp = diag.breusch_pagan_test()
    jb = diag.jarque_bera_test()
    dw = diag.durbin_watson_test()
    
    assert "statistic" in bp
    assert "p_value" in bp
    assert "statistic" in jb
    assert "statistic" in dw
