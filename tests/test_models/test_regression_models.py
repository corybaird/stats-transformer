import pytest
import pandas as pd
import numpy as np
import os
from stats_transformer.models.regression.robust_ols import RobustOLSModel
from stats_transformer.models.regression.panel import PanelRegressionModel
from stats_transformer.models.regression.iv import IV2SLSModel
from stats_transformer.models.regression.panel_iv import PanelIV2SLSModel

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
    
    rng = np.random.default_rng(42)

    n_entities = 50
    n_periods = 10

    entity = np.repeat(
        np.arange(n_entities),
        n_periods,
    )
    time = np.tile(
        np.arange(n_periods),
        n_entities,
    )
    n = len(entity)
    
    df = pd.DataFrame({
        "entity": entity,
        "date": time,
        "target": rng.random(size=n),
        "feature": rng.random(size=n),
    })    
    
    model = PanelRegressionModel(
        target="target",
        independent_variables=["feature"],
        entity_column="entity",
        time_column="date",
        entity_effects=True,
        cov_type="clustered",
        cluster_entity=True
    )
    
    metrics = model.fit(df)
    
    assert "rsquared" in metrics
    assert metrics["nobs"] == n
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

def test_panel_iv_model():
    # Simple Panel IV setting
    # Structural equation:
    # y_it = 2*x_it + 0.5*w_it + entity FE + time FE + error
    #
    # First stage:
    # x_it = 1.2*z_it + 0.3*w_it + entity FE + time FE + u_it

    rng = np.random.default_rng(42)

    n_entities = 50
    n_periods = 10

    entity = np.repeat(
        np.arange(n_entities),
        n_periods,
    )
    time = np.tile(
        np.arange(n_periods),
        n_entities,
    )
    n = len(entity)

    z = rng.normal(size=n) # instrument
    w = rng.normal(size=n) # exogenous control
    u = rng.normal(size=n)
    e = rng.normal(size=n)

    entity_effect_x = rng.normal(
        size=n_entities
    )[entity]
    time_effect_x = rng.normal(
        size=n_periods
    )[time]

    entity_effect_y = rng.normal(
        size=n_entities
    )[entity]
    time_effect_y = rng.normal(
        size=n_periods
    )[time]

    x = (
        1.2*z + 0.3*w + entity_effect_x + time_effect_x + u
    )

    y = (
        2.0*x + 0.5*w + entity_effect_y + time_effect_y + u + 0.2*e
    )

    df = pd.DataFrame({
        "entity": entity,
        "time": time,
        "y": y,
        "x": x,
        "z": z,
        "w": w,
    })

    model = PanelIV2SLSModel(
        target="y",
        independent_variables=["w"],
        endogenous=["x"],
        instruments=["z"],
        entity_column="entity",
        time_column="time",
        entity_effects=True,
        time_effects=True,
        cov_type="clustered",
        cluster_by="entity",
    )

    metrics = model.fit(df)

    assert model.model is not None
    assert metrics["num_observations"] == n

    assert model.model.params["x"] == pytest.approx(
        2.0,
        abs=0.15,
    )

    assert (
        metrics["first_stage"]["x"]["f.stat"]
        > 10
    )

    entity_dummies = [
        column
        for column in model.X_exog.columns
        if column.startswith("entity_")
    ]
    time_dummies = [
        column
        for column in model.X_exog.columns
        if column.startswith("time_")
    ]

    assert len(entity_dummies) == n_entities - 1
    assert len(time_dummies) == n_periods - 1

    metadata = model.get_model_metadata(metrics)

    assert "x" in metadata["coefficients"]
    assert metadata["summary"]["entity_effects"] is True
    assert metadata["summary"]["time_effects"] is True
    assert metadata["summary"]["cov_type"] == "clustered"