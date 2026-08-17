import pytest
import pandas as pd
import numpy as np
import os
from stats_transformer.models.regression.robust_ols import RobustOLSModel
from stats_transformer.models.regression.panel import PanelRegressionModel
from stats_transformer.models.regression.iv import IV2SLSModel
from stats_transformer.models.regression.panel_iv import PanelIV2SLSModel
from stats_transformer.models.regression.gmm import GMMModel
from examples.academic.coibion_gorodnichenko_2012 import CoibionGorodnichenko2012Replication

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

def _simulate_iv_data(seed=42, n=2000):
    np.random.seed(seed)
    z1 = np.random.normal(size=n)
    z2 = np.random.normal(size=n)
    u = np.random.normal(size=n)
    x_exog = np.random.normal(size=n)
    x_endog = 0.5 * z1 + 0.3 * z2 + 0.4 * u + np.random.normal(scale=0.5, size=n)
    y = 1.0 + 2.0 * x_exog - 1.5 * x_endog + u + np.random.normal(scale=0.3, size=n)
    return pd.DataFrame({"y": y, "x_exog": x_exog, "x_endog": x_endog, "z1": z1, "z2": z2})

def test_gmm_model_matches_linearmodels_ivgmm():
    from linearmodels.iv import IVGMM
    df = _simulate_iv_data()

    model = GMMModel(target="y", independent_variables=["x_exog"], endogenous=["x_endog"], instruments=["z1", "z2"], method="two_step", weighting="hac", bandwidth=0)
    metrics = model.fit(df)

    exog = pd.concat([pd.Series(1.0, index=df.index, name="const"), df["x_exog"]], axis=1)
    reference = IVGMM(dependent=df["y"], exog=exog, endog=df[["x_endog"]], instruments=df[["z1", "z2"]], weight_type="robust").fit(iter_limit=2, cov_type="robust")

    assert metrics["coefficients"]["const"] == pytest.approx(reference.params["const"], abs=1e-3)
    assert metrics["coefficients"]["x_exog"] == pytest.approx(reference.params["x_exog"], abs=1e-3)
    assert metrics["coefficients"]["x_endog"] == pytest.approx(reference.params["x_endog"], abs=1e-3)
    assert metrics["j_statistic"] == pytest.approx(reference.j_stat.stat, abs=0.05)

def test_gmm_model_methods_agree_under_correct_specification():
    df = _simulate_iv_data()
    coefficients = {}
    for method in ["one_step", "two_step", "iterated", "cue"]:
        model = GMMModel(target="y", independent_variables=["x_exog"], endogenous=["x_endog"], instruments=["z1", "z2"], method=method, weighting="hac", bandwidth=0)
        metrics = model.fit(df)
        coefficients[method] = metrics["coefficients"]["x_endog"]
    for method, coef in coefficients.items():
        assert coef == pytest.approx(-1.5, abs=0.05), f"{method} diverged: {coef}"

def test_gmm_j_test_does_not_reject_valid_instruments():
    df = _simulate_iv_data(seed=7)
    model = GMMModel(target="y", independent_variables=["x_exog"], endogenous=["x_endog"], instruments=["z1", "z2"], method="two_step")
    metrics = model.fit(df)
    assert metrics["j_pvalue"] > 0.05

def test_gmm_j_test_rejects_invalid_instrument():
    np.random.seed(7)
    n = 2000
    z1 = np.random.normal(size=n)
    z2 = np.random.normal(size=n)
    u = np.random.normal(size=n)
    x_exog = np.random.normal(size=n)
    x_endog = 0.5 * z1 + 0.3 * z2 + 0.4 * u + np.random.normal(scale=0.5, size=n)
    z2_invalid = z2 + 0.6 * u  # violates the exclusion restriction
    y = 1.0 + 2.0 * x_exog - 1.5 * x_endog + u + np.random.normal(scale=0.3, size=n)
    df = pd.DataFrame({"y": y, "x_exog": x_exog, "x_endog": x_endog, "z1": z1, "z2": z2_invalid})

    model = GMMModel(target="y", independent_variables=["x_exog"], endogenous=["x_endog"], instruments=["z1", "z2"], method="two_step")
    metrics = model.fit(df)
    assert metrics["j_pvalue"] < 0.01

def test_gmm_model_underidentified_raises():
    df = _simulate_iv_data(n=200)
    model = GMMModel(target="y", independent_variables=["x_exog"], endogenous=["x_endog", "z2"], instruments=[], method="two_step")
    with pytest.raises(ValueError, match="underidentified"):
        model.fit(df)

def test_coibion_gorodnichenko_2012_replication():
    result = CoibionGorodnichenko2012Replication().run()
    assert "metrics" in result
    assert result["metrics"]["num_observations"] > 0
    assert "forecast_revision" in result["metrics"]["coefficients"]

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