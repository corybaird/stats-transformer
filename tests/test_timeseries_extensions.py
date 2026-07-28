import numpy as np
import pandas as pd
from stats_transformer.models.timeseries.identification.blanchard_quah import BlanchardQuahModel
from stats_transformer.models.timeseries.identification.proxy_svar import ProxySVARModel
from stats_transformer.models.timeseries.identification.sign_zero import SignZeroSVARModel
from stats_transformer.models.timeseries.reduced_form.local_projections_iv import LocalProjectionsIVModel
from stats_transformer.models.timeseries.decompositions import TimeSeriesDecompositions
from examples.academic.var.stock_watson_2001 import StockWatson2001Replication
from examples.academic.var.blanchard_quah_1989 import BlanchardQuah1989Replication
from examples.academic.var.gertler_karadi_2015 import GertlerKaradi2015Replication
from examples.academic.var.jorda_taylor_2025 import JordaTaylor2025Replication

def create_synthetic_data():
    np.random.seed(42)
    n = 100
    y1 = np.cumsum(np.random.normal(size=n))
    y2 = np.cumsum(np.random.normal(size=n)) + 0.5 * y1
    z = np.random.normal(size=n) + 0.3 * y1
    df = pd.DataFrame({"y1": y1, "y2": y2, "z": z, "date": pd.date_range("2000-01-01", periods=n, freq="QE")})
    return df

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
