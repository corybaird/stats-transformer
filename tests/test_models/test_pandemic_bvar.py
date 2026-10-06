import numpy as np
import pandas as pd
import pytest
from stats_transformer.models.timeseries.reduced_form.pandemic_bvar import PandemicBVARModel

def test_pandemic_bvar_fit_and_metrics():
    np.random.seed(42)
    T = 60
    y1 = np.random.randn(T)
    y2 = 0.5 * y1 + np.random.randn(T)
    dates = pd.date_range("2015-01-01", periods=T, freq="QE")
    df = pd.DataFrame({"y1": y1, "y2": y2, "date": dates})
    pandemic_dates = ["2020-03-31", "2020-06-30", "2020-09-30"]
    model = PandemicBVARModel(target_variables=["y1", "y2"], date_column="date", lags=1, pandemic_periods=pandemic_dates, volatility_scale=5.0, n_draws=20, seed=42)
    model.fit(df)
    metrics = model.get_model_metrics()
    assert metrics["nobs"] == 59
    assert metrics["nvars"] == 2
    assert metrics["volatility_scale"] == 5.0
    summary = model.get_summary()
    assert "Pandemic BVAR Model" in summary
    pred = model.predict(df)
    assert pred.shape == (59, 2)
    run_metrics = model.run(df)
    assert run_metrics["nobs"] == 59
