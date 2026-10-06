import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from stats_transformer.models.timeseries.identification.bayesian_sign_zero import BayesianSignZeroSVARModel

def test_bayesian_sign_zero_fit_and_metrics(tmp_path):
    np.random.seed(42)
    T = 80
    y1 = np.random.randn(T)
    y2 = 0.4 * y1 + np.random.randn(T)
    y3 = -0.2 * y1 + 0.3 * y2 + np.random.randn(T)
    dates = pd.date_range("2010-01-01", periods=T, freq="QE")
    df = pd.DataFrame({"y1": y1, "y2": y2, "y3": y3, "date": dates})
    config_content = """
shocks:
  - s1
  - s2
  - s3
restrictions:
  - shock: s1
    response: y1
    type: sign
    value: "+"
    horizon: 0
"""
    config_path = tmp_path / "bsvar_config.yaml"
    config_path.write_text(config_content)
    model = BayesianSignZeroSVARModel(target_variables=["y1", "y2", "y3"], config_path=str(config_path), date_column="date", lags=1, n_draws=50, max_rotations_per_draw=20, required_accepts=3, seed=42)
    model.fit(df)
    metrics = model.get_model_metrics()
    assert metrics["accepted_draws"] >= 1
    assert metrics["total_draws"] >= 1
    rep = model.get_representative_draw()
    assert "rotation" in rep
    assert "irf" in rep
    summary = model.get_summary()
    assert "Bayesian Sign/Zero SVAR Model" in summary
    pred = model.predict(df)
    assert pred.shape[1] == 3
    fevd = model.compute_fevd(steps=5)
    assert fevd.shape == (5, 3, 3)
    hd, shocks = model.compute_hd()
    assert hd.shape[1] == 3
    assert shocks.shape[1] == 3
