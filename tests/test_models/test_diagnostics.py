import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.vector_ar.var_model import VAR
from stats_transformer.models.timeseries.diagnostics.residuals import ResidualDiagnostics
from stats_transformer.models.timeseries.diagnostics.stability import StabilityDiagnostics

def _synthetic_data(observations=120):
    generator = np.random.default_rng(42)
    shocks = generator.normal(size=(observations, 2))
    values = np.zeros((observations, 2))
    transition = np.array([[0.55, 0.10], [0.15, 0.40]])
    for index in range(1, observations):
        values[index] = transition @ values[index - 1] + shocks[index]
    data = pd.DataFrame(values, columns=["output", "inflation"])
    return data

def test_residual_diagnostics_serial_correlation():
    data = _synthetic_data()
    sm_model = VAR(data).fit(1)
    
    diag = ResidualDiagnostics(sm_model)
    res = diag.test_serial_correlation(lags=5)
    
    assert "portmanteau" in res
    assert "adjusted_portmanteau" in res
    assert res["portmanteau"]["df"] == 16 # 2^2 * (5 - 1)
    assert not np.isnan(res["portmanteau"]["statistic"])

def test_residual_diagnostics_normality():
    data = _synthetic_data()
    sm_model = VAR(data).fit(1)
    
    diag = ResidualDiagnostics(sm_model)
    res = diag.test_normality()
    
    assert "skewness" in res
    assert "kurtosis" in res
    assert "omnibus" in res
    assert res["omnibus"]["df"] == 4

def test_stability_diagnostics():
    data = _synthetic_data()
    sm_model = VAR(data).fit(1)
    
    diag = StabilityDiagnostics(sm_model)
    roots = diag.roots()
    
    # 2 equations, 1 lag -> 2 roots
    assert len(roots) == 2
    assert diag.is_stable()

    # ols_cusum is not implemented; it must raise rather than silently
    # return a zero process with fixed +/-1.0 bounds.
    with pytest.raises(NotImplementedError):
        diag.ols_cusum()
