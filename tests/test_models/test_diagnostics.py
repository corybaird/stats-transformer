import numpy as np
import pytest
from statsmodels.tsa.vector_ar.var_model import VAR
from stats_transformer.models.timeseries.diagnostics.residuals import ResidualDiagnostics
from stats_transformer.models.timeseries.diagnostics.stability import StabilityDiagnostics

def test_residual_diagnostics_serial_correlation(var1_data):
    data = var1_data()
    sm_model = VAR(data).fit(1)
    
    diag = ResidualDiagnostics(sm_model)
    res = diag.test_serial_correlation(lags=5)
    
    assert "portmanteau" in res
    assert "adjusted_portmanteau" in res
    assert res["portmanteau"]["df"] == 16 # 2^2 * (5 - 1)
    assert not np.isnan(res["portmanteau"]["statistic"])

def test_residual_diagnostics_normality(var1_data):
    data = var1_data()
    sm_model = VAR(data).fit(1)
    
    diag = ResidualDiagnostics(sm_model)
    res = diag.test_normality()
    
    assert "skewness" in res
    assert "kurtosis" in res
    assert "omnibus" in res
    assert res["omnibus"]["df"] == 4

def test_stability_diagnostics(var1_data):
    data = var1_data()
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
