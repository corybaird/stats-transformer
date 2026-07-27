import numpy as np
import pandas as pd
import pytest
from stats_transformer.models.timeseries.reduced_form.lag_selection import VARLagSelector
from stats_transformer.models.timeseries.reduced_form.restrictions import RestrictedVAR

def _synthetic_data(observations=120):
    generator = np.random.default_rng(42)
    shocks = generator.normal(size=(observations, 2))
    values = np.zeros((observations, 2))
    transition = np.array([[0.55, 0.10], [0.15, 0.40]])
    for index in range(1, observations):
        values[index] = transition @ values[index - 1] + shocks[index]
    data = pd.DataFrame(values, columns=["output", "inflation"])
    return data

def test_var_lag_selector_returns_criteria():
    data = _synthetic_data()
    selector = VARLagSelector(target_variables=["output", "inflation"], maxlags=4)
    selected = selector.fit(data)
    
    assert "aic" in selected
    assert "bic" in selected
    assert "fpe" in selected
    assert "hqic" in selected
    
    assert selector.criteria_history is not None
    assert len(selector.criteria_history) == 5 # lags 0 to 4
    assert set(selector.criteria_history.columns) == {"aic", "bic", "fpe", "hqic"}

def test_restricted_var_estimates_with_mask():
    data = _synthetic_data()
    
    # 2 variables, lag 1, trend 'c' -> 3 regressors (const, L1.output, L1.inflation)
    # Mask out L1.inflation in output equation (equation 0)
    mask = np.array([
        [True, True],   # const
        [True, True],   # L1.output
        [False, True],  # L1.inflation
    ])
    
    restricted_model = RestrictedVAR(data, mask=mask, maxlags=1, trend="c")
    res = restricted_model.fit()
    
    assert res.params.shape == (3, 2)
    # The restricted parameter should be exactly 0
    assert res.params[2, 0] == 0.0
    # Other parameters should be non-zero
    assert res.params[0, 0] != 0.0
    assert res.params[1, 0] != 0.0
    
    assert res.sigma_u.shape == (2, 2)
    assert res.resid.shape == (119, 2)
