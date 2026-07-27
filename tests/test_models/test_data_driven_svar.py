import numpy as np
import pandas as pd
import pytest
from stats_transformer.models.timeseries.identification.volatility import VolatilitySVARModel
from stats_transformer.models.timeseries.identification.independence import IndependenceSVARModel

def test_volatility_svar_model():
    np.random.seed(42)
    T = 200
    
    # structural shocks: unit variance in first 100, variance changes in second 100
    e1 = np.random.randn(T)
    e2 = np.random.randn(T)
    
    e1[100:] *= 2.0  # lambda1 = 4
    e2[100:] *= 0.5  # lambda2 = 0.25
    
    # Impact matrix B
    B = np.array([[1.0, 0.5], [0.3, 1.0]])
    
    u = B @ np.vstack([e1, e2])
    
    # Reduced form VAR(1) with these errors
    y = np.zeros((T, 2))
    for t in range(1, T):
        y[t] = 0.5 * y[t-1] + u[:, t]
        
    regime = np.concatenate([np.zeros(100), np.ones(100)])
    df = pd.DataFrame({'y1': y[:, 0], 'y2': y[:, 1], 'regime': regime})
    
    model = VolatilitySVARModel(
        target_variables=['y1', 'y2'],
        regime_column='regime',
        maxlags=1
    )
    
    model.df_clean = df
    model.build_model()
    
    metrics = model.get_model_metrics()
    assert 'lambda_diag' in metrics
    
    lambdas = metrics['lambda_diag']
    assert len(lambdas) == 2
    
def test_independence_svar_model():
    np.random.seed(42)
    T = 100
    
    # Independent non-Gaussian shocks (e.g., Laplace)
    e1 = np.random.laplace(size=T)
    e2 = np.random.laplace(size=T)
    
    B = np.array([[1.0, 0.8], [0.2, 1.0]])
    u = B @ np.vstack([e1, e2])
    
    y = np.zeros((T, 2))
    for t in range(1, T):
        y[t] = 0.5 * y[t-1] + u[:, t]
        
    df = pd.DataFrame({'y1': y[:, 0], 'y2': y[:, 1]})
    
    model = IndependenceSVARModel(
        target_variables=['y1', 'y2'],
        maxlags=1,
        n_starts=1 # fast test
    )
    
    model.df_clean = df
    model.build_model()
    
    metrics = model.get_model_metrics()
    assert 'opt_success' in metrics
    assert model.structural_impact is not None
    assert model.structural_impact.shape == (2, 2)
