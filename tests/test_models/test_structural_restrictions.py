import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from stats_transformer.models.timeseries.identification.sign_zero import SignZeroSVARModel
from stats_transformer.models.timeseries.identification.bootstrap import SVARBootstrap

def test_sign_zero_svar_model(tmp_path):
    # Create a synthetic dataset
    np.random.seed(42)
    T = 100
    y1 = np.random.randn(T)
    y2 = 0.5 * y1 + np.random.randn(T)
    y3 = -0.3 * y1 + 0.2 * y2 + np.random.randn(T)
    
    df = pd.DataFrame({'y1': y1, 'y2': y2, 'y3': y3})
    
    # Create a temporary config file
    config_content = """
variables:
  - y1
  - y2
  - y3
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
  - shock: s1
    response: y2
    type: sign
    value: "-"
    horizon: 0
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(config_content)
    
    # Test model
    model = SignZeroSVARModel(
        target_variables=['y1', 'y2', 'y3'],
        config_path=str(config_path),
        maxlags=1,
        max_draws=500,
        required_accepts=5
    )
    
    model.df_clean = df
    model.build_model()
    
    metrics = model.get_model_metrics()
    assert metrics['accepted_draws'] > 0
    assert metrics['total_draws'] > 0
    
    rep_draw = model.get_representative_draw()
    assert 'irf' in rep_draw
    assert 'rotation' in rep_draw
    
    # Check that restrictions hold in the accepted draws
    for irf in model.accepted_irfs:
        impact = irf[0] # horizon 0
        # s1 is index 0, y1 is index 0 -> +
        assert impact[0, 0] > 0
        # s1 is index 0, y2 is index 1 -> -
        assert impact[1, 0] < 0

def test_svar_bootstrap(tmp_path):
    np.random.seed(42)
    T = 50
    y1 = np.random.randn(T)
    y2 = 0.5 * y1 + np.random.randn(T)
    df = pd.DataFrame({'y1': y1, 'y2': y2})
    
    config_content = """
variables:
  - y1
  - y2
shocks:
  - s1
  - s2
restrictions:
  - shock: s1
    response: y1
    type: sign
    value: "+"
    horizon: 0
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(config_content)
    
    model = SignZeroSVARModel(
        target_variables=['y1', 'y2'],
        config_path=str(config_path),
        maxlags=1,
        max_draws=100,
        required_accepts=1 # small for fast test
    )
    model.df_clean = df
    model.build_model()
    
    bootstrap = SVARBootstrap(model, n_bootstrap=2, seed=42)
    results = bootstrap.run()
    
    assert len(results) > 0
    
    lower, upper = bootstrap.get_confidence_intervals()
    assert lower.shape == results[0]['irf'].shape
    assert upper.shape == results[0]['irf'].shape
