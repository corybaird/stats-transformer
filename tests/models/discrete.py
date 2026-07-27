import pytest
import pandas as pd
import numpy as np
from stats_transformer.models.discrete.logit import LogitModel

def test_logit_model():
    # Simple binary target with some noise to avoid singular matrix
    np.random.seed(42)
    n = 100
    x1 = np.random.normal(size=n)
    x2 = np.random.normal(size=n)
    # Probability
    p = 1 / (1 + np.exp(-(0.5 * x1 - 0.3 * x2 + np.random.normal(scale=0.1, size=n))))
    y = (p > 0.5).astype(int)
    
    df = pd.DataFrame({
        "target": y,
        "feature1": x1,
        "feature2": x2
    })
    
    model = LogitModel(
        target="target",
        independent_variables=["feature1", "feature2"]
    )
    
    metrics = model.fit(df)
    
    assert "pseudo_r_squared" in metrics
    assert metrics["num_observations"] == 100
    assert model.model is not None
