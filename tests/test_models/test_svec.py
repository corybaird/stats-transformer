import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.vector_ar.vecm import VECM
from stats_transformer.models.timeseries.structural.svec import SVEC, SVECModel


def test_svec_estimation():
    np.random.seed(42)
    y1 = np.cumsum(np.random.randn(120))
    y2 = y1 * 0.5 + np.random.randn(120)
    data = pd.DataFrame({"y1": y1, "y2": y2})

    vecm_model = VECM(data, k_ar_diff=1, coint_rank=1).fit()

    SR = np.array([[np.nan, 0.0], [np.nan, np.nan]])
    LR = np.array([[np.nan, np.nan], [np.nan, np.nan]])

    svec_fit = SVEC(vecm_model, SR=SR, LR=LR)
    matrices = svec_fit.get_structural_matrices()

    assert "SR" in matrices
    assert "LR" in matrices
    assert matrices["SR"].shape == (2, 2)
    assert np.isclose(matrices["SR"][0, 1], 0.0, atol=1e-5)
    assert np.isfinite(svec_fit.llf)


def test_svec_model_pipeline_interface():
    np.random.seed(42)
    y1 = np.cumsum(np.random.randn(100))
    y2 = y1 * 0.7 + np.random.randn(100)
    data = pd.DataFrame({"y1": y1, "y2": y2})

    SR = np.array([[np.nan, 0.0], [np.nan, np.nan]])
    model = SVECModel(target_variables=["y1", "y2"], k_ar_diff=1, coint_rank=1, SR=SR)
    metrics = model.fit(data)

    assert "num_observations" in metrics
    assert "log_likelihood" in metrics
    assert model.B_0.shape == (2, 2)
    assert "SVEC" in model.get_summary()
