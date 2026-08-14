import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.vector_ar.vecm import VECM
from stats_transformer.models.timeseries.structural.svec import SVEC

def test_svec_structural_estimation_not_implemented():
    np.random.seed(42)
    # Generate some VECM-like data (integrated)
    y1 = np.cumsum(np.random.randn(100))
    y2 = y1 * 0.5 + np.random.randn(100)
    data = pd.DataFrame({"y1": y1, "y2": y2})

    vecm_model = VECM(data, k_ar_diff=1, coint_rank=1).fit()

    # Define SR and LR restrictions
    SR = np.array([[np.nan, 0.0], [np.nan, np.nan]])
    LR = np.array([[np.nan, np.nan], [0.0, np.nan]])

    # SVEC's ML optimization is not implemented; constructing it must raise
    # rather than silently fabricate free parameters as 1.0.
    with pytest.raises(NotImplementedError):
        SVEC(vecm_model, SR=SR, LR=LR)
