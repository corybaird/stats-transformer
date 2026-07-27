import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM
from stats_transformer.models.timeseries.structural.svec import SVEC

def test_svec_matrices():
    np.random.seed(42)
    # Generate some VECM-like data (integrated)
    y1 = np.cumsum(np.random.randn(100))
    y2 = y1 * 0.5 + np.random.randn(100)
    data = pd.DataFrame({"y1": y1, "y2": y2})
    
    vecm_model = VECM(data, k_ar_diff=1, coint_rank=1).fit()
    
    # Define SR and LR restrictions
    SR = np.array([[np.nan, 0.0], [np.nan, np.nan]])
    LR = np.array([[np.nan, np.nan], [0.0, np.nan]])
    
    svec = SVEC(vecm_model, SR=SR, LR=LR)
    
    mats = svec.get_structural_matrices()
    assert "SR" in mats
    assert "LR" in mats
    
    # Check restrictions applied (since we used nan=1.0 for free params in the stub)
    assert mats["SR"][0, 1] == 0.0
    assert mats["LR"][1, 0] == 0.0
    
    # Check free params are 1.0 in the stub
    assert mats["SR"][0, 0] == 1.0
