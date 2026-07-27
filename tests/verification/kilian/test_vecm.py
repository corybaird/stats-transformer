import pytest
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import matlab.engine
    HAS_MATLAB = True
except ImportError:
    HAS_MATLAB = False

KILIAN_CODE_DIR = Path("references/matlab_benchmarks/kilian_2017")

@pytest.fixture(scope="module")
def matlab_session():
    if not HAS_MATLAB:
        pytest.skip("matlabengine not available.")
    eng = matlab.engine.start_matlab()
    if KILIAN_CODE_DIR.exists():
        eng.addpath(eng.genpath(str(KILIAN_CODE_DIR.absolute())), nargout=0)
    yield eng
    eng.quit()

@pytest.mark.skipif(not HAS_MATLAB, reason="Requires matlabengine")
def test_kilian_vecm_mle_match(matlab_session):
    if not KILIAN_CODE_DIR.exists():
        pytest.skip(f"Kilian textbook code not found in {KILIAN_CODE_DIR}")
        
    script_dir = KILIAN_CODE_DIR / "Code_Kilian" / "3" / "mle_unknown_beta"
    matlab_session.eval(f"cd('{script_dir.absolute()}');", nargout=0)
    matlab_session.eval("mle_unknown_beta;", nargout=0)
    
    y_matlab = matlab_session.workspace['y']
    y_py = np.array(y_matlab)
    
    from stats_transformer.models.timeseries.reduced_form.vecm import VECMModel
    
    df = pd.DataFrame(y_py, columns=["drgdp", "irate", "infl"])
    
    # Kilian p=4 means VAR(4), which is VECM with 3 lagged differences (k_ar_diff=3)
    # mle_unknown_beta computes Johansen MLE for VECM with unrestricted intercept
    model = VECMModel(target_variables=["drgdp", "irate", "infl"], k_ar_diff=3, deterministic="co")
    model.vecm_spec = __import__('statsmodels.tsa.vector_ar.vecm', fromlist=['VECM']).VECM(model.y, k_ar_diff=3, deterministic="co", coint_rank=2)
    model.model = model.vecm_spec.fit()
    
    alpha_py = model.model.alpha
    alpha_matlab = np.array(matlab_session.workspace['alpha'])
    
    # The signs of columns of alpha/beta might be flipped, which is mathematically equivalent, 
    # but we can check absolute values or the subspace spanned by beta.
    # For now, just test the residual covariance matrix SIGMAu which is invariant to normalization
    sigma_py = model.model.sigma_u
    sigma_matlab = np.array(matlab_session.workspace['SIGMAu'])
    
    np.testing.assert_allclose(sigma_py, sigma_matlab, rtol=1e-4, atol=1e-4)
