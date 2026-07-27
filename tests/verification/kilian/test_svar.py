import pytest
from pathlib import Path

try:
    import matlab.engine
    HAS_MATLAB = True
except ImportError:
    HAS_MATLAB = False

# Path where the user will drop the downloaded Kilian code
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

@pytest.mark.skipif(not HAS_MATLAB, reason="Requires matlabengine and local R2025b")
def test_kilian_svar_irf_match(matlab_session):
    if not KILIAN_CODE_DIR.exists():
        pytest.skip(f"Kilian textbook code not found in {KILIAN_CODE_DIR}")
        
    script_dir = KILIAN_CODE_DIR / "Code_Kilian" / "9" / "figure9_1_chol"
    matlab_session.eval(f"cd('{script_dir.absolute()}');", nargout=0)
    
    # Run the Kilian script which executes data.m, olsvarc, chol, irfvar and puts IRF into the workspace
    matlab_session.eval("figure9_1;", nargout=0)
    
    # 1. Fetch the data matrix `y` used by the MATLAB script
    y_matlab = matlab_session.workspace['y']
    y_py = np.array(y_matlab)
    
    # 2. Run Python SVAR estimation
    from stats_transformer.models.timeseries.identification.svar import SVARModel
    import pandas as pd
    
    df = pd.DataFrame(y_py, columns=["drpoil", "infl", "drgdp"])
    
    model = SVARModel(target_variables=["drpoil", "infl", "drgdp"], maxlags=4)
    model.fit(df)
    
    # Compute Python IRF (12 periods)
    # statsmodels IRF returns (periods, num_equations, num_shocks)
    # Kilian IRF returns (num_equations, periods+1) for the first shock only (real price of oil shock)
    irf_obj = model.model.irf(12)
    py_irf = irf_obj.irfs[:, :, 0].T  # transposed to match MATLAB's (num_equations, periods)
    
    # 3. Fetch the corresponding Kilian MATLAB IRF
    irf_matlab = np.array(matlab_session.workspace['IRF'])
    
    # 4. Assert np.allclose() between IRFs
    np.testing.assert_allclose(py_irf, irf_matlab, rtol=1e-5, atol=1e-5)
