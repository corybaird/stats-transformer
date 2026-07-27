import pytest
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import matlab.engine
    HAS_MATLAB = True
except ImportError:
    HAS_MATLAB = False

TOOLBOX_DIR = Path("references/matlab_benchmarks")

@pytest.fixture(scope="module")
def matlab_session():
    if not HAS_MATLAB:
        pytest.skip("matlabengine not available.")
    eng = matlab.engine.start_matlab()
    if (TOOLBOX_DIR / "VAR").exists():
        eng.addpath(eng.genpath(str(TOOLBOX_DIR.absolute())), nargout=0)
    yield eng
    eng.quit()

@pytest.mark.skipif(not HAS_MATLAB, reason="Requires matlabengine")
def test_local_projections_jt2025_ols_match(matlab_session):
    if not (TOOLBOX_DIR / "VAR").exists():
        pytest.skip(f"VAR-Toolbox code not found in {TOOLBOX_DIR}")
        
    data_path = TOOLBOX_DIR / "Replic" / "JT2025" / "JT2025_Data.xlsx"
    if not data_path.exists():
        pytest.skip(f"JT2025 sample data not found: {data_path}")
        
    # Ex5 is the LP-OLS dataset in Jordà and Taylor (2025) replication
    raw = pd.read_excel(data_path, sheet_name='Ex5', header=1, na_values=123456789)
    # The first column is dates
    mnem_OLS = ['lcpi', 'rr_shock', 'dlrgdp', 'dlcpi', 'dstir']
    
    # We only take rows from 3 onward
    data = raw.iloc[1:].reset_index(drop=True).copy()
    data[mnem_OLS] = data[mnem_OLS].apply(pd.to_numeric, errors="coerce")
    
    # Filter for 1985Q1-2007Q4 (which is indices matching the matlab code)
    # Let's just use the non-NaN portion of rr_shock for simplicity 
    # (or replicate exact indices if needed)
    data = data.dropna(subset=mnem_OLS).reset_index(drop=True)
    
    from stats_transformer.models.timeseries.reduced_form.local_projections import LocalProjectionsModel
    import matlab
    
    # Python LP model
    # Note: Python LPmodel does not automatically difference, so we test levels
    python_model = LocalProjectionsModel(target="lcpi", shock_var="rr_shock", horizon=18, controls=['dlrgdp', 'dlcpi', 'dstir'])
    python_model.fit(data)
    python_irf = python_model.irf_results
    python_irf_values = np.array([res['effect'] for res in python_irf])
    
    # Run MATLAB version
    options = matlab_session.LPoption(nargout=1)
    options["nsteps"] = 18
    options["longdiff"] = 0  # To match Python levels shift behavior
    options["impact"] = 1
    
    CTRL_OLS = matlab.double(data[['dlrgdp', 'dlcpi', 'dstir']].values.tolist())
    y_mat = matlab.double(data[['lcpi']].values.tolist())
    x_mat = matlab.double(data[['rr_shock']].values.tolist())
    
    matlab_result = matlab_session.LPmodel(y_mat, x_mat, CTRL_OLS, 0.0, 1.0, options, nargout=1)
    matlab_irf = np.array(matlab_result["IR"])
    
    # The arrays should match closely (using HC3 robust std errors could differ slightly, 
    # but point estimates should match exactly)
    np.testing.assert_allclose(python_irf_values, matlab_irf.flatten(), rtol=1e-4, atol=1e-4)
