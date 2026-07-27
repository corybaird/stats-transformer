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
def test_proxy_svar_gk2015_match(matlab_session):
    if not (TOOLBOX_DIR / "VAR").exists():
        pytest.skip(f"VAR-Toolbox code not found in {TOOLBOX_DIR}")
        
    data_path = TOOLBOX_DIR / "Replic" / "GK2015" / "GK2015_Data.xlsx"
    if not data_path.exists():
        pytest.skip(f"GK2015 sample data not found: {data_path}")
        
    # Read the data from excel (skip row 1, headers on row 2)
    raw = pd.read_excel(data_path, header=1, na_values=123456789)
    # The first column is dates, we only need the numeric data
    VARmnem = ['gs1', 'logcpi', 'logip', 'ebp']
    IVmnem = ['ff4_tc']
    
    # We only take rows from 3 onward according to MATLAB readcell handling
    data = raw.iloc[1:].reset_index(drop=True).copy()
    data[VARmnem] = data[VARmnem].apply(pd.to_numeric, errors="coerce")
    data[IVmnem] = data[IVmnem].apply(pd.to_numeric, errors="coerce")
    
    from stats_transformer.models.timeseries.identification.proxy_svar import ProxySVARModel
    import matlab
    
    python_model = ProxySVARModel(target_variables=VARmnem, instrument_variable="ff4_tc", maxlags=12)
    python_model.fit(data)
    
    # Python impact column
    python_impact = python_model.impact_column
    
    # Run MATLAB version
    options = matlab_session.VARoption(nargout=1)
    options["ident"] = "iv"
    options["IV"] = matlab.double(data[IVmnem].values.tolist())
    
    matlab_result = matlab_session.VARmodel(matlab.double(data[VARmnem].values.tolist()), 12.0, 1.0, options, nargout=1)
    # MATLAB returns IRbar of shape (nsteps, nvar, nshocks). For proxy SVAR, often there's 1 shock identified
    matlab_impact = np.array(matlab_result["IRbar"])[0, :, 0]
    
    # The python impact is equivalent to the impact response (IR at h=0). 
    # Signs can be opposite, so check absolute correlation or differences.
    # We scale both to be comparable or just test absolute difference if they use same scaling.
    python_impact = python_impact / python_impact[0]
    matlab_impact = matlab_impact / matlab_impact[0]
    
    np.testing.assert_allclose(python_impact, matlab_impact, rtol=1e-4, atol=1e-4)
