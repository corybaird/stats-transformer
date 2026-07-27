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
def test_var_toolbox_bq_match(matlab_session):
    if not (TOOLBOX_DIR / "VAR").exists():
        pytest.skip(f"VAR-Toolbox code not found in {TOOLBOX_DIR}")
        
    data_path = Path("data/examples/matlab_examples/BQ1989_Data.xlsx")
    if not data_path.exists():
        pytest.skip(f"Blanchard-Quah sample data not found: {data_path}")
        
    data = pd.read_excel(data_path).iloc[1:].reset_index(drop=True)
    variables = [column for column in data.columns if column not in ["Date", "date", "year", "quarter", "Unnamed: 0"]][:2]
    data[variables] = data[variables].apply(pd.to_numeric, errors="coerce")
    data_clean = data[variables].dropna()
    
    from stats_transformer.models.timeseries.identification.blanchard_quah import BlanchardQuahModel
    import matlab
    
    maxlags = 8
    python_model = BlanchardQuahModel(target_variables=variables, maxlags=maxlags)
    python_model.fit(data_clean)
    
    options = matlab_session.VARoption(nargout=1)
    options["ident"] = "long"
    options["inference"] = 0.0
    
    matlab_result = matlab_session.VARmodel(matlab.double(data_clean.values.tolist()), float(maxlags), 1.0, options, nargout=1)
    matlab_b0 = np.asarray(matlab_result["B"])
    
    np.testing.assert_allclose(python_model.B_0, matlab_b0, rtol=1e-10, atol=1e-10)
