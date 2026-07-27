import os
import json
import subprocess
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.api import VAR

R_AVAILABLE = os.environ.get("R_AVAILABLE", "0") == "1"

@pytest.mark.skipif(not R_AVAILABLE, reason="R environment not available")
def test_var_models_integration():
    # 1. Run R VAR script
    r_script = Path("tests/integration/r_scripts/var.R")
    subprocess.run(["Rscript", str(r_script)], check=True)
    
    # 2. Read results
    res_file = Path("tests/integration/var_results.json")
    assert res_file.exists()
    
    with open(res_file, "r") as f:
        r_res = json.load(f)
        
    df = pd.read_csv("tests/integration/data/canada.csv")
    
    # Fit VAR(2) with statsmodels
    model = VAR(df)
    results = model.fit(maxlags=2, trend='c')
    
    # Check that model fitted properly
    assert results.params is not None
