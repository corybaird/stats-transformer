import os
import json
import subprocess
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from stats_transformer.models.timeseries.identification.volatility import VolatilitySVARModel

# This test requires R and the svars package
R_AVAILABLE = os.environ.get("R_AVAILABLE", "0") == "1"

@pytest.mark.skipif(not R_AVAILABLE, reason="R environment not available")
def test_svars_volatility_integration():
    # 1. Run the R script to generate the data and the benchmark results
    r_script_path = Path("tests/integration/r_scripts/svars_benchmark.R")
    
    # We run Rscript
    subprocess.run(["Rscript", str(r_script_path)], check=True)
    
    # 2. Load the outputs
    data_path = Path("tests/integration/data_usa.csv")
    res_path = Path("tests/integration/svars_results.json")
    
    assert data_path.exists()
    assert res_path.exists()
    
    df = pd.read_csv(data_path)
    with open(res_path, 'r') as f:
        r_results = json.load(f)
        
    B_r = np.array(r_results["B_cv"])
    Lambda_r = np.array(r_results["Lambda_cv"])
    
    # 3. Create regime column for our Python model
    # R svars USA dataset break is at observation 79 (1979:3)
    # R is 1-indexed, so index 79 means 78 in 0-indexed Python.
    # The `id.cv` function with `SB=79` puts the first 79 observations in regime 1.
    # We must match exactly how svars handles the effective sample after VAR lags.
    # VAR(p=6) consumes the first 6 observations.
    # So the residuals have length T - 6.
    # The break at original index 79 corresponds to residual index 79 - 6 = 73.
    # Actually, R `svars` `id.cv` with `SB=79` means the break is *at* observation 79 of the ORIGINAL data.
    # Regime 1: 1 to 79. Regime 2: 80 to T.
    # Let's create a regime column on the original data.
    regime = np.zeros(len(df))
    regime[79:] = 1  # 80th observation and beyond (index 79) is regime 2
    df['regime'] = regime
    
    # 4. Run our VolatilitySVARModel
    model = VolatilitySVARModel(
        target_variables=['gdp', 'inf', 'ir'],
        regime_column='regime',
        maxlags=6
    )
    model.df_clean = df
    model.build_model()
    
    metrics = model.get_model_metrics()
    
    # 5. Compare Lambda
    Lambda_py = np.array(metrics['lambda_diag'])
    
    # Because of permutation indeterminacy, we might need to sort or align.
    # However, our alignment utility should put them in the same order as R if R uses Cholesky initialization/alignment.
    # If they are not perfectly aligned by our utility, we can sort them to compare.
    Lambda_r_sorted = np.sort(Lambda_r)
    Lambda_py_sorted = np.sort(Lambda_py)
    
    np.testing.assert_allclose(Lambda_py_sorted, Lambda_r_sorted, rtol=1e-2, atol=1e-2)
