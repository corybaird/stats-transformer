import os
import json
import subprocess
import pytest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
from pathlib import Path

R_AVAILABLE = os.environ.get("R_AVAILABLE", "0") == "1"

@pytest.mark.skipif(not R_AVAILABLE, reason="R environment not available")
def test_general_regressions_integration():
    # 1. Run R regression script
    r_script = Path("tests/integration/regression/regression.R")
    subprocess.run(["Rscript", str(r_script)], check=True)
    
    # 2. Read results
    res_file = Path("tests/integration/regression/regression_results.json")
    assert res_file.exists()
    
    with open(res_file, "r") as f:
        r_res = json.load(f)
        
    # Load dataset
    df = pd.read_csv("tests/integration/data/ghysels/ex2_regress_gdp_us.csv")
    
    # OLS Validation
    X = sm.add_constant(df[['ipr', 'su', 'pr', 'sr']])
    y = df['y']
    model_ols = sm.OLS(y, X).fit()
    
    # Compare OLS coefficients
    r_ols = np.array([r_res['ols_coef']['(Intercept)'], r_res['ols_coef']['ipr'], 
                      r_res['ols_coef']['su'], r_res['ols_coef']['pr'], r_res['ols_coef']['sr']])
    np.testing.assert_allclose(model_ols.params.values, r_ols, rtol=1e-5, atol=1e-5)
    
    # Robust standard errors (HC1)
    model_hc1 = sm.OLS(y, X).fit(cov_type='HC1')
    r_hc1 = np.array([r_res['se_hc1']['(Intercept)'], r_res['se_hc1']['ipr'], 
                      r_res['se_hc1']['su'], r_res['se_hc1']['pr'], r_res['se_hc1']['sr']])
    np.testing.assert_allclose(model_hc1.bse.values, r_hc1, rtol=1e-4, atol=1e-4)

    # Logit Validation
    df['y_bin'] = (df['y'] > 0).astype(int)
    X_logit = sm.add_constant(df[['ipr', 'su']])
    model_logit = sm.Logit(df['y_bin'], X_logit).fit(disp=0)
    r_logit = np.array([r_res['logit_coef']['(Intercept)'], r_res['logit_coef']['ipr'], r_res['logit_coef']['su']])
    np.testing.assert_allclose(model_logit.params.values, r_logit, rtol=1e-4, atol=1e-4)
