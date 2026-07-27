import os
import json
import subprocess
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from linearmodels.panel import PanelOLS, RandomEffects

R_AVAILABLE = os.environ.get("R_AVAILABLE", "0") == "1"

@pytest.mark.skipif(not R_AVAILABLE, reason="R environment not available")
def test_panel_models_integration():
    # 1. Run R panel script
    r_script = Path("tests/integration/r_scripts/panel.R")
    subprocess.run(["Rscript", str(r_script)], check=True)
    
    # 2. Read results
    res_file = Path("tests/integration/panel_results.json")
    assert res_file.exists()
    
    with open(res_file, "r") as f:
        r_res = json.load(f)
        
    df = pd.read_csv("tests/integration/data/grunfeld.csv")
    df = df.set_index(['firm', 'year'])
    
    # Panel OLS (Fixed Effects - Entity effects)
    exog = df[['value', 'capital']]
    mod_fe = PanelOLS(df['inv'], exog, entity_effects=True).fit()
    
    r_fe = np.array([r_res['fe_coef']['value'], r_res['fe_coef']['capital']])
    np.testing.assert_allclose(mod_fe.params.values, r_fe, rtol=1e-4, atol=1e-4)
