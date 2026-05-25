import pandas as pd
import numpy as np
import statsmodels.api as sm
from stats_transformer.models.base import ModelBase

class LocalProjectionsModel(ModelBase):
    def __init__(self, target, shock_var, horizon=8, controls=None, **kwargs):
        super().__init__(target=target, independent_variables=[shock_var] + (controls or []), **kwargs)
        self.shock_var = shock_var
        self.horizon = horizon
        self.controls = controls or []
        self.irf_results = []
        
    def build_model(self):
        self.irf_results = []
        y = self.df_clean[self.target]
        x = self.df_clean[self.independent_variables]
        x = sm.add_constant(x)
        
        for h in range(self.horizon + 1):
            y_h = y.shift(-h)
            valid_idx = y_h.notna()
            y_valid = y_h[valid_idx]
            x_valid = x.loc[valid_idx]
            
            model = sm.OLS(y_valid, x_valid)
            res = model.fit(cov_type='HC3')
            
            self.irf_results.append({
                'horizon': h,
                'effect': res.params[self.shock_var],
                'stderr': res.bse[self.shock_var],
                'lower_ci': res.conf_int().loc[self.shock_var, 0],
                'upper_ci': res.conf_int().loc[self.shock_var, 1],
                'pvalue': res.pvalues[self.shock_var]
            })

    def compute_irf(self):
        if not self.irf_results:
            self.build_model()
        return pd.DataFrame(self.irf_results)

    def summary_df(self):
        return self.compute_irf()

    def get_summary(self):
        return self.summary_df().to_string()

    def get_model_metrics(self):
        return {"horizon": self.horizon, "shock_var": self.shock_var}
