import pandas as pd
import numpy as np
import statsmodels.api as sm
from .regression import RegressionModel

class RobustOLSModel(RegressionModel):
    # Robust OLS forcing explicit covariance type definitions (HC1, HC3, HAC)

    def __init__(self, params_path=None, target=None, independent_variables=None, cov_type="HC3", add_entity_fixed_effects=False, entity_column=None, **kwargs):
        super().__init__(params_path=params_path, target=target, independent_variables=independent_variables, add_entity_fixed_effects=add_entity_fixed_effects, entity_column=entity_column, **kwargs)
        if self.params and not cov_type:
            self.cov_type = self.params.get("model", {}).get("robust_ols", {}).get("cov_type", "HC3")
        else:
            self.cov_type = cov_type

    def build_model(self, drop_na=True):
        if self.X is None or self.y is None:
            self.split_xy(drop_na=drop_na)

        x_numeric = self.X.select_dtypes(include=[np.number])
        if np.isnan(x_numeric.values).any() or np.isinf(x_numeric.values).any() or np.isnan(self.y.values).any() or np.isinf(self.y.values).any():
            if drop_na:
                mask = ~(np.isnan(x_numeric.values).any(axis=1) | np.isinf(x_numeric.values).any(axis=1) | np.isnan(self.y.values) | np.isinf(self.y.values))
                self.X = self.X[mask]
                self.y = self.y[mask]
            else:
                raise ValueError("Exogenous values contain inf or nans in robust OLS")

        self.model = sm.OLS(self.y, self.X).fit(cov_type=self.cov_type)
        return self.model
