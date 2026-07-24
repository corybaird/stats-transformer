import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from stats_transformer.models.base import ModelBase

class BlanchardQuahModel(ModelBase):

    def __init__(self, target_variables=None, date_column=None, maxlags=1, **kwargs):
        target = target_variables[0] if target_variables else "dummy"
        indep = target_variables[1:] if target_variables and len(target_variables) > 1 else ["dummy"]
        super().__init__(target=target, independent_variables=indep, **kwargs)
        self.target_variables = target_variables or []
        self.date_column = date_column
        self.time_column = date_column
        self.maxlags = maxlags
        self.var_result = None
        self.B_0 = None

    def _get_required_columns(self):
        cols = list(self.target_variables)
        if self.date_column:
            cols.append(self.date_column)
        return cols

    def build_model(self):
        if getattr(self, 'df_clean', None) is None:
            raise ValueError("No cleaned data available")
        if self.date_column and self.date_column in self.df_clean.columns:
            self.df_clean = self.df_clean.sort_values(self.date_column)
        self.y = self.df_clean[self.target_variables].astype(float)
        var_model = VAR(self.y)
        self.var_result = var_model.fit(maxlags=self.maxlags)
        self.model = self.var_result
        self._compute_blanchard_quah()
        return self.var_result

    def _compute_blanchard_quah(self):
        k = self.y.shape[1]
        p = self.var_result.k_ar
        params = self.var_result.params
        intercept_offset = 1 if 'const' in params.index else 0
        sum_A = np.zeros((k, k))
        for lag in range(p):
            A_lag = params.iloc[intercept_offset + lag * k : intercept_offset + (lag + 1) * k, :].values.T
            sum_A += A_lag
        inv_phi = np.linalg.inv(np.eye(k) - sum_A)
        sigma_u = self.var_result.sigma_u
        if type(sigma_u) == pd.DataFrame:
            sigma_u = sigma_u.values
        long_run_cov = inv_phi @ sigma_u @ inv_phi.T
        lower_chol = np.linalg.cholesky(long_run_cov)
        self.B_0 = np.linalg.inv(inv_phi) @ lower_chol

    def get_summary(self):
        if self.var_result is None:
            raise ValueError("Model not trained")
        return f"Blanchard-Quah SVAR Model\nB_0 Matrix:\n{self.B_0}\n\nVAR Summary:\n{self.var_result.summary()}"

    def get_model_metrics(self):
        if self.var_result is None:
            raise ValueError("Model not trained")
        return {
            "nobs": int(self.var_result.nobs),
            "aic": float(self.var_result.aic),
            "bic": float(self.var_result.bic)
        }

    def run(self, data):
        self.fit(data)
        return self.get_model_metadata()
