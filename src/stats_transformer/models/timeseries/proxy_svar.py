import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from stats_transformer.models.base import ModelBase

class ProxySVARModel(ModelBase):

    def __init__(self, target_variables=None, instrument_variable=None, date_column=None, maxlags=1, **kwargs):
        target = target_variables[0] if target_variables else "dummy"
        indep = target_variables[1:] if target_variables and len(target_variables) > 1 else ["dummy"]
        super().__init__(target=target, independent_variables=indep, **kwargs)
        self.target_variables = target_variables or []
        self.instrument_variable = instrument_variable
        self.date_column = date_column
        self.time_column = date_column
        self.maxlags = maxlags
        self.var_result = None
        self.impact_column = None

    def _get_required_columns(self):
        cols = list(self.target_variables)
        if self.instrument_variable and self.instrument_variable not in cols:
            cols.append(self.instrument_variable)
        if self.date_column:
            cols.append(self.date_column)
        return cols

    def build_model(self):
        if getattr(self, 'df_clean', None) is None:
            raise ValueError("No cleaned data available")
        if self.date_column and self.date_column in self.df_clean.columns:
            self.df_clean = self.df_clean.sort_values(self.date_column)
        self.y = self.df_clean[self.target_variables]
        var_model = VAR(self.y)
        self.var_result = var_model.fit(maxlags=self.maxlags)
        self.model = self.var_result
        if self.instrument_variable in self.df_clean.columns:
            self._estimate_proxy_svar()
        return self.var_result

    def _estimate_proxy_svar(self):
        residuals = self.var_result.resid
        z = self.df_clean.loc[residuals.index, self.instrument_variable].values
        u1 = residuals.iloc[:, 0].values
        valid_mask = ~np.isnan(z) & ~np.isnan(u1)
        z_valid = z[valid_mask]
        u1_valid = u1[valid_mask]
        stage1 = sm.OLS(u1_valid, sm.add_constant(z_valid)).fit()
        u1_hat = stage1.fittedvalues
        k = residuals.shape[1]
        b1_estimates = np.zeros(k)
        b1_estimates[0] = 1.0
        for i in range(1, k):
            ui_valid = residuals.iloc[:, i].values[valid_mask]
            stage2 = sm.OLS(ui_valid, u1_hat).fit()
            b1_estimates[i] = stage2.params[0]
        sigma_u = self.var_result.sigma_u
        if type(sigma_u) == pd.DataFrame:
            sigma_u = sigma_u.values
        sig11 = sigma_u[0, 0]
        scale = np.sqrt(max(sig11, 1e-8))
        self.impact_column = b1_estimates * scale

    def get_summary(self):
        if self.var_result is None:
            raise ValueError("Model not trained")
        return f"Proxy SVAR Model (Instrument: {self.instrument_variable})\nStructural Impact Column:\n{self.impact_column}\n\nVAR Summary:\n{self.var_result.summary()}"

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
