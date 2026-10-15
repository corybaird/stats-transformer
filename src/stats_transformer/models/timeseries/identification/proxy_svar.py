import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from stats_transformer.models.base import ModelBase

class ProxySVARModel(ModelBase):

    _is_multivariate = True

    def __init__(self, target_variables=None, instrument_variable=None, date_column=None, maxlags=1, **kwargs):
        super().__init__(**kwargs)
        self.target_variables = target_variables or getattr(self, "target_variables", [])
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

    def load_data(self, data):
        # The reduced-form VAR is estimated on the full sample of endogenous
        # variables: rows where the instrument is missing must survive ModelBase's
        # NaN-drop, so the instrument is held out of the shared row cleaning and
        # its true values (NaNs included) are read from self.df during estimation.
        instrument = self.instrument_variable
        if instrument:
            self.instrument_variable = None
        try:
            cleaned = super().load_data(data)
        finally:
            self.instrument_variable = instrument
        if instrument and instrument not in self.df.columns:
            raise ValueError(f"Required columns missing: ['{instrument}']")
        return cleaned

    def build_model(self):
        if getattr(self, 'df_clean', None) is None:
            raise ValueError("No cleaned data available")
        if self.date_column and self.date_column in self.df_clean.columns:
            self.df_clean = self.df_clean.sort_values(self.date_column)
        self.y = self.df_clean[self.target_variables]
        var_model = VAR(self.y)
        self.var_result = var_model.fit(maxlags=self.maxlags)
        self.model = self.var_result
        if self.instrument_variable in self.df.columns:
            self._estimate_proxy_svar()
        return self.var_result

    def _estimate_proxy_svar(self):
        residuals = self.var_result.resid
        # mirror ModelBase.load_data's index choice: a literal 'date' column
        # (with no date_column param) becomes df_clean's index, otherwise row
        # labels are the input frame's
        index_col = None if self.time_column else ('date' if 'date' in self.df.columns else None)
        if index_col:
            z = self.df.set_index(index_col)[self.instrument_variable].loc[residuals.index].to_numpy()
        else:
            z = self.df.loc[residuals.index, self.instrument_variable].to_numpy()
        u = residuals.to_numpy()
        # Gertler-Karadi (2015) / Mertens-Ravn (2014): with a proxy-external
        # instrument the impact ratios are the demeaned moments relative to the
        # instrumented residual, and the shock size follows the closed form
        # below; both use the common sample where the instrument is observed
        common = ~np.isnan(z) & ~np.isnan(u[:, 0])
        uc = u[common] - u[common].mean(axis=0)
        zc = z[common] - z[common].mean()
        n = int(common.sum())
        nparams = self.var_result.params.shape[0]
        sigma = uc.T @ uc / (n - nparams)
        s21 = (uc[:, 1:].T @ zc) / (uc[:, 0] @ zc)
        sig11 = sigma[0, 0]
        sig21 = sigma[1:, 0]
        sig22 = sigma[1:, 1:]
        q = np.outer(s21, s21) * sig11 - (np.outer(sig21, s21) + np.outer(s21, sig21)) + sig22
        s12s12 = (sig21 - s21 * sig11) @ np.linalg.solve(q, sig21 - s21 * sig11)
        b11 = np.sqrt(sig11 - s12s12)
        self.impact_column = np.r_[b11, b11 * s21]

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
