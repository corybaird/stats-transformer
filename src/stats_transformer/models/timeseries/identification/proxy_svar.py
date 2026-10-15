import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from stats_transformer.models.base import ModelBase

class ProxySVARModel(ModelBase):

    _is_multivariate = True

    def __init__(self, target_variables=None, instrument_variable=None, date_column=None, maxlags=1, **kwargs):
        super().__init__(**kwargs)
        self.target_variables = target_variables or []
        self.instrument_variable = instrument_variable
        self.date_column = date_column
        self.time_column = date_column
        self.maxlags = maxlags
        self.var_result = None
        self.impact_column = None
        self._instrument_missing = None

    def _get_required_columns(self):
        cols = list(self.target_variables)
        if self.instrument_variable and self.instrument_variable not in cols:
            cols.append(self.instrument_variable)
        if self.date_column:
            cols.append(self.date_column)
        return cols

    def load_data(self, data):
        # The reduced-form VAR is estimated on the full sample of endogenous
        # variables; rows where the instrument is missing only drop out of the
        # proxy moments. Mask the instrument's NaNs so ModelBase.load_data's
        # row cleaning keeps them, and record which rows are missing (indexed
        # the same way df_clean will be) so the true values can be restored.
        if isinstance(data, str):
            data = pd.read_csv(data) if data.endswith(".csv") else pd.read_parquet(data)
        else:
            data = data.copy()
        if self.instrument_variable and self.instrument_variable in data.columns:
            missing = data[self.instrument_variable].isna()
            if missing.any():
                data[self.instrument_variable] = data[self.instrument_variable].fillna(0.0)
                key_col = self.time_column or ('date' if 'date' in data.columns else None)
                key = data[key_col] if key_col else data.index
                self._instrument_missing = pd.Series(missing.to_numpy(), index=key)
        return super().load_data(data)

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
        # restore the instrument's true values (load_data masked the NaNs)
        z_col = self.df_clean[self.instrument_variable]
        if self._instrument_missing is not None:
            z_col = z_col.where(~self._instrument_missing.loc[self.df_clean.index])
            self.df_clean[self.instrument_variable] = z_col
        z = z_col.loc[residuals.index].to_numpy()
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
