import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
from stats_transformer.models.base import ModelBase

class LocalProjectionsIVModel(ModelBase):

    def __init__(self, target_variable=None, shock_variable=None, instrument_variable=None, control_variables=None, horizons=10, date_column=None, **kwargs):
        controls = control_variables or []
        super().__init__(target=target_variable or "dummy", independent_variables=[shock_variable or "dummy"] + controls, **kwargs)
        self.target_variable = target_variable
        self.shock_variable = shock_variable
        self.instrument_variable = instrument_variable
        self.control_variables = controls
        self.horizons = horizons
        self.date_column = date_column
        self.time_column = date_column
        self.irf_coefficients = []
        self.irf_std_errors = []

    def _get_required_columns(self):
        cols = [self.target_variable, self.shock_variable, self.instrument_variable] + list(self.control_variables)
        if self.date_column:
            cols.append(self.date_column)
        return list(dict.fromkeys(cols))

    def build_model(self):
        if getattr(self, 'df_clean', None) is None:
            raise ValueError("No cleaned data available")
        if self.date_column and self.date_column in self.df_clean.columns:
            self.df_clean = self.df_clean.sort_values(self.date_column)
        self._estimate_lp_iv()
        return self

    def _estimate_lp_iv(self):
        self.irf_coefficients = []
        self.irf_std_errors = []
        df = self.df_clean.copy()
        for h in range(self.horizons + 1):
            y_h = df[self.target_variable].shift(-h)
            x_shock = df[self.shock_variable]
            z_inst = df[self.instrument_variable]
            controls = df[self.control_variables].values if self.control_variables else None
            if controls is not None:
                exog = np.column_stack([x_shock, controls, np.ones(len(df))])
                instruments = np.column_stack([z_inst, controls, np.ones(len(df))])
            else:
                exog = np.column_stack([x_shock, np.ones(len(df))])
                instruments = np.column_stack([z_inst, np.ones(len(df))])
            valid_mask = ~np.isnan(y_h) & ~np.isnan(x_shock) & ~np.isnan(z_inst)
            if controls is not None:
                valid_mask = valid_mask & ~np.isnan(controls).any(axis=1)
            y_valid = y_h[valid_mask].values
            exog_valid = exog[valid_mask]
            inst_valid = instruments[valid_mask]
            iv_res = IV2SLS(y_valid, exog_valid, inst_valid).fit()
            self.irf_coefficients.append(float(iv_res.params[0]))
            self.irf_std_errors.append(float(iv_res.bse[0]))

    def get_summary(self):
        return f"Local Projections IV Model\nHorizons: {self.horizons}\nIRF Coefficients: {self.irf_coefficients}"

    def get_model_metrics(self):
        return {
            "horizons": self.horizons,
            "irf_h0": self.irf_coefficients[0] if self.irf_coefficients else None,
            "irf_h_max": self.irf_coefficients[-1] if self.irf_coefficients else None
        }

    def run(self, data):
        self.fit(data)
        return self.get_model_metadata()
