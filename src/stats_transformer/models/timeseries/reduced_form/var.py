import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from stats_transformer.models.base import ModelBase

class VARModel(ModelBase):
    _is_multivariate = True

    def __init__(self, target_variables=None, date_column=None, maxlags=None, ic=None, mask=None, **kwargs):
        super().__init__(**kwargs)
        model_params = self.params.get("model", {})
        self.target_variables = target_variables or getattr(self, "target_variables", []) or model_params.get("target_variables") or model_params.get("independent_variables", [])
        self.date_column = date_column or model_params.get("date_column")
        self.maxlags = maxlags or model_params.get("maxlags") or model_params.get("lags")
        self.ic = ic or model_params.get("ic")
        self.mask = mask if mask is not None else model_params.get("mask")

    def _get_required_columns(self):
        cols = list(self.target_variables)
        if self.date_column:
            cols.append(self.date_column)
        return cols

    def build_model(self, drop_na=True):
        if self.df_clean is None:
            raise ValueError("No cleaned data available")
        
        if self.date_column and self.date_column in self.df_clean.columns:
            self.df_clean = self.df_clean.sort_values(self.date_column)
            
        self.y = self.df_clean[self.target_variables]
        
        # Initialize and fit
        if self.mask is not None:
            from stats_transformer.models.timeseries.reduced_form.restrictions import RestrictedVAR
            # We assume a fixed maxlags if mask is provided, as IC selection doesn't apply to masks
            lags = self.maxlags if self.maxlags is not None else 1
            self.model = RestrictedVAR(self.y, mask=self.mask, maxlags=lags, trend="c").fit()
            self.var_spec = None
        else:
            self.var_spec = VAR(self.y)
            self.model = self.var_spec.fit(maxlags=self.maxlags, ic=self.ic)
        return self.model

    def fit(self, df, drop_na=True):
        required_cols = self._get_required_columns()
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        self.df_clean = df.copy()
        if drop_na:
            self.df_clean = self.df_clean.replace([np.inf, -np.inf], np.nan).dropna(subset=required_cols)

        self.build_model(drop_na=False)
        return self.get_model_metrics()

    def get_summary(self):
        if self.model is None:
            raise ValueError("Model not trained")
        return str(self.model.summary())

    def get_model_metrics(self):
        if self.model is None:
            raise ValueError("Model not trained")
        return {
            "aic": getattr(self.model, "aic", None),
            "bic": getattr(self.model, "bic", None),
            "hqic": getattr(self.model, "hqic", None),
            "fpe": getattr(self.model, "fpe", None),
            "num_observations": getattr(self.model, "nobs", None),
        }
