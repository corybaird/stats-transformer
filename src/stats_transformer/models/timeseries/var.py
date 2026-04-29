import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from src.stats_transformer.models.base import ModelBase

class VARModel(ModelBase):
    def __init__(self, target_variables=None, date_column=None, maxlags=None, ic=None, **kwargs):
        super().__init__(target=target_variables[0] if target_variables else "dummy", independent_variables=["dummy"], **kwargs)
        self.target_variables = target_variables or []
        self.date_column = date_column
        self.maxlags = maxlags
        self.ic = ic  # e.g., 'aic', 'bic', 'hqic', 'fpe'

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
            "aic": self.model.aic,
            "bic": self.model.bic,
            "hqic": self.model.hqic,
            "fpe": self.model.fpe,
            "num_observations": self.model.nobs,
        }
