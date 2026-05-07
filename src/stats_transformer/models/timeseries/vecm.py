import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM
from stats_transformer.models.base import ModelBase

class VECMModel(ModelBase):
    def __init__(self, target_variables=None, date_column=None, k_ar_diff=1, deterministic='n', **kwargs):
        super().__init__(target=target_variables[0] if target_variables else "dummy", independent_variables=["dummy"], **kwargs)
        self.target_variables = target_variables or []
        self.date_column = date_column
        self.k_ar_diff = k_ar_diff
        self.deterministic = deterministic

    def _get_required_columns(self):
        cols = list(self.target_variables)
        if self.date_column:
            cols.append(self.date_column)
        return cols

    def build_model(self, drop_na=True):
        if getattr(self, 'df_clean', None) is None:
            raise ValueError("No cleaned data available")
        
        if self.date_column and self.date_column in self.df_clean.columns:
            self.df_clean = self.df_clean.sort_values(self.date_column)
            
        self.y = self.df_clean[self.target_variables]
        
        self.vecm_spec = VECM(self.y, k_ar_diff=self.k_ar_diff, deterministic=self.deterministic)
        self.model = self.vecm_spec.fit()
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
            "num_observations": len(self.y)
        }
