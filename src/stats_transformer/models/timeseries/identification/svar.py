import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.svar_model import SVAR
from stats_transformer.models.base import ModelBase

class SVARModel(ModelBase):
    _is_multivariate = True

    def __init__(self, target_variables=None, date_column=None, svar_type='A', A=None, B=None, maxlags=None, **kwargs):
        super().__init__(**kwargs)
        self.target_variables = target_variables or []
        self.date_column = date_column
        self.svar_type = svar_type
        self.A = np.asarray(A) if A is not None else None
        self.B = np.asarray(B) if B is not None else None
        self.maxlags = maxlags

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
        k = self.y.shape[1]
        if self.svar_type == 'A' and self.A is None:
            A_mask = np.zeros((k, k), dtype=object)
            for i in range(k):
                for j in range(k):
                    if i == j:
                        A_mask[i, j] = 1.0
                    elif i > j:
                        A_mask[i, j] = 'E'
                    else:
                        A_mask[i, j] = 0.0
            self.A = A_mask
        elif self.svar_type == 'B' and self.B is None:
            B_mask = np.zeros((k, k), dtype=object)
            for i in range(k):
                for j in range(k):
                    if i >= j:
                        B_mask[i, j] = 'E'
                    else:
                        B_mask[i, j] = 0.0
            self.B = B_mask
        self.svar_spec = SVAR(self.y, svar_type=self.svar_type, A=self.A, B=self.B)
        self.model = self.svar_spec.fit(maxlags=self.maxlags)
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
        try:
            return str(self.model.summary())
        except AttributeError:
            return f"SVAR Model Results:\nEstimated A Matrix:\n{self.model.A}\n\nEstimated B Matrix:\n{self.model.B}"

    def get_model_metrics(self):
        if self.model is None:
            raise ValueError("Model not trained")
        return {
            "num_observations": self.model.nobs if hasattr(self.model, 'nobs') else len(self.y)
        }
