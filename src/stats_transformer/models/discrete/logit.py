import numpy as np
import pandas as pd
import statsmodels.api as sm
from stats_transformer.models.base import ModelBase

class LogitModel(ModelBase):
    def __init__(self, target=None, independent_variables=None, **kwargs):
        super().__init__(target=target, independent_variables=independent_variables, **kwargs)

    def _get_required_columns(self):
        return self.independent_variables + [self.target]

    def build_model(self, drop_na=True):
        if self.df_clean is None:
            raise ValueError("No cleaned data available")

        self.y = self.df_clean[self.target]
        self.X = self.df_clean[self.independent_variables]
        self.X = sm.add_constant(self.X)

        self.logit_spec = sm.Logit(self.y, self.X)
        self.model = self.logit_spec.fit(disp=False)
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
            "pseudo_r_squared": self.model.prsquared,
            "llr_pvalue": self.model.llr_pvalue,
            "aic": self.model.aic,
            "bic": self.model.bic,
            "num_observations": self.model.nobs,
        }
