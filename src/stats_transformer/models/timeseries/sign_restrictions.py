import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from stats_transformer.models.base import ModelBase

class SignRestrictionsSVARModel(ModelBase):

    def __init__(self, target_variables=None, sign_pattern=None, date_column=None, maxlags=1, max_draws=1000, **kwargs):
        target = target_variables[0] if target_variables else "dummy"
        indep = target_variables[1:] if target_variables and len(target_variables) > 1 else ["dummy"]
        super().__init__(target=target, independent_variables=indep, **kwargs)
        self.target_variables = target_variables or []
        self.sign_pattern = sign_pattern or []
        self.date_column = date_column
        self.time_column = date_column
        self.maxlags = maxlags
        self.max_draws = max_draws
        self.var_result = None
        self.accepted_rotations = []

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
        self.y = self.df_clean[self.target_variables]
        var_model = VAR(self.y)
        self.var_result = var_model.fit(maxlags=self.maxlags)
        self.model = self.var_result
        if self.sign_pattern:
            self._draw_sign_restrictions()
        return self.var_result

    def _draw_sign_restrictions(self):
        k = self.y.shape[1]
        sigma_u = self.var_result.sigma_u
        if type(sigma_u) == pd.DataFrame:
            sigma_u = sigma_u.values
        p_chol = np.linalg.cholesky(sigma_u)
        self.accepted_rotations = []
        for _ in range(self.max_draws):
            W = np.random.normal(size=(k, k))
            Q, R = np.linalg.qr(W)
            impact = p_chol @ Q
            valid = True
            for var_idx, target_sign in enumerate(self.sign_pattern):
                if target_sign == 1 and impact[var_idx, 0] <= 0:
                    valid = False
                    break
                elif target_sign == -1 and impact[var_idx, 0] >= 0:
                    valid = False
                    break
            if valid:
                self.accepted_rotations.append(impact)

    def get_summary(self):
        if self.var_result is None:
            raise ValueError("Model not trained")
        return f"Sign Restrictions SVAR Model\nAccepted Draws: {len(self.accepted_rotations)} / {self.max_draws}\n\nVAR Summary:\n{self.var_result.summary()}"

    def get_model_metrics(self):
        if self.var_result is None:
            raise ValueError("Model not trained")
        return {
            "nobs": int(self.var_result.nobs),
            "accepted_draws": len(self.accepted_rotations),
            "max_draws": self.max_draws
        }

    def run(self, data):
        self.fit(data)
        return self.get_model_metadata()
