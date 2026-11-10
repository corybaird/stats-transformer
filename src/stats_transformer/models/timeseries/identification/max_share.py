import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from stats_transformer.models.base import ModelBase

class MaxShareSVARModel(ModelBase):

    _is_multivariate = True

    def __init__(self, target_variables=None, date_column=None, maxlags=1, target_variable=None, target_index=0, min_horizon=0, max_horizon=40, zero_impact=False, **kwargs):
        super().__init__(**kwargs)
        self.target_variables = target_variables or getattr(self, "target_variables", [])
        self.date_column = self._param("date_column", date_column, None)
        self.time_column = self.date_column
        self.maxlags = self._param("maxlags", maxlags, 1)
        self.target_variable = self._param("target_variable", target_variable, None)
        self.target_index = self._param("target_index", target_index, 0)
        self.min_horizon = self._param("min_horizon", min_horizon, 0)
        self.max_horizon = self._param("max_horizon", max_horizon, 40)
        self.zero_impact = self._param("zero_impact", zero_impact, False)
        self.var_result = None
        self.impact_matrix = None
        self.Q = None
        self.max_share = None

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
        self.y = self.df_clean[self.target_variables].astype(float)
        self.var_result = VAR(self.y).fit(maxlags=self.maxlags)
        self.model = self.var_result
        self._identify_max_share()
        return self.var_result

    def _resolve_target_index(self):
        if self.target_variable is not None:
            if self.target_variable not in self.target_variables:
                raise ValueError(f"target_variable '{self.target_variable}' not in target_variables")
            return self.target_variables.index(self.target_variable)
        return int(self.target_index)

    def _identify_max_share(self):
        k = self.y.shape[1]
        i = self._resolve_target_index()
        if not (0 <= i < k):
            raise ValueError(f"target_index {i} out of range for {k} variables")
        P = np.linalg.cholesky(np.asarray(self.var_result.sigma_u))
        Phi = self.var_result.ma_rep(self.max_horizon)
        e = np.zeros(k)
        e[i] = 1.0
        V = np.zeros((k, k))
        for h in range(self.min_horizon, self.max_horizon + 1):
            vec = e @ Phi[h] @ P
            V += vec[:, None] @ vec[None, :]
        if self.zero_impact:
            # Barsky-Sims news shock: e_i' P q = 0, so maximize q'Vq on the
            # orthogonal complement of c = P' e_i instead of the full K-space.
            c = P.T @ e
            if k == 1:
                raise ValueError("zero_impact requires at least two variables")
            basis, _ = np.linalg.qr(np.column_stack([c, np.eye(k)]))
            Z = basis[:, 1:]
            vals, vecs = np.linalg.eigh(Z.T @ V @ Z)
            q = Z @ vecs[:, np.argmax(vals)]
        else:
            vals, vecs = np.linalg.eigh(V)
            q = vecs[:, np.argmax(vals)]
        # sign convention: the shock raises the target variable by max_horizon
        if (e @ Phi[self.max_horizon] @ P @ q) < 0:
            q = -q
        # complete [q, Q_perp]: QR of [q, I] has first column proportional to q,
        # the rest spans the orthogonal complement
        Q_perp, _ = np.linalg.qr(np.column_stack([q, np.eye(k)]))
        if Q_perp[:, 0] @ q < 0:
            Q_perp[:, 0] = -Q_perp[:, 0]
        self.Q = np.column_stack([q, Q_perp[:, 1:]])
        self.impact_matrix = P @ self.Q
        # V is PSD and indexed by Cholesky shocks; trace(V) is the cumulative
        # FEV of the target variable over the horizon window, so the share is
        # lambda_max / trace <= 1
        self.max_share = float(vals[-1] / np.trace(V)) if np.trace(V) > 0 else 0.0

    def irf(self, steps=20):
        if self.impact_matrix is None:
            raise ValueError("Model not trained")
        # rows (horizon, variable), columns structural shocks
        Phi = self.var_result.ma_rep(steps - 1)
        arr = Phi @ self.impact_matrix
        index = pd.MultiIndex.from_product([range(steps), self.target_variables], names=["horizon", "variable"])
        return pd.DataFrame(arr.reshape(steps * arr.shape[1], arr.shape[2]), index=index, columns=[f"shock_{j + 1}" for j in range(arr.shape[2])])

    def fevd(self, steps=20):
        if self.impact_matrix is None:
            raise ValueError("Model not trained")
        Phi = self.var_result.ma_rep(steps - 1)
        responses = Phi @ self.impact_matrix
        contrib = np.einsum('hij,hij->ij', responses, responses)
        total = contrib.sum(axis=1, keepdims=True)
        shares = np.divide(contrib, total, out=np.zeros_like(contrib), where=total > 0)
        cols = [f"shock_{j + 1}" for j in range(shares.shape[1])]
        return pd.DataFrame(shares, index=self.target_variables, columns=cols)

    def get_summary(self):
        if self.var_result is None:
            raise ValueError("Model not trained")
        target = self.target_variables[self._resolve_target_index()]
        return (f"Max-Share SVAR Model (target: {target}, horizons {self.min_horizon}-{self.max_horizon}, zero_impact: {self.zero_impact})\n"
                f"Max share of target FEV: {self.max_share}\n"
                f"Structural Impact Matrix:\n{self.impact_matrix}\n\nVAR Summary:\n{self.var_result.summary()}")

    def get_model_metadata(self, metrics=None):
        metadata = super().get_model_metadata(metrics)
        if self.impact_matrix is not None:
            metadata["impact_matrix"] = np.asarray(self.impact_matrix).tolist()
        if self.max_share is not None:
            metadata["max_share"] = self.max_share
        return metadata

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
