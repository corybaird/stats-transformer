import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from stats_transformer.models.base import ModelBase

class GMMModel(ModelBase):

    def __init__(self, target=None, independent_variables=None, endogenous=None, instruments=None, method="two_step", weighting="hac", bandwidth=None, iter_tol=1e-6, max_iter=100, **kwargs):
        independent_variables = independent_variables or []
        super().__init__(target=target, independent_variables=independent_variables, **kwargs)
        model_params = self.params.get("model", {}) if self.params else {}
        self.endogenous = model_params.get("endogenous", endogenous or [])
        self.instruments = model_params.get("instruments", instruments or [])
        self.method = model_params.get("method", method)
        self.weighting = model_params.get("weighting", weighting)
        self.bandwidth = model_params.get("bandwidth", bandwidth)
        self.iter_tol = iter_tol
        self.max_iter = max_iter
        self.exog_names = list(self.independent_variables)
        self.endog_names = list(self.endogenous)
        self.param_names = ["const"] + self.exog_names + self.endog_names

    def _get_required_columns(self):
        columns = list(self.independent_variables) + [self.target]
        for col in self.endogenous + self.instruments:
            if col not in columns:
                columns.append(col)
        if getattr(self, "entity_column", None) and self.entity_column not in columns:
            columns.append(self.entity_column)
        if getattr(self, "time_column", None) and self.time_column not in columns:
            columns.append(self.time_column)
        return columns

    def build_model(self):
        if self.df_clean is None:
            raise ValueError("No cleaned data available")

        y = self.df_clean[self.target].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(self.df_clean))] + [self.df_clean[c].to_numpy(dtype=float) for c in self.exog_names + self.endog_names])
        Z = np.column_stack([np.ones(len(self.df_clean))] + [self.df_clean[c].to_numpy(dtype=float) for c in self.exog_names] + [self.df_clean[c].to_numpy(dtype=float) for c in self.instruments])

        n, k = X.shape
        m = Z.shape[1]
        if m < k:
            raise ValueError(f"Model is underidentified: {m} instruments/moments for {k} parameters")

        self.y, self.X_mat, self.Z_mat = y, X, Z
        self.n_obs, self.n_params, self.n_moments = n, k, m

        beta_2sls = self._two_stage_least_squares(y, X, Z)

        if self.method == "one_step":
            beta_final = beta_2sls
            W_final = np.eye(m)
        elif self.method == "two_step":
            W1 = np.eye(m)
            beta_1 = self._gmm_estimate(y, X, Z, W1)
            W_final = self._weighting_matrix(y, X, Z, beta_1)
            beta_final = self._gmm_estimate(y, X, Z, W_final)
        elif self.method == "iterated":
            beta_prev = beta_2sls
            W = self._weighting_matrix(y, X, Z, beta_prev)
            for _ in range(self.max_iter):
                beta_new = self._gmm_estimate(y, X, Z, W)
                if np.max(np.abs(beta_new - beta_prev)) < self.iter_tol:
                    beta_prev = beta_new
                    break
                beta_prev = beta_new
                W = self._weighting_matrix(y, X, Z, beta_prev)
            beta_final = beta_prev
            W_final = W
        elif self.method == "cue":
            beta_final, W_final = self._continuously_updated_estimate(y, X, Z, beta_2sls)
        else:
            raise ValueError(f"Unknown GMM method '{self.method}'")

        self.beta = beta_final
        self.W = W_final
        self.residuals = y - X @ beta_final
        self.moments = Z * self.residuals[:, None]
        self.gbar = self.moments.mean(axis=0)

        S = self._spectral_density(self.moments)
        S_inv = np.linalg.pinv(S)
        G = -(Z.T @ X) / n
        avar = np.linalg.pinv(G.T @ S_inv @ G) / n
        self.vcov = avar
        self.se = np.sqrt(np.diag(avar))

        self.j_stat = float(n * self.gbar @ S_inv @ self.gbar)
        self.j_df = m - k
        self.j_pvalue = float(1 - stats.chi2.cdf(self.j_stat, self.j_df)) if self.j_df > 0 else None

        self.model = {"beta": beta_final, "W": W_final, "vcov": avar}
        return self.model

    def _two_stage_least_squares(self, y, X, Z):
        ZtZ_inv = np.linalg.pinv(Z.T @ Z)
        proj = Z @ ZtZ_inv @ Z.T
        XtPX = X.T @ proj @ X
        XtPy = X.T @ proj @ y
        return np.linalg.solve(XtPX, XtPy)

    def _gmm_estimate(self, y, X, Z, W):
        ZtX = Z.T @ X
        ZtY = Z.T @ y
        A = ZtX.T @ W @ ZtX
        b = ZtX.T @ W @ ZtY
        return np.linalg.solve(A, b)

    def _weighting_matrix(self, y, X, Z, beta):
        resid = y - X @ beta
        moments = Z * resid[:, None]
        S = self._spectral_density(moments)
        return np.linalg.pinv(S)

    def _spectral_density(self, moments):
        n = moments.shape[0]
        moments_centered = moments - moments.mean(axis=0)
        S = (moments_centered.T @ moments_centered) / n
        if self.weighting == "hac":
            bandwidth = self.bandwidth if self.bandwidth is not None else int(np.floor(4 * (n / 100) ** (2 / 9)))
            for lag in range(1, bandwidth + 1):
                weight = 1 - lag / (bandwidth + 1)
                gamma = (moments_centered[lag:].T @ moments_centered[:-lag]) / n
                S += weight * (gamma + gamma.T)
        return S

    def _continuously_updated_estimate(self, y, X, Z, beta_init):
        def objective(beta):
            resid = y - X @ beta
            moments = Z * resid[:, None]
            gbar = moments.mean(axis=0)
            S = self._spectral_density(moments)
            S_inv = np.linalg.pinv(S)
            return float(gbar @ S_inv @ gbar)

        result = minimize(objective, beta_init, method="BFGS")
        beta_final = result.x
        resid = y - X @ beta_final
        moments = Z * resid[:, None]
        S = self._spectral_density(moments)
        W_final = np.linalg.pinv(S)
        return beta_final, W_final

    def get_summary(self):
        if self.model is None:
            raise ValueError("Model not trained")
        lines = [f"GMM Estimation (method={self.method}, weighting={self.weighting})"]
        for name, coef, se in zip(self.param_names, self.beta, self.se):
            lines.append(f"  {name}: {coef:.6f} (se={se:.6f})")
        lines.append(f"J-statistic: {self.j_stat:.4f}, df={self.j_df}, p-value={self.j_pvalue}")
        return "\n".join(lines)

    def get_model_metrics(self):
        if self.model is None:
            raise ValueError("Model not trained")
        return {
            "method": self.method,
            "coefficients": dict(zip(self.param_names, self.beta.tolist())),
            "std_errors": dict(zip(self.param_names, self.se.tolist())),
            "j_statistic": self.j_stat,
            "j_df": self.j_df,
            "j_pvalue": self.j_pvalue,
            "num_observations": self.n_obs,
            "num_moments": self.n_moments,
            "num_params": self.n_params,
        }

    def run(self, data):
        self.fit(data)
        return self.get_model_metadata()
