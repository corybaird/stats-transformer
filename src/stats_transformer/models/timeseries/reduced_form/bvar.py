import numpy as np
import pandas as pd
from scipy.stats import invwishart
from stats_transformer.models.base import ModelBase

class BVARModel(ModelBase):
    _is_multivariate = True

    def __init__(self, target_variables=None, date_column=None, lags=1, lambda1=0.2, lambda2=0.5, lambda3=1.0, lambda4=100.0, n_draws=1000, seed=42, **kwargs):
        super().__init__(**kwargs)
        model_params = self.params.get("model", {})
        self.target_variables = target_variables or getattr(self, "target_variables", []) or model_params.get("target_variables") or model_params.get("independent_variables", [])
        self.date_column = date_column or model_params.get("date_column")
        self.time_column = self.date_column
        self.lags = model_params.get("lags", lags)
        self.lambda1 = model_params.get("lambda1", lambda1)
        self.lambda2 = model_params.get("lambda2", lambda2)
        self.lambda3 = model_params.get("lambda3", lambda3)
        self.lambda4 = model_params.get("lambda4", lambda4)
        self.n_draws = model_params.get("n_draws", n_draws)
        self.seed = model_params.get("seed", seed)
        self.posterior_B_mean = None
        self.posterior_V = None
        self.posterior_S = None
        self.posterior_nu = None
        self.B_draws = None
        self.Sigma_draws = None

    def _get_required_columns(self):
        cols = list(self.target_variables)
        if self.date_column and self.date_column not in cols:
            cols.append(self.date_column)
        return cols

    def build_model(self):
        if self.df_clean is None:
            raise ValueError("No cleaned data available")
        if self.date_column and self.date_column in self.df_clean.columns:
            self.df_clean = self.df_clean.sort_values(self.date_column)

        Y = self.df_clean[self.target_variables].to_numpy(dtype=float)
        n_obs_total, n_vars = Y.shape
        p = self.lags

        y, X = self._build_design_matrices(Y, p)
        T, k = X.shape

        sigma_hat = self._ar1_residual_std(Y)
        Lambda_prior, V_prior = self._minnesota_prior(n_vars, p, sigma_hat)
        S_prior = np.diag(sigma_hat ** 2)
        nu_prior = n_vars + 2

        V_prior_inv = np.linalg.inv(V_prior)
        V_post_inv = V_prior_inv + X.T @ X
        V_post = np.linalg.inv(V_post_inv)
        B_post = V_post @ (V_prior_inv @ Lambda_prior + X.T @ y)

        resid_prior = Lambda_prior.T @ V_prior_inv @ Lambda_prior
        resid_post = B_post.T @ V_post_inv @ B_post
        S_post = S_prior + y.T @ y + resid_prior - resid_post
        S_post = 0.5 * (S_post + S_post.T)
        nu_post = nu_prior + T

        self.posterior_B_mean = B_post
        self.posterior_V = V_post
        self.posterior_S = S_post
        self.posterior_nu = nu_post
        self.n_vars = n_vars
        self.n_obs = T

        self._draw_posterior()

        self.model = {"B_mean": B_post, "V": V_post, "S": S_post, "nu": nu_post}
        return self.model

    def _build_design_matrices(self, Y, p):
        T_total = Y.shape[0]
        n_vars = Y.shape[1]
        T = T_total - p
        y = Y[p:]
        X_lags = np.column_stack([Y[p - lag: T_total - lag] for lag in range(1, p + 1)])
        X = np.column_stack([np.ones(T), X_lags])
        return y, X

    def _ar1_residual_std(self, Y):
        sigmas = np.zeros(Y.shape[1])
        for i in range(Y.shape[1]):
            series = Y[:, i]
            y_t, y_lag = series[1:], series[:-1]
            X = np.column_stack([np.ones(len(y_lag)), y_lag])
            coef, *_ = np.linalg.lstsq(X, y_t, rcond=None)
            resid = y_t - X @ coef
            sigmas[i] = np.std(resid, ddof=X.shape[1]) if len(resid) > X.shape[1] else np.std(resid) + 1e-6
        return np.maximum(sigmas, 1e-6)

    def _minnesota_prior(self, n_vars, p, sigma_hat):
        # Natural-conjugate (Kadiyala & Karlsson 1997) Minnesota prior: the
        # regressor-variance matrix V is shared across all equations (a
        # requirement of the conjugate NIW form, since X is identical for
        # every equation), so "own" vs. "cross" tightness is a property of
        # each *regressor column* alone -- V does not vary by equation.
        # Column j at lag l is that regressor's own-lag coefficient in
        # variable j's equation and a cross-lag coefficient everywhere else,
        # so we price it by its role as a regressor: tightness shrinks with
        # lag length, and lambda2 governs the average cross-variable prior
        # relative to the own-variable prior across the whole system.
        k = 1 + n_vars * p
        Lambda_prior = np.zeros((k, n_vars))
        for i in range(n_vars):
            Lambda_prior[1 + i, i] = 1.0

        V_diag = np.zeros(k)
        V_diag[0] = (sigma_hat.mean() * self.lambda4) ** 2
        for lag in range(1, p + 1):
            for j in range(n_vars):
                row = 1 + (lag - 1) * n_vars + j
                own_tightness = (self.lambda1 / (lag ** self.lambda3)) ** 2
                cross_tightness = own_tightness * (self.lambda2 ** 2)
                weighted_tightness = (own_tightness + (n_vars - 1) * cross_tightness) / n_vars if n_vars > 1 else own_tightness
                V_diag[row] = weighted_tightness * (sigma_hat[j] ** 2)

        V_prior = np.diag(V_diag)
        return Lambda_prior, V_prior

    def _draw_posterior(self):
        rng = np.random.default_rng(self.seed)
        n_vars, k = self.n_vars, self.posterior_B_mean.shape[0]
        self.Sigma_draws = invwishart.rvs(df=self.posterior_nu, scale=self.posterior_S, size=self.n_draws, random_state=rng)
        if self.n_draws == 1:
            self.Sigma_draws = self.Sigma_draws[None, ...]
        self.B_draws = np.zeros((self.n_draws, k, n_vars))
        chol_V = np.linalg.cholesky(self.posterior_V)
        for d in range(self.n_draws):
            Sigma_d = self.Sigma_draws[d]
            Z = rng.standard_normal((k, n_vars))
            chol_Sigma = np.linalg.cholesky(Sigma_d)
            self.B_draws[d] = self.posterior_B_mean + chol_V @ Z @ chol_Sigma.T

    def get_summary(self):
        if self.model is None:
            raise ValueError("Model not trained")
        return f"Conjugate BVAR (Minnesota prior): n_vars={self.n_vars}, lags={self.lags}, n_obs={self.n_obs}, n_draws={self.n_draws}, lambda1={self.lambda1}"

    def get_model_metrics(self):
        if self.model is None:
            raise ValueError("Model not trained")
        return {
            "n_vars": self.n_vars,
            "lags": self.lags,
            "n_obs": self.n_obs,
            "n_draws": self.n_draws,
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
            "lambda3": self.lambda3,
            "lambda4": self.lambda4,
            "posterior_nu": float(self.posterior_nu),
        }

    def compute_irf(self, horizon=10, response=None, shock=None):
        if self.B_draws is None:
            raise ValueError("Model not trained")
        n_vars, p = self.n_vars, self.lags
        response_idx = self.target_variables.index(response) if response else 0
        shock_idx = self.target_variables.index(shock) if shock else 0

        rows = []
        for h in range(horizon + 1):
            draws_h = np.zeros(self.n_draws)
            for d in range(self.n_draws):
                B = self.B_draws[d]
                Sigma = self.Sigma_draws[d]
                companion, shock_vec = self._companion_form(B, Sigma, n_vars, p, shock_idx)
                irf_matrix = np.linalg.matrix_power(companion, h)
                draws_h[d] = (irf_matrix @ shock_vec)[response_idx]
            estimate = float(np.mean(draws_h))
            lower, upper = float(np.percentile(draws_h, 16)), float(np.percentile(draws_h, 84))
            rows.append({"horizon": h, "response": response, "shock": shock, "estimate": estimate, "std_error": float(np.std(draws_h)), "lower": lower, "upper": upper})
        return pd.DataFrame(rows)

    def _companion_form(self, B, Sigma, n_vars, p, shock_idx):
        A_coefs = B[1:].T.reshape(n_vars, p, n_vars)
        top = np.column_stack([A_coefs[:, lag, :] for lag in range(p)])
        companion = np.zeros((n_vars * p, n_vars * p))
        companion[:n_vars, :] = top
        if p > 1:
            companion[n_vars:, : n_vars * (p - 1)] = np.eye(n_vars * (p - 1))
        chol_Sigma = np.linalg.cholesky(Sigma)
        shock_vec = np.zeros(n_vars * p)
        shock_vec[:n_vars] = chol_Sigma[:, shock_idx]
        return companion, shock_vec

    def run(self, data):
        self.fit(data)
        return self.get_model_metadata()
