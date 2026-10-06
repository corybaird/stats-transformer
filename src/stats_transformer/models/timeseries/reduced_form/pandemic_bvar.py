import numpy as np
import pandas as pd
from scipy.stats import invwishart
from stats_transformer.models.base import ModelBase

class PandemicBVARModel(ModelBase):
    _is_multivariate = True

    def __init__(self, target_variables=None, date_column=None, lags=1, pandemic_periods=None, volatility_scale=10.0, lambda1=0.2, lambda2=0.5, lambda3=1.0, lambda4=100.0, n_draws=1000, seed=42, **kwargs):
        super().__init__(**kwargs)
        model_params = self.params.get("model", {})
        self.target_variables = target_variables or getattr(self, "target_variables", []) or model_params.get("target_variables") or model_params.get("independent_variables", [])
        self.date_column = date_column or model_params.get("date_column")
        self.time_column = self.date_column
        self.lags = model_params.get("lags", lags)
        self.pandemic_periods = pandemic_periods or model_params.get("pandemic_periods", [])
        self.volatility_scale = model_params.get("volatility_scale", volatility_scale)
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
        self.n_vars = 0
        self.n_obs = 0

    def _get_required_columns(self):
        cols = list(self.target_variables)
        if self.date_column and self.date_column not in cols:
            cols.append(self.date_column)
        return cols

    def _build_design_matrices(self, Y, p):
        t_total = Y.shape[0]
        t_obs = t_total - p
        y = Y[p:]
        x_lags = np.column_stack([Y[p - lag: t_total - lag] for lag in range(1, p + 1)])
        x = np.column_stack([np.ones(t_obs), x_lags])
        return y, x

    def _ar1_residual_std(self, Y):
        sigmas = np.zeros(Y.shape[1])
        for i in range(Y.shape[1]):
            series = Y[:, i]
            y_t, y_lag = series[1:], series[:-1]
            x_mat = np.column_stack([np.ones(len(y_lag)), y_lag])
            coef, *_ = np.linalg.lstsq(x_mat, y_t, rcond=None)
            resid = y_t - x_mat @ coef
            sigmas[i] = np.std(resid, ddof=x_mat.shape[1]) if len(resid) > x_mat.shape[1] else np.std(resid) + 1e-6
        return np.maximum(sigmas, 1e-6)

    def _minnesota_prior(self, n_vars, p, sigma_hat):
        k = 1 + n_vars * p
        lambda_prior = np.zeros((k, n_vars))
        for i in range(n_vars):
            lambda_prior[1 + i, i] = 1.0
        v_diag = np.zeros(k)
        v_diag[0] = (sigma_hat.mean() * self.lambda4) ** 2
        for lag in range(1, p + 1):
            for j in range(n_vars):
                row = 1 + (lag - 1) * n_vars + j
                own_tightness = (self.lambda1 / (lag ** self.lambda3)) ** 2
                cross_tightness = own_tightness * (self.lambda2 ** 2)
                weighted_tightness = (own_tightness + (n_vars - 1) * cross_tightness) / n_vars if n_vars > 1 else own_tightness
                v_diag[row] = weighted_tightness * (sigma_hat[j] ** 2)
        v_prior = np.diag(v_diag)
        return lambda_prior, v_prior

    def _apply_pandemic_weights(self, y, X):
        t_obs = y.shape[0]
        weights = np.ones(t_obs)
        if self.date_column and self.pandemic_periods and self.date_column in self.df_clean.columns:
            dates = self.df_clean[self.date_column].values[self.lags:]
            pan_set = set(str(p) for p in self.pandemic_periods)
            for t_idx in range(t_obs):
                d_str = str(dates[t_idx])
                if any(p in d_str for p in pan_set):
                    weights[t_idx] = 1.0 / np.sqrt(self.volatility_scale)
        W = np.diag(weights)
        return W @ y, W @ X

    def build_model(self):
        if self.df_clean is None:
            raise ValueError("No cleaned data available")
        if self.date_column and self.date_column in self.df_clean.columns:
            self.df_clean = self.df_clean.sort_values(self.date_column)
        Y = self.df_clean[self.target_variables].to_numpy(dtype=float)
        n_obs_total, n_vars = Y.shape
        p = self.lags
        y_raw, X_raw = self._build_design_matrices(Y, p)
        y, X = self._apply_pandemic_weights(y_raw, X_raw)
        T, k = X.shape
        self.n_vars = n_vars
        self.n_obs = T
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
        self._draw_posterior()
        self.model = B_post
        return self.model

    def _draw_posterior(self):
        rng = np.random.default_rng(self.seed)
        n_vars = self.n_vars
        k_dim = self.posterior_B_mean.shape[0]
        self.Sigma_draws = invwishart.rvs(df=self.posterior_nu, scale=self.posterior_S, size=self.n_draws, random_state=rng)
        if self.n_draws == 1:
            self.Sigma_draws = self.Sigma_draws[None, ...]
        self.B_draws = np.zeros((self.n_draws, k_dim, n_vars))
        chol_V = np.linalg.cholesky(self.posterior_V)
        for d in range(self.n_draws):
            Sigma_d = self.Sigma_draws[d]
            Z = rng.standard_normal((k_dim, n_vars))
            chol_Sigma = np.linalg.cholesky(Sigma_d)
            self.B_draws[d] = self.posterior_B_mean + chol_V @ Z @ chol_Sigma.T

    def fit(self, df, drop_na=True):
        required_cols = self._get_required_columns()
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        self.df = df
        self.df_clean = df[required_cols].dropna().copy() if drop_na else df[required_cols].copy()
        return self.build_model()

    def predict(self, df):
        if self.posterior_B_mean is None:
            raise ValueError("Model not fitted")
        Y = df[self.target_variables].to_numpy(dtype=float)
        _, X = self._build_design_matrices(Y, self.lags)
        pred = X @ self.posterior_B_mean
        return pd.DataFrame(pred, columns=self.target_variables)

    def get_summary(self):
        if self.posterior_B_mean is None:
            raise ValueError("Model not fitted")
        return f"Pandemic BVAR Model (Cascaldi-Garcia 2022)\nVariables: {self.target_variables}\nPandemic Periods: {self.pandemic_periods}\nVolatility Scale: {self.volatility_scale}"

    def get_model_metrics(self):
        if self.posterior_B_mean is None:
            raise ValueError("Model not fitted")
        metrics = {}
        metrics["nobs"] = int(self.n_obs)
        metrics["nvars"] = int(self.n_vars)
        metrics["volatility_scale"] = float(self.volatility_scale)
        return metrics

    def run(self, data):
        self.fit(data)
        return self.get_model_metrics()
