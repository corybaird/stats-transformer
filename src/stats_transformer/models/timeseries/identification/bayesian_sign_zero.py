import numpy as np
import pandas as pd
import yaml
from itertools import permutations
from scipy.stats import invwishart
from stats_transformer.models.base import ModelBase

class BayesianSignZeroSVARModel(ModelBase):
    _is_multivariate = True

    def __init__(self, target_variables=None, config_path=None, date_column=None, lags=1, lambda1=0.2, lambda2=0.5, lambda3=1.0, lambda4=100.0, n_draws=500, max_rotations_per_draw=20, required_accepts=50, enable_permutation=True, seed=42, **kwargs):
        super().__init__(**kwargs)
        model_params = self.params.get("model", {})
        self.target_variables = target_variables or getattr(self, "target_variables", []) or model_params.get("target_variables") or model_params.get("independent_variables", [])
        self.config_path = config_path or model_params.get("config_path")
        self.date_column = date_column or model_params.get("date_column")
        self.time_column = self.date_column
        self.lags = model_params.get("lags", lags)
        self.lambda1 = model_params.get("lambda1", lambda1)
        self.lambda2 = model_params.get("lambda2", lambda2)
        self.lambda3 = model_params.get("lambda3", lambda3)
        self.lambda4 = model_params.get("lambda4", lambda4)
        self.n_draws = model_params.get("n_draws", n_draws)
        self.max_rotations_per_draw = model_params.get("max_rotations_per_draw", max_rotations_per_draw)
        self.required_accepts = model_params.get("required_accepts", required_accepts)
        self.enable_permutation = model_params.get("enable_permutation", enable_permutation)
        self.seed = model_params.get("seed", seed)
        self.shocks = []
        self.restrictions = []
        self.narrative_restrictions = []
        self.posterior_B_mean = None
        self.posterior_V = None
        self.posterior_S = None
        self.posterior_nu = None
        self.B_draws = None
        self.Sigma_draws = None
        self.accepted_rotations = []
        self.accepted_irfs = []
        self.accepted_B = []
        self.accepted_Sigma = []
        self.total_draws = 0
        self.n_obs = 0
        if self.config_path:
            self._load_config()

    def _load_config(self):
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        self.shocks = config.get("shocks", [])
        self.restrictions = config.get("restrictions", [])
        self.narrative_restrictions = config.get("narrative_restrictions", [])

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

    def _compute_ma_matrices(self, B_matrix, steps, n_vars, p):
        phis = np.zeros((steps, n_vars, n_vars))
        phis[0] = np.eye(n_vars)
        lag_coefs = [B_matrix[1 + l * n_vars: 1 + (l + 1) * n_vars, :].T for l in range(p)]
        for h in range(1, steps):
            acc = np.zeros((n_vars, n_vars))
            for l in range(min(h, p)):
                acc += lag_coefs[l] @ phis[h - 1 - l]
            phis[h] = acc
        return phis

    def _evaluate_candidate(self, candidate_irfs, shock_to_idx, var_to_idx):
        for r in self.restrictions:
            s_idx = shock_to_idx[r["shock"]]
            v_idx = var_to_idx[r["response"]]
            r_type = r["type"]
            h_def = r.get("horizon", 0)
            horizons = range(h_def[0], h_def[1] + 1) if type(h_def) == list else [h_def]
            for h in horizons:
                val = candidate_irfs[h, v_idx, s_idx]
                if r_type == "sign":
                    if r["value"] == "+" and val <= 0:
                        return False
                    if r["value"] == "-" and val >= 0:
                        return False
                elif r_type == "zero":
                    if abs(val) > 1e-10:
                        return False
        return True

    def _check_narrative(self, Q, p_chol_inv, residuals, shock_to_idx):
        u_t = residuals.T
        e_t = Q.T @ p_chol_inv @ u_t
        dates = self.df_clean[self.date_column].values if self.date_column else np.arange(u_t.shape[1])
        date_to_idx = {d: i for i, d in enumerate(dates[-u_t.shape[1]:])}
        for r in self.narrative_restrictions:
            s_idx = shock_to_idx[r["shock"]]
            target_date = r["date"]
            if target_date not in date_to_idx:
                return False
            val = e_t[s_idx, date_to_idx[target_date]]
            if r["type"] == "sign":
                if r["value"] == "+" and val <= 0:
                    return False
                if r["value"] == "-" and val >= 0:
                    return False
        return True

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
        self._sample_and_identify(X, y, n_vars, p)
        self.model = B_post
        return self.model

    def _sample_and_identify(self, X, y, n_vars, p):
        rng = np.random.default_rng(self.seed)
        k_dim = self.posterior_B_mean.shape[0]
        chol_V = np.linalg.cholesky(self.posterior_V)
        var_to_idx = {v: i for i, v in enumerate(self.target_variables)}
        shock_to_idx = {s: i for i, s in enumerate(self.shocks)} if self.shocks else {v: i for i, v in enumerate(self.target_variables)}
        max_h = 0
        for r in self.restrictions:
            h_val = r.get("horizon", 0)
            max_h = max(max_h, h_val[1] if type(h_val) == list else h_val)
        steps = max_h + 1
        draws = 0
        accepts = 0
        self.accepted_rotations = []
        self.accepted_irfs = []
        self.accepted_B = []
        self.accepted_Sigma = []
        while draws < self.n_draws and accepts < self.required_accepts:
            draws += 1
            Sigma_d = invwishart.rvs(df=self.posterior_nu, scale=self.posterior_S, random_state=rng)
            Z = rng.standard_normal((k_dim, n_vars))
            chol_Sigma = np.linalg.cholesky(Sigma_d)
            B_d = self.posterior_B_mean + chol_V @ Z @ chol_Sigma.T
            phis = self._compute_ma_matrices(B_d, steps, n_vars, p)
            u_resid = y - X @ B_d
            p_chol = np.linalg.cholesky(Sigma_d)
            p_chol_inv = np.linalg.inv(p_chol)
            for _ in range(self.max_rotations_per_draw):
                W = rng.normal(size=(n_vars, n_vars))
                Q, R = np.linalg.qr(W)
                Q = Q @ np.diag(np.sign(np.diag(R)))
                impact = p_chol @ Q
                candidate_irfs = np.zeros((steps, n_vars, n_vars))
                for h in range(steps):
                    candidate_irfs[h] = phis[h] @ impact
                matched_Q = None
                matched_irfs = None
                if not self.restrictions:
                    matched_Q = Q
                    matched_irfs = candidate_irfs
                elif self._evaluate_candidate(candidate_irfs, shock_to_idx, var_to_idx):
                    matched_Q = Q
                    matched_irfs = candidate_irfs
                elif self.enable_permutation and n_vars <= 8:
                    for perm in permutations(range(n_vars)):
                        perm_Q = Q[:, perm]
                        perm_impact = p_chol @ perm_Q
                        perm_irfs = np.zeros((steps, n_vars, n_vars))
                        for h in range(steps):
                            perm_irfs[h] = phis[h] @ perm_impact
                        if self._evaluate_candidate(perm_irfs, shock_to_idx, var_to_idx):
                            matched_Q = perm_Q
                            matched_irfs = perm_irfs
                            break
                if matched_Q is not None:
                    valid_narrative = True
                    if self.narrative_restrictions:
                        valid_narrative = self._check_narrative(matched_Q, p_chol_inv, u_resid, shock_to_idx)
                    if valid_narrative:
                        self.accepted_rotations.append(matched_Q)
                        self.accepted_irfs.append(matched_irfs)
                        self.accepted_B.append(B_d)
                        self.accepted_Sigma.append(Sigma_d)
                        accepts += 1
                        break
        self.total_draws = draws

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
        p = self.lags
        _, X = self._build_design_matrices(Y, p)
        pred = X @ self.posterior_B_mean
        return pd.DataFrame(pred, columns=self.target_variables)

    def compute_fevd(self, steps=20):
        if not self.accepted_rotations:
            raise ValueError("No accepted draws available")
        n_vars = len(self.target_variables)
        n_acc = len(self.accepted_rotations)
        all_fevd = np.zeros((n_acc, steps, n_vars, n_vars))
        for idx in range(n_acc):
            B_d = self.accepted_B[idx]
            Q_d = self.accepted_rotations[idx]
            Sigma_d = self.accepted_Sigma[idx]
            p_chol = np.linalg.cholesky(Sigma_d)
            impact = p_chol @ Q_d
            phis = self._compute_ma_matrices(B_d, steps, n_vars, self.lags)
            structural_ma = np.zeros((steps, n_vars, n_vars))
            for h in range(steps):
                structural_ma[h] = phis[h] @ impact
            for h in range(steps):
                for i in range(n_vars):
                    for j in range(n_vars):
                        contrib = np.sum([structural_ma[tau, i, j] ** 2 for tau in range(h + 1)])
                        all_fevd[idx, h, i, j] = contrib
                    total_var = np.sum(all_fevd[idx, h, i, :])
                    if total_var > 0:
                        all_fevd[idx, h, i, :] /= total_var
        return np.median(all_fevd, axis=0)

    def compute_hd(self):
        if not self.accepted_rotations:
            raise ValueError("No accepted draws available")
        rep = self.get_representative_draw()
        idx = rep["index"]
        B_d = self.accepted_B[idx]
        Q_d = self.accepted_rotations[idx]
        Sigma_d = self.accepted_Sigma[idx]
        p_chol = np.linalg.cholesky(Sigma_d)
        impact = p_chol @ Q_d
        inv_impact = np.linalg.inv(impact)
        Y = self.df_clean[self.target_variables].to_numpy(dtype=float)
        y, X = self._build_design_matrices(Y, self.lags)
        residuals = y - X @ B_d
        t_obs, n_vars = residuals.shape
        structural_shocks = (inv_impact @ residuals.T).T
        phis = self._compute_ma_matrices(B_d, t_obs, n_vars, self.lags)
        hd = np.zeros((t_obs, n_vars, n_vars))
        for t in range(t_obs):
            for j in range(n_vars):
                for tau in range(t + 1):
                    imp = phis[t - tau] @ impact[:, j]
                    hd[t, :, j] += imp * structural_shocks[tau, j]
        return hd, structural_shocks

    def get_representative_draw(self):
        if not self.accepted_irfs:
            raise ValueError("No accepted draws available")
        all_irfs = np.array(self.accepted_irfs)
        median_irf = np.median(all_irfs, axis=0)
        std_irf = np.std(all_irfs, axis=0)
        std_irf[std_irf == 0] = 1e-10
        distances = np.sum(((all_irfs - median_irf) / std_irf) ** 2, axis=(1, 2, 3))
        best_idx = int(np.argmin(distances))
        result = {}
        result["index"] = best_idx
        result["rotation"] = self.accepted_rotations[best_idx]
        result["irf"] = self.accepted_irfs[best_idx]
        return result

    def get_summary(self):
        if self.posterior_B_mean is None:
            raise ValueError("Model not fitted")
        n_acc = len(self.accepted_rotations)
        return f"Bayesian Sign/Zero SVAR Model\nAccepted Draws: {n_acc} / {self.total_draws}\nVariables: {self.target_variables}"

    def get_model_metrics(self):
        if self.posterior_B_mean is None:
            raise ValueError("Model not fitted")
        metrics = {}
        metrics["nobs"] = int(self.n_obs)
        metrics["accepted_draws"] = int(len(self.accepted_rotations))
        metrics["total_draws"] = int(self.total_draws)
        return metrics

    def run(self, data):
        self.fit(data)
        return self.get_model_metrics()
