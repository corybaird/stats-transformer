import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import cholesky
from statsmodels.tsa.vector_ar.vecm import VECM
from stats_transformer.models.base import ModelBase


def _orthogonal_complement(M: np.ndarray) -> np.ndarray:
    K, r = M.shape
    if r == 0:
        return np.eye(K)
    if r >= K:
        return np.empty((K, 0))
    q, _ = np.linalg.qr(M, mode="complete")
    return q[:, r:]


class SVEC:
    """
    Structural Vector Error Correction (SVEC) model.
    Estimates short-run (SR) and long-run (LR) structural impact matrices
    via Maximum Likelihood matching R vars::SVEC (King et al. 1991).
    """

    def __init__(self, vecm_result, SR=None, LR=None, max_iter=1000, r=None):
        self.vecm = vecm_result
        self.sigma_u = np.asarray(vecm_result.sigma_u)
        self.neqs = self.sigma_u.shape[0]
        self.nobs = getattr(vecm_result, "nobs", 100)

        self.coint_rank = r if r is not None else getattr(vecm_result, "coint_rank", 1)
        self.SR = np.asarray(SR, dtype=float) if SR is not None else np.full((self.neqs, self.neqs), np.nan)
        self.LR = np.asarray(LR, dtype=float) if LR is not None else np.full((self.neqs, self.neqs), np.nan)
        self.max_iter = max_iter

        self.beta = getattr(vecm_result, "beta", None)
        self.alpha = getattr(vecm_result, "alpha", None)
        self.gamma = getattr(vecm_result, "gamma", None)

        self.Xi = self._compute_long_run_impact_multiplier()
        self._fit_structural()

    def _compute_long_run_impact_multiplier(self) -> np.ndarray:
        K = self.neqs
        r = self.coint_rank
        if r == 0 or r >= K or self.alpha is None or self.beta is None:
            return np.eye(K)

        alpha_mat = np.asarray(self.alpha)
        beta_mat = np.asarray(self.beta)
        if alpha_mat.ndim == 1:
            alpha_mat = alpha_mat.reshape(K, r)
        if beta_mat.ndim == 1:
            beta_mat = beta_mat.reshape(K, r)

        alpha_orth = _orthogonal_complement(alpha_mat)
        beta_orth = _orthogonal_complement(beta_mat)

        if alpha_orth.shape[1] == 0 or beta_orth.shape[1] == 0:
            return np.eye(K)

        gamma_sum = np.zeros((K, K))
        if self.gamma is not None:
            gamma_arr = np.asarray(self.gamma)
            if gamma_arr.ndim == 3:
                gamma_sum = np.sum(gamma_arr, axis=0)
            elif gamma_arr.ndim == 2:
                if gamma_arr.shape == (K, K):
                    gamma_sum = gamma_arr
                elif gamma_arr.shape[0] == K and gamma_arr.shape[1] % K == 0:
                    p_diff = gamma_arr.shape[1] // K
                    for l in range(p_diff):
                        gamma_sum += gamma_arr[:, l * K : (l + 1) * K]
                elif gamma_arr.shape[1] == K and gamma_arr.shape[0] % K == 0:
                    p_diff = gamma_arr.shape[0] // K
                    for l in range(p_diff):
                        gamma_sum += gamma_arr[l * K : (l + 1) * K, :]

        Gamma = np.eye(K) - gamma_sum
        mid = alpha_orth.T @ Gamma @ beta_orth
        try:
            mid_inv = np.linalg.inv(mid)
            Xi = beta_orth @ mid_inv @ alpha_orth.T
        except np.linalg.LinAlgError:
            Xi = beta_orth @ np.linalg.pinv(mid) @ alpha_orth.T
        return Xi

    def _fit_structural(self):
        K = self.neqs
        T = self.nobs
        sigma_u = self.sigma_u
        Xi = self.Xi

        sr_mask = ~np.isnan(self.SR)
        lr_mask = ~np.isnan(self.LR)

        try:
            L = cholesky(sigma_u, lower=True)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(sigma_u)
            eigvals = np.maximum(eigvals, 1e-6)
            L = eigvecs @ np.diag(np.sqrt(eigvals))

        b0_init = L.flatten()

        def objective(b_flat):
            B = b_flat.reshape((K, K))
            try:
                sign, logdet = np.linalg.slogdet(B @ B.T)
                if sign <= 0 or logdet <= -100:
                    return 1e10
                inv_BB = np.linalg.inv(B @ B.T)
                nll = 0.5 * T * (logdet + np.trace(inv_BB @ sigma_u))
                return nll
            except (np.linalg.LinAlgError, ValueError):
                return 1e10

        constraints = []
        for i in range(K):
            for j in range(K):
                if sr_mask[i, j]:
                    target_val = float(self.SR[i, j])
                    idx = i * K + j
                    constraints.append({
                        "type": "eq",
                        "fun": lambda b, idx=idx, target_val=target_val: b[idx] - target_val
                    })

        for i in range(K):
            for j in range(K):
                if lr_mask[i, j]:
                    target_val = float(self.LR[i, j])
                    row_xi = Xi[i, :]
                    col_j = j
                    constraints.append({
                        "type": "eq",
                        "fun": lambda b, row_xi=row_xi, col_j=col_j, target_val=target_val: np.dot(row_xi, b.reshape((K, K))[:, col_j]) - target_val
                    })

        opt_res = minimize(
            objective,
            b0_init,
            method="SLSQP",
            constraints=constraints,
            options={"maxiter": self.max_iter, "ftol": 1e-9}
        )

        if not opt_res.success:
            opt_res = minimize(
                objective,
                b0_init,
                method="Nelder-Mead",
                options={"maxiter": self.max_iter * 2}
            )

        B_est = opt_res.x.reshape((K, K))
        for i in range(K):
            for j in range(K):
                if sr_mask[i, j]:
                    B_est[i, j] = self.SR[i, j]

        self.SR_est = B_est
        self.LR_est = Xi @ B_est
        self.opt_res = opt_res
        self.bse = np.full((K, K), np.nan)

        try:
            sign, logdet_bb = np.linalg.slogdet(B_est @ B_est.T)
            inv_bb = np.linalg.inv(B_est @ B_est.T)
            self.llf = -0.5 * T * (K * np.log(2 * np.pi) + logdet_bb + np.trace(inv_bb @ sigma_u))
        except (np.linalg.LinAlgError, ValueError):
            self.llf = -np.inf

        sign_s, logdet_s = np.linalg.slogdet(sigma_u)
        self.llf_unrestricted = -0.5 * T * (K * np.log(2 * np.pi) + logdet_s + K)
        self.lr_stat = max(0.0, 2.0 * (self.llf_unrestricted - self.llf))

    def get_structural_matrices(self):
        return {
            "SR": self.SR_est,
            "LR": self.LR_est
        }


class SVECModel(ModelBase):
    _is_multivariate = True

    def __init__(self, target_variables=None, date_column=None, k_ar_diff=1, coint_rank=1, deterministic="n", SR=None, LR=None, max_iter=1000, **kwargs):
        super().__init__(**kwargs)
        if target_variables:
            self.target_variables = target_variables
        elif self.params:
            self.target_variables = self.params.get("model", {}).get("target_variables", [])
            if not self.target_variables and self.target:
                self.target_variables = [self.target] + (self.independent_variables or [])
        else:
            self.target_variables = []

        self.date_column = date_column
        self.k_ar_diff = k_ar_diff
        self.coint_rank = coint_rank
        self.deterministic = deterministic
        self.SR = SR
        self.LR = LR
        self.max_iter = max_iter
        self.svec_fit = None

    def _get_required_columns(self):
        cols = list(self.target_variables)
        if self.date_column:
            cols.append(self.date_column)
        return cols

    def build_model(self, drop_na=True):
        if getattr(self, "df_clean", None) is None:
            raise ValueError("No cleaned data available")
        if self.date_column and self.date_column in self.df_clean.columns:
            self.df_clean = self.df_clean.sort_values(self.date_column)

        self.y = self.df_clean[self.target_variables]
        self.vecm_spec = VECM(self.y, k_ar_diff=self.k_ar_diff, coint_rank=self.coint_rank, deterministic=self.deterministic)
        self.vecm_result = self.vecm_spec.fit()
        self.model = self.vecm_result
        self.svec_fit = SVEC(self.vecm_result, SR=self.SR, LR=self.LR, max_iter=self.max_iter, r=self.coint_rank)
        self.B_0 = self.svec_fit.SR_est
        self.LR_est = self.svec_fit.LR_est
        return self.svec_fit

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
        if self.svec_fit is None:
            raise ValueError("Model not trained")
        summary_text = (
            f"SVEC Structural Error Correction Model\n"
            f"Cointegration Rank: {self.coint_rank}\n"
            f"Log-Likelihood: {self.svec_fit.llf:.4f}\n"
            f"LR Over-identification Test Stat: {self.svec_fit.lr_stat:.4f}\n\n"
            f"Estimated Short-Run Impact Matrix B (SR):\n{self.svec_fit.SR_est}\n\n"
            f"Estimated Long-Run Impact Matrix (LR = Xi @ B):\n{self.svec_fit.LR_est}"
        )
        return summary_text

    def get_model_metrics(self):
        if self.svec_fit is None:
            raise ValueError("Model not trained")
        return {
            "num_observations": len(self.y),
            "log_likelihood": float(self.svec_fit.llf),
            "coint_rank": int(self.coint_rank),
            "lr_stat": float(self.svec_fit.lr_stat)
        }
