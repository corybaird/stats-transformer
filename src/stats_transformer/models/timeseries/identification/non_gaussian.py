import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from statsmodels.tsa.api import VAR
from stats_transformer.models.base import ModelBase
from stats_transformer.models.timeseries.identification.alignment import align_to_cholesky


class NonGaussianSVARModel(ModelBase):
    """
    Data-driven SVAR identification via Non-Gaussian Maximum Likelihood (Lanne, Meitz, & Saikkonen 2010, R svars::id.ng).
    Assumes structural innovations follow independent Student-t distributions.
    """

    _is_multivariate = True

    def __init__(self, target_variables=None, date_column=None, maxlags=1, n_starts=5, seed=42, **kwargs):
        super().__init__(**kwargs)
        if target_variables:
            self.target_variables = target_variables
        elif self.params:
            self.target_variables = self.params.get("model", {}).get("target_variables", [])
            if not self.target_variables and self.target:
                self.target_variables = [self.target] + (self.independent_variables or [])
        else:
            self.target_variables = []

        model_params = self.params.get("model", {})
        self.date_column = date_column or model_params.get("date_column")
        self.maxlags = model_params.get("maxlags", maxlags)
        self.n_starts = model_params.get("n_starts", n_starts)
        self.seed = model_params.get("seed", seed)

        self.var_result = None
        self.structural_impact = None
        self.B_0 = None
        self.df_nu = None
        self.llf = None
        self.optimization_status = None

    def _get_required_columns(self):
        cols = list(self.target_variables)
        if self.date_column:
            cols.append(self.date_column)
        return cols

    def _construct_q(self, theta, K):
        Q = np.eye(K)
        idx = 0
        for i in range(K):
            for j in range(i + 1, K):
                G = np.eye(K)
                G[i, i] = np.cos(theta[idx])
                G[j, j] = np.cos(theta[idx])
                G[i, j] = -np.sin(theta[idx])
                G[j, i] = np.sin(theta[idx])
                Q = Q @ G
                idx += 1
        return Q

    def _log_likelihood_t(self, params, p_chol, resid):
        T, K = resid.shape
        n_angles = int(K * (K - 1) / 2)

        theta = params[:n_angles]
        nu = params[n_angles:]

        Q = self._construct_q(theta, K)
        B = p_chol @ Q

        try:
            B_inv = np.linalg.inv(B)
            sign, logdet = np.linalg.slogdet(B)
            if sign <= 0:
                return 1e10
        except np.linalg.LinAlgError:
            return 1e10

        e_t = (B_inv @ resid.T).T

        ll = -T * logdet
        for i in range(K):
            nu_i = nu[i]
            if nu_i <= 2.01:
                return 1e10
            c_nu = gammaln((nu_i + 1.0) / 2.0) - gammaln(nu_i / 2.0) - 0.5 * np.log(np.pi * (nu_i - 2.0))
            ll_i = T * c_nu - ((nu_i + 1.0) / 2.0) * np.sum(np.log(1.0 + (e_t[:, i] ** 2) / (nu_i - 2.0)))
            ll += ll_i

        return -ll

    def _identify_shocks(self):
        resid = self.var_result.resid.values if isinstance(self.var_result.resid, pd.DataFrame) else self.var_result.resid
        T, K = resid.shape
        n_angles = int(K * (K - 1) / 2)

        Sigma = np.cov(resid.T)
        p_chol = np.linalg.cholesky(Sigma)

        best_nll = np.inf
        best_params = None
        best_res = None
        rng = np.random.default_rng(self.seed)

        bounds = [(0, 2 * np.pi)] * n_angles + [(2.1, 50.0)] * K

        for _ in range(self.n_starts):
            theta_init = rng.uniform(0, 2 * np.pi, size=n_angles)
            nu_init = rng.uniform(4.0, 15.0, size=K)
            init_params = np.concatenate([theta_init, nu_init])

            res = minimize(
                self._log_likelihood_t,
                init_params,
                args=(p_chol, resid),
                method="L-BFGS-B",
                bounds=bounds
            )
            if res.fun < best_nll:
                best_nll = res.fun
                best_params = res.x
                best_res = res

        best_theta = best_params[:n_angles]
        best_nu = best_params[n_angles:]

        Q = self._construct_q(best_theta, K)
        B = p_chol @ Q

        self.structural_impact = align_to_cholesky(B, p_chol)
        self.B_0 = self.structural_impact
        self.df_nu = best_nu
        self.llf = -float(best_nll)
        self.optimization_status = best_res

    def build_model(self, drop_na=True):
        if getattr(self, "df_clean", None) is None:
            raise ValueError("No cleaned data available")
        if self.date_column and self.date_column in self.df_clean.columns:
            self.df_clean = self.df_clean.sort_values(self.date_column)

        self.y = self.df_clean[self.target_variables]
        var_model = VAR(self.y)
        self.var_result = var_model.fit(maxlags=self.maxlags)
        self.model = self.var_result
        self._identify_shocks()
        return self.var_result

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
        if self.structural_impact is None:
            raise ValueError("Model not trained")
        summary_text = (
            f"Non-Gaussian Maximum Likelihood SVAR (Lanne et al. 2010)\n"
            f"Optimization Success: {self.optimization_status.success}\n"
            f"Log-Likelihood: {self.llf:.4f}\n"
            f"Estimated Student-t Degrees of Freedom (nu): {np.round(self.df_nu, 3)}\n\n"
            f"Estimated Structural Impact Matrix B_0:\n{self.structural_impact}\n\n"
            f"Reduced-Form VAR Summary:\n{self.var_result.summary()}"
        )
        return summary_text

    def get_model_metrics(self):
        if self.structural_impact is None:
            raise ValueError("Model not trained")
        return {
            "nobs": int(self.var_result.nobs),
            "log_likelihood": float(self.llf),
            "opt_success": bool(self.optimization_status.success),
            "df_nu_mean": float(np.mean(self.df_nu))
        }
