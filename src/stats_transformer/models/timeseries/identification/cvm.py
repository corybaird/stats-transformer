import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tsa.api import VAR
from stats_transformer.models.base import ModelBase
from stats_transformer.models.timeseries.identification.alignment import align_to_cholesky


class CVMSVARModel(ModelBase):
    """
    Data-driven SVAR identification via Cramér-von Mises distance (Huo & Székely 2016, R svars::id.cvm).
    Minimizes the Cramér-von Mises independence criterion across recovered structural shock pairs.
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
        self.optimization_status = None

    def _get_required_columns(self):
        cols = list(self.target_variables)
        if self.date_column:
            cols.append(self.date_column)
        return cols

    def _cramer_von_mises_stat(self, x, y):
        T = len(x)
        x_le = (x[:, None] <= x[None, :])
        y_le = (y[:, None] <= y[None, :])

        F_joint = (x_le & y_le).mean(axis=1)
        F_x = x_le.mean(axis=1)
        F_y = y_le.mean(axis=1)

        diff = F_joint - F_x * F_y
        cvm_val = float(np.sum(diff ** 2) / T)
        return cvm_val

    def _objective(self, theta, p_chol, resid):
        K = resid.shape[1]
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

        B = p_chol @ Q
        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            return 1e10

        e_t = (B_inv @ resid.T).T
        total_cvm = 0.0
        for i in range(K):
            for j in range(i + 1, K):
                total_cvm += self._cramer_von_mises_stat(e_t[:, i], e_t[:, j])
        return total_cvm

    def _identify_shocks(self):
        resid = self.var_result.resid.values if isinstance(self.var_result.resid, pd.DataFrame) else self.var_result.resid
        K = resid.shape[1]
        n_angles = int(K * (K - 1) / 2)

        Sigma = np.cov(resid.T)
        p_chol = np.linalg.cholesky(Sigma)

        best_obj = np.inf
        best_theta = None
        best_res = None
        rng = np.random.default_rng(self.seed)

        for _ in range(self.n_starts):
            theta_init = rng.uniform(0, 2 * np.pi, size=n_angles)
            res = minimize(
                self._objective,
                theta_init,
                args=(p_chol, resid),
                method="L-BFGS-B",
                bounds=[(0, 2 * np.pi)] * n_angles
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_theta = res.x
                best_res = res

        Q = np.eye(K)
        idx = 0
        for i in range(K):
            for j in range(i + 1, K):
                G = np.eye(K)
                G[i, i] = np.cos(best_theta[idx])
                G[j, j] = np.cos(best_theta[idx])
                G[i, j] = -np.sin(best_theta[idx])
                G[j, i] = np.sin(best_theta[idx])
                Q = Q @ G
                idx += 1

        B = p_chol @ Q
        self.structural_impact = align_to_cholesky(B, p_chol)
        self.B_0 = self.structural_impact
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
            f"Cramér-von Mises Distance SVAR (Huo & Székely 2016)\n"
            f"Optimization Success: {self.optimization_status.success}\n\n"
            f"Estimated Structural Impact Matrix B_0:\n{self.structural_impact}\n\n"
            f"Reduced-Form VAR Summary:\n{self.var_result.summary()}"
        )
        return summary_text

    def get_model_metrics(self):
        if self.structural_impact is None:
            raise ValueError("Model not trained")
        return {
            "nobs": int(self.var_result.nobs),
            "opt_success": bool(self.optimization_status.success),
            "opt_criterion": float(self.optimization_status.fun)
        }
