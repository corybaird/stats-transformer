import numpy as np
import pandas as pd
from scipy.optimize import minimize
from stats_transformer.models.base import ModelBase


class STVARModel(ModelBase):
    """
    Smooth Transition Vector Autoregression (STVAR) model.
    Dynamics transition continuously between two states based on a logistic
    smooth transition function G(st-d; gamma, c) (Teräsvirta & Yang 2014).
    """

    _is_multivariate = True

    def __init__(self, target_variables=None, date_column=None, transition_variable=None, lags=1, delay=1, gamma=None, c=None, intercept=True, max_iter=500, **kwargs):
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
        self.transition_variable = transition_variable
        self.lags = lags
        self.delay = delay
        self.gamma = gamma
        self.c = c
        self.intercept = intercept
        self.max_iter = max_iter

        self.regime_1_params = None
        self.regime_2_params = None
        self.transition_weights = None
        self.residuals = None
        self.fittedvalues = None

    def _get_required_columns(self):
        cols = list(self.target_variables)
        if self.transition_variable and self.transition_variable not in cols:
            cols.append(self.transition_variable)
        if self.date_column and self.date_column not in cols:
            cols.append(self.date_column)
        return cols

    def _construct_lag_matrix(self, y_mat, tr_series):
        T, K = y_mat.shape
        p = self.lags
        d = self.delay
        max_lag = max(p, d)

        Y_eff = y_mat[max_lag:]
        T_eff = len(Y_eff)

        X_list = []
        if self.intercept:
            X_list.append(np.ones((T_eff, 1)))

        for lag in range(1, p + 1):
            X_list.append(y_mat[max_lag - lag : T - lag])

        X_eff = np.hstack(X_list)
        tr_eff = tr_series[max_lag - d : T - d]
        return Y_eff, X_eff, tr_eff

    def _logistic_weight(self, tr_eff, gamma, c, scale):
        z = -gamma * (tr_eff - c) / max(1e-6, scale)
        z = np.clip(z, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(z))

    def _estimate_conditional_ols(self, Y, X, G):
        G_col = G.reshape((-1, 1))
        X1 = (1.0 - G_col) * X
        X2 = G_col * X
        X_aug = np.hstack([X1, X2])

        try:
            params = np.linalg.lstsq(X_aug, Y, rcond=None)[0]
            fitted = X_aug @ params
            resids = Y - fitted
            ssr = np.sum(resids ** 2)
            m = X.shape[1]
            p1 = params[:m, :]
            p2 = params[m:, :]
            return p1, p2, fitted, resids, ssr
        except Exception:
            return None, None, None, None, np.inf

    def _optimize_transition_parameters(self, Y_eff, X_eff, tr_eff):
        scale = np.std(tr_eff) if np.std(tr_eff) > 1e-6 else 1.0
        c_candidates = np.percentile(tr_eff, [20, 35, 50, 65, 80])
        gamma_candidates = [0.5, 1.0, 2.0, 5.0, 10.0]

        best_ssr = np.inf
        best_init = (1.0, float(np.median(tr_eff)))

        for g in gamma_candidates:
            for c_val in c_candidates:
                G = self._logistic_weight(tr_eff, g, c_val, scale)
                _, _, _, _, ssr = self._estimate_conditional_ols(Y_eff, X_eff, G)
                if ssr < best_ssr:
                    best_ssr = ssr
                    best_init = (g, c_val)

        def objective(params):
            g_opt, c_opt = params[0], params[1]
            if g_opt <= 0:
                return 1e10
            G = self._logistic_weight(tr_eff, g_opt, c_opt, scale)
            _, _, _, _, ssr = self._estimate_conditional_ols(Y_eff, X_eff, G)
            return ssr

        res = minimize(
            objective,
            best_init,
            bounds=[(0.001, 50.0), (np.min(tr_eff), np.max(tr_eff))],
            method="L-BFGS-B",
            options={"maxiter": self.max_iter}
        )

        if res.success:
            return float(res.x[0]), float(res.x[1])
        return float(best_init[0]), float(best_init[1])

    def build_model(self, drop_na=True):
        if getattr(self, "df_clean", None) is None:
            raise ValueError("No cleaned data available")
        if self.date_column and self.date_column in self.df_clean.columns:
            self.df_clean = self.df_clean.sort_values(self.date_column)

        y_df = self.df_clean[self.target_variables]
        y_mat = y_df.values.astype(float)

        tr_var_name = self.transition_variable or self.target_variables[0]
        tr_series = self.df_clean[tr_var_name].values.astype(float)

        Y_eff, X_eff, tr_eff = self._construct_lag_matrix(y_mat, tr_series)
        T_eff, K = Y_eff.shape
        self.scale = float(np.std(tr_eff)) if np.std(tr_eff) > 1e-6 else 1.0

        if self.gamma is None or self.c is None:
            self.gamma, self.c = self._optimize_transition_parameters(Y_eff, X_eff, tr_eff)

        self.transition_weights = self._logistic_weight(tr_eff, self.gamma, self.c, self.scale)
        p1, p2, fitted, resids, ssr = self._estimate_conditional_ols(Y_eff, X_eff, self.transition_weights)

        self.regime_1_params = p1
        self.regime_2_params = p2
        self.fittedvalues = pd.DataFrame(fitted, columns=self.target_variables)
        self.residuals = pd.DataFrame(resids, columns=self.target_variables)
        self.total_ssr = float(ssr)
        self.nobs = T_eff

        k_params = X_eff.shape[1] * K * 2 + 2
        self.aic = float(T_eff * np.log(self.total_ssr / T_eff) + 2 * k_params)
        self.bic = float(T_eff * np.log(self.total_ssr / T_eff) + np.log(T_eff) * k_params)

        lag_names = []
        if self.intercept:
            lag_names.append("const")
        for lag in range(1, self.lags + 1):
            for v in self.target_variables:
                lag_names.append(f"L{lag}.{v}")

        self.regime_1_df = pd.DataFrame(self.regime_1_params, index=lag_names, columns=self.target_variables)
        self.regime_2_df = pd.DataFrame(self.regime_2_params, index=lag_names, columns=self.target_variables)
        self.model = self
        self.params = self.regime_1_df
        return self

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
        if self.regime_1_params is None:
            raise ValueError("Model not trained")
        summary_text = (
            f"Smooth Transition Vector Autoregression (STVAR)\n"
            f"Lags: {self.lags}, Delay: {self.delay}\n"
            f"Transition Variable: {self.transition_variable or self.target_variables[0]}\n"
            f"Smoothness Gamma: {self.gamma:.4f}, Location c: {self.c:.4f}\n"
            f"Mean Transition Weight: {np.mean(self.transition_weights):.4f}\n"
            f"AIC: {self.aic:.4f}, BIC: {self.bic:.4f}, SSR: {self.total_ssr:.4f}\n\n"
            f"Regime 1 Baseline Coefficients (G = 0):\n{self.regime_1_df}\n\n"
            f"Regime 2 Upper Coefficients (G = 1):\n{self.regime_2_df}"
        )
        return summary_text

    def get_model_metrics(self):
        if self.regime_1_params is None:
            raise ValueError("Model not trained")
        return {
            "num_observations": int(self.nobs),
            "gamma": float(self.gamma),
            "c": float(self.c),
            "lags": int(self.lags),
            "delay": int(self.delay),
            "aic": float(self.aic),
            "bic": float(self.bic),
            "ssr": float(self.total_ssr),
            "mean_weight": float(np.mean(self.transition_weights))
        }
