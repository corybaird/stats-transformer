import numpy as np
import pandas as pd
from stats_transformer.models.base import ModelBase


class TVARModel(ModelBase):
    """
    Two-Regime Threshold Vector Autoregression (TVAR) model.
    Dynamics switch between two linear regimes based on a threshold variable
    st-d relative to estimated threshold gamma (Hansen 2011, Balke 2000).
    """

    _is_multivariate = True

    def __init__(self, target_variables=None, date_column=None, threshold_variable=None, lags=1, delay=1, trim=0.15, gamma=None, intercept=True, **kwargs):
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
        self.threshold_variable = threshold_variable or model_params.get("threshold_variable")
        self.lags = model_params.get("lags", lags)
        self.delay = model_params.get("delay", delay)
        self.trim = model_params.get("trim", trim)
        self.gamma = model_params.get("gamma", gamma)
        self.intercept = model_params.get("intercept", intercept)

        self.regime_1_params = None
        self.regime_2_params = None
        self.regime_1_cov = None
        self.regime_2_cov = None
        self.regime_shares = None
        self.residuals = None
        self.fittedvalues = None

    def _get_required_columns(self):
        cols = list(self.target_variables)
        if self.threshold_variable and self.threshold_variable not in cols:
            cols.append(self.threshold_variable)
        if self.date_column and self.date_column not in cols:
            cols.append(self.date_column)
        return cols

    def _construct_lag_matrix(self, y_mat, th_series):
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
        th_eff = th_series[max_lag - d : T - d]
        return Y_eff, X_eff, th_eff

    def _estimate_regime_ols(self, Y, X):
        try:
            params = np.linalg.lstsq(X, Y, rcond=None)[0]
            resids = Y - X @ params
            cov = (resids.T @ resids) / max(1, len(Y) - X.shape[1])
            return params, resids, cov
        except Exception:
            return None, None, None

    def _grid_search_threshold(self, Y_eff, X_eff, th_eff):
        T_eff, K = Y_eff.shape
        num_regressors = X_eff.shape[1]
        min_obs = max(num_regressors + 2, int(np.floor(self.trim * T_eff)))

        unique_th = np.sort(np.unique(th_eff))
        lower_idx = int(np.floor(self.trim * len(unique_th)))
        upper_idx = int(np.ceil((1.0 - self.trim) * len(unique_th)))

        candidate_gammas = unique_th[lower_idx:upper_idx]
        if len(candidate_gammas) == 0:
            candidate_gammas = [np.median(th_eff)]

        best_gamma = candidate_gammas[0]
        best_ssr = np.inf

        for cand in candidate_gammas:
            mask1 = th_eff <= cand
            mask2 = ~mask1

            if np.sum(mask1) < min_obs or np.sum(mask2) < min_obs:
                continue

            Y1, X1 = Y_eff[mask1], X_eff[mask1]
            Y2, X2 = Y_eff[mask2], X_eff[mask2]

            p1, r1, _ = self._estimate_regime_ols(Y1, X1)
            p2, r2, _ = self._estimate_regime_ols(Y2, X2)

            if p1 is None or p2 is None:
                continue

            ssr = np.sum(r1 ** 2) + np.sum(r2 ** 2)
            if ssr < best_ssr:
                best_ssr = ssr
                best_gamma = cand

        return best_gamma

    def build_model(self, drop_na=True):
        if getattr(self, "df_clean", None) is None:
            raise ValueError("No cleaned data available")
        if self.date_column and self.date_column in self.df_clean.columns:
            self.df_clean = self.df_clean.sort_values(self.date_column)

        y_df = self.df_clean[self.target_variables]
        y_mat = y_df.values.astype(float)

        th_var_name = self.threshold_variable or self.target_variables[0]
        th_series = self.df_clean[th_var_name].values.astype(float)

        Y_eff, X_eff, th_eff = self._construct_lag_matrix(y_mat, th_series)
        T_eff, K = Y_eff.shape

        if self.gamma is None:
            self.gamma = self._grid_search_threshold(Y_eff, X_eff, th_eff)

        mask1 = th_eff <= self.gamma
        mask2 = ~mask1

        if np.sum(mask1) == 0 or np.sum(mask2) == 0:
            mask1 = np.ones(T_eff, dtype=bool)
            mask2 = np.ones(T_eff, dtype=bool)

        p1, r1, cov1 = self._estimate_regime_ols(Y_eff[mask1], X_eff[mask1])
        p2, r2, cov2 = self._estimate_regime_ols(Y_eff[mask2], X_eff[mask2])

        self.regime_1_params = p1
        self.regime_2_params = p2
        self.regime_1_cov = cov1
        self.regime_2_cov = cov2
        self.regime_shares = {
            "regime_1": float(np.mean(mask1)),
            "regime_2": float(np.mean(mask2))
        }

        fitted = np.zeros_like(Y_eff)
        resids = np.zeros_like(Y_eff)
        fitted[mask1] = X_eff[mask1] @ p1
        fitted[mask2] = X_eff[mask2] @ p2
        resids = Y_eff - fitted

        self.fittedvalues = pd.DataFrame(fitted, columns=self.target_variables)
        self.residuals = pd.DataFrame(resids, columns=self.target_variables)
        self.total_ssr = float(np.sum(resids ** 2))
        self.nobs = T_eff

        k_params = X_eff.shape[1] * K * 2
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
            f"Two-Regime Threshold Vector Autoregression (TVAR)\n"
            f"Lags: {self.lags}, Delay: {self.delay}\n"
            f"Threshold Variable: {self.threshold_variable or self.target_variables[0]}\n"
            f"Estimated Threshold Gamma: {self.gamma:.4f}\n"
            f"Regime 1 Share (<= gamma): {self.regime_shares['regime_1']:.2%}\n"
            f"Regime 2 Share (> gamma):  {self.regime_shares['regime_2']:.2%}\n"
            f"AIC: {self.aic:.4f}, BIC: {self.bic:.4f}, SSR: {self.total_ssr:.4f}\n\n"
            f"Regime 1 Coefficients:\n{self.regime_1_df}\n\n"
            f"Regime 2 Coefficients:\n{self.regime_2_df}"
        )
        return summary_text

    def get_model_metrics(self):
        if self.regime_1_params is None:
            raise ValueError("Model not trained")
        return {
            "num_observations": int(self.nobs),
            "gamma": float(self.gamma),
            "delay": int(self.delay),
            "lags": int(self.lags),
            "aic": float(self.aic),
            "bic": float(self.bic),
            "ssr": float(self.total_ssr),
            "regime_1_share": float(self.regime_shares["regime_1"]),
            "regime_2_share": float(self.regime_shares["regime_2"])
        }
