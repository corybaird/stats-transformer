import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM
from stats_transformer.models.base import ModelBase


class TVECMModel(ModelBase):
    """
    Two-Regime Threshold Vector Error Correction Model (TVECM).
    Combines cointegration with non-linear threshold regime switching
    on the error correction term (Hansen & Seo 2002, Stigler 2010).
    """

    _is_multivariate = True

    def __init__(self, target_variables=None, date_column=None, k_ar_diff=1, coint_rank=1, delay=1, trim=0.15, gamma=None, deterministic="n", **kwargs):
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
        self.delay = delay
        self.trim = trim
        self.gamma = gamma
        self.deterministic = deterministic

        self.beta = None
        self.regime_1_params = None
        self.regime_2_params = None
        self.regime_1_cov = None
        self.regime_2_cov = None
        self.regime_shares = None
        self.residuals = None
        self.fittedvalues = None

    def _get_required_columns(self):
        cols = list(self.target_variables)
        if self.date_column:
            cols.append(self.date_column)
        return cols

    def _construct_tvecm_design_matrices(self, y_levels, ect_series):
        T, K = y_levels.shape
        p_diff = self.k_ar_diff
        d = self.delay
        max_lag = max(p_diff + 1, d)

        dY = np.diff(y_levels, axis=0)
        T_d = len(dY)

        dY_eff = dY[max_lag - 1 :]
        T_eff = len(dY_eff)

        ect_eff = ect_series[max_lag - 1 - d : T - 1 - d]

        X_list = [np.ones((T_eff, 1)), ect_eff.reshape((T_eff, -1))]
        for lag in range(1, p_diff + 1):
            X_list.append(dY[max_lag - 1 - lag : T_d - lag])

        X_eff = np.hstack(X_list)
        return dY_eff, X_eff, ect_eff

    def _estimate_regime_ols(self, Y, X):
        try:
            params = np.linalg.lstsq(X, Y, rcond=None)[0]
            resids = Y - X @ params
            cov = (resids.T @ resids) / max(1, len(Y) - X.shape[1])
            return params, resids, cov
        except Exception:
            return None, None, None

    def _grid_search_threshold(self, dY_eff, X_eff, ect_eff):
        T_eff = len(dY_eff)
        num_regressors = X_eff.shape[1]
        min_obs = max(num_regressors + 2, int(np.floor(self.trim * T_eff)))

        unique_th = np.sort(np.unique(ect_eff))
        lower_idx = int(np.floor(self.trim * len(unique_th)))
        upper_idx = int(np.ceil((1.0 - self.trim) * len(unique_th)))

        candidate_gammas = unique_th[lower_idx:upper_idx]
        if len(candidate_gammas) == 0:
            candidate_gammas = [np.median(ect_eff)]

        best_gamma = candidate_gammas[0]
        best_ssr = np.inf

        for cand in candidate_gammas:
            mask1 = ect_eff <= cand
            mask2 = ~mask1

            if np.sum(mask1) < min_obs or np.sum(mask2) < min_obs:
                continue

            Y1, X1 = dY_eff[mask1], X_eff[mask1]
            Y2, X2 = dY_eff[mask2], X_eff[mask2]

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
        y_levels = y_df.values.astype(float)
        T, K = y_levels.shape

        self.vecm_baseline = VECM(y_df, k_ar_diff=self.k_ar_diff, coint_rank=self.coint_rank, deterministic=self.deterministic).fit()
        self.beta = self.vecm_baseline.beta

        ect = y_levels @ self.beta
        if ect.ndim > 1:
            ect_series = ect[:, 0]
        else:
            ect_series = ect

        dY_eff, X_eff, ect_eff = self._construct_tvecm_design_matrices(y_levels, ect_series)
        T_eff = len(dY_eff)

        if self.gamma is None:
            self.gamma = self._grid_search_threshold(dY_eff, X_eff, ect_eff)

        mask1 = ect_eff <= self.gamma
        mask2 = ~mask1

        if np.sum(mask1) == 0 or np.sum(mask2) == 0:
            mask1 = np.ones(T_eff, dtype=bool)
            mask2 = np.ones(T_eff, dtype=bool)

        p1, r1, cov1 = self._estimate_regime_ols(dY_eff[mask1], X_eff[mask1])
        p2, r2, cov2 = self._estimate_regime_ols(dY_eff[mask2], X_eff[mask2])

        self.regime_1_params = p1
        self.regime_2_params = p2
        self.regime_1_cov = cov1
        self.regime_2_cov = cov2
        self.regime_shares = {
            "regime_1": float(np.mean(mask1)),
            "regime_2": float(np.mean(mask2))
        }

        fitted = np.zeros_like(dY_eff)
        fitted[mask1] = X_eff[mask1] @ p1
        fitted[mask2] = X_eff[mask2] @ p2
        resids = dY_eff - fitted

        diff_col_names = [f"D.{v}" for v in self.target_variables]
        self.fittedvalues = pd.DataFrame(fitted, columns=diff_col_names)
        self.residuals = pd.DataFrame(resids, columns=diff_col_names)
        self.total_ssr = float(np.sum(resids ** 2))
        self.nobs = T_eff

        k_params = X_eff.shape[1] * K * 2
        self.aic = float(T_eff * np.log(self.total_ssr / T_eff) + 2 * k_params)
        self.bic = float(T_eff * np.log(self.total_ssr / T_eff) + np.log(T_eff) * k_params)

        var_names = ["const", "ECT"]
        for lag in range(1, self.k_ar_diff + 1):
            for v in self.target_variables:
                var_names.append(f"LD{lag}.{v}")

        self.regime_1_df = pd.DataFrame(self.regime_1_params, index=var_names, columns=diff_col_names)
        self.regime_2_df = pd.DataFrame(self.regime_2_params, index=var_names, columns=diff_col_names)
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
            f"Two-Regime Threshold Vector Error Correction Model (TVECM)\n"
            f"Cointegration Rank: {self.coint_rank}, Difference Lags: {self.k_ar_diff}\n"
            f"Estimated Beta Cointegrating Vector:\n{self.beta}\n"
            f"Estimated Threshold Gamma on ECT: {self.gamma:.4f}\n"
            f"Regime 1 Share (ECT <= gamma): {self.regime_shares['regime_1']:.2%}\n"
            f"Regime 2 Share (ECT > gamma):  {self.regime_shares['regime_2']:.2%}\n"
            f"AIC: {self.aic:.4f}, BIC: {self.bic:.4f}, SSR: {self.total_ssr:.4f}\n\n"
            f"Regime 1 Coefficients (ECT <= gamma):\n{self.regime_1_df}\n\n"
            f"Regime 2 Coefficients (ECT > gamma):\n{self.regime_2_df}"
        )
        return summary_text

    def get_model_metrics(self):
        if self.regime_1_params is None:
            raise ValueError("Model not trained")
        return {
            "num_observations": int(self.nobs),
            "gamma": float(self.gamma),
            "coint_rank": int(self.coint_rank),
            "k_ar_diff": int(self.k_ar_diff),
            "aic": float(self.aic),
            "bic": float(self.bic),
            "ssr": float(self.total_ssr),
            "regime_1_share": float(self.regime_shares["regime_1"]),
            "regime_2_share": float(self.regime_shares["regime_2"])
        }
