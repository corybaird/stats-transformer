import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from stats_transformer.models.base import ModelBase

class DynamicFactorModel(ModelBase):
    _is_multivariate = True

    def __init__(self, target_variables=None, date_column=None, n_factors=1, factor_lags=1, max_iter=100, tol=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.target_variables = target_variables or []
        self.date_column = date_column
        self.time_column = date_column
        self.n_factors = n_factors
        self.factor_lags = factor_lags
        self.max_iter = max_iter
        self.tol = tol
        self.scaler = StandardScaler()
        self.loglikelihood_history = []
        self.n_iter_ = 0
        self.converged_ = False

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

        X = self.df_clean[self.target_variables].to_numpy(dtype=float)
        X_std = self.scaler.fit_transform(X)
        T, N = X_std.shape
        k, p = self.n_factors, self.factor_lags

        Lambda, A, Q, R = self._initialize_params(X_std, T, N, k, p)

        prev_ll = -np.inf
        self.loglikelihood_history = []
        for iteration in range(self.max_iter):
            smoothed_means, smoothed_covs, lag1_covs, ll = self._e_step(X_std, Lambda, A, Q, R, T, N, k, p)
            Lambda, A, Q, R = self._m_step(X_std, smoothed_means, smoothed_covs, lag1_covs, T, N, k, p)
            self.loglikelihood_history.append(ll)
            self.n_iter_ = iteration + 1
            if abs(ll - prev_ll) < self.tol * (1 + abs(prev_ll)):
                self.converged_ = True
                prev_ll = ll
                break
            prev_ll = ll

        smoothed_means, smoothed_covs, lag1_covs, ll = self._e_step(X_std, Lambda, A, Q, R, T, N, k, p)
        self.loglikelihood_history.append(ll)

        self.Lambda = Lambda
        self.A = A
        self.Q = Q
        self.R = R
        self.factors_ = smoothed_means[:, :k]
        self.final_loglikelihood = ll
        self.model = {"Lambda": Lambda, "A": A, "Q": Q, "R": R}
        return self.model

    def _initialize_params(self, X_std, T, N, k, p):
        pca = PCA(n_components=k)
        factors_pca = pca.fit_transform(X_std)
        Lambda = pca.components_.T.copy()

        state_dim = k * p
        A = np.zeros((state_dim, state_dim))
        if T > p:
            Y = factors_pca[p:, :]
            Z = np.column_stack([factors_pca[p - lag - 1: T - lag - 1, :] for lag in range(p)])
            coef, *_ = np.linalg.lstsq(Z, Y, rcond=None)
            A[:k, :] = coef.T
        if p > 1:
            A[k:, : state_dim - k] = np.eye(state_dim - k)

        resid = factors_pca[p:] - Z @ coef if T > p else factors_pca
        Q_top = np.atleast_2d(np.cov(resid.T)) if resid.shape[0] > 1 else np.eye(k) * 0.1
        Q = np.zeros((state_dim, state_dim))
        Q[:k, :k] = Q_top + np.eye(k) * 1e-6

        fitted = factors_pca @ Lambda.T
        residuals = X_std - fitted
        R = np.diag(np.var(residuals, axis=0) + 1e-6)

        return Lambda, A, Q, R

    def _kalman_filter(self, X_std, Lambda, A, Q, R, T, N, k, p):
        state_dim = k * p
        Lambda_full = np.zeros((N, state_dim))
        Lambda_full[:, :k] = Lambda

        filtered_means = np.zeros((T, state_dim))
        filtered_covs = np.zeros((T, state_dim, state_dim))
        predicted_means = np.zeros((T, state_dim))
        predicted_covs = np.zeros((T, state_dim, state_dim))

        x_pred = np.zeros(state_dim)
        P_pred = np.eye(state_dim) * 1.0
        ll = 0.0

        for t in range(T):
            predicted_means[t] = x_pred
            predicted_covs[t] = P_pred

            y_t = X_std[t]
            innovation = y_t - Lambda_full @ x_pred
            S = Lambda_full @ P_pred @ Lambda_full.T + R
            S_inv = np.linalg.inv(S)

            sign, logdet = np.linalg.slogdet(S)
            ll += -0.5 * (N * np.log(2 * np.pi) + logdet + innovation @ S_inv @ innovation)

            K = P_pred @ Lambda_full.T @ S_inv
            x_filt = x_pred + K @ innovation
            P_filt = P_pred - K @ Lambda_full @ P_pred

            filtered_means[t] = x_filt
            filtered_covs[t] = P_filt

            x_pred = A @ x_filt
            P_pred = A @ P_filt @ A.T + Q

        return filtered_means, filtered_covs, predicted_means, predicted_covs, ll

    def _e_step(self, X_std, Lambda, A, Q, R, T, N, k, p):
        state_dim = k * p
        filtered_means, filtered_covs, predicted_means, predicted_covs, ll = self._kalman_filter(X_std, Lambda, A, Q, R, T, N, k, p)

        smoothed_means = np.zeros((T, state_dim))
        smoothed_covs = np.zeros((T, state_dim, state_dim))
        lag1_covs = np.zeros((T, state_dim, state_dim))

        smoothed_means[-1] = filtered_means[-1]
        smoothed_covs[-1] = filtered_covs[-1]

        for t in range(T - 2, -1, -1):
            P_pred_next = predicted_covs[t + 1]
            J = filtered_covs[t] @ A.T @ np.linalg.inv(P_pred_next)
            smoothed_means[t] = filtered_means[t] + J @ (smoothed_means[t + 1] - predicted_means[t + 1])
            smoothed_covs[t] = filtered_covs[t] + J @ (smoothed_covs[t + 1] - P_pred_next) @ J.T
            lag1_covs[t + 1] = J @ smoothed_covs[t + 1]

        return smoothed_means, smoothed_covs, lag1_covs, ll

    def _m_step(self, X_std, smoothed_means, smoothed_covs, lag1_covs, T, N, k, p):
        state_dim = k * p

        Ezz = np.zeros((state_dim, state_dim))
        for t in range(T):
            Ezz += smoothed_covs[t] + np.outer(smoothed_means[t], smoothed_means[t])
        Ezz /= T

        Ezz_k = Ezz[:k, :k]
        Exz = np.zeros((N, k))
        Exx_diag = np.zeros(N)
        for t in range(T):
            Exz += np.outer(X_std[t], smoothed_means[t, :k])
            Exx_diag += X_std[t] ** 2
        Exz /= T
        Exx_diag /= T
        Lambda_new = Exz @ np.linalg.pinv(Ezz_k)

        # R = E[x x'] - Lambda_new E[z x'], the closed-form residual variance that
        # accounts for posterior state uncertainty via Ezz (Shumway & Stoffer, 2017, S6.3).
        # Using only the point-estimate residual (x_t - Lambda z_t)^2 omits the
        # Lambda P_t Lambda' correction term and understates R, which breaks EM's
        # guaranteed monotonic increase in the observed-data log-likelihood.
        R_diag = Exx_diag - np.sum(Lambda_new * Exz, axis=1)
        R_new = np.diag(np.maximum(R_diag, 1e-6))

        A_new = np.zeros((state_dim, state_dim))
        Q_new = np.zeros((state_dim, state_dim))
        if T > 1:
            S11 = np.zeros((state_dim, state_dim))
            S10 = np.zeros((state_dim, state_dim))
            S00 = np.zeros((state_dim, state_dim))
            for t in range(1, T):
                S11 += smoothed_covs[t] + np.outer(smoothed_means[t], smoothed_means[t])
                S10 += lag1_covs[t] + np.outer(smoothed_means[t], smoothed_means[t - 1])
                S00 += smoothed_covs[t - 1] + np.outer(smoothed_means[t - 1], smoothed_means[t - 1])
            S11 /= (T - 1)
            S10 /= (T - 1)
            S00 /= (T - 1)

            A_top = S10[:k, :] @ np.linalg.pinv(S00)
            A_new[:k, :] = A_top
            if p > 1:
                A_new[k:, : state_dim - k] = np.eye(state_dim - k)

            Q_top = S11[:k, :k] - A_top @ S10[:k, :].T
            Q_new[:k, :k] = 0.5 * (Q_top + Q_top.T) + np.eye(k) * 1e-6

        return Lambda_new, A_new, Q_new, R_new

    def get_summary(self):
        if self.model is None:
            raise ValueError("Model not trained")
        return f"Dynamic Factor Model: n_factors={self.n_factors}, factor_lags={self.factor_lags}, n_iter={self.n_iter_}, converged={self.converged_}, loglikelihood={self.final_loglikelihood:.4f}"

    def get_model_metrics(self):
        if self.model is None:
            raise ValueError("Model not trained")
        return {
            "n_factors": self.n_factors,
            "factor_lags": self.factor_lags,
            "n_iter": self.n_iter_,
            "converged": self.converged_,
            "loglikelihood": float(self.final_loglikelihood),
            "num_observations": int(self.factors_.shape[0]),
        }

    def compute_factors(self):
        if self.model is None:
            raise ValueError("Model not trained")
        columns = [f"factor_{i + 1}" for i in range(self.n_factors)]
        result = pd.DataFrame(self.factors_, columns=columns)
        if self.date_column and self.date_column in self.df_clean.columns:
            result[self.date_column] = self.df_clean[self.date_column].to_numpy()
        return result

    def run(self, data):
        self.fit(data)
        return self.get_model_metadata()
