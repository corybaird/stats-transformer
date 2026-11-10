import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from stats_transformer.models.timeseries.reduced_form.local_projections import LocalProjectionsModel


class _LPExtensionBase(LocalProjectionsModel):
    def _horizon_cov(self, h):
        cov_type_norm = self.cov_type.upper() if isinstance(self.cov_type, str) else str(self.cov_type)
        if cov_type_norm in ("HAC", "NW", "NEWEY_WEST", "NEWEYWEST"):
            return "HAC", {"maxlags": h + 1, **(self.cov_kwds or {})}
        if cov_type_norm == "STANDARD":
            return "nonrobust", None
        return self.cov_type, dict(self.cov_kwds) if self.cov_kwds else None

    def _fit_horizon(self, y_h, x, h):
        valid_idx = y_h.notna() & x.notna().all(axis=1)
        y_valid = y_h[valid_idx]
        x_valid = x.loc[valid_idx]
        cov_type, cov_kwds = self._horizon_cov(h)
        model = sm.OLS(y_valid, x_valid)
        return model.fit(cov_type=cov_type, cov_kwds=cov_kwds) if cov_kwds is not None else model.fit(cov_type=cov_type)

    def _irf_row(self, h, res, var, coef="effect"):
        ci = res.conf_int().loc[var]
        return {
            "horizon": h,
            "coef": float(res.params[var]),
            "std_err": float(res.bse[var]),
            "ci_lower": float(ci.loc[0]),
            "ci_upper": float(ci.loc[1]),
        }

    def compute_irf(self):
        if not hasattr(self, "irf_df") or self.irf_df is None:
            self.build_model()
        return self.irf_df

    def summary_df(self):
        return self.compute_irf()


class StateDependentLPModel(_LPExtensionBase):
    def __init__(self, target, shock_var, horizon=8, controls=None, transition_variable=None, transition="threshold", threshold=0.0, gamma=1.0, center=0.0, date_column=None, cumulative=False, lags=0, cov_type="HAC", cov_kwds=None, **kwargs):
        super().__init__(target=target, shock_var=shock_var, horizon=horizon, controls=controls, date_column=date_column, cumulative=cumulative, lags=lags, cov_type=cov_type, cov_kwds=cov_kwds, **kwargs)
        if transition not in ("threshold", "logistic"):
            raise ValueError(f"transition must be 'threshold' or 'logistic', got {transition!r}")
        if transition_variable is None:
            raise ValueError("transition_variable must be specified")
        self.transition_variable = transition_variable
        self.transition = transition
        self.threshold = threshold
        self.gamma = gamma
        self.center = center
        self.irf_regime1_df = None
        self.irf_regime2_df = None

    def _get_required_columns(self):
        return super()._get_required_columns() + [self.transition_variable]

    def _transition_weights(self, z):
        if self.transition == "threshold":
            return (z > self.threshold).astype(float)
        return 1.0 / (1.0 + np.exp(-self.gamma * (z - self.center)))

    def build_model(self):
        if self.date_column and self.date_column in self.df_clean.columns:
            self.df_clean = self.df_clean.sort_values(self.date_column)
        z = self.df_clean[self.transition_variable]
        w = self._transition_weights(z)
        n_r1 = int((w > 1e-12).sum())
        n_r2 = int((1.0 - w > 1e-12).sum())
        if n_r1 == 0 or n_r2 == 0:
            raise ValueError(f"Transition variable {self.transition_variable!r} leaves an empty regime (r1={n_r1}, r2={n_r2})")
        x = sm.add_constant(self.df_clean[[self.shock_var] + self.controls])
        if self.lags > 0:
            for lag in range(1, self.lags + 1):
                x[f"{self.shock_var}_lag{lag}"] = self.df_clean[self.shock_var].shift(lag)
                for c in self.controls:
                    x[f"{c}_lag{lag}"] = self.df_clean[c].shift(lag)
        x1 = x.mul(w, axis=0).add_suffix("_r1")
        x2 = x.mul(1.0 - w, axis=0).add_suffix("_r2")
        xi = pd.concat([x1, x2], axis=1)
        y = self.df_clean[self.target]
        rows1, rows2 = [], []
        for h in range(self.horizon + 1):
            y_h = self._target_at_horizon(y, h)
            res = self._fit_horizon(y_h, xi, h)
            for suffix, rows in (("_r1", rows1), ("_r2", rows2)):
                var = f"{self.shock_var}{suffix}"
                ci = res.conf_int().loc[var]
                rows.append({
                    "horizon": h,
                    "effect": float(res.params[var]),
                    "stderr": float(res.bse[var]),
                    "lower_ci": float(ci.loc[0]),
                    "upper_ci": float(ci.loc[1]),
                    "pvalue": float(res.pvalues[var]),
                })
        self.irf_regime1_df = pd.DataFrame(rows1)
        self.irf_regime2_df = pd.DataFrame(rows2)
        self.irf_df = pd.concat([
            self.irf_regime1_df.assign(regime=1),
            self.irf_regime2_df.assign(regime=2),
        ], ignore_index=True)

    def compute_irf(self):
        if self.irf_df is None:
            self.build_model()
        return self.irf_df

    def get_model_metrics(self):
        r1 = self.irf_regime1_df
        r2 = self.irf_regime2_df
        return {
            "horizon": self.horizon,
            "transition": self.transition,
            "transition_variable": self.transition_variable,
            "irf_r1_h0": float(r1["effect"].iloc[0]),
            "irf_r2_h0": float(r2["effect"].iloc[0]),
        }

    def get_summary(self):
        return self.compute_irf().to_string()


class SmoothLPModel(_LPExtensionBase):
    def __init__(self, target, shock_var, horizon=8, controls=None, penalty_lambda=None, n_basis=4, date_column=None, cumulative=False, lags=0, **kwargs):
        super().__init__(target=target, shock_var=shock_var, horizon=horizon, controls=controls, date_column=date_column, cumulative=cumulative, lags=lags, **kwargs)
        self.penalty_lambda = penalty_lambda
        self.n_basis = min(n_basis, horizon + 1)
        self.selected_lambda = None
        self.irf_df = None

    def _basis_matrix(self):
        H = self.horizon
        grid = np.zeros(1) if H == 0 else np.arange(H + 1) / H
        return np.vander(grid, N=self.n_basis, increasing=True)

    def _second_difference(self, H):
        D = np.zeros((max(H - 1, 1), H + 1))
        for i in range(H - 1):
            D[i, i] = 1.0
            D[i, i + 1] = -2.0
            D[i, i + 2] = 1.0
        return D

    def build_model(self):
        if self.date_column and self.date_column in self.df_clean.columns:
            self.df_clean = self.df_clean.sort_values(self.date_column)
        H = self.horizon
        B = self._basis_matrix()
        n_basis = self.n_basis
        regressors = [self.shock_var] + self.controls
        x_base = self.df_clean[regressors]
        if self.lags > 0:
            for lag in range(1, self.lags + 1):
                x_base[f"{self.shock_var}_lag{lag}"] = self.df_clean[self.shock_var].shift(lag)
                for c in self.controls:
                    x_base[f"{c}_lag{lag}"] = self.df_clean[c].shift(lag)
        x_base = sm.add_constant(x_base)
        smooth_cols = list(x_base.columns[1:])
        n_smooth = len(smooth_cols)
        y = self.df_clean[self.target]
        z_blocks, y_blocks, dummy_blocks = [], [], []
        for h in range(H + 1):
            y_h = self._target_at_horizon(y, h)
            valid_idx = y_h.notna() & x_base.notna().all(axis=1)
            xv = x_base.loc[valid_idx].drop(columns="const")
            z_blocks.append(np.kron(xv.values, B[h]))
            dummy = np.zeros((int(valid_idx.sum()), H + 1))
            dummy[:, h] = 1.0
            dummy_blocks.append(dummy)
            y_blocks.append(y_h[valid_idx].values)
        Z = np.hstack([np.vstack(z_blocks), np.vstack(dummy_blocks)])
        y_stack = np.concatenate(y_blocks)
        n_obs = len(y_stack)
        ZTZ = Z.T @ Z
        ZTy = Z.T @ y_stack
        # Horizon-specific intercepts stay unpenalized; every coefficient path is
        # smoothed with a second-difference penalty on the IRF across horizons.
        P_smooth = np.kron(np.eye(n_smooth), B.T @ self._second_difference(H).T @ self._second_difference(H) @ B)
        P = np.block([[P_smooth, np.zeros((P_smooth.shape[0], H + 1))], [np.zeros((H + 1, P_smooth.shape[0])), np.zeros((H + 1, H + 1))]])
        if self.penalty_lambda is None:
            self.selected_lambda, best_gcv = None, np.inf
            for lam in np.logspace(-3, 5, 33):
                A = np.linalg.inv(ZTZ + lam * P)
                resid = y_stack - Z @ (A @ ZTy)
                hat = np.einsum("ij,ji->", Z, A @ Z.T)
                gcv = float(resid @ resid) / n_obs / max(1.0 - hat / n_obs, 1e-12) ** 2
                if gcv < best_gcv:
                    best_gcv, self.selected_lambda = gcv, float(lam)
            lam = self.selected_lambda
        else:
            lam = self.penalty_lambda
        A = np.linalg.inv(ZTZ + lam * P)
        theta = A @ ZTy
        resid = y_stack - Z @ theta
        rank = np.linalg.matrix_rank(Z)
        sigma2 = float(resid @ resid) / max(n_obs - rank, 1)
        V = sigma2 * A @ ZTZ @ A
        path = np.zeros((H + 1, n_smooth))
        se = np.zeros((H + 1, n_smooth))
        for h in range(H + 1):
            for j in range(n_smooth):
                l_vec = np.zeros(n_smooth * n_basis + (H + 1))
                l_vec[j * n_basis:(j + 1) * n_basis] = B[h]
                path[h, j] = float(l_vec @ theta)
                se[h, j] = np.sqrt(max(float(l_vec @ V @ l_vec), 0.0))
        self.irf_df = pd.DataFrame({
            "horizon": range(H + 1),
            "coef": path[:, 0],
            "std_err": se[:, 0],
            "ci_lower": path[:, 0] - 1.96 * se[:, 0],
            "ci_upper": path[:, 0] + 1.96 * se[:, 0],
        })
        self.irf_paths = pd.DataFrame(path, columns=smooth_cols)
        self.model = None

    def get_model_metrics(self):
        irf = self.compute_irf()
        return {
            "horizon": self.horizon,
            "penalty_lambda": self.penalty_lambda if self.penalty_lambda is not None else self.selected_lambda,
            "irf_h0": float(irf["coef"].iloc[0]),
            "irf_h_max": float(irf["coef"].iloc[-1]),
        }

    def get_summary(self):
        return self.compute_irf().to_string()


class QuantileLPModel(_LPExtensionBase):
    def __init__(self, target, shock_var, horizon=8, controls=None, quantiles=(0.1, 0.5, 0.9), date_column=None, cumulative=False, lags=0, **kwargs):
        super().__init__(target=target, shock_var=shock_var, horizon=horizon, controls=controls, date_column=date_column, cumulative=cumulative, lags=lags, **kwargs)
        if not quantiles or not all(0.0 < q < 1.0 for q in quantiles):
            raise ValueError(f"quantiles must be non-empty values strictly between 0 and 1, got {quantiles!r}")
        self.quantiles = tuple(float(q) for q in quantiles)
        self.irf_by_quantile = {}
        self.irf_df = None

    def build_model(self):
        if self.date_column and self.date_column in self.df_clean.columns:
            self.df_clean = self.df_clean.sort_values(self.date_column)
        y = self.df_clean[self.target]
        x = self._build_design_matrix()
        frames = []
        for q in self.quantiles:
            rows = []
            for h in range(self.horizon + 1):
                y_h = self._target_at_horizon(y, h)
                valid_idx = y_h.notna() & x.notna().all(axis=1)
                res = QuantReg(y_h[valid_idx], x.loc[valid_idx]).fit(q=q, max_iter=5000)
                row = self._irf_row(h, res, self.shock_var)
                row["quantile"] = q
                rows.append(row)
                frames.append(row)
            self.irf_by_quantile[q] = pd.DataFrame(rows)
        self.irf_df = pd.DataFrame(frames)[["horizon", "quantile", "coef", "std_err", "ci_lower", "ci_upper"]]

    def compute_irf(self):
        if self.irf_df is None:
            self.build_model()
        return self.irf_df

    def get_model_metrics(self):
        median_key = min(self.quantiles, key=lambda q: abs(q - 0.5))
        irf = self.irf_by_quantile[median_key]
        return {
            "horizon": self.horizon,
            "quantiles": list(self.quantiles),
            "irf_h0": float(irf["coef"].iloc[0]),
            "irf_h_max": float(irf["coef"].iloc[-1]),
        }

    def get_summary(self):
        return self.compute_irf().to_string()
