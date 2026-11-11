import numpy as np
import pandas as pd
from stats_transformer.models.base import ModelBase
from stats_transformer.models.timeseries.reduced_form.var import VARModel

class _VARXResults:
    def __init__(self, params, resid, aic, bic, nobs, df_resid):
        self.params = params
        self.resid = resid
        self.aic = float(aic)
        self.bic = float(bic)
        self.nobs = int(nobs)
        self.df_resid = int(df_resid)

class VARXModel(ModelBase):
    _is_multivariate = True

    def __init__(self, target_variables=None, exog_variables=None, independent_variables=None, date_column=None, maxlags=2, exog_lags=0, **kwargs):
        super().__init__(**kwargs)
        model_params = self.params.get("model", {})
        self.target_variables = list(target_variables or getattr(self, "target_variables", []) or model_params.get("target_variables") or [])
        self.exog_variables = list(exog_variables or model_params.get("exog_variables") or independent_variables or model_params.get("independent_variables") or [])
        self.date_column = date_column or model_params.get("date_column")
        self.maxlags = self._param("maxlags", maxlags, 2)
        self.exog_lags = self._param("exog_lags", exog_lags, 0)
        self.A_matrices = None
        self.B_matrices = None
        self.intercept = None
        self.sigma_u = None
        self.history = None
        self.resid = None
        self.coef_df = None
        self.model = None

    def _get_required_columns(self):
        cols = list(self.target_variables)
        cols += [c for c in self.exog_variables if c not in cols]
        if self.date_column and self.date_column not in cols:
            cols.append(self.date_column)
        return cols

    def _term_names(self, K, Kx):
        names = ["const"]
        names += [f"L{i}.{v}" for i in range(1, int(self.maxlags) + 1) for v in self.target_variables]
        names += [f"L{j}.{v}" for j in range(0, int(self.exog_lags) + 1) for v in self.exog_variables]
        return names

    def build_model(self, drop_na=True):
        if self.df_clean is None:
            raise ValueError("No cleaned data available")
        if self.date_column and self.date_column in self.df_clean.columns:
            self.df_clean = self.df_clean.sort_values(self.date_column)
        self.y = self.df_clean[self.target_variables]
        K = len(self.target_variables)
        Kx = len(self.exog_variables)
        p = int(self.maxlags)
        q = int(self.exog_lags)
        Yv = self.y.values.astype(float)
        T = Yv.shape[0]
        Xv = self.df_clean[self.exog_variables].values.astype(float) if Kx else np.zeros((T, 0))
        if Kx == 0:
            q = 0
            self.exog_lags = 0
        start = max(p, q)
        n = T - start
        if n <= 1:
            raise ValueError(f"Not enough observations ({T}) for maxlags={p}, exog_lags={q}")
        y_blocks = np.hstack([Yv[start - i:T - i] for i in range(1, p + 1)]) if p else np.zeros((n, 0))
        x_blocks = np.hstack([Xv[start - j:T - j] for j in range(0, q + 1)]) if Kx else np.zeros((n, 0))
        rhs = np.column_stack([np.ones(n), y_blocks, x_blocks])
        lhs = Yv[start:]
        beta, _, _, _ = np.linalg.lstsq(rhs, lhs, rcond=None)
        resid = lhs - rhs @ beta
        k = rhs.shape[1]
        dof = n - k
        if dof <= 0:
            raise ValueError(f"Not enough degrees of freedom: {n} observations for {k} regressors per equation")
        sigma_u = resid.T @ resid / dof
        self.A_matrices = np.array([beta[1 + (i - 1) * K:1 + i * K, :].T for i in range(1, p + 1)]) if p else np.zeros((0, K, K))
        self.B_matrices = [beta[1 + p * K + j * Kx:1 + p * K + (j + 1) * Kx, :].T for j in range(q + 1)]
        self.intercept = beta[0].copy()
        self.sigma_u = sigma_u
        self.resid = resid
        self.history = Yv[-p:]
        self.coef_df = pd.DataFrame(beta, index=self._term_names(K, Kx), columns=self.target_variables)
        sigma_mle = resid.T @ resid / n
        sign, ld = np.linalg.slogdet(sigma_mle)
        if sign <= 0:
            raise ValueError("Residual covariance is singular")
        free_params = p * K * K + K + (q + 1) * K * Kx
        aic = ld + 2.0 * free_params / n
        bic = ld + np.log(n) * free_params / n
        self.model = _VARXResults(params=self.coef_df, resid=resid, aic=aic, bic=bic, nobs=n, df_resid=dof)
        return self.model

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
        if self.model is None:
            raise ValueError("Model not trained")
        sigma_df = pd.DataFrame(self.sigma_u, index=self.target_variables, columns=self.target_variables)
        lines = [
            f"VARX Model (target_lags={self.maxlags}, exog_lags={self.exog_lags})",
            "",
            self.coef_df.round(6).to_string(),
            "",
            f"Observations: {self.model.nobs}",
            f"AIC: {self.model.aic:.4f}",
            f"BIC: {self.model.bic:.4f}",
            "",
            "Residual covariance:",
            sigma_df.round(6).to_string(),
        ]
        return "\n".join(lines)

    def get_model_metrics(self):
        if self.model is None:
            raise ValueError("Model not trained")
        return {
            "aic": self.model.aic,
            "bic": self.model.bic,
            "num_observations": self.model.nobs,
            "maxlags": int(self.maxlags),
            "exog_lags": int(self.exog_lags),
        }

    def dynamic_multipliers(self, steps=12):
        if self.A_matrices is None:
            raise ValueError("Model not trained")
        K = len(self.target_variables)
        Kx = len(self.exog_variables)
        if Kx == 0:
            raise ValueError("Model has no exogenous variables")
        mults = np.zeros((steps, K, Kx))
        for h in range(steps):
            m = np.zeros((K, Kx))
            if h <= self.exog_lags:
                m = m + self.B_matrices[h]
            for i in range(1, min(len(self.A_matrices), h) + 1):
                m = m + self.A_matrices[i - 1] @ mults[h - i]
            mults[h] = m
        if Kx == 1:
            columns = list(self.target_variables)
        else:
            columns = [f"{tv}.{ex}" for ex in self.exog_variables for tv in self.target_variables]
        return pd.DataFrame(mults.reshape(steps, K * Kx), columns=columns, index=pd.RangeIndex(steps, name="h"))

    def forecast(self, steps, exog_future=None):
        if self.A_matrices is None:
            raise ValueError("Model not trained")
        Kx = len(self.exog_variables)
        p = len(self.A_matrices)
        q = self.exog_lags if Kx else 0
        if exog_future is None:
            # ponytail: hold last exog value constant as the future path; pass exog_future for a real path
            Xf = np.zeros((steps, Kx)) if Kx else np.zeros((steps, 0))
            if Kx:
                Xf = np.repeat(self.df_clean[self.exog_variables].values[-1:], steps, axis=0)
        else:
            Xf = exog_future.values if isinstance(exog_future, pd.DataFrame) else np.asarray(exog_future, dtype=float)
            Xf = np.atleast_2d(Xf)
            if Kx and Xf.shape != (steps, Kx):
                raise ValueError(f"exog_future must have shape ({steps}, {Kx}), got {Xf.shape}")
        Yv = self.y.values.astype(float)
        prev = [Yv[-i] for i in range(p, 0, -1)]
        out = np.zeros((steps, len(self.target_variables)))
        for h in range(steps):
            val = self.intercept.copy()
            for i in range(1, p + 1):
                val = val + self.A_matrices[i - 1] @ prev[-i]
            for j in range(0, q + 1):
                if h - j >= 0:
                    val = val + self.B_matrices[j] @ Xf[h - j]
            out[h] = val
            prev.append(val)
        return pd.DataFrame(out, columns=self.target_variables, index=pd.RangeIndex(1, steps + 1))

class ConditionalForecaster:

    def __init__(self, model, active_shocks=None, var_names=None):
        self.active_shocks = active_shocks
        self.A_matrices, self.intercept, self.sigma_u, self.history, self.var_names = self._extract_inputs(model, var_names)

    def _extract_inputs(self, model, var_names):
        if isinstance(model, VARModel):
            res = model.model
            if res is None:
                raise ValueError("Model not trained")
            A = np.asarray(res.coefs, dtype=float)
            history = model.y.values[-A.shape[0]:].astype(float)
            return A, np.asarray(res.intercept, dtype=float), np.asarray(res.sigma_u, dtype=float), history, list(model.target_variables)
        if isinstance(model, VARXModel):
            if model.A_matrices is None:
                raise ValueError("Model not trained")
            return np.asarray(model.A_matrices, dtype=float), np.asarray(model.intercept, dtype=float), np.asarray(model.sigma_u, dtype=float), np.asarray(model.history, dtype=float), list(model.target_variables)
        if isinstance(model, (tuple, list)) and len(model) == 4:
            A, intercept, sigma_u, history = [np.asarray(part, dtype=float) for part in model]
            if A.ndim == 2:
                A = A[None, :, :]
            K = A.shape[1]
            names = list(var_names) if var_names else [f"y{i + 1}" for i in range(K)]
            return A, intercept, sigma_u, history, names
        raise TypeError("model must be a fitted VARModel, VARXModel or (A_matrices, intercept, sigma_u, history) tuple")

    def _resolve_var(self, var_key, K):
        if isinstance(var_key, str):
            if var_key in self.var_names:
                return self.var_names.index(var_key)
            raise ValueError(f"Unknown variable {var_key!r}; known: {self.var_names}")
        idx = int(var_key)
        if not 0 <= idx < K:
            raise ValueError(f"Variable index {idx} out of range for {K} endogenous variables")
        return idx

    def _parse_constraints(self, constraints, K):
        items = {}
        if isinstance(constraints, pd.DataFrame):
            for var_key in constraints.index:
                for col in constraints.columns:
                    val = constraints.loc[var_key, col]
                    if pd.notna(val):
                        items[(self._resolve_var(var_key, K), int(col))] = float(val)
        elif isinstance(constraints, dict):
            for var_key, path in constraints.items():
                vi = self._resolve_var(var_key, K)
                for h, val in (path or {}).items():
                    h = int(h)
                    if h < 1:
                        raise ValueError(f"Constraint step {h} invalid: steps must be 1-based and positive")
                    items[(vi, h)] = float(val)
        else:
            raise TypeError("constraints must be a dict {var: {step: value}} or a DataFrame (rows=variables, columns=steps)")
        return items

    def _active_indices(self, K, H):
        if not self.active_shocks:
            return list(range(H * K))
        idxs = []
        for shock in self.active_shocks:
            vi = self._resolve_var(shock, K)
            idxs.extend(h * K + vi for h in range(H))
        return sorted(set(idxs))

    def _baseline(self, H):
        p = self.A_matrices.shape[0]
        hist = list(np.asarray(self.history, dtype=float)[-p:])
        out = np.zeros((H, len(self.var_names)))
        for h in range(H):
            val = np.asarray(self.intercept, dtype=float).copy()
            for i in range(1, p + 1):
                val = val + self.A_matrices[i - 1] @ hist[-i]
            out[h] = val
            hist.append(val)
        return out

    def _ma_coefficients(self, H):
        A = self.A_matrices
        K = A.shape[1]
        psi = [np.eye(K)]
        for s in range(1, H):
            acc = np.zeros((K, K))
            for i in range(1, min(A.shape[0], s) + 1):
                acc = acc + A[i - 1] @ psi[s - i]
            psi.append(acc)
        return psi

    def run(self, constraints, horizon=None):
        K = self.A_matrices.shape[1]
        cons = self._parse_constraints(constraints, K)
        max_h = max((h for _, h in cons), default=0)
        H = max(max_h, int(horizon) if horizon else 0)
        if H <= 0:
            raise ValueError("Constraints are empty: provide constraints or a horizon")
        base = self._baseline(H)
        psi = self._ma_coefficients(H)
        n_c = len(cons)
        C = np.zeros((n_c, H * K))
        r = np.zeros(n_c)
        for row, ((vi, h), target) in enumerate(sorted(cons.items())):
            block = h - 1
            C[row, block * K + vi] = 1.0
            r[row] = target - base[block, vi]
        M = np.zeros((H * K, H * K))
        for h in range(H):
            for s in range(h + 1):
                M[h * K:(h + 1) * K, s * K:(s + 1) * K] = psi[h - s]
        CM = C @ M
        s_idx = self._active_indices(K, H)
        T = CM[:, s_idx]
        # Minimum-norm structural shocks in the whitened (unit-variance) metric:
        # eps* = R'(R R')^-1 (r - base); pinv keeps the solve stable when T T' is rank-deficient
        eps_S = T.T @ np.linalg.pinv(T @ T.T) @ r
        eps_full = np.zeros(H * K)
        eps_full[s_idx] = eps_S
        cond = base.copy()
        for h in range(H):
            dev = np.zeros(K)
            for s in range(h + 1):
                dev = dev + psi[h - s] @ eps_full[s * K:(s + 1) * K]
            cond[h] = base[h] + dev
        forecast_df = pd.DataFrame(cond, columns=self.var_names, index=pd.RangeIndex(1, H + 1))
        shock_names = [f"{v}_shock" for v in self.var_names]
        shocks_df = pd.DataFrame(eps_full.reshape(H, K), columns=shock_names, index=pd.RangeIndex(1, H + 1))
        return forecast_df, shocks_df
