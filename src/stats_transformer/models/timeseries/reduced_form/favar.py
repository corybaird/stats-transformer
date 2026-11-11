import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
from stats_transformer.models.base import ModelBase

class FAVARModel(ModelBase):
    # Factor-Augmented VAR (Bernanke, Boivin and Eliasz, 2005)
    _is_multivariate = True

    def __init__(self, observed_variables=None, panel_variables=None, slow_variables=None, n_factors=3, maxlags=2, date_column=None, **kwargs):
        super().__init__(**kwargs)
        model_params = self.params.get("model", {})
        self.observed_variables = observed_variables or model_params.get("observed_variables") or model_params.get("target_variables") or model_params.get("independent_variables", [])
        self.panel_variables = panel_variables or model_params.get("panel_variables") or model_params.get("independent_variables", [])
        self.target_variables = list(self.observed_variables)
        self.slow_variables = slow_variables or model_params.get("slow_variables")
        self.n_factors = model_params.get("n_factors", n_factors)
        self.maxlags = model_params.get("maxlags", maxlags)
        self.date_column = date_column or model_params.get("date_column")
        self.scaler = StandardScaler()
        self.slow_scaler = StandardScaler()
        self.state_names = []
        self.loadings = None

    def _get_required_columns(self):
        return list(self.observed_variables) + list(self.panel_variables)

    def _purge_factors(self, C_hat):
        # BBE two-step rotation: regress the panel common component on slow
        # factors and the observed variables, then remove the Y_t component so
        # the rotated factors are purged of the observables' direct influence
        Y = self.df_clean[self.observed_variables].to_numpy(dtype=float)
        F_hat = C_hat
        if self.slow_variables:
            slow = self.slow_scaler.fit_transform(self.df_clean[self.slow_variables].to_numpy(dtype=float))
            n_slow = min(self.n_factors, len(self.slow_variables))
            F_slow = PCA(n_components=n_slow, svd_solver="full").fit_transform(slow)
            design = np.column_stack([np.ones(len(C_hat)), F_slow, Y])
            coef, *_ = np.linalg.lstsq(design, C_hat, rcond=None)
            b_Y = coef[1 + n_slow:]
            F_hat = C_hat - Y @ b_Y
        return F_hat

    def _estimate_loadings(self, F_hat, X_std):
        # Observation loading matrix mapping the state vector [F_hat, Y] to the
        # standardized panel: X_std ~ [F_hat, Y] @ B, loadings = B.T
        design = np.column_stack([F_hat, self.df_clean[self.observed_variables].to_numpy(dtype=float)])
        B, *_ = np.linalg.lstsq(design, X_std, rcond=None)
        return B.T

    def _panel_irfs(self, steps):
        # Orthogonalized (Cholesky) state IRFs; observables ordered last, so
        # latent factors do not respond to the observable shock on impact
        orth_irfs = self.model.irf(steps).orth_irfs
        panel_irfs = np.einsum("ik,hkj->hij", self.loadings, orth_irfs)
        scale = self.scaler.scale_
        return panel_irfs * scale[None, :, None]

    def build_model(self, drop_na=True):
        if self.df_clean is None:
            raise ValueError("No cleaned data available")
        if self.slow_variables:
            missing_slow = [c for c in self.slow_variables if c not in self.panel_variables]
            if missing_slow:
                raise ValueError(f"Slow variables not in panel: {missing_slow}")
        if self.n_factors >= len(self.panel_variables):
            raise ValueError(f"n_factors ({self.n_factors}) must be smaller than the number of panel variables ({len(self.panel_variables)})")

        if self.date_column and self.date_column in self.df_clean.columns:
            self.df_clean = self.df_clean.sort_values(self.date_column)

        self.y = self.df_clean[self.observed_variables]
        X_std = self.scaler.fit_transform(self.df_clean[self.panel_variables].to_numpy(dtype=float))
        C_hat = PCA(n_components=self.n_factors, svd_solver="full").fit_transform(X_std)
        F_hat = self._purge_factors(C_hat)

        self.state_names = [f"factor_{i + 1}" for i in range(self.n_factors)] + list(self.observed_variables)
        self.target_variables = list(self.state_names)
        Z = pd.DataFrame(np.column_stack([F_hat, self.df_clean[self.observed_variables].to_numpy(dtype=float)]), index=self.df_clean.index, columns=self.state_names)

        self.loadings = self._estimate_loadings(F_hat, X_std)
        self.var_spec = VAR(Z)
        self.model = self.var_spec.fit(maxlags=self.maxlags, trend="c")
        return self.model

    def fit(self, df, drop_na=True):
        required_cols = self._get_required_columns()
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        self.df_clean = df.copy().replace([np.inf, -np.inf], np.nan)
        if drop_na:
            self.df_clean = self.df_clean.dropna(subset=required_cols)

        self.build_model(drop_na=False)
        return self.get_model_metrics()

    def irf(self, steps=12):
        if self.model is None or self.loadings is None:
            raise ValueError("Model not trained")
        panel_irfs = self._panel_irfs(steps)
        columns = pd.MultiIndex.from_product([self.panel_variables, self.state_names], names=["response", "shock"])
        return pd.DataFrame(panel_irfs.reshape(steps + 1, -1), index=pd.Index(range(steps + 1), name="horizon"), columns=columns)

    def fevd(self, steps=12):
        if self.model is None or self.loadings is None:
            raise ValueError("Model not trained")
        panel_irfs = self._panel_irfs(steps)
        contrib = np.cumsum(panel_irfs ** 2, axis=0)[-1]
        shares = contrib / contrib.sum(axis=1, keepdims=True)
        return pd.DataFrame(shares, index=list(self.panel_variables), columns=self.state_names)

    def get_summary(self):
        if self.model is None:
            raise ValueError("Model not trained")
        loading_text = ", ".join(list(self.panel_variables))
        observed_text = ", ".join(list(self.observed_variables))
        return f"FAVAR Model\nStates: {', '.join(self.state_names)}\nPanel ({len(self.panel_variables)}): {loading_text}\nObserved: {observed_text}\nSlow variables: {list(self.slow_variables) if self.slow_variables else None}\n\nVAR Summary:\n{self.model.summary()}"

    def get_model_metrics(self):
        if self.model is None:
            raise ValueError("Model not trained")
        return {
            "aic": float(self.model.aic),
            "bic": float(self.model.bic),
            "hqic": float(self.model.hqic),
            "fpe": float(self.model.fpe),
            "num_observations": int(self.model.nobs),
            "n_factors": int(self.n_factors),
            "n_panel_variables": len(self.panel_variables),
            "n_slow_variables": len(self.slow_variables) if self.slow_variables else 0,
        }
